# Face alignment and crop demo
# Uses MTCNN, FaceBoxes or Retinaface as a face detector;
# Support different backbones, include PFLD, MobileFaceNet, MobileNet;
# Retinaface+MobileFaceNet gives the best peformance
# Cunjian Chen (ccunjian@gmail.com), Feb. 2021

from __future__ import division
import argparse
import torch
import os
import cv2
import numpy as np
from common.utils import BBox,drawLandmark,drawLandmark_multiple
from models.basenet import MobileNet_GDConv
from models.pfld_compressed import PFLDInference
from models.mobilefacenet import MobileFaceNet
from FaceBoxes import FaceBoxes
from Retinaface import Retinaface
from PIL import Image
import matplotlib.pyplot as plt
from MTCNN import detect_faces
import glob
import time
from utils.align_trans import get_reference_facial_points, warp_and_crop_face
from scipy.spatial import ConvexHull
from skimage import draw
import tqdm
import sys, traceback
import imageio.v3 as iio

parser = argparse.ArgumentParser(description='PyTorch face landmark')
# Datasets
parser.add_argument('--backbone', default='PFLD', type=str,
                    help='choose which backbone network to use: MobileNet, PFLD, MobileFaceNet')
parser.add_argument('--detector', default='Retinaface', type=str,
                    help='choose which face detector to use: MTCNN, FaceBoxes, Retinaface')

args = parser.parse_args()
mean = np.asarray([ 0.485, 0.456, 0.406 ])
std = np.asarray([ 0.229, 0.224, 0.225 ])

crop_size= 112
scale = crop_size / 112.
reference = get_reference_facial_points(default_square = True) * scale

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

def extract_video_frame(
    video_path, frame_path=None, frame_range=(306, 2135), downsample=True
):
    """Extract frames from the given video

    Extract each frame from the given video file and store them into '.jpg' format. It
    extracts every frame of the video. If the given frame path exsits, it overwrites
    the contents if users choose that.

    Args:
            video_path (str): Required. The path of video file.

            frame_path (str): Required. The path to store extracted frames. If the path exists, it tries to
                                    remove it by asking the user.

    Raises:
            OSError: If the given video path is incorrect, or the video cannot be opened by
                            Opencv.
            ValueError: If the given specified range out of range
    """

    frames = []
    count = 0
    for idx, frame in enumerate(iio.imiter(video_path)):
        if frame is None:
            continue

        if downsample and count % 2 == 0:
            count += 1
            continue

        if frame_range is None or (
            idx >= frame_range[0] and idx < frame_range[1]
        ):
            if frame_path is None:
                
                frames.append(frame)
            else:
                fname = "frame_{:0>4d}.png".format(idx)
                ofname = os.path.join(frame_path, fname)
                ret = cv2.imwrite(ofname, frame)

        count += 1

        #iio.imwrite(f"extracted_images/frame{idx:03d}.jpg", frame)
    '''
    cap = cv2.VideoCapture()
    cap.open(video_path)
    if not cap.isOpened():
        raise OSError("Failed to open input video")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if frame_range is not None:
        if frame_range[1] > frame_count:
            raise ValueError("Requested frame range is longer than the video")
    print('debug', frame_count, frame_range)
    for frameId in range(int(frame_count)):
        ret, frame = cap.read()

        if frame is None:
            continue

        if downsample and count % 2 == 0:
            count += 1
            continue

        if frame_range is None or (
            frameId >= frame_range[0] and frameId < frame_range[1]
        ):
            if frame_path is None:
                
                frames.append(frame)
            else:
                fname = "frame_{:0>4d}.png".format(frameId)
                ofname = os.path.join(frame_path, fname)
                ret = cv2.imwrite(ofname, frame)

        count += 1

    cap.release()
    '''
    return frames


def extract_fer_dataset(dataset_path, des_path_root, args):

    if args.backbone=='MobileNet':
        out_size = 224
    else:
        out_size = 112 
    model = load_model()
    model = model.eval()
    if args.detector=='FaceBoxes':
        face_boxes = FaceBoxes()
    elif args.detector=='Retinaface':
        retinaface=Retinaface.Retinaface()    
    else:
        print('Error: not suppored detector')    

    if not os.path.exists(des_path_root):
        os.makedirs(des_path_root)

    # Subject
    for target in ['train', 'test']:
        subject_path = os.path.join(dataset_path, target)

        video_file_path = None

        for file in tqdm.tqdm(os.listdir(subject_path)):
            
            try:
                # File paths
                assert file.endswith(".avi") or file.endswith(".mp4")
                video_file_path = os.path.join(subject_path, file)            
                #video_file_path = '/home/yoon/data/fer/self-supervised-learning/train/004508680.avi'

                raw_frames = extract_video_frame(
                    video_file_path, frame_path=None, frame_range=None, downsample=False
                )

                for i, img in enumerate(tqdm.tqdm(raw_frames, leave=False)):
                    height,width,_=img.shape
                    if args.detector=='MTCNN':
                        # perform face detection using MTCNN
                        faces, landmarks = detect_faces(img)
                    elif args.detector=='FaceBoxes':
                        faces = face_boxes(img)
                    elif args.detector=='Retinaface':
                        faces = retinaface(img)            
                    else:
                        print('Error: not suppored detector')        
                    if len(faces)==0:
                        print('NO face is detected!', file, i)
                        continue

                    face = faces[0] # 첫번째 얼굴만 사용
                    
                    if face[4]<0.9: # remove low confidence detection
                        continue
                    x1=face[0]
                    y1=face[1]
                    x2=face[2]
                    y2=face[3]
                    w = x2 - x1 + 1
                    h = y2 - y1 + 1
                    size = int(min([w, h])*1.2)
                    cx = x1 + w//2
                    cy = y1 + h//2
                    x1 = cx - size//2
                    x2 = x1 + size
                    y1 = cy - size//2
                    y2 = y1 + size

                    dx = max(0, -x1)
                    dy = max(0, -y1)
                    x1 = max(0, x1)
                    y1 = max(0, y1)

                    edx = max(0, x2 - width)
                    edy = max(0, y2 - height)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    new_bbox = list(map(int, [x1, x2, y1, y2]))
                    new_bbox = BBox(new_bbox)
                    cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
                    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                        cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)            
                    cropped_face = cv2.resize(cropped, (out_size, out_size))

                    if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
                        continue
                    test_face = cropped_face.copy()
                    test_face = test_face/255.0
                    if args.backbone=='MobileNet':
                        test_face = (test_face-mean)/std
                    test_face = test_face.transpose((2, 0, 1))
                    test_face = test_face.reshape((1,) + test_face.shape)
                    input = torch.from_numpy(test_face).float()
                    input= torch.autograd.Variable(input)
                    start = time.time()
                    if args.backbone=='MobileFaceNet':
                        landmark = model(input)[0].cpu().data.numpy()
                    else:
                        landmark = model(input).cpu().data.numpy()
                    end = time.time()
                    #print('Time: {:.6f}s.'.format(end - start))
                                        
                    landmark = landmark.reshape(-1,2)
                    landmark = new_bbox.reprojectLandmark(landmark)
                    ROI_face = ConvexHull(landmark).vertices
                    face_lm = poly2mask(
                        landmark[ROI_face, 1], landmark[ROI_face, 0], img, (64, 64)
                    )                                        

                    # Save file
                    des_path = os.path.join(des_path_root, target)
                    if not os.path.exists(des_path):
                        os.makedirs(des_path)
                    folder_path = os.path.join(des_path, file[:-4])
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                        
                    file_path = os.path.join(folder_path, str(i) + '.jpg')
                    iio.imwrite(file_path, face_lm)
                    #cv2.imwrite(file_path,face_lm)

            

            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                print(e, "\n")

                # Print origin trace info
                with open("./fer_extraction_error.txt", "a") as myfile:
                    myfile.write("Session: {}\n".format(subject_path))
                    exc_info = sys.exc_info()
                    traceback.print_exception(*exc_info, file=myfile)
                    myfile.write("\n")
                    del exc_info


def poly2mask(vertex_row_coords, vertex_col_coords, frame, crop_shape):
    h, w, c = frame.shape
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, (h, w)
    )
    cropped_frame = np.zeros(frame.shape, dtype=np.uint8)

    if fill_row_coords.size == 0 or fill_col_coords.size == 0:
        pass
    else:
        cropped_frame[fill_row_coords, fill_col_coords] = frame[
            fill_row_coords, fill_col_coords
        ]
        cropped_frame = cropped_frame[
            min(fill_row_coords) : max(fill_row_coords),
            min(fill_col_coords) : max(fill_col_coords),
        ]

    # Resize frame with range(0, 255) in uint8 format
    img = Image.fromarray(cropped_frame)
    cropped_frame = np.array(img.resize(crop_shape), dtype=np.uint8)

    return cropped_frame

def load_model():
    if args.backbone=='MobileNet':
        model = MobileNet_GDConv(136)
        model = torch.nn.DataParallel(model)
        # download model from https://drive.google.com/file/d/1Le5UdpMkKOTRr1sTp4lwkw8263sbgdSe/view?usp=sharing
        checkpoint = torch.load('checkpoint/mobilenet_224_model_best_gdconv_external.pth.tar', map_location=map_location)
        print('Use MobileNet as backbone')
    elif args.backbone=='PFLD':
        model = PFLDInference() 
        # download from https://drive.google.com/file/d/1gjgtm6qaBQJ_EY7lQfQj3EuMJCVg9lVu/view?usp=sharing
        checkpoint = torch.load('checkpoint/pfld_model_best.pth.tar', map_location=map_location)
        print('Use PFLD as backbone') 
        # download from https://drive.google.com/file/d/1T8J73UTcB25BEJ_ObAJczCkyGKW5VaeY/view?usp=sharing
    elif args.backbone=='MobileFaceNet':
        model = MobileFaceNet([112, 112],136)   
        checkpoint = torch.load('checkpoint/mobilefacenet_model_best.pth.tar', map_location=map_location)      
        print('Use MobileFaceNet as backbone')         
    else:
        print('Error: not suppored backbone')    
    model.load_state_dict(checkpoint['state_dict'])
    return model

INPATH='/home/yoon/data/face/VGG-Face2/data'
OUTPATH='/home/yoon/data/face/VGG-Face2/face_extract'
  
if __name__ == '__main__':  
    extract_fer_dataset('/home/yoon/data/fer/self-supervised-learning', '/home/yoon/data/fer/self-supervised-learning/faces', args)
                
