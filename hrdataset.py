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
from ast import expr_context
import os
import csv
import sys
import time
import datetime
import traceback
import xml.dom.minidom

from tqdm import tqdm

import cv2

from PIL import Image

from skimage import draw

from scipy.spatial import ConvexHull
from scipy.signal import resample

import numpy as np

import torch
import json
import pyVHR as vhr

EXG1_CHANNEL_IDX = 32
EXG2_CHANNEL_IDX = 33
EXG3_CHANNEL_IDX = 34

# Face alignment and crop demo
# Uses MTCNN, FaceBoxes or Retinaface as a face detector;
# Support different backbones, include PFLD, MobileFaceNet, MobileNet;
# Retinaface+MobileFaceNet gives the best peformance
# Cunjian Chen (ccunjian@gmail.com), Feb. 2021



parser = argparse.ArgumentParser(description='PyTorch face landmark')
# Datasets
parser.add_argument('--backbone', default='PFLD', type=str,
                    help='choose which backbone network to use: MobileNet, PFLD, MobileFaceNet')
parser.add_argument('--detector', default='Retinaface', type=str,
                    help='choose which face detector to use: MTCNN, FaceBoxes, Retinaface')
parser.add_argument('--datatype', default='mahnob', type=str,)
args = parser.parse_args()

if args.detector=='FaceBoxes':
    face_boxes = FaceBoxes()
elif args.detector=='Retinaface':
    retinaface=Retinaface.Retinaface()    
else:
    print('Error: not suppored detector')    

FACESIZE=112

mean = np.asarray([ 0.485, 0.456, 0.406 ])
std = np.asarray([ 0.229, 0.224, 0.225 ])

crop_size= FACESIZE
scale = crop_size / FACESIZE
reference = get_reference_facial_points(default_square = True) * scale

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

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
        model = MobileFaceNet([FACESIZE, FACESIZE],136)   
        checkpoint = torch.load('checkpoint/mobilefacenet_model_best.pth.tar', map_location=map_location)      
        print('Use MobileFaceNet as backbone')         
    else:
        print('Error: not suppored backbone')    
    model.load_state_dict(checkpoint['state_dict'])
    return model

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

    return frames


def align_face(frames, model):

    align_frames = np.empty((len(frames), FACESIZE, FACESIZE, 3), dtype=np.uint8)
    for i, img in enumerate(tqdm(frames, leave=False)):
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
            continue        

        face = faces[0] # 첫번째 얼굴만 사용
        
        #if face[4]<0.9: # remove low confidence detection
        #    continue
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
        cropped_face = cv2.resize(cropped, (FACESIZE, FACESIZE))

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
        if args.backbone=='MobileFaceNet':
            landmark = model(input)[0].cpu().data.numpy()
        else:
            landmark = model(input).cpu().data.numpy()
        #print('Time: {:.6f}s.'.format(end - start))
                            
        landmark = landmark.reshape(-1,2)
        landmark = new_bbox.reprojectLandmark(landmark)
        ROI_face = ConvexHull(landmark).vertices
        face_lm = poly2mask(
            landmark[ROI_face, 1], landmark[ROI_face, 0], img, (FACESIZE, FACESIZE)
        )                                            
        align_frames[i] = face_lm

    return align_frames

def extract_hr_from_ecg(file_path, channel_idx, begin=5, end=35):
    import pyedflib
    import heartpy as hp
    
    signals, signals_headers, header = pyedflib.highlevel.read_edf(
        file_path, ch_nrs=channel_idx, verbose=False
    )

    sample_rate = signals_headers[0]["sample_rate"]
    start_idx = int(begin * sample_rate)
    end_idx = int(end * sample_rate)
    data = signals[0][start_idx:end_idx]
    filtered = hp.filter_signal(
        data, cutoff=0.05, sample_rate=sample_rate, filtertype="notch"
    )
    resampled_data = resample(filtered, len(filtered) * 2)

    # Run analysis
    wd, m = hp.process(hp.scale_data(resampled_data), sample_rate * 2)

    return m["bpm"]


def extract_mahnob_hci_dataset(dataset_path, des_path_root):
    if not os.path.exists(des_path_root):
        os.makedirs(des_path_root)

    start = time.time()
    sessions = os.listdir(dataset_path)
    sessions.sort()

    # Detect face landmarks model
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        device="cuda:1",
        flip_input=False,
        face_detector="blazeface",
    )

    # Sessions
    for session in tqdm(sessions, desc="Extract MAHNOB-HCI Dataset"):
        session_path = os.path.join(dataset_path, session)        
        video_file_path = None
        ecg_file_path = None
        mata_data_file_path = None

        try:
            # File paths
            for file in os.listdir(session_path):
                if file.endswith(".avi"):
                    video_file_path = os.path.join(session_path, file)
                elif file.endswith(".bdf"):
                    ecg_file_path = os.path.join(session_path, file)
                elif file.endswith(".xml"):
                    mata_data_file_path = os.path.join(session_path, file)

            if (
                video_file_path is None
                or ecg_file_path is None
                or mata_data_file_path is None
            ):
                raise OSError("Files are incomplete in {}".format(session_path))

            # Extract ground truth HR
            hr_gt = extract_hr_from_ecg(ecg_file_path, EXG2_CHANNEL_IDX)
            if np.isnan(hr_gt):
                raise ValueError("Ground truth heart rate value is NaN")
            
            # Extract frames
            raw_frames = extract_video_frame(video_file_path)
            # Align face
            align_frames = align_face(raw_frames, fa)

            # Get subject ID
            doc = xml.dom.minidom.parse(mata_data_file_path)
            subject_id = doc.getElementsByTagName("subject")[0].attributes["id"].value
            # Save file
            save_path = os.path.join(des_path_root, subject_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = os.path.join(save_path, session)
            np.savez(file_path, frames=align_frames, hr=hr_gt)
            print(
                "{} saved with frames: {}; HR: {}!\n".format(
                    file_path, align_frames.shape, hr_gt
                )
            )

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print(e, "\n")

            # Print origin trace info
            with open("./mahnob_hci_dataset_error.txt", "a") as myfile:
                myfile.write("Session: {}\n".format(session_path))
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info, file=myfile)
                myfile.write("\n")
                del exc_info

    duration = str(datetime.timedelta(seconds=time.time() - start))
    print("It takes {} time for extracting MAHNOB-HCI dataset".format(duration))

def mahnob_delete_bw(dataset_path):
    import shutil
    sessions = os.listdir(dataset_path)
    sessions.sort()

    # Detect face landmarks model
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        device="cuda:1",
        flip_input=False,
        face_detector="blazeface",
    )
    numavi = 0
    numbdf = 0
    # Sessions
    
    for session in tqdm(sessions, desc="Extract MAHNOB-HCI Dataset"):
        session_path = os.path.join(dataset_path, session)        
        video_file_path = None

        # File paths
        cnt = 0
        bdf_exist = False
        avi_exist = False
        for file in os.listdir(session_path):
            if file.endswith(".avi"):
                if 'BW' in file:
                    video_file_path = os.path.join(session_path, file)
                    print(video_file_path)
                    os.remove(video_file_path)
                else:
                    cnt = cnt+1
                    numavi += 1
                avi_exist = True
            elif file.endswith(".bdf"):
                numbdf += 1
                bdf_exist = True
        
        if not bdf_exist:
            print('no bdf', session_path)
            shutil.rmtree(session_path)
        if not avi_exist:
            print('no avi', session_path)
            shutil.rmtree(session_path)
    print('avi', numavi, 'bdf', numbdf)
        
def extract_pure_dataset(dataset_path, des_path_root):
    model = load_model()
    model = model.eval()
    if not os.path.exists(des_path_root):
        os.makedirs(des_path_root)

    start = time.time()
    sessions = os.listdir(dataset_path)
    sessions.sort()

    # Sessions
    for session in tqdm(sessions, desc="Extract PURE Dataset"):
        session_path = os.path.join(dataset_path, session)        
        if not os.path.isdir(session_path):
            continue

        jsonpath = os.path.join(dataset_path, session + '.json')
        with open(jsonpath, 'r') as f:
            jsonread = json.load(f)
            hr_inf = jsonread["/FullPackage"]
            image_inf = jsonread["/Image"]
        raw_frames = []
        hr_gt = []
        # File paths

        for i, ii in enumerate(image_inf):
            file = 'Image' + str(ii["Timestamp"]) + '.png'
            raw_frames.append(cv2.imread(os.path.join(session_path, file)))
            hr_gt.append(hr_inf[i]["Value"]["pulseRate"])

        # Align face
        align_frames = align_face(raw_frames, model)

        n_vid = len(hr_gt) // 150
        for i in range(n_vid):
            start = i * 150
            end = (i + 1) * 150

            # Calculate gt
            cur_gtHR = np.mean(hr_gt[start:end])
            # Save file
            des_path = os.path.join(des_path_root, session)
            if not os.path.exists(des_path):
                os.makedirs(des_path)
            file_path = os.path.join(des_path, str(i))
            np.savez(file_path, frames=align_frames[start:end], hr=cur_gtHR)
            print(
                "{} saved with frames: {}; HR: {}!\n".format(
                    file_path, align_frames.shape, cur_gtHR
                )
            )

    duration = str(datetime.timedelta(seconds=time.time() - start))
    print("It takes {} time for extracting MAHNOB-HCI dataset".format(duration))


def extract_vipl_hr_dataset(dataset_path, des_path_root):
    if not os.path.exists(des_path_root):
        os.makedirs(des_path_root)

    start = time.time()
    subjects = os.listdir(dataset_path)
    subjects.sort()

    # Detect face landmarks model
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        device="cuda:1",
        flip_input=False,
        face_detector="blazeface",
    )

    # Subject
    for subject in tqdm(subjects, desc="Extract VIPL-HR-V2 Dataset"):
        subject_path = os.path.join(dataset_path, subject)

        video_file_path = []
        mata_data_file_path = None

        try:
            # File paths
            for file in os.listdir(subject_path):
                if file.endswith(".avi"):
                    video_file_path.append(os.path.join(subject_path, file))
                elif file.endswith(".csv"):
                    mata_data_file_path = os.path.join(subject_path, file)

            if len(video_file_path) != 5 or mata_data_file_path is None:
                raise OSError("Files are incomplete in {}".format(subject_path))
            video_file_path.sort()

            # Extract ground truth HR
            hr_gt = None
            fps = None
            with open(mata_data_file_path) as f:
                csv_reader = csv.reader(f, delimiter=",")
                line_count = 0
                for row in csv_reader:
                    if line_count == 1:
                        hr_gt = row[1:]
                    elif line_count == 2:
                        fps = row[1:]

                    line_count += 1

            if len(hr_gt) != 5 or len(fps) != 5:
                raise ValueError(
                    "Ground truth heart rate value or FPS value is INCORRECT!"
                )

            for idx, vid in enumerate(video_file_path):
                # Extract frames
                raw_frames = extract_video_frame(
                    vid, frame_range=None, downsample=False
                )

                # Align face
                align_frames = align_face(raw_frames, fa)

                # Save file
                if not os.path.exists(des_path_root):
                    os.makedirs(des_path_root)
                file_path = os.path.join(
                    des_path_root,
                    "{}_{}".format(vid.split("/")[-2], vid.split("/")[-1][:6]),
                )
                np.savez(file_path, frames=align_frames, hr=hr_gt[idx], fps=fps[idx])
                print(
                    "{} saved with frames: {}; HR: {}; FPS: {}!\n".format(
                        file_path, align_frames.shape, hr_gt[idx], fps[idx]
                    )
                )

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print(e, "\n")

            # Print origin trace info
            with open("./vipl_hr_v2_dataset_error.txt", "a") as myfile:
                myfile.write("Session: {}\n".format(subject_path))
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info, file=myfile)
                myfile.write("\n")
                del exc_info

    duration = str(datetime.timedelta(seconds=time.time() - start))
    print("It takes {} time for extracting VIPL-HR-V2 dataset".format(duration))

def target_bpm_idx(timesGT, timesGT_idx, target):
    '''
    sec초에 가장 가까운 idx를 반환한다.
    '''
    min_target_time_delta = 100
    target_time_idx = -1
    for j in range(timesGT_idx, len(timesGT)):
        delta = abs(timesGT[j] - target)
        if delta < min_target_time_delta:
            min_target_time_delta = delta
            target_time_idx = j
        else:
            break
        
    return target_time_idx

def extract_cohface_dataset(dataset_path, des_path_root):
    model = load_model()
    model = model.eval()
    if not os.path.exists(des_path_root):
        os.makedirs(des_path_root)
    
    # sp = vhr.extraction.sig_processing.SignalProcessing()

    dataset_name = 'cohface'                   # the name of the python class handling it 
    video_DIR = dataset_path  # dir containing videos
    BVP_DIR = dataset_path    # dir containing BVPs GT

    dataset = vhr.datasets.datasetFactory(dataset_name, videodataDIR=video_DIR, BVPdataDIR=BVP_DIR)
    allvideo = dataset.videoFilenames    

    # print the list of video names with the progressive index (idx)
    wsize = 1           # seconds of video processed (with overlapping) for each estimate # 150(frame)/20(fps) = 7.5sec, 7초 동안의 
    assert len(dataset.sigFilenames) == len(allvideo), '%d vs %d' % (len(dataset.sigFilenames) , len(allvideo))

    # Sessions
    for v in tqdm(range(len(allvideo))):
        print('processing', allvideo[v])
        fname = dataset.getSigFilename(v)
        sigGT = dataset.readSigfile(fname)
        bpmGT, timesGT = sigGT.getBPM(wsize)
        videoFileName = dataset.getVideoFilename(v)
        fps = vhr.extraction.get_fps(videoFileName)
        raw_frames = extract_video_frame(
                videoFileName, frame_range=None, downsample=True
            )
        align_frames = align_face(raw_frames, model)
        #align_frames = align_face_batch(frames, fa)
        n_vid = len(align_frames)//150 #150 frame 단위.. 다른 데이터셋들처럼
        for i in range(n_vid): 
            start = i  * 150
            end = (i + 1) * 150

            # Calculate gt
            cur_gtHR = bpmGT[int(i*150/fps)]
            # Save file
            vfns = videoFileName.split('/')
            session = vfns[-3] + '_' + vfns[-2]
            des_path = os.path.join(des_path_root, session)
            if not os.path.exists(des_path):
                os.makedirs(des_path)
            file_path = os.path.join(des_path, str(i))
            if len(align_frames[int(start):int(end)]) < 150:
                print('frames are short')
                continue
            np.savez(file_path, frames=align_frames[int(start):int(end)], hr=cur_gtHR)
            print(
                "{} saved with frames: {}; HR: {} for {}!\n".format(
                    file_path, align_frames.shape, cur_gtHR, fname
                )
            )
        print('done', allvideo[v])


def extract_mahnob_dataset(dataset_path, des_path_root):
    model = load_model()
    model = model.eval()
    
    if not os.path.exists(des_path_root):
        os.makedirs(des_path_root)

    dataset_name = 'mahnob'                   # the name of the python class handling it 
    video_DIR = dataset_path  # dir containing videos
    BVP_DIR = dataset_path    # dir containing BVPs GT

    dataset = vhr.datasets.datasetFactory(dataset_name, videodataDIR=video_DIR, BVPdataDIR=BVP_DIR)
    allvideo = dataset.videoFilenames    

    # 학습/테스트셋 분류를 위한 데이터별 id 추출
    sessions = os.listdir(dataset_path)
    sessions.sort()
    sesstoid = {}

    for session in sessions:
        session_path = os.path.join(dataset_path, session)        
        
        mata_data_file_path = None

        # File paths
        for file in os.listdir(session_path):               
            if file.endswith(".xml"):
                mata_data_file_path = os.path.join(session_path, file)

        if (
            mata_data_file_path is None
        ):
            raise OSError("Files are incomplete in {}".format(session_path))

        # Get subject ID
        doc = xml.dom.minidom.parse(mata_data_file_path)
        subject_id = doc.getElementsByTagName("subject")[0].attributes["id"].value
        sesstoid[session] = subject_id

    # print the list of video names with the progressive index (idx)
    wsize = 5          # seconds of video processed (with overlapping) for each estimate # 150(frame)/20(fps) = 7.5sec, 7초 동안의 
    #assert len(dataset.sigFilenames) > 0
    # Sessions
    assert len(dataset.sigFilenames) == len(allvideo), '%d vs %d' % (len(dataset.sigFilenames) , len(allvideo))
    for v in tqdm(range(len(allvideo))):    
        try:    
            print('Processing', allvideo[v])
            fname = dataset.getSigFilename(v)
            sigGT = dataset.readSigfile(fname)
            bpmGT, timesGT = sigGT.getBPM(wsize)
            videoFileName = dataset.getVideoFilename(v)
            
            raw_frames = extract_video_frame(
                videoFileName, frame_range=None, downsample=True
            )
            align_frames = align_face(raw_frames, model)
            
            if len(align_frames) != len(raw_frames):
                print('faces are not extracted')
                continue

            n_vid = len(align_frames)//150 #150 frame 단위.. 다른 데이터셋들처럼
            timesGT_idx = 0

            for i in range(n_vid): 
                start = i  * 150
                end = (i + 1) * 150
                timesGT_idx = target_bpm_idx(timesGT, timesGT_idx, i*5)
                # Calculate gt
                if len(bpmGT) < timesGT_idx:
                    print('len bpmGT < timesGT_idx')
                    break
                cur_gtHR = bpmGT[timesGT_idx]
                # Save file
                vfns = videoFileName.split('/')
                session = vfns[-3] + '_' + vfns[-2]
                des_path = os.path.join(des_path_root, sesstoid[vfns[-2]])
                if not os.path.exists(des_path):
                    os.makedirs(des_path)
                file_path = os.path.join(des_path, vfns[-2] + '_' + str(i))
                if len(align_frames[int(start):int(end)]) < 150:
                    print('frames are short')
                    continue
                np.savez(file_path, frames=align_frames[int(start):int(end)], hr=cur_gtHR)
                print(
                    "{} saved with frames: {}; HR: {}!\n".format(
                        file_path, align_frames[int(start):int(end)].shape, cur_gtHR
                    )
                )
                timesGT_idx += 1
            print('Done', allvideo[v])

        except Exception as e:
            print(e, "\n")
            traceback.print_exc()
            # Print origin trace info
            with open("./mahnob_dataset_error.txt", "a") as myfile:
                myfile.write("File: {}\n".format(fname))
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info, file=myfile)
                myfile.write("\n")
                del exc_info

def extract_ubfc_dataset(dataset_path, des_path_root):
    model = load_model()
    model = model.eval()

    if not os.path.exists(des_path_root):
        os.makedirs(des_path_root)

    start = time.time()
    subjects = os.listdir(dataset_path)
    subjects.sort()

    # Subject
    for subject in tqdm(subjects, desc="Extract UBFC-rPPG Dataset"):
        subject_path = os.path.join(dataset_path, subject)

        video_file_path = None
        mata_data_file_path = None

        try:
            # File paths
            for file in os.listdir(subject_path):
                if file.endswith(".avi"):
                    video_file_path = os.path.join(subject_path, file)
                elif file.endswith(".txt"):
                    mata_data_file_path = os.path.join(subject_path, file)

            if video_file_path is None or mata_data_file_path is None:
                raise OSError("Files are incomplete in {}".format(subject_path))

            # Extract ground truth HR
            gtHR = None
            with open(mata_data_file_path, "r") as f:
                gtdata = [[float(l) for l in line.split()] for line in f.readlines()]
                gtHR = gtdata[1]

            if gtHR is None:
                raise ValueError("Ground truth heart rate value value is INCORRECT!")

            n_vid = len(gtHR) // 150

            for i in range(n_vid):
                start = i * 150
                end = (i + 1) * 150
                # Extract frames
                raw_frames = extract_video_frame(
                    video_file_path, frame_range=(start, end), downsample=False
                )

                # Align face
                align_frames = align_face(raw_frames, model)

                # Calculate gt
                cur_gtHR = np.mean(gtHR[start:end])

                # Save file
                des_path = os.path.join(des_path_root, subject)
                if not os.path.exists(des_path):
                    os.makedirs(des_path)
                file_path = os.path.join(des_path, str(i))
                np.savez(file_path, frames=align_frames, hr=cur_gtHR)
                print(
                    "{} saved with frames: {}; HR: {}!\n".format(
                        file_path, align_frames.shape, cur_gtHR
                    )
                )

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print(e, "\n")

            # Print origin trace info
            with open("./ubfc_rppg_dataset_error.txt", "a") as myfile:
                myfile.write("Session: {}\n".format(subject_path))
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info, file=myfile)
                myfile.write("\n")
                del exc_info

    duration = str(datetime.timedelta(seconds=time.time() - start))
    print("It takes {} time for extracting UBFC-rPPG dataset".format(duration))


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


def cal_rois(frame, landmark, crop_shape):
    ROI_face = ConvexHull(landmark).vertices
    ROI_forehead = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    ROI_cheek_left1 = [0, 1, 2, 31, 41, 0]
    ROI_cheek_left2 = [2, 3, 4, 5, 48, 31, 2]
    ROI_cheek_right1 = [16, 15, 14, 35, 46, 16]
    ROI_cheek_right2 = [14, 13, 12, 11, 54, 35, 14]
    ROI_mouth = [5, 6, 7, 8, 9, 10, 11, 54, 55, 56, 57, 58, 59, 48, 5]

    all_ROIs = np.empty((7, crop_shape[0], crop_shape[1], 3), dtype=np.uint8)

    forehead = landmark[ROI_forehead, :]
    left_eye = np.mean(landmark[36:42, :], axis=0)
    right_eye = np.mean(landmark[42:48, :], axis=0)
    eye_distance = np.linalg.norm(left_eye - right_eye)
    tmp = (
        np.mean(landmark[17:22, :], axis=0) + np.mean(landmark[22:27, :], axis=0)
    ) / 2 - (left_eye + right_eye) / 2
    tmp = eye_distance / np.linalg.norm(tmp) * 0.6 * tmp
    ROI_forehead = np.concatenate(
        [
            forehead,
            forehead[np.newaxis, -1, :] + tmp,
            forehead[np.newaxis, 0, :] + tmp,
            forehead[np.newaxis, 0, :],
        ],
        axis=0,
    )

    all_ROIs[0] = poly2mask(
        landmark[ROI_face, 1], landmark[ROI_face, 0], frame, crop_shape
    )
    all_ROIs[1] = poly2mask(ROI_forehead[:, 1], ROI_forehead[:, 0], frame, crop_shape)
    all_ROIs[2] = poly2mask(
        landmark[ROI_cheek_left1, 1], landmark[ROI_cheek_left1, 0], frame, crop_shape
    )
    all_ROIs[3] = poly2mask(
        landmark[ROI_cheek_left2, 1], landmark[ROI_cheek_left2, 0], frame, crop_shape
    )
    all_ROIs[4] = poly2mask(
        landmark[ROI_cheek_right1, 1], landmark[ROI_cheek_right1, 0], frame, crop_shape
    )
    all_ROIs[5] = poly2mask(
        landmark[ROI_cheek_right2, 1], landmark[ROI_cheek_right2, 0], frame, crop_shape
    )
    all_ROIs[6] = poly2mask(
        landmark[ROI_mouth, 1], landmark[ROI_mouth, 0], frame, crop_shape
    )

    return all_ROIs


def write_config_to_file(config, save_path):
    """Record and save current parameter settings
    Parameters
    ----------
    config : object of class `Parameters`
            Object of class `Parameters`
    save_path : str
            Path to save the file
    """

    with open(os.path.join(save_path, "config.txt"), "w") as file:
        for arg in vars(config):
            file.write(str(arg) + ": " + str(getattr(config, arg)) + "\n")


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

#ycyoon 추가
if __name__=='__main__':
    if args.datatype == 'mahnob':
        extract_mahnob_dataset('/home/yoon/data/PPG/mahnob/Sessions', 'mahnob')
    elif args.datatype == 'ubfc':
        extract_ubfc_dataset('/home/yoon/data/PPG/UBFC/UBFC_DATASET/DATASET_2/', 'data')
    elif args.datatype == 'pure':
        extract_pure_dataset('/home/yoon/data/PPG/PURE', 'pure')
    elif args.datatype == 'cohface':
        extract_cohface_dataset('/home/yoon/data/PPG/cohface/', 'cohface')

    #extract_mahnob_hci_dataset('/home/yoon/data/PPG/mahnob/Sessions', 'data2')    
    #mahnob_delete_bw('/home/yoon/data/PPG/mahnob/Sessions')