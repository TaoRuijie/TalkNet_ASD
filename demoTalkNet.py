import subprocess
import sys
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
import tqdm
import torch
import argparse
import glob
import subprocess
import warnings
import cv2
import pickle
import numpy
import pdb
import math
import python_speech_features
import mediapipe as mp
import matplotlib.pyplot as plt


import cProfile
import pstats
from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect import SceneManager, open_video, ContentDetector, ThresholdDetector
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from talkNet import talkNet

warnings.filterwarnings("ignore")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)


parser = argparse.ArgumentParser(
    description="TalkNet Demo or Columnbia ASD Evaluation")

parser.add_argument('--videoName',             type=str,
                    default="001",   help='Demo video name')
parser.add_argument('--videoFolderInput',           type=str,
                    default="demo",  help='Path for inputs')
parser.add_argument('--videoFolderOutput',           type=str,
                    default="output_dir",  help='Path for tmps and outputs')
parser.add_argument('--pretrainModel',         type=str,
                    default="pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')
parser.add_argument('--fps',                   type=float,
                    default=25,   help='Desired FPS')
parser.add_argument('--frame_size',                   type=int,
                    default=256,   help='Desired frame size')

parser.add_argument('--angleThreshold',                   type=int,
                    default=10,   help='Desired threshold for yaw')
parser.add_argument('--contentDetectorThreshold',                   type=float,
                    default=27.0,   help='Desired frame size')
parser.add_argument('--thresholdDetectorThreshold',                   type=float,
                    default=30.0,   help='Desired frame size')
# parser.add_argument('--frame_size',                   type=int,
                    # default=256,   help='Desired frame size')
# parser.add_argument('--frame_size',                   type=int,
                    # default=256,   help='Desired frame size')

parser.add_argument('--nDataLoaderThread',     type=int,
                    default=10,   help='Number of workers')
parser.add_argument('--facedetScale',          type=float, default=0.25,
                    help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,
                    default=40,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=5,
                    help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,
                    default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float,
                    default=0.40, help='Scale bounding box')

parser.add_argument('--start',                 type=int,
                    default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,
                    help='The duration of the video, when set as 0, will extract the whole video')

parser.add_argument('--evalCol',               dest='evalCol',
                    action='store_true', help='Evaluate on Columnbia dataset')
parser.add_argument('--colSavePath',           type=str,
                    default="/data08/col",  help='Path for inputs, tmps and outputs')

args = parser.parse_args()

if os.path.isfile(args.pretrainModel) == False:  # Download the pretrained model
    Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
    cmd = "gdown --id %s -O %s" % (Link, args.pretrainModel)
    subprocess.call(cmd, shell=True, stdout=None)

if args.evalCol == True:
    # The process is: 1. download video and labels(I have modified the format of labels to make it easiler for using)
    # 	              2. extract audio, extract video frames
    #                 3. scend detection, face detection and face tracking
    #                 4. active speaker detection for the detected face clips
    #                 5. use iou to find the identity of each face clips, compute the F1 results
    # The step 1 to 3 will take some time (That is one-time process). It depends on your cpu and gpu speed. For reference, I used 1.5 hour
    # The step 4 and 5 need less than 10 minutes
    # Need about 20G space finally
    # ```
    args.videoName = 'col'
    args.videoFolder = args.colSavePath
    args.savePath = os.path.join(args.videoFolder, args.videoName)
    args.videoPath = os.path.join(args.videoFolder, args.videoName + '.mp4')
    args.duration = 0
    if os.path.isfile(args.videoPath) == False:  # Download video
        link = 'https://www.youtube.com/watch?v=6GzxbrO0DHM&t=2s'
        cmd = "youtube-dl -f best -o %s '%s'" % (args.videoPath, link)
        output = subprocess.call(cmd, shell=True, stdout=None)
    if os.path.isdir(args.videoFolder + '/col_labels') == False:  # Download label
        link = "1Tto5JBt6NsEOLFRWzyZEeV6kCCddc6wv"
        cmd = "gdown --id %s -O %s" % (link,
                                       args.videoFolder + '/col_labels.tar.gz')
        subprocess.call(cmd, shell=True, stdout=None)
        cmd = "tar -xzvf %s -C %s" % (args.videoFolder +
                                      '/col_labels.tar.gz', args.videoFolder)
        subprocess.call(cmd, shell=True, stdout=None)
        os.remove(args.videoFolder + '/col_labels.tar.gz')
else:
    args.videoPath = glob.glob(os.path.join(
        args.videoFolderInput, args.videoName + '.*'))[0]
    # args.savePath = os.path.join(args.videoFolderOutput, args.videoName)
    args.savePath = args.videoFolderOutput

def scene_detect(args):
    # CPU: Scene detection, output is the list of each shot's time duration
    video = open_video(args.videoFilePath)

    sceneManager = SceneManager()

    sceneManager.add_detector(ContentDetector(threshold=args.contentDetectorThreshold, min_scene_len=30))
    sceneManager.add_detector(ThresholdDetector(threshold=args.thresholdDetectorThreshold))

    sceneManager.detect_scenes(video)
    sceneList = sceneManager.get_scene_list()

    savePath = os.path.join(args.pyworkPath, 'scene.pckl')
    if not sceneList:
        # Fallback: If no scenes detected, create a single "scene" from start to end
        sceneList = [(0, video.frame_count)]
    with open(savePath, 'wb') as file:
        pickle.dump(sceneList, file)
        sys.stderr.write(f"{args.videoFilePath} - scenes detected: {len(sceneList)}\n")
    
    return sceneList


def inference_video(args):
    # GPU: Face detection, output is the list contains the face location and score in this frame
    DET = S3FD(device='cuda')
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(
            imageNumpy, conf_th=0.9, scales=[args.facedetScale])
        dets.append([])
        for bbox in bboxes:
            # dets has the frames info, bbox info, conf info
            dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]
                                                     ).tolist(), 'conf': bbox[-1]})
        sys.stderr.write('%s-%05d; %d dets\r' %
                         (args.videoFilePath, fidx, len(dets[-1])))
    savePath = os.path.join(args.pyworkPath, 'faces.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(dets, fil)
    return dets


def bb_intersection_over_union(boxA, boxB, evalCol=False):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol == True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def track_shot(args, sceneFaces):
    # CPU: Face tracking
    iouThres = 0.5     # Minimum IOU between consecutive face detections
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(
                        face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            frameNum = numpy.array([f['frame'] for f in track])
            bboxes = numpy.array([numpy.array(f['bbox']) for f in track])
            frameI = numpy.arange(frameNum[0], frameNum[-1]+1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = numpy.stack(bboxesI, axis=1)
            if max(numpy.mean(bboxesI[:, 2]-bboxesI[:, 0]), numpy.mean(bboxesI[:, 3]-bboxesI[:, 1])) > args.minFaceSize:
                tracks.append({'frame': frameI, 'bbox': bboxesI})
    return tracks


def crop_video(args, track, cropFile):
    # CPU: crop the face clips
    flist = glob.glob(os.path.join(
        args.pyframesPath, '*.jpg'))  # Read the frames
    flist.sort()
    vOut = cv2.VideoWriter(
        cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), args.fps, (args.frame_size, args.frame_size))  # Write video
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:  # Read the tracks
        dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2)
        dets['y'].append((det[1]+det[3])/2)  # crop center x
        dets['x'].append((det[0]+det[2])/2)  # crop center y
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs = args.cropScale
        bs = dets['s'][fidx]   # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
        image = cv2.imread(flist[frame])
        frame = numpy.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)),
                          'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi  # BBox center Y
        mx = dets['x'][fidx] + bsi  # BBox center X
        face = frame[int(my-bs):int(my+bs*(1+2*cs)),
                     int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        vOut.write(cv2.resize(face, (args.frame_size, args.frame_size)))
    audioTmp = cropFile + '.wav'
    audioStart = (track['frame'][0]) / args.fps
    audioEnd = (track['frame'][-1]+1) / args.fps
    vOut.release()
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" %
               (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp))
    output = subprocess.call(
        command, shell=True, stdout=None)  # Crop audio file
    _, audio = wavfile.read(audioTmp)
    command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" %
               (cropFile, audioTmp, args.nDataLoaderThread, cropFile))  # Combine audio and video file
    output = subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + 't.avi')
    return {'track': track, 'proc_track': dets}


def extract_MFCC(file, outPath):
    # CPU: extract mfcc
    sr, audio = wavfile.read(file)
    # (N_frames, 13)   [1s = 100 frames]
    mfcc = python_speech_features.mfcc(audio, sr)
    featuresPath = os.path.join(
        outPath, file.split('/')[-1].replace('.wav', '.npy'))
    numpy.save(featuresPath, mfcc)


def evaluate_network(files, args):
    # GPU: active speaker detection by pretrained TalkNet
    s = talkNet()
    s.loadParameters(args.pretrainModel)
    sys.stderr.write("Model %s loaded from previous state! \r\n" %
                     args.pretrainModel)
    s.eval()
    allScores = []
    # durationSet = {1,2,4,6} # To make the result more reliable
    # Use this line can get more reliable result
    durationSet = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}
    for file in tqdm.tqdm(files, total=len(files)):
        fileName = os.path.splitext(file.split(
            '/')[-1])[0]  # Load audio and video
        _, audio = wavfile.read(os.path.join(
            args.pycropPath, fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(
            audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        video = cv2.VideoCapture(os.path.join(
            args.pycropPath, fileName + '.avi'))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (args.frame_size, args.frame_size))
                face = face[int(112-(112/2)):int(112+(112/2)),
                            int(112-(112/2)):int(112+(112/2))]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = numpy.array(videoFeature)
        length = min((audioFeature.shape[0] - audioFeature.shape[0] %
                     4) / 100, videoFeature.shape[0] / args.fps)
        audioFeature = audioFeature[:int(round(length * 100)), :]
        videoFeature = videoFeature[:int(round(length * args.fps)), :, :]
        allScore = []  # Evaluation use TalkNet
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(
                        audioFeature[i * duration * 100:(i+1) * duration * 100, :]).unsqueeze(0).cuda()
                    inputV = torch.FloatTensor(
                        videoFeature[i * duration * args.fps: (i+1) * duration * args.fps, :, :]).unsqueeze(0).cuda()
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)
                    embedA, embedV = s.model.forward_cross_attention(
                        embedA, embedV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    score = s.lossAV.forward(out, labels=None)
                    scores.extend(score)
            allScore.append(scores)
        allScore = numpy.round(
            (numpy.mean(numpy.array(allScore), axis=0)), 1).astype(float)
        allScores.append(allScore)
    return allScores


def visualization(tracks, scores, args):
    # CPU: visulize the result for video format
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            # average smoothing
            s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]
            s = numpy.mean(s)
            faces[frame].append({'track': tidx, 'score': float(s), 's': track['proc_track']['s']
                                [fidx], 'x': track['proc_track']['x'][fidx], 'y': track['proc_track']['y'][fidx]})
    firstImage = cv2.imread(flist[0])
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'),
                           cv2.VideoWriter_fourcc(*'XVID'), args.fps, (fw, fh))
    colorDict = {0: 0, 1: 255}
    for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist)):
        image = cv2.imread(fname)
        # image = cv2.resize(image, )
        for face in faces[fidx]:
            clr = colorDict[int((face['score'] >= 0))]
            txt = round(face['score'], 1)
            cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(
                face['x']+face['s']), int(face['y']+face['s'])), (0, clr, 255-clr), 10)
            cv2.putText(image, '%s' % (txt), (int(face['x']-face['s']), int(
                face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, clr, 255-clr), 5)
        vOut.write(image)
    vOut.release()
    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" %
               (os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'),
                args.nDataLoaderThread, os.path.join(args.pyaviPath, 'video_out.avi')))
    output = subprocess.call(command, shell=True, stdout=None)


def evaluate_col_ASD(tracks, scores, args):
    txtPath = args.videoFolder + '/col_labels/fusion/*.txt'  # Load labels
    predictionSet = {}
    for name in {'long', 'bell', 'boll', 'lieb', 'sick', 'abbas'}:
        predictionSet[name] = [[], []]
    dictGT = {}
    txtFiles = glob.glob("%s" % txtPath)
    for file in txtFiles:
        lines = open(file).read().splitlines()
        idName = file.split('/')[-1][:-4]
        for line in lines:
            data = line.split('\t')
            frame = int(int(data[0]) / 29.97 * args.fps)
            x1 = int(data[1])
            y1 = int(data[2])
            x2 = int(data[1]) + int(data[3])
            y2 = int(data[2]) + int(data[3])
            gt = int(data[4])
            if frame in dictGT:
                dictGT[frame].append([x1, y1, x2, y2, gt, idName])
            else:
                dictGT[frame] = [[x1, y1, x2, y2, gt, idName]]
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))  # Load files
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            # average smoothing
            s = numpy.mean(
                score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)])
            faces[frame].append({'track': tidx, 'score': float(s), 's': track['proc_track']['s']
                                [fidx], 'x': track['proc_track']['x'][fidx], 'y': track['proc_track']['y'][fidx]})
    for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist)):
        if fidx in dictGT:  # This frame has label
            for gtThisFrame in dictGT[fidx]:  # What this label is ?
                faceGT = gtThisFrame[0:4]
                labelGT = gtThisFrame[4]
                idGT = gtThisFrame[5]
                ious = []
                for face in faces[fidx]:  # Find the right face in my result
                    faceLocation = [int(face['x']-face['s']), int(face['y']-face['s']),
                                    int(face['x']+face['s']), int(face['y']+face['s'])]
                    faceLocation_new = [int(face['x']-face['s']) // 2, int(face['y']-face['s']) // 2, int(
                        face['x']+face['s']) // 2, int(face['y']+face['s']) // 2]
                    iou = bb_intersection_over_union(
                        faceLocation_new, faceGT, evalCol=True)
                    if iou > 0.5:
                        ious.append([iou, round(face['score'], 2)])
                if len(ious) > 0:  # Find my result
                    ious.sort()
                    labelPredict = ious[-1][1]
                else:
                    labelPredict = 0
                x1 = faceGT[0]
                y1 = faceGT[1]
                width = faceGT[2] - faceGT[0]
                predictionSet[idGT][0].append(labelPredict)
                predictionSet[idGT][1].append(labelGT)
    names = ['long', 'bell', 'boll', 'lieb', 'sick', 'abbas']  # Evaluate
    names.sort()
    F1s = 0
    for i in names:
        scores = numpy.array(predictionSet[i][0])
        labels = numpy.array(predictionSet[i][1])
        scores = numpy.int64(scores > 0)
        F1 = f1_score(labels, scores)
        ACC = accuracy_score(labels, scores)
        if i != 'abbas':
            F1s += F1
            print("%s, ACC:%.2f, F1:%.2f" % (i, 100 * ACC, 100 * F1))
    print("Average F1:%.2f" % (100 * (F1s / 5)))


def extract_segment(track_path, start_frame, end_frame, output_path_video, output_path_audio):
    # Convert start_frame and end_frame to time (in seconds)
    start_time = start_frame / args.fps
    end_time = end_frame / args.fps

    # FFmpeg command to extract video with audio trimming
    command_video = f'ffmpeg -accurate_seek -i "{track_path}.avi" -ss {start_time} -to {end_time} -c:v libx264 -c:a aac "{output_path_video}" -loglevel panic'

    # FFmpeg command to extract audio separately
    command_audio = f'ffmpeg -accurate_seek -i "{track_path}.avi" -ss {start_time} -to {end_time} -vn -c:a aac "{output_path_audio}" -loglevel panic'

    # Execute command for video and audio extraction
    try:
        # Extract video
        subprocess.run(command_video, shell=True, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print(f"Video segment extracted successfully: {output_path_video}")

        # Extract audio
        subprocess.run(command_audio, shell=True, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print(f"Audio segment extracted successfully: {output_path_audio}")

    except subprocess.CalledProcessError as e:
        print(f"Error extracting segment: {e}")


# Main function
def main():
    # This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
    # ```
    # .
    # ├── pyavi
    # │   ├── audio.wav (Audio from input video)
    # │   ├── video.avi (Copy of the input video)
    # │   ├── video_only.avi (Output video without audio)
    # │   └── video_out.avi  (Output video with audio)
    # ├── pycrop (The detected face videos and audios)
    # │   ├── 000000.avi
    # │   ├── 000000.wav
    # │   ├── 000001.avi
    # │   ├── 000001.wav
    # │   └── ...
    # ├── pyframes (All the video frames in this video)
    # │   ├── 000001.jpg
    # │   ├── 000002.jpg
    # │   └── ...
    # |── pyfilter (Output clipped videoes)
    # └── pywork
    #     ├── faces.pckl (face detection result)
    #     ├── scene.pckl (scene detection result)
    #     ├── scores.pckl (ASD result)
    #     └── tracks.pckl (face tracking result)
    # ```

    # Initialization
    args.pyaviPath = os.path.join(args.savePath, 'pyavi')
    args.pyframesPath = os.path.join(args.savePath, 'pyframes')
    args.pyworkPath = os.path.join(args.savePath, 'pywork')
    args.pycropPath = os.path.join(args.savePath, 'pycrop')
    args.pyfilteredVideo = os.path.join(args.savePath, 'pyfilter', 'video')
    args.pyfilteredAudio = os.path.join(args.savePath, 'pyfilter', 'audio')
    
    # if os.path.exists(args.savePath):
    #     rmtree(args.savePath)
    
    # The path for the input video, input audio, output video
    os.makedirs(args.pyaviPath, exist_ok=True)
    os.makedirs(args.pyframesPath, exist_ok=True)  # Save all the video frames
    # Save the results in this process by the pckl method
    os.makedirs(args.pyworkPath, exist_ok=True)
    # Save the detected face clips (audio+video) in this process
    os.makedirs(args.pycropPath, exist_ok=True)
    # Save the detected face clips (audio+video) in this process
    os.makedirs(args.pyfilteredVideo, exist_ok=True)
    # Save the detected face clips (audio+video) in this process
    os.makedirs(args.pyfilteredAudio, exist_ok=True)

    # Extract video
    args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
    # If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
    if args.duration == 0:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" %
                   (args.videoPath, args.nDataLoaderThread, args.videoFilePath))
    else:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" %
                   (args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.videoFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Extract the video and save in %s \r\n" % (args.videoFilePath))

    # Extract audio
    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" %
               (args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Extract the audio and save in %s \r\n" % (args.audioFilePath))

    # Extract the video frames
    command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" %
               (args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg')))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Extract the frames and save in %s \r\n" % (args.pyframesPath))

    # Scene detection for the video frames
    scene = scene_detect(args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Scene detection and save in %s \r\n" % (args.pyworkPath))

    # Face detection for the video frames
    faces = inference_video(args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Face detection and save in %s \r\n" % (args.pyworkPath))

    # Face tracking
    allTracks, vidTracks = [], []
    for shot in scene:
        # Discard the shot frames less than minTrack frames
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack:
            # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
            allTracks.extend(track_shot(
                args, faces[shot[0].frame_num:shot[1].frame_num]))
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Face track and detected %d tracks \r\n" % len(allTracks))

    # Face clips cropping
    for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks)):
        vidTracks.append(crop_video(
            args, track, os.path.join(args.pycropPath, '%05d' % ii)))
    savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(vidTracks, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Face Crop and saved in %s tracks \r\n" % args.pycropPath)
    fil = open(savePath, 'rb')
    vidTracks = pickle.load(fil)

    # Active Speaker Detection by TalkNet
    files = glob.glob("%s/*.avi" % args.pycropPath)
    files_audio = glob.glob("%s/*.wav" % args.pycropPath)

    files.sort()
    scores = evaluate_network(files, args)
    savePath = os.path.join(args.pyworkPath, 'scores.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(scores, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Scores extracted and saved in %s \r\n" % args.pyworkPath)

    # Frame rate of the video (assumed 25 FPS)
    MIN_SEGMENT_FRAMES = 2 * args.fps  # Minimum segment length in frames
    MAX_SEGMENT_FRAMES = 5 * args.fps  # Maximum segment length in frames

    filtered_segments = []
    count_segments = 0
    # Process each track and its corresponding score
    for ii, (track, score_array) in tqdm.tqdm(enumerate(zip(allTracks, scores)), total=len(allTracks)):
        start_frame = None
        end_frame = None
        segment_frames = []

        for frame_idx, score in enumerate(score_array):

            if score > 0:
                frame_number = track['frame'][frame_idx]
                frame_path = os.path.join(args.pyframesPath, f"{(frame_number+1):06d}.jpg")
                image = cv2.imread(frame_path)

                # Check if the image was loaded successfully
                if image is None:
                    print(f"Warning: Frame {frame_number} could not be loaded! Skipping...")
                    continue

                # # Display the frame using Matplotlib
                # plt.imshow(image)
                # plt.axis("off")  # Turn off axes for a cleaner display
                # plt.show()

                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                img_h, img_w, _ = image.shape
                face_2d = []
                face_3d = []
                y=0
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                                if idx ==1:
                                    nose_2d = (lm.x * img_w,lm.y * img_h)
                                    nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
                                x,y = int(lm.x * img_w),int(lm.y * img_h)

                                face_2d.append([x,y])
                                face_3d.append(([x,y,lm.z]))

                        #Get 2d, 3d Coord
                        face_2d = numpy.array(face_2d,dtype=numpy.float64)
                        face_3d = numpy.array(face_3d,dtype=numpy.float64)

                        # Camera matrix (intrinsic parameters)
                        focal_length = 1 * img_w
                        cam_matrix = numpy.array([[focal_length,0,img_h/2],
                                            [0,focal_length,img_w/2],
                                            [0,0,1]])
                        # No lens distortion
                        distortion_matrix = numpy.zeros((4,1),dtype=numpy.float64)

                        # SolvePnP to calculate rotation and translation vectors
                        _,rotation_vec,_ = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)

                        #getting rotational of face
                        rmat,_ = cv2.Rodrigues(rotation_vec)

                        angles,_,_,_,_,_ = cv2.RQDecomp3x3(rmat)
                        y = angles[1] * 360				
                              

                if abs(y) < args.angleThreshold:
                    # Start a new segment if not already started
                    if start_frame is None:
                        start_frame = frame_idx
                    end_frame = frame_idx

                    # Check if segment length exceeds the maximum allowed duration
                    if (end_frame - start_frame + 1) > MAX_SEGMENT_FRAMES:
                        # Save the current valid segment
                        segment_frames.append((start_frame, end_frame))
                        start_frame = None  # Reset for next segment
                else:
                    # End the current segment if the score is not positive
                    if start_frame is not None:
                        # Save only if segment is long enough
                        if (end_frame - start_frame + 1) >= MIN_SEGMENT_FRAMES:
                            segment_frames.append((start_frame, end_frame))
                        start_frame = None
            
            else:
                # End the current segment if the score is not positive
                if start_frame is not None:
                    # Save only if segment is long enough
                    if (end_frame - start_frame + 1) >= MIN_SEGMENT_FRAMES:
                        segment_frames.append((start_frame, end_frame))
                    start_frame = None

        # Handle last segment if it ends positively
        if start_frame is not None and (end_frame - start_frame + 1) >= MIN_SEGMENT_FRAMES:
            segment_frames.append((start_frame, end_frame))

        count_segments += len(segment_frames)
        if segment_frames:
            # Extract and save each valid segment
            # for seg_idx, (seg_start, seg_end) in enumerate(segment_frames):
            seg_idx = 0
            seg_start, seg_end = segment_frames[0]
            segment_video_path = os.path.join(
                args.pyfilteredVideo, f"{args.videoName}_track_{ii:05d}_segment_{seg_idx:02d}.avi")
            segment_audio_path = os.path.join(
                args.pyfilteredAudio, f"{args.videoName}_track_{ii:05d}_segment_{seg_idx:02d}.wav")
            track_path = os.path.join(args.pycropPath, '%05d' % ii)
            extract_segment(track_path, seg_start+10, seg_end-10, segment_video_path, segment_audio_path)
            filtered_segments.append(segment_video_path)

    print("Found ", count_segments, " Segments")
    # Save filtered segments metadata
    savePath = os.path.join(args.pyworkPath, 'filtered_segments.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(filtered_segments, fil)
    sys.stderr.write(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} Filtered segments saved in {savePath}\n")

    # if args.evalCol == True:
    #     # The columnbia video is too big for visualization. You can still add the `visualization` funcition here if you want
    #     evaluate_col_ASD(vidTracks, scores, args)
    #     quit()
    # else:
    #     # Visualization, save the result as the new video
    #     visualization(vidTracks, scores, args)

    # At the end of the main function
    folders_to_keep = [args.pyfilteredVideo, args.pyfilteredAudio]
    folders_to_delete = [args.pyaviPath, args.pyframesPath, args.pyworkPath, args.pycropPath]

    for folder in folders_to_delete:
        if folder not in folders_to_keep and os.path.exists(folder):
            rmtree(folder)
    sys.stderr.write(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} Removed unnecessary folders after processing.\n")


if __name__ == '__main__':
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()  # Run your script here
    # profiler.disable()

    # Save the profiling data to a file
    # profiler.dump_stats("profiling_results.prof")

    # # Optional: Print profiling stats to the console
    # stats = pstats.Stats(profiler)
    # stats.strip_dirs()
    # stats.sort_stats("cumulative")  # Sort by cumulative time
    # stats.print_stats(20)  # Print the top 20 time-consuming functions
