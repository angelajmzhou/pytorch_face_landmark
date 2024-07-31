# Face alignment and crop demo
# Uses MTCNN, FaceBoxes or Retinaface as a face detector;
# Support different backbones, include PFLD, MobileFaceNet, MobileNet;
# Retinaface+MobileFaceNet gives the best peformance
# Cunjian Chen (ccunjian@gmail.com), Feb. 2021
from __future__ import division
import argparse
import glob
import os
import shutil
import sys
import time

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import distance as dist
import torch

from common.utils import BBox, drawLandmark, drawLandmark_multiple
from FaceBoxes import FaceBoxes
from Retinaface import Retinaface
from MTCNN import detect_faces
from blur_detection import detect_blur_fft, variance_of_laplacian, hwd_blur_detect
from models.basenet import MobileNet_GDConv
from models.pfld_compressed import PFLDInference
from models.mobilefacenet import MobileFaceNet
from SPIGA.spiga.demo.visualize.plotter import Plotter
from SPIGA.spiga.inference.config import ModelConfig
from SPIGA.spiga.inference.framework import SPIGAFramework
from utils.align_trans import get_reference_facial_points, warp_and_crop_face
from filediff import compare_folders

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def SPIGA_eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[7])
    B = dist.euclidean(eye[2], eye[6])
    C = dist.euclidean(eye[3], eye[5])
    D = dist.euclidean(eye[0], eye[4])
    ear = (A + B + C) / (3.0 * D)
    return ear


parser = argparse.ArgumentParser(description='Blink and Blur Detection')
# Datasets
parser.add_argument('--backbone', default='MobileFaceNet', type=str,
                    help='choose which backbone network to use: MobileNet, PFLD, MobileFaceNet, SPIGA')
parser.add_argument('--detector', default='Retinaface', type=str,
                    help='choose which face detector to use: MTCNN, FaceBoxes, Retinaface')
parser.add_argument('--spigaconfig', default='wflw', type=str,
                    help='choose SPIGA config: wflw, 300wpublic, 300wprivate, merlrav, cofw68')
parser.add_argument('--showlandmark', default=False, type=bool,
                    help='choose whether to show facial landmarking on output images')
parser.add_argument('--blurdetect', default='fft', type=str,
                    help='choose blur detection method: fft, vol, hwd')
parser.add_argument('--filepath', default='data/', type=str,
                    help='image directory in data/*.JPG (or other filetype) format')
args = parser.parse_args()
mean = np.asarray([ 0.485, 0.456, 0.406 ])
std = np.asarray([ 0.229, 0.224, 0.225 ])

crop_size= 112
scale = crop_size / 112.
reference = get_reference_facial_points(default_square = True) * scale
total=0
count=0

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
        model.load_state_dict(checkpoint['state_dict'])
    elif args.backbone=='PFLD':
        model = PFLDInference() 
        # download from https://drive.google.com/file/d/1gjgtm6qaBQJ_EY7lQfQj3EuMJCVg9lVu/view?usp=sharing
        checkpoint = torch.load('checkpoint/pfld_model_best.pth.tar', map_location=map_location)
        print('Use PFLD as backbone') 
        model.load_state_dict(checkpoint['state_dict'])
        # download from https://drive.google.com/file/d/1T8J73UTcB25BEJ_ObAJczCkyGKW5VaeY/view?usp=sharing
    elif args.backbone=='MobileFaceNet':
        model = MobileFaceNet([112, 112],136)   
        checkpoint = torch.load('checkpoint/mobilefacenet_model_best.pth.tar', map_location=map_location) 
        model.load_state_dict(checkpoint['state_dict'])     
        print('Use MobileFaceNet as backbone') 
    elif args.backbone =='SPIGA':
        model = SPIGAFramework(ModelConfig(args.spigaconfig))
        print("Use SPIGA as backbone")        
    else:
        print('Error: not supported backbone')    
    return model

if __name__ == '__main__':
    if os.path.exists('faces_detected'):
        shutil.rmtree('faces_detected')
    if os.path.exists('not_viable'):
        shutil.rmtree('not_viable')
    if os.path.exists('viable'):
        shutil.rmtree('viable')
    os.makedirs('faces_detected')
    os.makedirs('viable')
    os.makedirs('not_viable')
    if args.backbone=='MobileNet':
        out_size = 224
    else:
        out_size = 112 
    model = load_model()
    if args.backbone != 'SPIGA':
        model = model.eval()
    filenames=glob.glob(args.filepath)

    for imgname in filenames:
        print(imgname)
        img = cv2.imread(imgname)
        org_img = Image.open(imgname)
        height,width,_=img.shape
        if args.detector=='MTCNN':
            # perform face detection using MTCNN
            image = Image.open(imgname)
            faces, landmarks = detect_faces(image)
        elif args.detector=='FaceBoxes':
            face_boxes = FaceBoxes()
            faces = face_boxes(img)
        elif args.detector=='Retinaface':
            retinaface=Retinaface.Retinaface()    
            faces = retinaface(img)            
        else:
            print('Error: not supported detector')        
        ratio=0
        if len(faces)==0:
            print('NO face is detected!')
            continue
        for k, face in enumerate(faces): 
            if face[4]<0.9: # remove low confidence detection
                continue

            #calculate bbox
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
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            blurry=False

            if args.blurdetect == ' fft':
                mean, blurry = detect_blur_fft(gray, imgname, size= 60, thresh = -1)
                img = cv2.putText(img,("mean: " + str(mean)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA) 
            elif args.blurdetect == ' vol':
                blurry = variance_of_laplacian(gray, thresh = 10)
            elif args.blurdetect == ' hwd':
                #low threshold -> more likely blur classification 
                #low minZero -> less likely
                per, blurry = hwd_blur_detect(gray)
                img = cv2.putText(img,("Per: " + str(per)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA) 
            else:
                print("Invalid blur detection method.")
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
            elif args.backbone == 'SPIGA':
                features = model.inference(img, [[new_bbox.x,new_bbox.y,new_bbox.w,new_bbox.h]])
                landmark = np.array(features['landmarks'][0])
            else:
                landmark = model(input).cpu().data.numpy()
            end = time.time()
            print('Time: {:.6f}s.'.format(end - start))
            total += end-start
            count +=1
            landmark = landmark.reshape(-1,2)
            # reshape into any # rows, 2 columns
            if args.showlandmark:
                if args.backbone == 'SPIGA':
                    plotter = Plotter()
                    img = plotter.landmarks.draw_landmarks(img, landmark)
                    headpose = np.array(features['headpose'][0])
                    img = plotter.hpose.draw_headpose(img, [x1,y1,x2,y2], headpose[:3], headpose[3:], euler=True)
                landmark = new_bbox.reprojectLandmark(landmark)
                img = drawLandmark_multiple(img, new_bbox, landmark)

            # Calculate EAR for both eyes
            if args.backbone == 'SPIGA':
                left_eye = landmark[60:68]
                right_eye = landmark[68:76]
                left_ear = SPIGA_eye_aspect_ratio(left_eye)
                right_ear = SPIGA_eye_aspect_ratio(right_eye)
            else:
                left_eye = landmark[36:42]
                right_eye = landmark[42:48]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
            # Average the eye aspect ratio
            ear = (left_ear + right_ear) / 2.0
            print("EAR: "+str(ear))
            # Threshold for closed eyes (typically around 0.2)
            EYE_AR_THRESH = 0.2
            open=True
            if ear < EYE_AR_THRESH:
              open=False
            cv2.imwrite(os.path.join('faces_detected',os.path.basename(imgname)),img)
        #img = drawLandmark_multiple(img, new_bbox, facial5points)  # plot and show 5 points 
        if open and not blurry:
            cv2.imwrite(os.path.join('viable', os.path.basename(imgname)),img)# img
        else:
            cv2.imwrite(os.path.join('not_viable', os.path.basename(imgname)),img)# img
        cv2.imwrite(os.path.join('results',os.path.basename(imgname)),img)
    print("average time: {:.6f}".format(total/count))
    folder = ''
    if args.filepath[5:9] == 'open':
        folder = 'viable'
    if args.filepath[5:9] == 'clos':
        folder = 'not_viable'
        
    only_in_folder1, only_in_folder2, in_both_folders, count_folder1, count_folder2 = compare_folders(folder,args.filepath[:-5])

    print(f"\nTotal files in folder 1: {count_folder1}")
    print(f"Total files in folder 2: {count_folder2}")

    print("\nFiles only in folder 1:")
    for file in only_in_folder1:
        print(file)

    print("\nFiles only in folder 2:")
    for file in only_in_folder2:
        print(file)

    print("\nFiles that are in both folders:")
    for file in in_both_folders:
        print(file)
    print("\nPercent diff: "+ str(count_folder1/count_folder2))
    print("average time: "+str(total/count))
