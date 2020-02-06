# Face alignment demo
# Modified from https://github.com/lzx1413/pytorch_face_landmark
# cunjian@msu.edu
from __future__ import division
import argparse
import torch
import os
import cv2
import numpy as np
#import dlib
from common.utils import BBox,drawLandmark,drawLandmark_multiple
from models.basenet import ResNet, MobileNet_FCN
import matplotlib.pyplot as plt
from src import detect_faces
import glob
parser = argparse.ArgumentParser(description='PyTorch face landmark')
# Datasets
parser.add_argument('-img', '--image', default='face76', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu_id', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('-c', '--checkpoint', default='checkpoint/mobilenet_fcn_model_best.pth.tar', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

args = parser.parse_args()
mean = np.asarray([ 0.485, 0.456, 0.406 ])
std = np.asarray([ 0.229, 0.224, 0.225 ])

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

def load_model():
    #model = BaseNet()
    model = MobileNet_FCN(136)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoint, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__ == '__main__':
    out_size = 224
    model = load_model()
    model = model.eval()
    filenames=glob.glob("samples/12--Group/*.jpg")
    for imgname in filenames:
        print(imgname)
        img = cv2.imread(imgname)
        height,width,_=img.shape
        # perform face detection using MTCNN
        from PIL import Image
        image = Image.open(imgname)
        faces, landmarks = detect_faces(image)
        ratio=0
        if len(faces)==0:
            print('NO face is detected!')
            continue
        for k, face in enumerate(faces): 
            x1=face[0]
            y1=face[1]
            x2=face[2]
            y2=face[3]
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            size = int(max([w, h])*1.1)
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
            cropped_face = cv2.resize(cropped, (224, 224))

            if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
                continue
            test_face = cv2.resize(cropped_face,(224,224))
            test_face = test_face/255.0
            test_face = (test_face-mean)/std
            test_face = test_face.transpose((2, 0, 1))
            test_face = test_face.reshape((1,) + test_face.shape)
            input = torch.from_numpy(test_face).float()
            input= torch.autograd.Variable(input)
            landmark = model(input).cpu().data.numpy()
            landmark = landmark.reshape(-1,2)
            landmark = new_bbox.reprojectLandmark(landmark)
            img = drawLandmark_multiple(img, new_bbox, landmark)
  
        cv2.imwrite(os.path.join('results',os.path.basename(imgname)),img)

