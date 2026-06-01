"""
RUN Command: python -m source_code.vision.object_segmentation.object_segmentation
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
from collections import namedtuple
import math
import json
from source_code.vision.object_segmentation.image_preprocessing import threshold_image_RGB, threshold_image_HSV
from source_code.utility.paths import OBJ_SEGMENTATION_DIR

def I_pro(im):
    r_thresh, b_thresh ,g_thresh =  threshold_image_RGB(im)
    Area = [5000,500000]
    # Detect block
    extension =1.5 
    img_display = im.copy();
    lineWidth = 2
    MarkerSize = 8;
    min_area = Area[0]
    max_Area = Area[1]
    try:
        num_labels, labels,stats, centroids = cv2.connectedComponentsWithStats(r_thresh,connectivity=8, ltype = cv2.CV_32S)
        BlobFeature = namedtuple('BlobFeature',['label', 'area', 'uc', 'vc', 'bbox', 'theta', 'a', 'b'])
        m_red = []
        for i in range(1,num_labels):
            area = stats[i,cv2.CC_STAT_AREA]
            if(min_area<=area<=500000):
                uc, vc = centroids[i]
                bbox = stats[i,[cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
                blob_mask = (labels==i).astype('uint8')*255
                contours,_ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if(len(contours)>0):
                    ellipse_red = cv2.fitEllipse(contours[0])
                    (ex,ey),(a,b),theta = ellipse_red;
                    m_red.append(  BlobFeature(i,area,uc,vc,bbox,(theta*math.pi/180),a/2,b/2))
        #visualise
        
        for blob in m_red:
            if(blob.a>0 and blob.b>0):
                ellipse = ((blob.uc,blob.vc), (blob.a*2, blob.b*2), blob.theta*180/math.pi)
                cv2.ellipse(img_display,ellipse,(0,0,0),lineWidth)
        for blob in m_red:
            center = (int(blob.uc),int(blob.vc));
            cv2.circle(img_display,center,MarkerSize,(0,0,255),-1)
        if len(m_red)>0:
            x_m_red = m_red[0].uc
            y_m_red = m_red[0].vc
            m_red_center = np.array([x_m_red,y_m_red])
           
            theta_rad = m_red[0].theta
            x_min = x_m_red - math.cos(m_red[0].theta)*m_red[0].a*extension
            x_max = x_m_red + math.cos(m_red[0].theta)*m_red[0].a*extension
            y_min = y_m_red - math.sin(m_red[0].theta)*m_red[0].a*extension
            y_max = y_m_red + math.sin(m_red[0].theta)*m_red[0].a*extension
            cv2.line(img_display,(int(x_min),int(y_min)), (int(x_max), int(y_max)),(0,0,0),lineWidth)
    except Exception as e:
        print("No red block found")
    try:
        num_labels, labels,stats, centroids = cv2.connectedComponentsWithStats(g_thresh,connectivity=8, ltype = cv2.CV_32S);
        BlobFeature = namedtuple('BlobFeature',['label', 'area', 'uc', 'vc', 'bbox', 'theta', 'a', 'b'])
        m_green = []
        for i in range(1,num_labels):
            area = stats[i,cv2.CC_STAT_AREA]
            if(min_area<=area<=500000):
                uc, vc = centroids[i]
                bbox = stats[i,[cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
                blob_mask = (labels==i).astype('uint8')*255
                contours,_ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if(len(contours)>0):
                    ellipse_red = cv2.fitEllipse(contours[0])
                    (ex,ey),(a,b),theta = ellipse_red
                    m_green.append(  BlobFeature(i,area,uc,vc,bbox,(theta*math.pi/180),a/2,b/2))
        #visualise
        
        for blob in m_green:
            if(blob.a>0 and blob.b>0):
                ellipse = ((blob.uc,blob.vc), (blob.a*2, blob.b*2), blob.theta*180/math.pi)
                cv2.ellipse(img_display,ellipse,(0,0,0),lineWidth)
        for blob in m_green:
            center = (int(blob.uc),int(blob.vc));
            cv2.circle(img_display,center,MarkerSize,(0,255,0),-1)
        if len(m_green)>0:
            x_m_green = m_green[0].uc
            y_m_green = m_green[0].vc
            m_green_center = np.array([x_m_green,y_m_green])
           
            theta_rad = m_green[0].theta;
            x_min = x_m_green - math.cos(m_green[0].theta)*m_green[0].a*extension
            x_max = x_m_green + math.cos(m_green[0].theta)*m_green[0].a*extension
            y_min = y_m_green - math.sin(m_green[0].theta)*m_green[0].a*extension
            y_max = y_m_green + math.sin(m_green[0].theta)*m_green[0].a*extension
            cv2.line(img_display,(int(x_min),int(y_min)), (int(x_max), int(y_max)),(0,0,0),lineWidth)
    except Exception as e:
        print("No green block found")
    try:
        num_labels, labels,stats, centroids = cv2.connectedComponentsWithStats(b_thresh,connectivity=8, ltype = cv2.CV_32S)
        BlobFeature = namedtuple('BlobFeature',['label', 'area', 'uc', 'vc', 'bbox', 'theta', 'a', 'b'])
        m_blue = []
        for i in range(1,num_labels):
            area = stats[i,cv2.CC_STAT_AREA]
            if(min_area<=area<=500000):
                uc, vc = centroids[i];
                bbox = stats[i,[cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
                blob_mask = (labels==i).astype('uint8')*255
                contours,_ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if(len(contours)>0 ):
                    ellipse_red = cv2.fitEllipse(contours[0])
                    (ex,ey),(a,b),theta = ellipse_red
                    m_blue.append(  BlobFeature(i,area,uc,vc,bbox,(theta*math.pi/180),a/2,b/2))
        #visualise
        
        for blob in m_blue:
            if(blob.a>0 and blob.b>0):
                ellipse = ((blob.uc,blob.vc), (blob.a*2, blob.b*2), blob.theta*180/math.pi)
                cv2.ellipse(img_display,ellipse,(0,0,0),lineWidth)
        for blob in m_blue:
            center = (int(blob.uc),int(blob.vc))
            cv2.circle(img_display,center,MarkerSize,(255,0,0),-1)
        if len(m_blue)>0:
            x_m_blue = m_blue[0].uc
            y_m_blue = m_blue[0].vc
            m_blue_center = np.array([x_m_blue,y_m_blue])
           
            theta_rad = m_blue[0].theta
            x_min = x_m_blue - math.cos(m_blue[0].theta)*m_blue[0].a*extension
            x_max = x_m_blue + math.cos(m_blue[0].theta)*m_blue[0].a*extension
            y_min = y_m_blue - math.sin(m_blue[0].theta)*m_blue[0].a*extension
            y_max = y_m_blue + math.sin(m_blue[0].theta)*m_blue[0].a*extension
            cv2.line(img_display,(int(x_min),int(y_min)), (int(x_max), int(y_max)),(0,0,0),lineWidth)
            
    except Exception as e:
        print("No blue block found")
 
    processed_data = {
            "red_block_position": m_red_center.tolist(),
            "red_block_orientation": m_red[0].theta*(180/math.pi),
            "green_block_position": m_green_center.tolist(),
            "green_block_orientation": m_green[0].theta*(180/math.pi),
            "blue_block_position": m_blue_center.tolist(),
            "blue_block_orientation": m_blue[0].theta*(180/math.pi),
        }
    with open(OBJ_SEGMENTATION_DIR / "data" / "Pose.json",'w') as file:
        json.dump(processed_data,file,indent=3);

def main():
    I_pro(cv2.imread(OBJ_SEGMENTATION_DIR / "Img.png"))

if __name__ == "__main__":
    main()