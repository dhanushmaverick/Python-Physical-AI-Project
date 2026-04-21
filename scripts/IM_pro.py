import cv2;
import numpy as np;
from collections import namedtuple;
import math
def I_pro():
    im = cv2.imread("scripts/Img.jpeg");
    
    B, G, R = cv2.split(im)

    R = R.astype(np.float32)
    G = G.astype(np.float32)
    B = B.astype(np.float32)

    cv2.imshow("Image",im)
    cv2.waitKey(0)
    eps = 1e-6;
    r = R  / (R+G+B+eps) ;
    cv2.imshow("Red", r);
    cv2.waitKey(0)
    g = G  / (R+G+B+eps) ;
    cv2.imshow("Green", g);
    cv2.waitKey(0);
    b = B  / (R+G+B+eps) ;
    cv2.imshow("Blue", b);
    cv2.waitKey(0)

    
    r_thresh = (r > 0.5).astype('uint8') * 255
    #_,r_thresh =cv2.threshold(cv2.imread('scripts/Bin_img.jpeg',cv2.IMREAD_GRAYSCALE),127,255,cv2.THRESH_BINARY); #(r>0.6).astype('uint8')*255;
    cv2.imshow("Red Binary",r_thresh);
    cv2.waitKey(0)
    g_thresh = (g>0.4).astype('uint8')*255;
    cv2.imshow("Green Binary",g_thresh);
    cv2.waitKey(0)
    b_thresh = (b>0.5).astype('uint8')*255;
    cv2.imshow("Blue Binary", b_thresh)
    cv2.waitKey(0)
  
    # Detect block
    extension =1.5 ;
    img_display = im.copy();
    lineWidth = 2;
    MarkerSize = 8;
    try:
        num_labels, labels,stats, centroids = cv2.connectedComponentsWithStats(r_thresh,connectivity=8, ltype = cv2.CV_32S);
        BlobFeature = namedtuple('BlobFeature',['label', 'area', 'uc', 'vc', 'bbox', 'theta', 'a', 'b']);
        m_red = [];
        for i in range(1,num_labels):
            area = stats[i,cv2.CC_STAT_AREA];
            if(100<=area<=500000):
                uc, vc = centroids[i];
                bbox = stats[i,[cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]];
                blob_mask = (labels==i).astype('uint8')*255;
                contours,_ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);
                if(len(contours)>0):
                    ellipse_red = cv2.fitEllipse(contours[0]);
                    (ex,ey),(a,b),theta = ellipse_red;
                    m_red.append(  BlobFeature(i,area,uc,vc,bbox,theta*math.pi/180,a/2,b/2));
        #visualise
        
        for blob in m_red:
            if(blob.a>0 and blob.b>0):
                ellipse = ((blob.uc,blob.vc), (blob.a*2, blob.b*2), blob.theta*180/math.pi);
                cv2.ellipse(img_display,ellipse,(0,0,0),lineWidth);
        for blob in m_red:
            center = (int(blob.uc),int(blob.vc));
            cv2.circle(img_display,center,MarkerSize,(0,0,255),-1);
        if len(m_red)>0:
            x_m_red = m_red[0].uc;
            y_m_red = m_red[0].vc;
            m_red_center = np.array([x_m_red,y_m_red]);
           
            theta_rad = m_red[0].theta;
            x_min = x_m_red - math.cos(m_red[0].theta)*m_red[0].a*extension;
            x_max = x_m_red + math.cos(m_red[0].theta)*m_red[0].a*extension;
            y_min = y_m_red - math.sin(m_red[0].theta)*m_red[0].a*extension;
            y_max = y_m_red + math.sin(m_red[0].theta)*m_red[0].a*extension;
            cv2.line(img_display,(int(x_min),int(y_min)), (int(x_max), int(y_max)),(0,0,0),lineWidth);
    except Exception as e:
        print("No red block found");
    try:
        num_labels, labels,stats, centroids = cv2.connectedComponentsWithStats(g_thresh,connectivity=8, ltype = cv2.CV_32S);
        BlobFeature = namedtuple('BlobFeature',['label', 'area', 'uc', 'vc', 'bbox', 'theta', 'a', 'b']);
        m_green = [];
        for i in range(1,num_labels):
            area = stats[i,cv2.CC_STAT_AREA];
            if(500<=area<=500000):
                uc, vc = centroids[i];
                bbox = stats[i,[cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]];
                blob_mask = (labels==i).astype('uint8')*255;
                contours,_ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);
                if(len(contours)>0):
                    ellipse_red = cv2.fitEllipse(contours[0]);
                    (ex,ey),(a,b),theta = ellipse_red;
                    m_green.append(  BlobFeature(i,area,uc,vc,bbox,theta*math.pi/180,a/2,b/2));
        #visualise
        
        for blob in m_green:
            if(blob.a>0 and blob.b>0):
                ellipse = ((blob.uc,blob.vc), (blob.a*2, blob.b*2), blob.theta*180/math.pi);
                cv2.ellipse(img_display,ellipse,(0,0,0),lineWidth);
        for blob in m_green:
            center = (int(blob.uc),int(blob.vc));
            cv2.circle(img_display,center,MarkerSize,(0,255,0),-1);
        if len(m_green)>0:
            x_m_green = m_green[0].uc;
            y_m_green = m_green[0].vc;
            m_green_center = np.array([x_m_green,y_m_green]);
           
            theta_rad = m_green[0].theta;
            x_min = x_m_green - math.cos(m_green[0].theta)*m_green[0].a*extension;
            x_max = x_m_green + math.cos(m_green[0].theta)*m_green[0].a*extension;
            y_min = y_m_green - math.sin(m_green[0].theta)*m_green[0].a*extension;
            y_max = y_m_green + math.sin(m_green[0].theta)*m_green[0].a*extension;
            cv2.line(img_display,(int(x_min),int(y_min)), (int(x_max), int(y_max)),(0,0,0),lineWidth);
    except Exception as e:
        print("No green block found");
    try:
        num_labels, labels,stats, centroids = cv2.connectedComponentsWithStats(b_thresh,connectivity=8, ltype = cv2.CV_32S);
        BlobFeature = namedtuple('BlobFeature',['label', 'area', 'uc', 'vc', 'bbox', 'theta', 'a', 'b']);
        m_blue = [];
        for i in range(1,num_labels):
            area = stats[i,cv2.CC_STAT_AREA];
            if(500<=area<=500000):
                uc, vc = centroids[i];
                bbox = stats[i,[cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]];
                blob_mask = (labels==i).astype('uint8')*255;
                contours,_ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);
                if(len(contours)>0 ):
                    ellipse_red = cv2.fitEllipse(contours[0]);
                    (ex,ey),(a,b),theta = ellipse_red;
                    m_blue.append(  BlobFeature(i,area,uc,vc,bbox,theta*math.pi/180,a/2,b/2));
        #visualise
        
        for blob in m_blue:
            if(blob.a>0 and blob.b>0):
                ellipse = ((blob.uc,blob.vc), (blob.a*2, blob.b*2), blob.theta*180/math.pi);
                cv2.ellipse(img_display,ellipse,(0,0,0),lineWidth);
        for blob in m_blue:
            center = (int(blob.uc),int(blob.vc));
            cv2.circle(img_display,center,MarkerSize,(255,0,0),-1);
        if len(m_blue)>0:
            x_m_blue = m_blue[0].uc;
            y_m_blue = m_blue[0].vc;
            m_blue_center = np.array([x_m_blue,y_m_blue]);
           
            theta_rad = m_blue[0].theta;
            x_min = x_m_blue - math.cos(m_blue[0].theta)*m_blue[0].a*extension;
            x_max = x_m_blue + math.cos(m_blue[0].theta)*m_blue[0].a*extension;
            y_min = y_m_blue - math.sin(m_blue[0].theta)*m_blue[0].a*extension;
            y_max = y_m_blue + math.sin(m_blue[0].theta)*m_blue[0].a*extension;
            cv2.line(img_display,(int(x_min),int(y_min)), (int(x_max), int(y_max)),(0,0,0),lineWidth);
            
    except Exception as e:
        print("No blue block found");
    cv2.imshow("Block detection: ",img_display);
    cv2.waitKey(0);
I_pro();