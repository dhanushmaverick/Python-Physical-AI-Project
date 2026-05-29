import cv2
import numpy as np
from collections import namedtuple
from source_code.utility.paths import OBJ_SEGMENTATION_DIR


def threshold_image_RGB(im):
#Only 5 params can be customised: The threshold vals for R,G and B channels and the min and max areas for iblobs function.
    #im = cv2.imread(im)
    thresh_vals = [0.45, 0.39, 0.55];
    B, G, R = cv2.split(im)

    R = R.astype(np.float32)
    G = G.astype(np.float32)
    B = B.astype(np.float32)

   
    eps = 1e-6;
    r = R  / (R+G+B+eps) 
    
    g = G  / (R+G+B+eps) 
   
    b = B  / (R+G+B+eps) 
    
    
    r_thresh_val = thresh_vals[0]
    g_thresh_val = thresh_vals[1]
    b_thresh_val = thresh_vals[2]
    
    #area_min = 500;
    #area_max = 500000;
       

    r_thresh = (r > r_thresh_val).astype('uint8') * 255
    g_thresh = (g>g_thresh_val).astype('uint8')*255
    b_thresh = (b>b_thresh_val).astype('uint8')*255
     
     # Morphological cleanup
    kernel = np.ones((5,5), np.uint8)
    r_thresh = cv2.morphologyEx(r_thresh,cv2.MORPH_ERODE,kernel)
    r_thresh = cv2.morphologyEx(r_thresh,cv2.MORPH_OPEN,kernel)
    r_thresh = cv2.morphologyEx(r_thresh,cv2.MORPH_DILATE,kernel)

    g_thresh = cv2.morphologyEx(g_thresh,cv2.MORPH_ERODE,kernel)
    g_thresh = cv2.morphologyEx(g_thresh,cv2.MORPH_OPEN,kernel)
    g_thresh = cv2.morphologyEx(g_thresh,cv2.MORPH_DILATE,kernel)

    b_thresh = cv2.morphologyEx(b_thresh,cv2.MORPH_ERODE,kernel)
    b_thresh = cv2.morphologyEx(b_thresh, cv2.MORPH_OPEN, kernel)
    b_thresh = cv2.morphologyEx(b_thresh,cv2.MORPH_DILATE,kernel)


    #_,r_thresh =cv2.threshold(cv2.imread('scripts/Bin_img.jpeg',cv2.IMREAD_GRAYSCALE),127,255,cv2.THRESH_BINARY); #(r>0.6).astype('uint8')*255;
    
    return r_thresh, b_thresh, g_thresh

def threshold_image_HSV(im): # NON FUNCTIONAL !!! --> doesn't work for BLUE!
#Only 5 params can be customised: The threshold vals for R,G and B channels and the min and max areas for iblobs function.
    im = cv2.imread(str(im))

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(hsv)

    # Adaptive thresholds
    sat_thresh = np.mean(S) * 0.5
    val_thresh = np.mean(V) * 0.4

    # ---------------- RED ----------------
    red_mask1 = (
        (H >= 0) & (H <= 10) &
        (S > sat_thresh) &
        (V > val_thresh)
    )

    red_mask2 = (
        (H >= 170) & (H <= 179) &
        (S > sat_thresh) &
        (V > val_thresh)
    )
    r_thresh = ((red_mask1 | red_mask2).astype(np.uint8)) * 255

    # ---------------- GREEN ----------------
    green_mask = (
        (H >= 35) & (H <= 85) &
        (S > sat_thresh) &
        (V > val_thresh)
    )

    g_thresh = (green_mask.astype(np.uint8)) * 255

    # ---------------- BLUE ----------------
    blue_mask = (
        (H >= 100) & (H <= 115) &
        (S > sat_thresh) &
        (V > val_thresh)
)
    b_thresh = (blue_mask.astype(np.uint8)) * 255

    # Morphological cleanup
    kernel = np.ones((3,3), np.uint8)

    #r_thresh = cv2.morphologyEx(r_thresh,cv2.MORPH_OPEN,kernel)
    #g_thresh = cv2.morphologyEx(g_thresh,cv2.MORPH_OPEN,kernel)
    b_thresh = cv2.morphologyEx(b_thresh,cv2.MORPH_OPEN,kernel)

    # Visualize
    cv2.imshow("Red HSV", r_thresh)
    cv2.imshow("Green HSV", g_thresh)
    cv2.imshow("Blue HSV", b_thresh)

    cv2.waitKey(0)

    return r_thresh, b_thresh, g_thresh

def main():
    threshold_image_RGB(OBJ_SEGMENTATION_DIR /"Img.jpeg")
    #threshold_image_HSV(im=OBJ_SEGMENTATION_DIR /"Img.jpeg")

if __name__ == "__main__":
    main()