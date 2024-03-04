import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import math
import matplotlib.animation as animation
import os
import glob
from natsort import natsorted

def remove_noise(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size),0)


def filter_region(image, vertices):
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2])
    return cv2.bitwise_and(image, mask)


DATA_DIR = "C:\\Users\\luluk\\OneDrive - imt-atlantique\\Imt_courses\\PROCOM\\ia_racing_imt-main\\supervise\\dataset_drive8\\images"
IM_DIR = "C:\\Users\\luluk\\OneDrive - imt-atlantique\\Imt_courses\\PROCOM\\ia_racing_imt-main\\supervise\\dataset_drive4\\images_preprocessed"
# Create the new image folder if it doesn't exist
if not os.path.exists(IM_DIR):
    os.makedirs(IM_DIR)



# Load each image and store them in an array
images = []
images = [cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2RGB) for file in natsorted(glob.glob(DATA_DIR+"\\*.jpg"))]
# for file in image_files:
#     image = cv2.imread(file, cv2.IMREAD_COLOR)
#     images.append(image)

#######################################################
# Now you have an array of images from the data directory
plt.imshow(images[0], animated=True)
plt.show(block=False)
plt.pause(0.005)
plt.clf()
#######################################################
# Define the source and destination points for the perspective transform
xs=[10, 150, 118, 35]
ys=[69,73,60,60]
src = np.float32([(xs[0], ys[0]), (xs[1], ys[1]), (xs[2], ys[2]), (xs[3], ys[3])])
# Destination points are to be parallel, taking into account the image size
dst = np.float32([[40, 120],
                [120, 120],
                [120, 0],
                [40, 0]])

# Calculate the transformation matrix and it's inverse transformation
M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)

#######################################################
TRESH_MIN = np.array([0, 75, 0],np.uint8)
TRESH_MAX = np.array([180, 255, 20],np.uint8)
TRESHY_MIN = np.array([0, 210, 210],np.uint8)
TRESHY_MAX = np.array([200, 255, 255],np.uint8)
TRESH3_MIN = np.array([0, 0, 0],np.uint8)
TRESH3_MAX = np.array([120, 180, 180],np.uint8)
#######################################################
verticesx=np.array([[(0, 60), (160, 60), (160, 120), (0, 120)]])

################################
# Lane detection configuration #
################################
nwindows = 40
window_height = math.floor(images[0].shape[0]//nwindows)

# Set the width of the windows +/- margin
margin = 5

# Set minimum number of pixels found to recenter window
minpix = 6

#######################################################
# Define conversions in x and y from pixels space to meters
ym_per_pix = 0.18/120 # meters per pixel in y dimension
xm_per_pix = 0.5/80 # meters per pixel in x dimension

#######################################################
video = cv2.VideoWriter("video_CV.avi", 0, 20, (480,360))

#######################################################
leftx_current = 40
rightx_current = 120
#######################################################
radstock = []

start_time = time.time()

p=0
r=0
l=0
#######################################################
for i in range(len(images)):
    imagex=images[i]
    ##########################################################
    # Convert to grayscale, blur, and perform edge detection #
    ##########################################################
    test_image = imagex
    #test_image = remove_noise(image=test_image, kernel_size=(1, 1))
    test_image = cv2.warpPerspective(test_image, M, (160,120))
    # test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HLS)
    test_image = 255-cv2.inRange(test_image, TRESH3_MIN, TRESH3_MAX)
    # #stock = test_image
    # #test_image = test_image-cv2.inRange(imagex, TRESHY_MIN, TRESHY_MAX)
    # #test_image = filter_region(image=test_image,vertices=verticesx)
    # # #stock =filter_region(image=stock,vertices=verticesx)

    # # #########################
    # # # Perspective transform #
    # # #########################
    # # test_image = cv2.warpPerspective(test_image, M, (160,120))
    
    # #test_image = cv2.rotate(test_image, cv2.ROTATE_180)
    # # #stock = cv2.warpPerspective(stock, M, (160,120))

    ####################################################
    # Take a histogram of the bottom half of the image #
    ####################################################
    try:

        histogram = np.sum(test_image[0:10,:], axis=0)
        histogrambis = np.sum(test_image[100:120,:], axis=0)
        midpoint = math.floor((histogram.nonzero()[0][0]+histogram.nonzero()[0][-1])//2)
        ###############################################################
        # Find the peak of the left and right halves of the histogram #
        ###############################################################
        if i==0 or histogram[max(leftx_base-margin,0):min(leftx_base+margin,160)+1]==0 or histogram[max(rightx_base-margin,0):min(rightx_base+margin,160)+1]==0 or histogram.nonzero()[0][-1]-histogram.nonzero()[0][0]>85 or l>=5 or r>=5:          
            leftx_base = midpoint-histogram[:midpoint][::-1].nonzero()[0][0]-3
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint + 3
            if rightx_base-leftx_base<60:
                if np.mean(histogrambis.nonzero()[0])>80:
                    leftx_base=0
                else:
                    rightx_base=160           
    except:
        pass

    # Set height of windows
    leftx_current = leftx_base
    rightx_current = rightx_base
    nonzeroy = np.array(test_image.nonzero()[0])
    nonzerox = np.array(test_image.nonzero()[1])
    if i==0 or ((left_lanex_inds.shape==(0,) and left_laney_inds.shape == (0,)) and (right_lanex_inds.shape == (0,) and right_laney_inds.shape == (0,))):
        left_lanex_inds = []
        right_lanex_inds = []
        left_laney_inds = []
        right_laney_inds = []
        try:
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_high = (window+1)*window_height
                win_y_low = window*window_height

                win_xleft_low = math.floor(max(leftx_current - margin,0))
                win_xleft_high = math.floor(min(leftx_current + margin,test_image.shape[1]))
                win_xright_low = math.floor(max(rightx_current - margin,0))
                win_xright_high = math.floor(min(rightx_current + margin,test_image.shape[1]))

            
                if win_xleft_low>=(win_y_low-25)/3.1:
                    # Identify the nonzero pixels in x and y within the window #
                    good_lefty_inds = nonzeroy[((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high))]
                    good_leftx_inds = nonzerox[((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high))]

                    # Append these indices to the lists
                    left_laney_inds.append(good_lefty_inds)
                    left_lanex_inds.append(good_leftx_inds)

                    # If you found > minpix pixels, recenter next window on their mean position
                    if len(good_leftx_inds) > minpix:
                        leftx_current = math.ceil(np.mean(good_leftx_inds))

                if win_xright_high<=(700-win_y_low)/4.7:
                    good_righty_inds = nonzeroy[((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high))]
                    good_rightx_inds = nonzerox[((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high))]

                    #Append these indices to the lists
                    right_laney_inds.append(good_righty_inds)
                    right_lanex_inds.append(good_rightx_inds)

                    # If you found > minpix pixels, recenter next window on their mean position
                    if len(good_rightx_inds) > minpix:
                        rightx_current = math.ceil(np.mean(good_rightx_inds))

            # Concatenate the arrays of indices
            left_laney_inds = np.concatenate(left_laney_inds)
            right_laney_inds = np.concatenate(right_laney_inds)
            left_lanex_inds = np.concatenate(left_lanex_inds)
            right_lanex_inds = np.concatenate(right_lanex_inds)
        except:
            left_laney_inds = []
            right_laney_inds = []
            left_lanex_inds = []
            right_lanex_inds = []
    else:
        try:
            left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                        left_fit[1]*nonzeroy + left_fit[2] + margin)))
            left_lanex_inds = nonzerox[left_lane_inds]
            left_laney_inds = nonzeroy[left_lane_inds]
            leftx_base = math.ceil(np.mean(left_lanex_inds[:10]))
        except:
            left_lane_inds = []
            left_lanex_inds = []
            left_laney_inds = []
        try:     
            right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                        right_fit[1]*nonzeroy + right_fit[2] + margin)))
            right_lanex_inds = nonzerox[right_lane_inds]
            right_laney_inds = nonzeroy[right_lane_inds]
            rightx_base = math.ceil(np.mean(right_lanex_inds[:10]))

        except:
            right_lane_inds = []
            right_lanex_inds = []
            right_laney_inds = []

    try:
        left_fit = np.polyfit(left_laney_inds, left_lanex_inds, 2)
        ltrue=True
        l=0
    except:
        left_fit = [0,0,0]
        ltrue=False
        l+=1
    try: 
        right_fit = np.polyfit(right_laney_inds,right_lanex_inds, 2)
        rtrue=True
        r=0
    except:
        right_fit = [0,0,0]
        rtrue=False
        r+=1
    


    # plt.imshow(cv2.warpPerspective(imagex,M,(160,120)))
    if rtrue:
        try:
            xrs = np.linspace(0, test_image.shape[1], 100)
            yrs = right_fit[0] * xrs ** 2 + right_fit[1] * xrs + right_fit[2]
            y_eval_r = np.max(yrs)
            y_max = test_image.shape[0]
            right_fit_cr = np.polyfit(yrs*ym_per_pix, xrs*xm_per_pix, 2)
            right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_r*ym_per_pix + right_fit_cr[1])**2)**1.5) / (2*right_fit_cr[0])
            right_x_pos = right_fit[0]*y_max**2 + right_fit[1]*y_max + right_fit[2]
        except:
            right_fit_cr = [0,0,0]
            right_curverad = np.inf
            right_x_pos = np.inf
            xrs = []
            yrs = []
    else:
        right_fit_cr = [0,0,0]
        right_curverad = np.inf
        left_x_pos = np.inf
        xrs = []
        yrs = []

    if ltrue:
        try:
            xls = np.linspace(0, test_image.shape[1], 100)
            yls = left_fit[0] * xls ** 2 + left_fit[1] * xls + left_fit[2]
            y_eval_l = np.max(yls)
            y_max = test_image.shape[0]
            left_fit_cr = np.polyfit(yls*ym_per_pix, xls*xm_per_pix, 2)
            left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_l*ym_per_pix + left_fit_cr[1])**2)**1.5) / (2*left_fit_cr[0])
            left_x_pos = left_fit[0]*y_max**2 + left_fit[1]*y_max + left_fit[2]
        except:
            left_fit_cr = [0,0,0]
            left_curverad = np.inf
            left_x_pos = np.inf
            xls = []
            yls = []
    else:
        left_fit_cr = [0,0,0]
        left_curverad = np.inf
        right_x_pos = np.inf
        xls = []
        yls = []

    # Calculate the x position of the center of the lane 
    if left_x_pos != np.inf and right_x_pos != np.inf:
        center_lanes_x_pos = (left_x_pos + right_x_pos)//2

    elif left_x_pos != np.inf:
        center_lanes_x_pos = left_x_pos+42

    elif right_x_pos != np.inf:
        center_lanes_x_pos = right_x_pos-42 


    # Calculate the deviation between the center of the lane and the center of the picture
    # The car is assumed to be placed in the center of the picture
    # If the deviation is negative, the car is on the felt hand side of the center of the lane
    veh_pos = ((test_image.shape[1]//2) - center_lanes_x_pos) * xm_per_pix

    warp_zero = np.zeros_like(test_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    line = np.dstack((warp_zero, warp_zero, warp_zero))
    linebis = np.dstack((warp_zero, warp_zero, warp_zero))


    if ltrue:
        pts_left_lane = np.array([(np.vstack([np.array(yls)+2.5, np.array(xls)])).T],dtype=np.int32)
        pts_left = np.array([(np.vstack([np.array(yls), np.array(xls)])).T],dtype=np.int32)
    else:
        pts_left = np.array([],dtype=np.int32)
        pts_left_lane = np.array([],dtype=np.int32)

    if rtrue:
        pts_right_lane = np.array([np.flipud((np.vstack([np.array(yrs)-2.5, np.array(xrs)])).T)],dtype=np.int32)
        pts_right = np.array([np.flipud((np.vstack([np.array(yrs), np.array(xrs)])).T)],dtype=np.int32)
    else:
        pts_right = np.array([],dtype=np.int32)
        pts_right_lane = np.array([],dtype=np.int32)


    
    try:
        pts_lane = np.hstack((pts_left_lane, pts_right_lane))
        cv2.fillPoly(color_warp, pts_lane.round(0), (0, 255, 0))
    except:
        pass
    if ltrue:
        cv2.polylines(line, pts_left.round(0), isClosed=False, color=(255, 0, 0), thickness=5)
        cv2.polylines(linebis, pts_left.round(0), isClosed=False, color=(255, 255, 255), thickness=5)


    if rtrue:
        cv2.polylines(line, pts_right.round(0), isClosed=False, color=(255, 0, 0), thickness=5)
        cv2.polylines(linebis, pts_right.round(0), isClosed=False, color=(255, 255, 255), thickness=5)



    tosave = color_warp | linebis
    # Save the image in the new folder
    nametosave = str(str(i)+"_cam_image_array_.jpg")
    #cv2.imwrite(os.path.join(IM_DIR, nametosave), tosave)
        
    # Warp the blank back to original image space using inverse perspective matrix (M_inv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (imagex.shape[1], imagex.shape[0]))
    newwarpline = cv2.warpPerspective(line, M_inv, (imagex.shape[1], imagex.shape[0]))
    newwarpmask = cv2.warpPerspective(linebis, M_inv, (imagex.shape[1], imagex.shape[0]))
    # Combine the result with the original image
    out_img = imagex
    mask = newwarpmask & imagex
    out_img = out_img - mask
    out_img = cv2.addWeighted(out_img, 1, newwarpline, 1, 0)
    out_img = cv2.addWeighted(out_img, 1, newwarp, 0.4, 0)
    out_img = cv2.line(out_img, (80, 70), (80, 120), (255, 0, 0), 1)

    resized = cv2.resize(out_img, (480,360), interpolation = cv2.INTER_LANCZOS4)
    rad = (left_curverad+right_curverad)/2
    if abs(left_curverad)>=1.2 and abs(right_curverad)<1.2:
        rad=np.sign(right_curverad)*0.25+right_curverad
    elif abs(right_curverad)>=1.2 and abs(left_curverad)<1.2:
        rad=np.sign(left_curverad)*0.25+left_curverad
    elif abs(left_curverad)>=1.2 and abs(right_curverad)>=1.2:
        rad=np.inf
    # if rad!=np.inf:
    #     if len(radstock)>=3:
    #         radstock.pop(0)
    #     radstock.append(rad)
    #     radmean=np.mean(radstock)
    #     k=0
    # elif k>=5:
    #     radmean=np.inf
    #     k+=1
    # else:
    #     radmean=np.mean(radstock)
    #     k+=1
    cv2.putText(resized,'Curve Radius [m]: '+ str(np.round(rad,2)),(10,15), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),1,cv2.LINE_AA)
    cv2.putText(resized,'Center Offset [m]: '+ str(np.round(veh_pos,2)),(10,30), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(resized,'Lfit: '+str(np.round(left_fit_cr[0],2)),(10,45), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(resized,'Rfit: '+str(np.round(right_fit_cr[0],2)),(10,60), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),1,cv2.LINE_AA)
    # cv2.putText(resized,'image n '+str(i),(10,75), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),1,cv2.LINE_AA)
    plt.imshow(resized,animated=True)
    plt.draw()
    plt.pause(0.1-time.time()+start_time)
    plt.clf()
    plt.title('Fps:'+str(np.round(1/(time.time()-start_time),1)),loc='left')
    start_time = time.time()
    #video.write(resized)

video.release()
