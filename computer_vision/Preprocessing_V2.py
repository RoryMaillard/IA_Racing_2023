import cv2
import glob
from natsort import natsorted
import numpy as np
import math

def load_images(data_dir):
    images_list = []
    image_files = natsorted(glob.glob(data_dir + "\\*.jpg"))
    for file in image_files:
        images_list.append(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB))
    return images_list

def perspective_transform(image, inv=False):
    src = np.float32([(10, 69), (150, 73), (118, 60), (35, 60)])
    dst = np.float32([[40, 120], [120, 120], [120, 0], [40, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    if inv:
        image_warped = cv2.warpPerspective(image, M_inv, (160, 120))
    else:
        image_warped = cv2.warpPerspective(image, M, (160, 120))
    return image_warped

def mask_image(image):
    TRESH_MIN = np.array([0, 0, 0],np.uint8)
    TRESH_MAX = np.array([120, 180, 180],np.uint8)
    image_masked=255-cv2.inRange(image, TRESH_MIN, TRESH_MAX)
    return image_masked

def calculate_fits(image, nwindows, margin, minpix, hist_height=10, calculate_all=True, leftx_base=40, rightx_base=120, left_fit=[0,0,0], right_fit=[0,0,0]):
    nonzeroy = np.array(image.nonzero()[0])
    nonzerox = np.array(image.nonzero()[1])

    # Calculate everything
    if calculate_all:
        try:
            histogram_up = np.sum(image[0:hist_height,:], axis=0)
            histogram_down = np.sum(image[image.shape[0]-hist_height:,:], axis=0)
            out_img = np.dstack((image, image, image))*255
            midpoint = math.floor((histogram_up.nonzero()[0][0]+histogram_up.nonzero()[0][-1])//2)
            if histogram_up[max(leftx_base-margin,0):min(leftx_base+margin,160)+1]==0 or histogram_up[max(rightx_base-margin,0):min(rightx_base+margin,160)+1]==0 or histogram_up.nonzero()[0][-1]-histogram_up.nonzero()[0][0]>85:         
                leftx_base = midpoint-histogram_up[:midpoint][::-1].nonzero()[0][0]-3
                rightx_base = np.argmax(histogram_up[midpoint:]) + midpoint + 3
                if rightx_base-leftx_base<60:
                    if np.mean(histogram_down.nonzero()[0])>image.shape[1]/2:
                        leftx_base=0
                    else:
                        rightx_base=160
        except:
            pass

        leftx_current = leftx_base
        rightx_current = rightx_base
        window_height = math.floor(image.shape[0]/nwindows)
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = (window+1)*window_height
            win_y_high = window*window_height
            win_xleft_low = math.floor(max(leftx_current - margin,0))
            win_xleft_high = math.floor(min(leftx_current + margin,image.shape[1]))
            win_xright_low = math.floor(max(rightx_current - margin,0))
            win_xright_high = math.floor(min(rightx_current + margin,image.shape[1]))
            cv2.rectangle(out_img, (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), (0,255,0), 2)
            cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high), (0,255,0), 2)
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

        left_laney_inds = np.concatenate(left_laney_inds)
        right_laney_inds = np.concatenate(right_laney_inds)
        left_lanex_inds = np.concatenate(left_lanex_inds)
        right_lanex_inds = np.concatenate(right_lanex_inds)

    # Calculate only using previous fits
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
        ltrue=False
    except:
        left_fit = [0,0,0]
        ltrue=True
    try: 
        right_fit = np.polyfit(right_laney_inds,right_lanex_inds, 2)
        rtrue=False
    except:
        right_fit = [0,0,0]
        rtrue=True

    calculate_all = ltrue and rtrue

    return left_fit, right_fit, left_lane_inds, right_lane_inds, calculate_all, out_img, leftx_base, rightx_base, ltrue, rtrue
    
def calculate_curvature_and_pos(image, left_fit, right_fit, ym_per_pix, xm_per_pix, rtrue, ltrue):
    if rtrue:
        try:
            xrs = np.linspace(0, image.shape[1], 100)
            yrs = right_fit[0] * xrs ** 2 + right_fit[1] * xrs + right_fit[2]
            y_eval_r = np.max(yrs)
            y_max = image.shape[0]
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
            xls = np.linspace(0, image.shape[1], 100)
            yls = left_fit[0] * xls ** 2 + left_fit[1] * xls + left_fit[2]
            y_eval_l = np.max(yls)
            y_max = image.shape[0]
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
    if left_x_pos != np.inf and right_x_pos != np.inf:
        center_lanes_x_pos = (left_x_pos + right_x_pos)//2

    elif left_x_pos != np.inf:
        center_lanes_x_pos = left_x_pos+42

    elif right_x_pos != np.inf:
        center_lanes_x_pos = right_x_pos-42

    veh_pos = ((image.shape[1]//2) - center_lanes_x_pos) * xm_per_pix

    return left_curverad, right_curverad, veh_pos, left_fit_cr, right_fit_cr, xrs, yrs, xls, yls

def plotting_images(image, out_img, left_curverad, right_curverad, veh_pos, xrs, yrs, xls, yls, rtrue, ltrue):
    warp_zero = np.zeros_like(image).astype(np.uint8)
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

    cv2.putText(out_img, "Left radius of curvature: " + str(math.floor(left_curverad)) + "m", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(out_img, "Right radius of curvature: " + str(math.floor(right_curverad)) + "m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(out_img, "Vehicle position: " + str(round(veh_pos, 2)) + "m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return out_img

def lane_detection(images, ym_per_pix, xm_per_pix, warp=True, mask=True, nwindows=40, margin=5, minpix=6, hist_height=10):
    left_fit = None
    right_fit = None
    for i in range(len(images)):
        image_init = images[i]
        image_prepro = image_init

        if warp:
            image_prepro = perspective_transform(image_prepro)

        if mask:
            image_prepro = mask_image(image_prepro)
        
        left_fit, right_fit, left_lane_inds, right_lane_inds, calculate_all, out_img, leftx_base, rightx_base, ltrue, rtrue = calculate_fits(image_prepro, nwindows, margin, minpix, hist_height, True, 40, 120, left_fit, right_fit)
        left_curverad, right_curverad, veh_pos, left_fit_cr, right_fit_cr, xrs, yrs, xls, yls = calculate_curvature_and_pos(image_prepro, left_fit, right_fit, ym_per_pix, xm_per_pix, rtrue, ltrue)
        image_prepro = plotting_images(image_prepro, out_img, left_curverad, right_curverad, veh_pos, xrs, yrs, xls, yls, rtrue, ltrue)
        
    return image_prepro

def main():
    data_dir = "/RoryMaillard/IA_Racing_2023/Visuel"
    images = load_images(data_dir)

    
    nwindows = 40
    margin = 5
    minpix = 6
    hist_height = 10
    ym_per_pix = 0.18/120
    xm_per_pix = 0.5/80
    
    images = lane_detection(images, nwindows, margin, minpix, ym_per_pix, xm_per_pix, hist_height)
    
    # Display images
    
main()
