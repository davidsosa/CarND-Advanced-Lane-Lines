import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline
import pickle
import glob
# prepare object points
import matplotlib.patches as patches

# Read in an image and grayscale it
image = mpimg.imread('./test_images/test3.jpg')

#plt.figure(0)
#plt.imshow(image)
#plt.show()

dist_pickle = pickle.load( open( "dist_pickle.p", "rb" ) )
imgpoints = dist_pickle["imgpoints"]
objpoints = dist_pickle["objpoints"]
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def abs_sobel_thresh(img, orient='y',sobel_kernel=9,thresh=(0, 255)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient=='x':
        sobelgrad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    elif orient=='y':    
        sobelgrad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)       
    abs_sobelgrad = np.absolute(sobelgrad)
    scaled_sobel = np.uint8(255*abs_sobelgrad/np.max(abs_sobelgrad))   
    binary_output = np.zeros_like(scaled_sobel)
    
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(np.power(sobelx,2) + np.power(sobely,2))

    scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(15, 1.3)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel))
    
    sobel_atan = np.arctan2(abs_sobely,abs_sobelx)

    scaled_sobel = np.uint8(255*sobel_atan/np.max(sobel_atan))
    binary_output = np.zeros_like(sobel_atan)

    binary_output[(sobel_atan >= thresh[0]) & (sobel_atan <= thresh[1])] = 1
    return binary_output

# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

hls_binary = hls_select(image, thresh=(150, 255))

grad_binary = abs_sobel_thresh(image, orient='x', sobel_kernel=7, thresh=(50,150))
mag_binary = mag_thresh(image, sobel_kernel=7, mag_thresh=(50, 150))
dir_binary = dir_threshold(image, sobel_kernel=7, thresh=(0.5, 1.3))
hls_binary = hls_select(image,thresh=(120, 255))

ksize = 7
# Choose a larger odd number to smooth gradient measurements
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(50, 200))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(50, 200))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)  | hls_binary==1 )] = 1

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 1
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero

    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image

imshape = combined.shape  
img_x_center = imshape[1]/2

mask_left_bottom = [img_x_center-550, imshape[0]]
mask_right_bottom = [img_x_center+550, imshape[0]]

mask_right_upper = [img_x_center+100, 400]
mask_left_upper = [img_x_center-100, 400]

vertices = np.array([[ (mask_left_bottom[0],mask_left_bottom[1]),(mask_right_bottom[0],mask_right_bottom[1]),(mask_right_upper[0],mask_right_upper[1]), (mask_left_upper[0],mask_left_upper[1] ) ]] , dtype=np.int32)

masked = region_of_interest(image,vertices)

#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
#f.tight_layout()
#ax1.imshow(image)
#ax1.set_title('Original image', fontsize=20)
#ax2.imshow(combined,cmap='gray')
#ax2.set_title('Thresholded image', fontsize=20)
#plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.)
#plt.savefig('./myimages/threholded_image.png', bbox_inches='tight')
#plt.show()

undistorted = cv2.undistort(image, mtx, dist, None, mtx)

#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
#f.tight_layout()
#ax1.imshow(image)
#ax1.set_title('Original image', fontsize=20)
#ax2.imshow(undistorted)
#ax2.set_title('Undistorted image', fontsize=20)
#plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.)
#plt.savefig('./myimages/undistorted_image.png', bbox_inches='tight')
##plt.show()

left_bottom = [img_x_center-600, imshape[0]]
right_bottom = [img_x_center+600, imshape[0]]

right_upper = [img_x_center+80, 450]
left_upper = [img_x_center-80, 450]

x = [left_bottom[0], right_bottom[0], right_upper[0], left_upper[0], left_bottom[0]]
y = [left_bottom[1], right_bottom[1], right_upper[1], left_upper[1], left_bottom[1]]

left_bottom_dest = [img_x_center-250, imshape[0]]
right_bottom_dest = [img_x_center+250, imshape[0]]

right_upper_dest = [img_x_center+250, 0]
left_upper_dest = [img_x_center-250, 0]
print(imshape)
x_dest = [left_bottom_dest[0], right_bottom_dest[0], right_upper_dest[0], left_upper_dest[0], left_bottom_dest[0]]
y_dest = [left_bottom_dest[1], right_bottom_dest[1], right_upper_dest[1], left_upper_dest[1], left_bottom_dest[1]]

def change_street_perspective(img,mtx,dist):
    img = region_of_interest(img, vertices)
    img = cv2.undistort(img, mtx, dist, None, mtx)
    #gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_size = (img.shape[1], img.shape[0])
    offset = 10 # offset for dst points
    src = np.float32([left_upper, right_upper, right_bottom, left_bottom  ])

    #print("source points",left_upper,right_upper,left_bottom,right_bottom)
    dst = np.float32([ left_upper_dest,
                       right_upper_dest, 
                       right_bottom_dest, 
                       left_bottom_dest])
 
    #print("destination points",left_upper_dest,right_upper_dest,left_bottom_dest,right_bottom_dest)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size = (img.shape[1],img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv


combined_thr = region_of_interest(combined, vertices)

warped_color, M_color, Minv_color = change_street_perspective(undistorted, mtx, dist)
warped, M, Minv = change_street_perspective(combined, mtx, dist)

#f1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
#f1.tight_layout()
#plt.title("Perspective")
#ax1.set_title('Normal Perspective ', fontsize=20)
#
#ax1.plot(x, y, 'b--', lw=4)
#ax1.imshow(combined,cmap='gray')
#
#ax2.set_title("Bird's Eye Perspective", fontsize=20)
#ax2.imshow(warped,cmap='gray')
#ax2.plot(x_dest, y_dest, 'b--', lw=4)
#
#plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.)
##plt.show()
#plt.savefig('./myimages/birdseye_threshold.png', bbox_inches='tight')
#
#f2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
#f2.tight_layout()
#plt.title("Perspective")
#ax1.set_title('Normal Perspective ', fontsize=20)
#
#ax1.plot(x, y, 'b--', lw=4)
#ax1.imshow(undistorted)
#
#ax2.set_title("Bird's Eye Perspective", fontsize=20)
#ax2.imshow(warped_color)
#ax2.plot(x_dest, y_dest, 'b--', lw=4)
#
#plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.)
#plt.savefig('./myimages/birdseye_color.png', bbox_inches='tight')
##plt.show()

#plt.figure(10)
histogram = np.sum(warped[ warped.shape[0]//2:, :], axis=0)
#plt.plot(histogram)
#plt.savefig('./myimages/histogram.png', bbox_inches='tight')
##plt.show()

## Create an output image to draw on and visualize the result
out_img = np.dstack((warped, warped, warped))*255

midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(warped.shape[0]/nwindows)

# Identify the x and y positions of all nonzero pixels in the image
nonzero = warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

#plt.figure()
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = warped.shape[0] - (window+1)*window_height
    win_y_high = warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
                  (255,255,255), 2)
    cv2.rectangle(out_img, (win_xright_high,win_y_high), (win_xright_low,win_y_low),
                  (255,255,255), 2)  
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
       
# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

## Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

## Generate x and y values for plotting
ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

#out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds],0] = 255   # paint left line red
#out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds],1] = 255 # paint right line red

plt.figure(3)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.imshow(out_img[:,:,1],cmap='gray')
#plt.imshow(out_img,cmap='jet')
plt.show()

# Create an image to draw on and an image to show the selection window
out_img = np.dstack((warped, warped, warped))*255
window_img = np.zeros_like(out_img)
# Color in left and right line pixels
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
plt.figure(4)
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
plt.imshow(result)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()


leftx = np.array([y + np.random.randint(-50, high=51) for y in left_fitx])
rightx = np.array([y + np.random.randint(-50, high=51) for y in right_fitx])

plt.figure(5)
mark_size = 3
plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
plt.xlim(0, 1280)
plt.ylim(0, 720)
plt.plot(left_fitx, ploty, color='green', linewidth=3)
plt.plot(right_fitx, ploty, color='green', linewidth=3)
plt.gca().invert_yaxis() # to visualize as we do the images
plt.show()

y_eval = np.max(ploty)
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
print(left_curverad, right_curverad)

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Fit new polynomials to x,y in world spaceleft_fitx
left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
# Now our radius of curvature is in meters
print(left_curverad, 'm', right_curverad, 'm')
# Example values: 632.1 m    626.2 m

plt.figure(6)
# Create an image to draw the lines on
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
undist = image.copy()
# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


# assume car is at center
car_x_position = img_x_center
lane_center_position = (right_fitx[-1] + left_fitx[-1]) /2
center_dist = (car_x_position - lane_center_position) * xm_per_pix

print("(right_fitx[0] + left_fitx[0]) /2",(right_fitx[-1] + left_fitx[-1]) /2)
print("Car position",car_x_position)
print("car_x_position - lane_center_position",car_x_position - lane_center_position)
print("xm_per_pix",xm_per_pix)
print("center_dist",center_dist)

cv2.putText(result,'Left Curve R:'+str(round(left_curverad,2))+' m ', (25,75), cv2.FONT_ITALIC, 2, (0,0,0),3 )
cv2.putText(result,'Right Curve R:'+str(round(right_curverad,2))+' m ', (25,175), cv2.FONT_ITALIC, 2, (0,0,0),3 )
cv2.putText(result,'Center Dist:'+str(round(center_dist,2))+' m ', (25,250), cv2.FONT_ITALIC, 2, (0,0,0),3 )

plt.imshow(result)
plt.savefig('./myimages/final_image.png', bbox_inches='tight')
#plt.show()
