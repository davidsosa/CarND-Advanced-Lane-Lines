## Writeup - Advanced Line Finding by David Sosa 
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. (Done)
* Apply a distortion correction to raw images. (Done)
* Use color transforms, gradients, etc., to create a thresholded binary image. (Done)
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/original_undistorted.png "Undistorted"
[image2]: ./output_images/undistorted_image.png "Road Transformed"
[image3]: ./output_images/threholded_image.png "Binary Example"
[image4]: ./output_images/birdseye_color.png "Warp Example Color"
[image5]: ./output_images/birdseye_threshold.png "Warp Example Black"  Visual"
[image6]: ./output_images/histogram.png "Histogram"
[image7]: ./output_images/fitter_lines_wsquares_white.png "Fitted lines"
[image8]: ./output_images/final_image.png "final_image"


[video1]: ["Video"](https://youtu.be/01PSaf6LpNk)

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 9 through 54 of the file called `calibration.py` which is included in the GitHub repository.   

I prepared "object points", which are the (x, y, z) coordinates of the chessboard corners in the world. I assume the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 20 through 66 in `pipeline_vid.py`).  Here's an example of my output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `change_stree_perspective(img,mtx,dist)`, which appears in lines 108 through 152 in the file `pipeline_vid.py`.  The `change_stree_perspective(img,mtx,dist)` function takes as inputs an image (`img`), the camera matrix (`mtx`) and the distortion coefficients (`dist`) . I chose the the source and destination points in the following manner:

```python
imshape = combined.shape  
img_x_center = imshape[1]/2

left_bottom = [img_x_center-600, imshape[0]]
right_bottom = [img_x_center+600, imshape[0]]

right_upper = [img_x_center+50, 450]
left_upper = [img_x_center-50, 450]

left_bottom_dest = [img_x_center-250, imshape[0]]
right_bottom_dest = [img_x_center+250, imshape[0]]

right_upper_dest = [img_x_center+250, 0]
left_upper_dest = [img_x_center-250, 0]

src = np.float32([left_upper, right_upper, right_bottom, left_bottom  ])						
dst = np.float32([ left_upper_dest,right_upper_dest,right_bottom_dest,left_bottom_dest])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 560, 450      | 390, 0        | 
| 720, 450      | 890, 0      |
| 40, 720       | 390, 720      |
| 1240, 720     | 890, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. The image has been additionally masked in order to get rid of parts of the image that are not needed to identify the lanes. Two images are shown in the following:  masked with color and masked with thresholds.  

![alt text][image4]

![alt text][image5].

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane-line pixels were identified using the line-finding method: peaks in a histogram. After applying calibration, thresholding, masking and a perspective transform. A histogram then is taken along all the colums in the lower half of the image like this:

```python
histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
```
with the resulting histogram shown.  
![alt text][image6]

After obtaining the histogram, sliding windows are placed around the line centers to find and follow the lines up to the top of of the frame. After finishind this process all the found pixels are concatenated and their positions are extracted. With the left and right line pixels at hand it is possible to fit a 2nd order polynomial by doing:

```python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

The result of this process is shown in the following figure where the left and right lines show blue and red color respectively. The windows and the fitted line are also shown.

![alt text][image7]
 
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 305 through 316 in my code in `pipeline_vid.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 342 through 345 in my code in `pipeline_vid.py` in the function `pipeline()`.  Here is an example of my result on a test image:
![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/01PSaf6LpNk)
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I tried the pipeline in one of the harder challenge videos and it failed a lot. I think that several other threshold could be adjusted to get rid of other gradients that appear in the image. One could also think of a requiring that the lines do not exceed a certain known distance. This way no lines which are outside a given range would be found and fitted.   