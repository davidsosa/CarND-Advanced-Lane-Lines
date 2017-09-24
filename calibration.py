import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline
import pickle
import glob

# prepare object points
nx = 9
ny = 6

#load all images using glob
images = glob.glob('./camera_cal/calibration*.jpg')

objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points on an image plane

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # reshape objp matrix

print("How many images did I read?",len(images)) 

for fname in images:
    # Read in image
    img = mpimg.imread(fname)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
   
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp) 
        
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imshow(img)
        #plt.show()

print("In how many images could I read corners?",len(imgpoints))

# Test on an image
img = mpimg.imread('./camera_cal/calibration3.jpg')
def cal_undistort(img, objpoints, imgpoints):    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)   
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    print(gray.shape[::-1]) 
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

undistorted = cal_undistort(img, objpoints, imgpoints)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# prepare dictionary to save for later
dist_pickle = {}
dist_pickle['objpoints'] = objpoints 
dist_pickle['imgpoints'] = imgpoints
dist_pickle['mtx'] = mtx
dist_pickle['dist'] = dist

pickle.dump(dist_pickle,open("dist_pickle.p","wb"))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original image', fontsize=20)
ax2.imshow(undistorted)
ax2.set_title('Undistorted image', fontsize=20)
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.)
plt.savefig('./myimages/original_undistorted.png', bbox_inches='tight')
#plt.show()



