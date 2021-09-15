#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=cA8K8dl-E6k


# Brute-Force Matching with ORB Descriptors

import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

src = str(sys.argv[1])
target = str(sys.argv[2])

print(src)
print(target)

im1 = cv2.imread(src)          # Image that needs to be registered.
im2 = cv2.imread(target) # trainImage

rows, cols, dim = im1.shape
#output_im = cv2.warpPerspective(output_im,M,(int(cols*1.5),int(rows*1.5)))
scale_percent = 40 # percent of original size
width = int(im1.shape[1] * scale_percent / 100)
height = int(im1.shape[0] * scale_percent / 100)
dim = (width, height)
im1 = cv2.resize(im1, dim, interpolation = cv2.INTER_AREA)
im2 = cv2.resize(im2, dim, interpolation = cv2.INTER_AREA)

img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Initiate ORB detector
orb = cv2.ORB_create(50)  #Registration works with at least 50 points

# find the keypoints and descriptors with orb
kp1, des1 = orb.detectAndCompute(img1, None)  #kp1 --> list of keypoints
kp2, des2 = orb.detectAndCompute(img2, None)

im1_keypoints = cv2.drawKeypoints(im1, kp1, None, flags=None)
im2_keypoints = cv2.drawKeypoints(im2, kp2, None, flags=None)
#Brute-Force matcher takes the descriptor of one feature in first set and is 
#matched with all other features in second set using some distance calculation.
# create Matcher object

cv2.imshow("im1_keypoints", im1_keypoints)
cv2.imshow("im2_keypoints", im2_keypoints)
cv2.waitKey(0)

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

# Match descriptors.
matches = matcher.match(des1, des2, None)  #Creates a list of all matches, just like keypoints

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

#Like we used cv2.drawKeypoints() to draw keypoints, 
#cv2.drawMatches() helps us to draw the matches. 
#https://docs.opencv.org/3.0-beta/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
# Draw first 10 matches.
img3 = cv2.drawMatches(im1,kp1, im2, kp2, matches[:10], None)

cv2.imshow("Matches image", img3)
cv2.waitKey(0)

#Now let us use these key points to register two images. 
#Can be used for distortion correction or alignment
#For this task we will use homography. 
# https://docs.opencv.org/3.4.1/d9/dab/tutorial_homography.html

# Extract location of good matches.
# For this we will use RANSAC.
#RANSAC is abbreviation of RANdom SAmple Consensus, 
#in summary it can be considered as outlier rejection method for keypoints.
#http://eric-yuan.me/ransac/
#RANSAC needs all key points indexed, first set indexed to queryIdx
#Second set to #trainIdx. 

points1 = np.zeros((len(matches), 2), dtype=np.float32)  #Prints empty array of size equal to (matches, 2)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
   points1[i, :] = kp1[match.queryIdx].pt    #gives index of the descriptor in the list of query descriptors
   points2[i, :] = kp2[match.trainIdx].pt    #gives index of the descriptor in the list of train descriptors

#Now we have all good keypoints so we are ready for homography.   
# Find homography
#https://en.wikipedia.org/wiki/Homography_(computer_vision)
  
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
height, width, channels = im2.shape
im1Reg = cv2.warpPerspective(im1, h, (width, height))  #Applies a perspective transformation to an image.
   
print("Estimated homography : \n",  h)

cv2.imshow("Registered image", im1Reg)
cv2.waitKey()