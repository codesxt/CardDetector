# A card detector using OpenCV and Python
# This code uses ORB to extract features and BRIEF to compute descriptors
# OpenCV Version: 2.4.9
# Python Version: 2.7.8

import numpy as np
import cv2

# Sets the minimum of matches required to consider the card as detected
MIN_MATCH_COUNT = 25

# Configures input from the webcam and opens the template file to detect
# It sets the template as grayscale to make the processing faster
cap = cv2.VideoCapture(0)
template_file = "template.png"
template = cv2.imread(template_file)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Initializes feature detector as ORB and feature descriptor as BRIEF
detector = cv2.FeatureDetector_create("ORB")
descriptor = cv2.DescriptorExtractor_create("BRIEF")

# ORB feature detector detects features from the template Image
# BRIEF then computes descriptors for the detected features
tpl_keypoints = detector.detect(template)
(tpl_keypoints, tpl_descriptors) = descriptor.compute(template, tpl_keypoints)

# A matcher object is initialized to match features from the webcam
# input with the template image features
matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')

while cap.isOpened():
	ret, img = cap.read()

	# Input image preprocessing
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.equalizeHist(img)

	# The detector object detects features from the input image, then extracts
	# descriptors from the image and matches them to the template descriptors
	img_keypoints = detector.detect(img)
	(img_keypoints, img_descriptors) = descriptor.compute(img, img_keypoints)
	matches = matcher.match(tpl_descriptors, img_descriptors)

	# This stage extracts the best matches from the poorer ones
	dist = [m.distance for m in matches]
	thres_dist = (sum(dist) / len(dist)) * 0.6
	good = [m for m in matches if m.distance < thres_dist]

	# If the best matches are more than the MIN_MATCH_COUNT variable, then a card is
	# detected
	if len(good) > MIN_MATCH_COUNT:
		# The points of matched features are stored in numpy arrays
		src_pts = np.float32([ tpl_keypoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ img_keypoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		# An homography is calculated to obtain the perspective transform
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()
		h,w = template.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

		# The perspective from the matched points is transformed according
		# to the homography matrix M and contours are drawn over the input image
		dst = cv2.perspectiveTransform(pts,M)
		cv2.polylines(img,[np.int32(dst)],True,255,3, cv2.CV_AA)
	else:
		print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
		matchesMask = None


	cv2.imshow('Template', template)
	cv2.imshow('Card Detection', img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()
