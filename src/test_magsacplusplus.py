#! /usr/bin/env/ python
#
################################################################################## 
# Author: 
#   Alessio Xompero: a.xompero@gmail.com
#
#  Created Date: 2022/06/08
# Modified Date: 2022/06/08
#
#####################################################################################
# MIT License
#
# Copyright (c) 2022 Alessio
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#####################################################################################

import os
import sys
import glob
import argparse
import numpy as np

import random

import cv2 # OpenCV - make sure the OpenCV version is 4.5

from tqdm import tqdm

from pdb import set_trace as bp

# Taken from https://www.programcreek.com/python/?code=PacktPublishing%2FPractical-Computer-Vision%2FPractical-Computer-Vision-master%2FChapter08%2F08_compute_F_mat.py
# and https://github.com/danini/magsac/blob/master/examples/example_fundamental_matrix.ipynb



def CheckOpenCvVersion():
	(major, minor, _) = cv2.__version__.split(".")
	
	if (int(major) < 4):
		print('OpenCV version should be at least than 4.5 to run USAC estimators!')
		return False
	elif (int(major) >= 4) & (int(minor) < 5):
		print('OpenCV version should be at least than 4.5 to run USAC estimators!')
		return False
	else:
		return True



def ComputeSIFTfeatures(filename, max_n_kps):

	# load image
	img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)

	# create sift object
	siftdet = cv2.SIFT_create(max_n_kps)    
	
	kps, des = siftdet.detectAndCompute(img,None)
	
	return img, kps, des

def ComputeORBkeypoints(filename, max_n_kps):
	"""
	Reads image from filename and computes ORB keypoints
	Returns image, keypoints and descriptors. 
	"""
	# load image
	img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)


	# create orb object
	orb = cv2.ORB_create(max_n_kps)

	# set parameters 
	orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
	orb.setWTA_K(3)

	# detect keypoints
	kp = orb.detect(img,None)

	# for detected keypoints compute descriptors. 
	kp, des = orb.compute(img, kp)

	return img, kp, des


def BruteForceMatcher(des1, des2, opt):
	"""
	Brute force matcher to match ORB feature descriptors
	"""

	tentatives = []

	if opt.feature == 'orb':
		# create BFMatcher object
		bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
		# Match descriptors.
		matches = bf.match(des1,des2)

		# Sort them in the order of their distance.
		matches = sorted(matches, key = lambda x:x.distance)

		dist_th = opt.dist_th

		
		for i,(m) in enumerate(matches):
			if m.distance < dist_th:
				tentatives.append(m)
	
	elif opt.feature == 'sift':
		bf = cv2.BFMatcher()

		SNN_threshold = opt.snn_th
		matches = bf.knnMatch(des1, des2, k=2)

		# Apply ratio test
		snn_ratios = []
		for m, n in matches:
		    if m.distance < SNN_threshold * n.distance:
		        tentatives.append(m)
		        snn_ratios.append(m.distance / n.distance)

		# Sort the points according to the SNN ratio.
		# This step is required both for PROSAC and P-NAPSAC.
		sorted_indices = np.argsort(snn_ratios)
		tentatives = list(np.array(tentatives)[sorted_indices])

	
	return tentatives



def PrintUsacParameters(usac_params):
	print('USAC Parameters')

	print('randomGeneratorState: {:d}'.format(usac_params.randomGeneratorState))
	print('confidence: {:f}'.format(usac_params.confidence))
	print('maxIterations: {:d}'.format(usac_params.maxIterations))
	print('threshold: {:f}'.format(usac_params.threshold))
	print('isParallel: {:d}'.format(int(usac_params.isParallel)))
	print('loIterations: {:d}'.format(usac_params.loIterations))
	print('loSampleSize: {:d}'.format(usac_params.loSampleSize)) 

	if usac_params.score == cv2.SCORE_METHOD_RANSAC:
		print('score: SCORE_METHOD_RANSAC')
	elif usac_params.score == cv2.SCORE_METHOD_MSAC:
		print('score: SCORE_METHOD_MSAC')
	elif usac_params.score == cv2.SCORE_METHOD_MAGSAC:
		print('score: SCORE_METHOD_MAGSAC')
	elif usac_params.score == cv2.SCORE_METHOD_LMEDS:
		print('score: SCORE_METHOD_LMEDS')

	if usac_params.loMethod == cv2.LOCAL_OPTIM_NULL:
		print('loMethod: LOCAL_OPTIM_NULL')
	elif usac_params.loMethod == cv2.LOCAL_OPTIM_INNER_LO:
		print('loMethod: LOCAL_OPTIM_INNER_LO')
	elif usac_params.loMethod == cv2.LOCAL_OPTIM_INNER_AND_ITER_LO:
		print('loMethod: LOCAL_OPTIM_INNER_AND_ITER_LO')
	elif usac_params.loMethod == cv2.LOCAL_OPTIM_GC:
		print('loMethod: LOCAL_OPTIM_GC')
	elif usac_params.loMethod == cv2.LOCAL_OPTIM_SIGMA:
		print('loMethod: LOCAL_OPTIM_SIGMA')

	if usac_params.neighborsSearch == cv2.NEIGH_FLANN_KNN:
		print('neighborsSearch: NEIGH_FLANN_KNN')
	elif usac_params.neighborsSearch == cv2.NEIGH_GRID:
		print('neighborsSearch: NEIGH_GRID')
	elif usac_params.neighborsSearch == cv2.NEIGH_FLANN_RADIUS:
		print('neighborsSearch: NEIGH_FLANN_RADIUS')

	if usac_params.sampler == cv2.SAMPLING_UNIFORM:
		print('sampler: SAMPLING_UNIFORM')
	elif usac_params.sampler == cv2.SAMPLING_PROGRESSIVE_NAPSAC:
		print('sampler: SAMPLING_PROGRESSIVE_NAPSAC')
	elif usac_params.sampler == cv2.SAMPLING_NAPSAC:
		print('sampler: SAMPLING_NAPSAC')
	elif usac_params.sampler == cv2.SAMPLING_PROSAC:
		print('sampler: SAMPLING_PROSAC')


'''
uncommenting the line: usac_params.randomGeneratorState = random.randint(0,1000000)
results in the same bahviour of
F, status	=	cv2.findFundamentalMat(pts1, pts2, cv2.USAC_MAGSAC, ransacReprojThreshold, confidence, maxIters)
where the estimated fundamental matrix and number of inliers is 
repeatable across iterations
'''
def FindFundamentalMatMAGSACplusplus(pts1, pts2, ransacReprojThreshold, confidence, maxIters):
	# set OpenCV USAC parameters for MAGSAC++
	usac_params = cv2.UsacParams()

	usac_params.randomGeneratorState = random.randint(0,1000000)
	usac_params.confidence = confidence
	usac_params.maxIterations = maxIters
	usac_params.loMethod = cv2.LOCAL_OPTIM_SIGMA
	usac_params.score = cv2.SCORE_METHOD_MAGSAC
	usac_params.threshold = ransacReprojThreshold
	# usac_params.isParallel = False # False is deafult
	usac_params.loIterations = 10
	usac_params.loSampleSize = 50
	usac_params.neighborsSearch = cv2.NEIGH_GRID
	usac_params.sampler = cv2.SAMPLING_UNIFORM
 
	# PrintUsacParameters(usac_params)

	# Compute fundamental matrix
	F, status	=	cv2.findFundamentalMat(pts1, pts2, usac_params)

	return F, status

'''
uncommenting the line: usac_params.randomGeneratorState = random.randint(0,1000000)
results in the same bahviour of
F, status	=	cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold, confidence, maxIters)
where the estimated fundamental matrix and number of inliers is 
repeatable across iterations
'''
def FindFundamentalMatRANSAC(pts1, pts2, ransacReprojThreshold, confidence, maxIters):
	# set OpenCV USAC parameters for MAGSAC++
	usac_params = cv2.UsacParams()

	usac_params.randomGeneratorState = random.randint(0,1000000)
	usac_params.confidence = confidence
	usac_params.maxIterations = maxIters
	usac_params.loMethod = cv2.LOCAL_OPTIM_NULL
	usac_params.score = cv2.SCORE_METHOD_RANSAC
	usac_params.threshold = ransacReprojThreshold
	# usac_params.isParallel = False # False is deafult
	# usac_params.loIterations = 10
	# usac_params.loSampleSize = 50
	# usac_params.neighborsSearch = cv2.NEIGH_FLANN_RADIUS
	usac_params.sampler = cv2.SAMPLING_UNIFORM
 
	# PrintUsacParameters(usac_params)

	# Compute fundamental matrix
	F, status	=	cv2.findFundamentalMat(pts1, pts2, usac_params)
	# F, status	=	cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold, confidence, maxIters)

	return F, status


def FindRobustFundamentalMat(pts1, pts2, flag, ransacReprojThreshold, confidence, maxIters):
	if flag == 'RANSAC':
		F, status = FindFundamentalMatRANSAC(pts1, pts2, ransacReprojThreshold, confidence, maxIters)
	elif flag == 'MAGSAC++':
		F, status = FindFundamentalMatMAGSACplusplus(pts1, pts2, ransacReprojThreshold, confidence, maxIters)

	return F, status


def ComputeFundamentalMatrix(filename1, filename2, opt):
	"""
	Takes in filenames of two input images 
	Return Fundamental matrix computes 
	using 8 point algorithm with MAGSAC++ robust fitting estimator
	"""

	ransacReprojThreshold = opt.ransacReprojThreshold
	confidence = opt.conf
	maxIters = opt.maxIters

	if opt.feature == 'orb':
		# compute ORB keypoints and descriptor for each image
		img1, kp1, des1 = ComputeORBkeypoints(filename1, opt.max_n_kps)
		img2, kp2, des2 = ComputeORBkeypoints(filename2, opt.max_n_kps)
	elif opt.feature == 'sift':
		# compute SIFT keypoints and descriptor for each image
		img1, kp1, des1 = ComputeSIFTfeatures(filename1, opt.max_n_kps)
		img2, kp2, des2 = ComputeSIFTfeatures(filename2, opt.max_n_kps)

	# compute keypoint matches using descriptor
	matches = BruteForceMatcher(des1, des2, opt)

	# extract points
	pts1 = []
	pts2 = []
	for i,(m) in enumerate(matches):
		pts2.append(kp2[m.trainIdx].pt)
		pts1.append(kp1[m.queryIdx].pt)
	pts1  = np.asarray(pts1)
	pts2 = np.asarray(pts2)


	# Compute fundamental matrix
	F, status	=	FindRobustFundamentalMat(pts1, pts2, opt.SACestimator, ransacReprojThreshold, confidence, maxIters)

	if F is None or F.shape == (1, 1):
		print('No fundamental matrix found')
		return np.zeros((3, 3, 1), dtype = "uint8"), 0

	if F.shape[0] > 3:
		# more than one matrix found, just pick the first
		print('More than one matrix found')
		print(F)
		F = F[0:3, 0:3]

	ninliers =  np.sum(status)

	if ninliers >= opt.min_num_inliers:
		return F, ninliers
	else:
		return np.zeros((3, 3, 1), dtype = "uint8"), ninliers



if __name__ == '__main__':

	print('Initialising:')
	print('Python {}.{}'.format(sys.version_info[0], sys.version_info[1]))
	print('OpenCV {}'.format(cv2.__version__))

	if not CheckOpenCvVersion():
		exit(1)

	# Arguments
	parser = argparse.ArgumentParser(description='RANSAC and MAGSAC++ test')
	parser.add_argument('--dataset', default='EuR5', type=str)
	parser.add_argument('--datapath', default='', type=str)
	parser.add_argument('--respath', default='', type=str)
	parser.add_argument('--n_runs', default='100', type=int)

	parser.add_argument('--min_num_inliers', default='15', type=int)
	parser.add_argument('--ransacReprojThreshold', default='2.0', type=float)
	parser.add_argument('--conf', default='0.99', type=float)
	parser.add_argument('--maxIters', default='1000', type=int)

	parser.add_argument('--max_n_kps', default='1000', type=int)

	parser.add_argument('--dist_th', default='50', type=int)
	parser.add_argument('--snn_th', default='0.6', type=float)

	parser.add_argument('--feature', default='orb', type=str, choices=['sift','orb'])

	parser.add_argument('--SACestimator', default='RANSAC', type=str, choices=['RANSAC','MAGSAC++'])

	args = parser.parse_args()

	datapath=args.datapath
	respath=args.respath
	n_runs=args.n_runs
	dataset=args.dataset

	#compute F matrix between two images
	file_list = sorted(glob.glob(os.path.join(datapath,dataset,'*.png')))

	for r in tqdm(range(1,n_runs+1)):
		# rnd = random.random()
		# print(rnd)
		
		F, ninliers = ComputeFundamentalMatrix(file_list[0], file_list[1], args)
		print('{:d}: {:d} - {}'.format(r,ninliers, F.flatten('C'))) # flatten in row-major (C-style) order

