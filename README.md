# test-robustfundamentalmat-opencv-python
Simple Python script for testing the robust estimation of the fundamental matrix between two images with RANSAC and MAGSAC++ in OpenCV, and reproducibility across 100 runs.

The script is a personal adaption of codes taken from:
* [Practical Computer Vision](https://www.programcreek.com/python/?code=PacktPublishing%2FPractical-Computer-Vision%2FPractical-Computer-Vision-master%2FChapter08%2F08_compute_F_mat.py)
* [MAGSAC](https://github.com/danini/magsac/blob/master/examples/example_fundamental_matrix.ipynb)



# Table of Contents

1. [Setup](#setup)
2. [Demo](#demo)
3. [Arguments](#arguments)
4. [Known issues](#known-issues)
5. [Enquiries, Question and Comments](#enquiries-question-and-comments)
6. [Reference](#references)
7. [Licence](#licence)


## Setup
* Ubuntu 18.04 LTS
* OpenCV: 4.5.5
* Python: 3.10

Create a conda environment name USAC with py-opencv and tqdm packages:

```
conda create --name USAC python=3.10
conda install -c conda-forge py-opencv tqdm
```



## Demo

From Linux terminal, run to activate the created conda environment and launch the testing Python script:
```
source run_USAC.sh
```

The script will run on two example images from the sequence Machine Hall 05 of the [public EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) located in [data/EuR5](data).

Values of the arguments can be changed directly in the Python script or passed within the bash script.

The demo runs automatically the robust estimator MAGSAC++. This can be changed to RANSAC by commenting and uncommenting the line with cv2.findFundamentalMat().

## Arguments
* _n_runs_: number of runs (default: 5)
* _min_num_inliers_: minimum number of inliers to accept the estimated fundamental matrix (default: 15)
* _ransacReprojThreshold_: maximum reprojection error allowed for RANSAC (default: 2.0)
* _conf_: confidence for RANSAC (default: 0.99)
* _maxIters_: maximum number of iterations for RANSAC (default: 1000)
* _max_n_kps_: maximum number of keypoints to detect in an image (default: 1000)
* _dist_th_: threshold on the Hamming distance for ORB features (default: 50)
* _snn_th_: threshold for the Lowe's ratio test or Second Nearest Neighbour (default: 0.6)
* _feature_: type of local image feature to use, for example SIFT or ORB (default: orb)
* _SACestimator_: algorithm to use for robustly estimating the fundamental matrix, for example RANSAC or MAGSAC++ (default: MAGSAC++)

## Known issues

The script returns the same fundamental matrix and number of inliers across the 100 runs without changing any value of the parameters. This behaviour is unexpected due to the sampling approach of the estimators (RANSAC, MAGSAC++). I posted a [question](https://forum.opencv.org/t/ransac-like-estimators-not-random-across-multiple-runs/9086) on the OpenCV Forum to know more about this behaviour. 

The behaviour is due to a random seed initialises always to zero when calling:
```
F, status	=	cv2.findFundamentalMat(pts1, pts2, cv2.USAC_MAGSAC, ransacReprojThreshold, confidence, maxIters)
```
or
```
F, status	=	cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold, confidence, maxIters)
```

I would reccomend to avoid using the above functions. To fix the issue and folliwing the reply provided in OpenCV Forum, I would suggest using the following portions of code (here reported only for MAGSAC++):
```
def FindFundamentalMatMAGSACplusplus(pts1, pts2, ransacReprojThreshold, confidence, maxIters):
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

  F, status = cv2.findFundamentalMat(pts1, pts2, usac_params)

  return F, status
```

If you comment the line about setting the randomGenerateState, the code will generate the same output  as the previous command:
```
F, status	=	cv2.findFundamentalMat(pts1, pts2, cv2.USAC_MAGSAC, ransacReprojThreshold, confidence, maxIters)
```

Setting the randomGenerateState to random integers will restore the standard non-deterministic behaviour of these estimators, providing different results for each run. 


## Enquiries, Question and Comments

If you have any further enquiries, question, or comments, please contact a.xompero@gmail.com If you would like to file a bug report or a feature request, use the Github issue tracker. 

## References

D. Barath, J. Noskova, M. Ivashechkin, J. Matas, **MAGSAC++, a fast, reliable and accurate robust estimator**, CVPR 2020  
[[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Barath_MAGSAC_a_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.pdf)] [[code](https://github.com/danini/magsac)]

D. Mishkin, **Evaluating OpenCV new RANSACs**, Blog post [[link](https://ducha-aiki.github.io/wide-baseline-stereo-blog/2021/05/17/OpenCV-New-RANSACs.html)]


## Licence

This work is licensed under the MIT License. To view a copy of this license, see [LICENSE](LICENSE).
