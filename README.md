# test-robustfundamentalmat-opencv-python
Simple Python script for testing the robust estimation of the fundamental matrix between two images with RANSAC and MAGSAC++ in OpenCV, and reproducibility across 100 runs.

The script is a personal adaption of codes taken from:
* [Practical Computer Vision] (https://www.programcreek.com/python/?code=PacktPublishing%2FPractical-Computer-Vision%2FPractical-Computer-Vision-master%2FChapter08%2F08_compute_F_mat.py)
* [MAGSAC](https://github.com/danini/magsac/blob/master/examples/example_fundamental_matrix.ipynb)

## Setup
* Ubuntu 18.04 LTS
* OpenCV: 4.5.5
* Python: 3.10

## Demo

From Linux terminal, run to activate the created conda environment and launch the testing Python script:
```
source run_USAC.sh
```

The script will run on two example images from the sequence Machine Hall 05 of the [public EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) located in [data/EuR5](data).

Values of the arguments can be changed directly in the Python script or passed within the bash script.

The demo runs automatically the robust estimator MAGSAC++. This can be changed to RANSAC by commenting and uncommenting the line with cv2.findFundamentalMat().

## Arguments
* n_runs: number of runs (default: 5)
* min_num_inliers: minimum number of inliers to accept the estimated fundamental matrix (default: 15)
* ransacReprojThreshold: maximum reprojection error allowed for RANSAC (default: 2.0)
* conf: confidence for RANSAC (default: 0.99)
* maxIters: maximum number of iterations for RANSAC (default: 1000)
* max_n_kps: maximum number of keypoints to detect in an image (default: 1000)
* dist_th: threshold on the Hamming distance for ORB features (default: 50)
* snn_th: threshold for the Lowe's ratio test or Second Nearest Neighbour (default: 0.6)
* feature: type of local image feature to use, for example SIFT or ORB (default: orb)

## Known issues

The script returns the same fundamental matrix and number of inliers across the 100 runs without changing any value of the parameters. This behaviour is unexpected due to the sampling approach of the estimators (RANSAC, MAGSAC++). I posted a [question](https://forum.opencv.org/t/ransac-like-estimators-not-random-across-multiple-runs/9086) on the OpenCV to know more about this behaviour. 


## Enquiries, Question and Comments

If you have any further enquiries, question, or comments, please contact a.xompero@gmail.com If you would like to file a bug report or a feature request, use the Github issue tracker. 

## References

D. Barath, J. Noskova, M. Ivashechkin, J. Matas, **MAGSAC++, a fast, reliable and accurate robust estimator**, CVPR 2020  
[[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Barath_MAGSAC_a_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.pdf)] [[code](https://github.com/danini/magsac)]



## Licence

This work is licensed under the MIT License. To view a copy of this license, see [LICENSE](LICENSE).
