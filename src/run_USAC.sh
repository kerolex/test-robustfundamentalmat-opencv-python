#!/bin/bash
#
################################################################################## 
# Author: 
#   Alessio Xompero: a.xompero@gmail.com
#
#  Created Date: 2022/06/08
# Modified Date: 2022/06/09
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
#
USAC_DIR=$HOME/Desktop/USAC
DATAPATH=$USAC_DIR/data/
RESPATH=$USAC_DIR/results/

DATASET=EuR5


## PARAMETERS
N_RUNS=100
MIN_N_INLIERS=15
RANSAC_REPROJ_TH=2.0
CONFIDENCE=0.99
MAX_ITERS=1000
MAX_N_KPS=1000
DIST_TH=50
SNN_TH=0.6
LOCAL_FEAT='orb'
SAC_ESTIMATOR='MAGSAC++'

############################################
source activate USAC

python test_magsacplusplus.py \
	--respath  							$RESPATH 					\
	--datapath 							$DATAPATH 				\
	--dataset  							$DATASET 					\
	--n_runs   							$N_RUNS 					\
	--min_num_inliers 			$MIN_N_INLIERS 		\
	--ransacReprojThreshold $RANSAC_REPROJ_TH \
	--conf 									$CONFIDENCE 			\
	--maxIters 							$MAX_ITERS 				\
	--max_n_kps							$MAX_N_KPS 				\
	--dist_th								$DIST_TH 					\
	--snn_th								$SNN_TH 					\
	--feature								$LOCAL_FEAT				\
	--SACestimator					$SAC_ESTIMATOR 				

echo "Finished!"

conda deactivate
