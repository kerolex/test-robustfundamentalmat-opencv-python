#!/bin/bash

C3OD_DIR=$HOME/Desktop/USAC
DATAPATH=$C3OD_DIR/data/
RESPATH=$C3OD_DIR/results/

DATASET=EuR5
N_RUNS=100


############################################
source activate C3OD

python test_magsacplusplus.py --respath $RESPATH --datapath $DATAPATH --dataset $DATASET --n_runs $N_RUNS

echo "Finished!"

conda deactivate