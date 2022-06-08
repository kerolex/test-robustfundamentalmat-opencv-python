#!/bin/bash

USAC_DIR=$HOME/Desktop/USAC
DATAPATH=$USAC_DIR/data/
RESPATH=$USAC_DIR/results/

DATASET=EuR5
N_RUNS=100


############################################
source activate USAC

python test_magsacplusplus.py --respath $RESPATH --datapath $DATAPATH --dataset $DATASET --n_runs $N_RUNS

echo "Finished!"

conda deactivate
