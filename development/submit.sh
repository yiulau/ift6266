#! /bin/bash

export jobid=23
cp -r experiment experiment$jobid
cd experiment$jobid

export feature_maps=(32 32 64 64 128 256)
export mlp_hiddens=(256)
export conv_sizes=(5 5 5 4 4 4)
export pool_sizes=(2 2 2 2 2 2)
export image_size=260
export batch_size=8
# number of batches/epochs before stopping
#num_batches=5
export num_epochs=50

export from_server=0
export live_plot=0
        
qsub -N jobname$jobid script
cp ../submit.sh submit$jobid.sh
