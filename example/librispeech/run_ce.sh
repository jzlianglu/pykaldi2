python ../../bin/train_ce.py -config configs/ce.yaml \
    -data configs/data.yaml \
    -exp_dir exp/tr460_blstm_3x512_dp02 \
    -lr 0.0001 \
    -batch_size 1 \
    -sweep_size 460 \
    -aneal_lr_epoch 3 \
    -num_epochs 8 \
    -aneal_lr_ratio 0.5 

