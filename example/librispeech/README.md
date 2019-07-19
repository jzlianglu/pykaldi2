# Librispeech example

We assume that you have run the Kaldi [Librispeech recipe](https://github.com/kaldi-asr/kaldi/blob/master/egs/librispeech/s5/run.sh) up to the end of GMM stages. You also need to build the denominator graph for sequence training. 

## CE training

The top-level script for CE training is "run\_ce.sh", which is like 

  ```
    python ../../bin/train_ce.py -config configs/ce.yaml \
    -data configs/data.yaml \
    -exp_dir exp/tr460_blstm_3x512_dp02 \
    -lr 0.0001 \
    -batch_size 1 \
    -sweep_size 460 \
    -aneal_lr_epoch 3 \
    -num_epochs 8 \
    -aneal_lr_ratio 0.5 

  ```


