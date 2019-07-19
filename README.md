# pykaldi2

PyKaldi2 is a speech toolkit that is built based on [Kaldi](http://kaldi-asr.org/) and [PyTorch](https://pytorch.org/). It relies on [PyKaldi](https://github.com/pykaldi/pykaldi) -- the Python wrapper of Kaldi, to access Kaldi functionalities. The key features of PyKaldi2 are one-the-fly lattice generation for lattice-based sequence training, on-the-fly data simulation and on-the-fly alignment gereation. 

## How to install

PyKaldi2 runs on top of the [Horovod](https://github.com/horovod/horovod) and PyKaldi libraries. The dockerfile is provided to customarize the envriorment. To use the repo, do the following three steps. 

1. Clone the repo by

   ```
     git clone https://github.com/jzlianglu/pykaldi2.git
   ```
2. Build the docker image, simply run

  ```
    docker build -t horovod-pykaldi -f docker/dockerfile 
  ```

3. Activate the docker image, for example

  ```
    NV_GPU=0,1,2,3 nvidia-docker run -v `pwd`:`pwd` -w `pwd` --shm-size=32G -i -t horovod-pykaldi
  ```

If you want to run multi-GPU jobs using Horovod on a single machine,  the command is like

  ```
    horovodrun -np 4 -H localhost:4 sh run_ce.sh 
  ```
Please refer to [Horovod](https://github.com/horovod/horovod) for running cross-machine distributed jobs. 

## Training speed

We measured the training speed of PyKaldi2 on Librispeech dataset with Tesla V100 GPUs. We used BLSTM acoustic models with 3 hidden layers and each layer has 512 hidden units. The table below shows the training speed in our experiments. iRTF means the inverted real time factor, i.e., the amount of data (in terms of hours) we can process per hour. The minibatch is of the shape as batch_size*seq_len*feat_dim. For CE training, the seq_len is 80, which means we cut the utterance into chunks of 80 frames. For MMI training, the sequence length is variable, so it is denoted as *.  

| model | loss | bsxlen    | #GPUs |iRTF |
|------ | -----| ----------| ------|---- |
|       | CE   | 64x80     |  1    | 190 |
|       | CE   | 64x80     |  4    | 220 |
|       | CE   | 256x80    |  4    | 520 |
| BLSTM | CE   | 1024x80   | 16    | 1356|
|       | MMI  | 1 x *     | 1     | 11.6|
|       | MMI  | 4 x *     | 1     | 16.7|
|       | MMI  | 4 x *     | 4     | 34.5|
|       | MMI  | 16 x *    | 4     | 50.4|


## Cross-entropy training

An example of runing a cross-entropy job is

  ```
   python train_ce.py -config configs/ce.yaml \  
    -data configs/data.yaml \                 
    -exp_dir exp/tr960_blstm_3x512_dp02 \     
    -lr 0.0002 \                              
    -batch_size 64 \                          
    -sweep_size 960 \                         
    -aneal_lr_epoch 3 \                       
    -num_epochs 8 \                           
    -aneal_lr_ratio 0.5                 
  ```

## Sequence training

An example of runing a sequence traing job is
 
  ```   
python train_se.py -config configs/mmi.yaml \
    -data configs/data.yaml \
    -exp_dir exp/tr960_blstm_3x512_dp02 \
    -criterion "mmi" \
    -seed_model exp/tr960_blstm_3x512_dp02/model.7.tar \
    -prior_path /datadisk2/lial/librispeech/s5/exp/tri6b_ali_tr960/final.occs \
    -trans_model /datadisk2/lial/librispeech/s5/exp/tri6b_ali_tr960/final.mdl \
    -den_dir /datadisk2/lial/librispeech/s5/exp/tri6b_denlats_960_cleaned/dengraph \
    -lr 0.000001 \
    -ce_ratio 0.1 \
    -max_grad_norm 5 \
    -batch_size 4 \
  ```

## Reference

Liang Lu, Xiong Xiao, Zhuo Chen, Yifan Gong, "PyKaldi2: Yet another speech toolkit based on Kaldi and PyTorch", arxiv, 2019
