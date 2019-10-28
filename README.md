# pykaldi2

PyKaldi2 is a speech toolkit that is built based on [Kaldi](http://kaldi-asr.org/) and [PyTorch](https://pytorch.org/). It relies on [PyKaldi](https://github.com/pykaldi/pykaldi) - the Python wrapper of Kaldi, to access Kaldi functionalities. The key features of PyKaldi2 are one-the-fly lattice generation for lattice-based sequence training, on-the-fly data simulation and on-the-fly alignment gereation. A beta version lattice-free MMI (LFMMI) training script is also provided.  

## How to install

PyKaldi2 runs on top of the [Horovod](https://github.com/horovod/horovod) and PyKaldi libraries. The dockerfile is provided to customarize the envriorment. To use the repo, do the following three steps. 

1. Clone the repo by

  ```
    git clone https://github.com/jzlianglu/pykaldi2.git
  ```
2. Build the docker image, simply run

  ```
    docker build -t horovod-pykaldi docker 
  ```
   Alternatively, you can download the docker image by

  ```
    docker pull pykaldi2docker/horovod-pykaldi:torch1.2
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

We measured the training speed of PyKaldi2 on Librispeech dataset with Tesla V100 GPUs. We used BLSTM acoustic models with 3 hidden layers and each layer has 512 hidden units. The table below shows the training speed in our experiments. iRTF means the inverted real time factor, i.e., the amount of data (in terms of hours) we can process per hour. The minibatch is of the shape as batch-size x seq-len x feat-dim. For CE training, the seq-len is 80, which means we cut the utterance into chunks of 80 frames. For MMI training, the sequence length is variable, so it is denoted as *.  

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
|       | MMI  | 64 x *    | 16    | 176 |

## Example

To use PyKaldi2, you need to run the Kaldi speech toolkit up to the end of GMM training stages. PyKaldi2 will rely on the alignments and the denominator graph from the GMM system for CE and SE training. An example of the Librispeech system is given in the [example](https://github.com/jzlianglu/pykaldi2/tree/master/example) directory. 

## Future works

Currently, the toolkit is still in the early stage, and we are still improving it. The dimensions that we are looking at include
 1. More efficent dataloader, to support large-scale dataset.
 2. More efficent distributed training. Horovod has sub-linear speedup when it runs on the cross-machine distributed training mode, which could be improved.
 3. Lattice-free MMI, the state-of-the-art approach in Kaldi
 4. Joint frontend and backend optimization. 
 5. Support more neural network models

If you are intersted to contribute to this line of research, please contact Liang Lu (email address is provided in the arxiv paper). 

## Disclaimer

This is not an official Microsoft product

## Reference

Liang Lu, Xiong Xiao, Zhuo Chen, Yifan Gong, "[PyKaldi2: Yet another speech toolkit based on Kaldi and PyTorch](https://arxiv.org/abs/1907.05955)", arxiv, 2019
