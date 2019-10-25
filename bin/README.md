# Top-level training script

Currently, we have implemented cross-entropy training and sequence-discriminative training of LSTM- and Transformer-based acoustic models. The scripts are expected to be customizable. The users can adapt these scripts to train other acoustic models or design a new training pipeline.

## LSTM-based acoustic model


1. train\_ce.py

   Cross-entropy training for LSTM-based acoustic model. We used chunk-wise dataloader, i.e., segment the utterances into the same size of chunk (e.g., 80 frams). This is for efficiency, but may not be the best choice for recognition accuracy. It can be changed to whole utterance level dataloader for LSTM training.

2. train\_se.py

   Sequence training for LSTM acoustic model, with fixed alignments

3. train\_se2.py

   Sequence training for LSTM acoustic model with on-the-fly alignment generation. However, from our results, this apporach does not outperform train\_se.py

4. train\_chain.py

    A lattice-free MMI (LFMMI) training script for a LSTM based acoustic model. This is only the beta version, and it does not work better than the lattice-based apporach in the toolkit yet. We need to develop a new dataloader that prepare minibatches suitable for LFMMI.

## Transformer-based acoustic model

1. train\_transformer\_ce.py 

   Cross-entropy training for a transformer-based acoustic model.

2. train\_transformer\_se.py

   Sequence training for transformer-based acoustic model with fixed alignments. 

## Lattice generation and decoding

1. latgen.py
   
   The script that can do lattice generation. 

2. dump\_loglikes.py

   The script that dumps log-likelihoods for scroing. 

## Reference

Liang Lu, "[A Transformer with Interleaved Self-attention and Convolution for Hybrid Acoustic Models](https://arxiv.org/abs/1910.10352)", arxiv, 2019
