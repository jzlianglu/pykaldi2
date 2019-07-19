# Librispeech example

We assume that you have run the Kaldi [Librispeech recipe](https://github.com/kaldi-asr/kaldi/blob/master/egs/librispeech/s5/run.sh) up to the end of GMM stages. You also need to build the denominator graph for sequence training. 

## CE training

The bash script for CE training is **run\_ce.sh**, which is like 

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
The ce.yaml file contains the data and model configurations. The data.yaml file contains the data, which are raw waveforms, and labels. The data.yaml in the above example is like

  ```
   clean_source:                                                            
  1:                                                                     
    type: Librispeech                                                    
    wav: /datadisk2/lial/LibriSpeechWav/train-clean-100.zip              
    label: /datadisk2/lial/LibriSpeechWav/train-960-pdf-ids.txt          
    aux_label: /datadisk2/lial/LibriSpeechWav/train-960-trans-ids.txt    
  2:                                                                     
    type: Librispeech                                                    
    wav: /datadisk2/lial/LibriSpeechWav/train-clean-360.zip              
    aux_label: /datadisk2/lial/LibriSpeechWav/train-960-trans-ids.txt    
    label: /datadisk2/lial/LibriSpeechWav/train-960-pdf-ids.txt          
  3:                                                                     
    type: Librispeech                                                    
    wav: /datadisk2/lial/LibriSpeechWav/train-other-500.zip              
    aux_label: /datadisk2/lial/LibriSpeechWav/train-960-trans-ids.txt    
    label: /datadisk2/lial/LibriSpeechWav/train-960-pdf-ids.txt
  ```
The zip files contains the raw waveforms, and *trans-ids* and *pdf-ids* are Kaldi labels in the form of transition ids and tied HMM pdf ids, which look like

  ```
   100-121669-0001 0 43 43 43 43 43 43 43 44 44 44 44 44 44 44 44 ...
  ``` 
The *pdf-ids* are used for CE training, and *trans-ids* are used for SE training. In fact, given Kaldi transition model, we can convert *trans-ids* to *pdf-ids*, how this is slow in Python. Currently we use two versions of the labels. The labels are in plain text at the moment. We are working on supporting data and labels in the form of HDF5.

If you want to apply dynamic data simulation, you need to provide the noise and RIR sources such as

  ```
   dir\_noise:
    1:
      type: Noise
      wav: Path-to\noise.zip
   rir:
    1:
      type: RIR
      wav: Path-to\rir.zip
  ```

## SE training

The bash script for SE training is **run_se.sh**, which is like

  ```
   python ../../bin/train_se.py -config configs/se.yaml \
    -data configs/data.yaml \
    -exp_dir /datadisk2/lial/release/exp/tr960_blstm_3x512_dp02/ \
    -criterion "smbr" \
    -seed_model /datadisk2/lial/release/exp/tr960_blstm_3x512_dp02/model.7.tar \
    -prior_path /datadisk2/lial/librispeech/s5/exp/tri6b_ali_tr960/final.occs \
    -trans_model /datadisk2/lial/librispeech/s5/exp/tri6b_ali_tr960/final.mdl \
    -den_dir /datadisk2/lial/librispeech/s5/exp/tri6b_denlats_960_cleaned/dengraph \
    -lr 0.000001 \
    -ce_ratio 0.1 \
    -max_grad_norm 5 \
    -batch_size 4 \
    -num_epochs 1  
  ```
The *prior_path*, *trans\_model* and *den_dir* are from the Kaldi setup. Trained with 960 hours of data, we obtained the following results in our experiments.

|model   |  loss | dev-clean |  dev-other | test-clean | test-other |
|        | CE    | 4.6       | 13.4       | 5.1        | 13.5       |
|  BLSTM | MMI   | 4.3       | 12.1       | 4.8        | 12.5       |
|        | sMBR  | 4.3       | 12.3       | 4.9        |            |

