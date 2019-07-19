
python ../../bin/train_se.py -config configs/se.yaml \
    -data data.yaml \
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
                                                                              
