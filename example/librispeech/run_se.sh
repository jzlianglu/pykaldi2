
#python train_se.py -config /datadisk2/lial/release/configs/mmi.yaml \
#    -data /datadisk2/lial/release/configs/data.yaml \
#    -exp_dir /datadisk2/lial/release/exp/tr460_blstm_3x512_dp02 \
#    -criterion "mmi" \
#    -seed_model /datadisk2/lial/release/exp/tr460_blstm_3x512_dp02/model.7.tar \
#    -prior_path /datadisk2/lial/librispeech/s5/exp/tri5b_ali_clean_460/final.occs \
#    -trans_model /datadisk2/lial/librispeech/s5/exp/tri5b/final.mdl \
#    -den_dir /datadisk2/lial/librispeech/s5/exp/tri5b_denlats_clean_460/dengraph \
#    -lr 0.000002 \
#    -ce_ratio 0.1 \
#    -max_grad_norm 5 \
#    -batch_size 4 \
#    -num_epochs 1 

python train_se.py -config configs/se.yaml \
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
                                                                              

# -transform exp/tr460_blstm_3x512_dp02.v2/transform.pkl \
