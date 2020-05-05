#!/bin/bash


# Copyright 2013    Yajie Miao    Carnegie Mellon University
# Apache 2.0

# Decode the DNN model. The [srcdir] in this script should be the same as dir in
# build_nnet_pfile.sh. Also, the DNN model has been trained and put in srcdir.
# All these steps will be done automatically if you run the recipe file run-dnn.sh

# Modified 2018 Mirco Ravanelli Univeristé de Montréal - Mila



## Begin configuration section
stage=0

# Reading the options in the cfg file
#source < (grep = $cfg_file | sed 's/ *= */=/g')

num_threads=12 
min_active=200                                                    
max_active=7000                                                   
max_mem=50000000                                                  
beam=15.0                                                         
latbeam=8.0                                                       
acwt=0.10                                                         
max_arcs=-1                                                       
skip_scoring=false                                                
scoring_opts="--cmd run.pl --min_lmwt 4 --max_lmwt 20"            
norm_vars=False                                                   
alidir=/datadisk2/lial/librispeech/s5/exp/tri5b
#data=/datadisk2/lial/librispeech/s5/data/dev_clean 
graphdir=/datadisk2/lial/librispeech/s5/exp/tri5b/graph_tgsmall   


echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 5)"
   echo "Usage: steps/decode_dnn.sh [options] <graph-dir> <data-dir> <ali-dir> <decode-dir>"
   echo " e.g.: steps/decode_dnn.sh exp/tri4/graph data/test exp/tri4_ali exp/tri4_dnn/decode"
   echo "main options (for others, see top of script file)"
   echo "  --stage                                  # starts from which stage"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --acwt <acoustic-weight>                 # default 0.1 ... used to get posteriors"
   echo "  --num-threads <n>                        # number of threads to use, default 4."
   echo "  --parallel-opts <opts>                   # e.g. '-pe smp 4' if you supply --num-threads 4"
   echo "  --scoring-opts <opts>                    # options to local/score.sh"
   exit 1;
fi

out_folder=$1
data=$2
log_prob=$3


dir=`echo $out_folder | sed 's:/$::g'` # remove any trailing slash.
srcdir=`dirname $dir`; # assume model directory one level up from decoding directory.

if [ $stage -le 0 ]; then

mkdir -p $dir/log

nj=1

echo $nj > $dir/num_jobs

# Some checks.  Note: we don't need $srcdir/tree but we expect
# it should exist, given the current structure of the scripts.
for f in $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

    finalfeats="ark:$log_prob"
    latgen-faster-mapped-parallel --num-threads=$num_threads --min-active=$min_active --max-active=$max_active --max-mem=$max_mem --beam=$beam --lattice-beam=$latbeam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt $alidir/final.mdl $graphdir/HCLG.fst "$finalfeats" "ark:|gzip -c > $dir/lat.1.gz" &> $dir/log/decode.1.log
wait

fi

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "$0: not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh $scoring_opts $data $graphdir $dir
  local/score_ctm.sh $scoring_opts $data $graphdir $dir $alidir/final.mdl
fi

exit 0;
