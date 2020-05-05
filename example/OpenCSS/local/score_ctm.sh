#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#           2014  Guoguo Chen
#           2019  Liang Lu
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
cmd=local/run.pl
stage=0
decode_mbr=true
word_ins_penalty=0.0,0.5,1.0
min_lmwt=7
max_lmwt=15
iter=final
#end configuration section.

echo "$0 $@"  # Print the command line for logging

. local/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: local/score.sh [--cmd (run.pl|queue.pl...)] <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --decode_mbr (true/false)       # maximum bayes risk decoding (confusion network)."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1;
fi

lang_or_graph=$1
dir=$2
model=$3

for f in $dir/lat.1.gz; do
  [ ! -f $f ] && echo "score.sh: no such file $f" && exit 1;
done

mkdir -p $dir/scoring/log

for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/best_path.LMWT.$wip.log \
    lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
    lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
    lattice-1best ark:- ark:- \| \
    lattice-align-words $lang_or_graph/phones/word_boundary.int $model ark:- ark:- \| \
    nbest-to-ctm ark:- - \| \
    local/int2sym.pl -f 5 $lang_or_graph/words.txt ">&" $dir/LMWT_$wip.ctm
done

exit 0;
