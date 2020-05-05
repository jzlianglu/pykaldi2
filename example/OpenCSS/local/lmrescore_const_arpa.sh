#!/bin/bash

# Copyright 2014  Guoguo Chen
#           2019  Liang Lu
# Apache 2.0

# This script rescores lattices with the ConstArpaLm format language model.

# Begin configuration section.
cmd=$PYKALDIPATH/example/OpenCSS/local/run.pl
skip_scoring=false
stage=1
scoring_opts=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ $# != 4 ]; then
   echo "Does language model rescoring of lattices (remove old LM, add new LM)"
   echo "Usage: $0 [options] <old-lang-dir> <new-lang-dir> \\"
   echo "                   <input-decode-dir> <output-decode-dir>"
   echo "options: [--cmd (run.pl|queue.pl [queue opts])]"
   exit 1;
fi

oldlang=$1
newlang=$2
indir=$3
outdir=$4

oldlm=$oldlang/G.fst
newlm=$newlang/G.carpa
! cmp $oldlang/words.txt $newlang/words.txt &&\
  echo "$0: Warning: vocabularies may be incompatible."
[ ! -f $oldlm ] && echo "$0: Missing file $oldlm" && exit 1;
[ ! -f $newlm ] && echo "$0: Missing file $newlm" && exit 1;
! ls $indir/lat.* >/dev/null &&\
  echo "$0: No lattices input directory $indir" && exit 1;

if ! cmp -s $oldlang/words.txt $newlang/words.txt; then
  echo "$0: $oldlang/words.txt and $newlang/words.txt differ: make sure you know what you are doing.";
fi

oldlmcommand="fstproject --project_output=true $oldlm |"

mkdir -p $outdir/log

if [ $stage -le 1 ]; then
  $cmd $outdir/log/rescorelm.log \
    lattice-lmrescore --lm-scale=-1.0 \
    ark:$indir/lat.ark "$oldlmcommand" ark:-  \| \
    lattice-lmrescore-const-arpa --lm-scale=1.0 \
    ark:- "$newlm" "ark,t:|gzip -c>$outdir/lat.1.gz" || exit 1;
fi

exit 0;
