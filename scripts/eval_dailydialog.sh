#!/bin/sh

mf=${1}
filename=${2}
batch_size=${3:-32}
split_batch=${4:-1}

parlai ed \
  -t dailydialog:no_start \
  -mf ${mf} \
  --batchsize ${batch_size} \
  --report-filename ${filename} \
  --inference greedy \
  --beam-size 1 \
  --dm-samples 6688 \
  --num-buckets 20 \
  --beam-block-ngram 3 \
  --beam-context-block-ngram 3 \
  --split-batch ${split_batch}
