#!/bin/sh

model_file=${1}/${5}-balancing-bsz${2}-lr${3}
batch_size=${2}
lr=${3}
init_model=${4}
dataset=${5}


parlai tdr \
    -t ${dataset} \
    -veps 0.25 \
    --embedding-size 512 \
    --n-layers 8 \
    --ffn-size 2048 \
    --dropout 0.1 \
    --n-heads 16 \
    --learn-positional-embeddings True \
    --n-positions 512 \
    --variant xlm \
    --activation gelu \
    --attention-dropout 0.0 \
    --relu-dropout 0.0 \
    --batchsize ${batch_size} \
    --model transformer/dm_generator \
    --history-add-global-end-token end \
    --label-truncate 128 \
    -lr ${lr} \
    --lr-scheduler reduceonplateau \
    --lr-scheduler-patience 10 \
    --optimizer adam \
    --model-parallel true \
    --save-after-valid true \
    --text-truncate 512 \
    --truncate 512 \
    --warmup_updates 100 \
    --fp16 true \
    --fp16-impl mem_efficient \
    --update-freq 1 \
    --gradient-clip 0.1 \
    --skip-generation true \
    --model-file ${model_file} \
    --tensorboard-log true \
    --init-model ${init_model} \
    \
    `# tokenizer params` \
    --dict-file zoo:tutorial_transformer_generator/model.dict \
    --dict-tokenizer bpe \
    --dict-lower true \
    \
    `# validation` \
    --eval-batchsize ${batch_size} \
    -veps 0.25 \
    -vme 10000 \
    -vp 10 \
    --validation-metric ppl \
    --validation-metric-mode min \
    --save-after-valid true \
    --log-every-n-secs 20 \
    --validation-patience 50 \
    \
    `# miscs` \
    --inference greedy \
    --beam-size 1 \
    --dm-every-n-epochs 1.0 \
    --dm-samples 5000 \
    --dm-option balancing \
    --num-buckets 20 \
    --use-generated-texts true \
    --use-fixed-hist true \
