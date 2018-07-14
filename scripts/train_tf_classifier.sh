#!/bin/bash

python -m hansardparser.plenaryparser.build_training_set.mk_lines_corpus \
  -v 2 \
  --fmt seq \
  --rm_flatworld_tags \
  --input data/generated/plenaryparser/hansard_txt_hand_labels_w_text.csv

OUTPATH=experiments/plenaryparser/line_classifier/classifier$(ls experiments/plenaryparser/line_classifier | wc -l | tr -d " ")
CORPUS=data/generated/plenaryparser/text2vec/corpora/$(ls -rt data/generated/plenaryparser/text2vec/corpora | tail -n 1)
BUILDER=data/generated/plenaryparser/text2vec/builders/$(ls -rt data/generated/plenaryparser/text2vec/builders | tail -n 1)

# CNN
python -m hansardparser.plenaryparser.classify.tf.train \
    -v 2 \
    --outpath $OUTPATH \
    --classifier cnn \
    --restore \
    --corpus $CORPUS \
    --builder $BUILDER \
    --n_examples 6000 \
    --n_eval_examples 100 \
    --n_epochs 20 --batch_size 100 --learning_rate 0.001 \
    --eval_every 50 --save_every 50 \
    --n_features 100 \
    --embed_size 50 \
    --dropout_p_keep 1.0 \
    --n_filters 16 32 \
    --filter_size 5 5 \
    --filter_stride 1 1 \
    --pool_size 2 2 \
    --pool_stride 2 2 \
    --dense_size 128 \
    --batch_normalize_dense \
    --batch_normalize_embeds

# RNN
python -m hansardparser.plenaryparser.classify.tf.train \
    -v 2 \
    --outpath $OUTPATH \
    --classifier rnn \
    --restore \
    --corpus $CORPUS \
    --builder $BUILDER \
    --n_examples 6000 \
    --n_eval_examples 50 \
    --n_epochs 80 --batch_size 50 --learning_rate 0.001 \
    --eval_every 100 --save_every 100 \
    --n_features 200 \
    --embed_size 50 \
    --cell_type gru \
    --n_hidden 100 100 \
    --use_attention \
    --clip_gradients --max_grad_norm 5.0 \
    --dropout_p_keep 0.75 \
    --batch_normalize_embeds
