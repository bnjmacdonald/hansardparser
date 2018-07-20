#!/bin/bash

cd hansardparser/

USR_DIR="plenaryparser/classify/tf"
PROBLEM="hansard_line_speaker_span"
DATA_DIR="../data/generated/t2t_data/$PROBLEM"
TMP_DIR="../data/temp/t2t_datagen/$PROBLEM"
DECODE_DIR="../data/generated/t2t_decode/$PROBLEM"
MODEL="aligned"
HPARAMS_SET="aligned_base"
HPARAMS="batch_size=50,num_hidden_layers=1,hidden_size=64,learning_rate=0.001,max_length=100"
DECODE_HPARAMS="batch_size=10,beam_size=1"
TRAIN_DIR="../experiments/t2t_train/$PROBLEM/$MODEL/$HPARAMS_SET"

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR $DECODE_DIR

t2t-datagen \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM

t2t-trainer \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams=$HPARAMS \
    --hparams_set=$HPARAMS_SET \
    --output_dir=$TRAIN_DIR \
    --train_steps=1000 \
    --local_eval_frequency=1000 \
    --eval_use_test_set true


# file containing lines to be predicted.
DECODE_FILE="../data/tests/raw/speaker_names.txt"
# DECODE_FILE="../data/tests/generated/plenaryparser/hansard_txt_hand_labels_w_text.csv"
DECODE_OUTPUT_FILE=$DECODE_DIR/decode.txt

t2t-decoder \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS_SET \
    --decode_hparams=$DECODE_HPARAMS \
    --hparams=$HPARAMS \
    --output_dir=$TRAIN_DIR \
    --decode_from_file=$DECODE_FILE \
    --decode_to_file=$DECODE_OUTPUT_FILE


t2t-exporter \
    --t2t_usr_dir=$USR_DIR \
    --model=$MODEL \
    --hparams_set=$HPARAMS_SET \
    --hparams=$HPARAMS \
    --problem=$PROBLEM \
    --data_dir=$DATA_DIR \
    --output_dir=$TRAIN_DIR
