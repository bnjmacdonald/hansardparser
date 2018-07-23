#!/bin/bash
# Commands for a tensor2tensor (t2t) pipeline to be run locally.
# 
# Steps:
#   - split data into train/dev/test sets.
#   - serialize data on disk.
#   - train tensorflow model.
#   - (optional) decode examples using trained model.
#   - export model for serving.
#   - upload model to google cloud storage.
#   - create/update model on ml-engine.
#   - (optional) request batch/online predictions.
# 
# Notes:
#   - use $PROBLEM_PREDICT when you want to export a model that can receive an
#       array of strings as input.
# 
# Todos:
#   - TODO: implement script for inspecting trained model performance on unseen
#       labeled data.
#   - TODO: implement script for inspecting trained model performance on unseen
#       unlabeled data.

# you must be in this directory:
cd /Users/bnjmacdonald/Documents/current/projects/hansardparser


# problem: hansard_line_type4
SPLIT_LINES_FILEPATH="data/generated/plenaryparser/hansard_txt_hand_labels_w_text.csv"
SPLIT_LINES_OUTPATH="data/generated/plenaryparser/hansard_txt_hand_labels_w_text_splits"
USR_DIR="hansardparser/plenaryparser/TxtParser/LineLabeler/t2t_problem"
PROBLEM="hansard_line_type4"
PREDICT_PROBLEM="hansard_line_type4_predict"
TMP_DIR=$SPLIT_LINES_OUTPATH
# TMP_DIR="data/tests/hansards/txt/1987"
DECODE_HPARAMS="batch_size=10"
DECODE_FILE="data/tests/raw/hansard_lines.txt"

MODEL="lstm_encoder"
HPARAMS_SET="lstm_attention"
HPARAMS="batch_size=100,num_hidden_layers=2,hidden_size=128,max_input_seq_length=0,dropout=0.25"

MODEL="transformer_encoder"
HPARAMS_SET="transformer_base"
HPARAMS="batch_size=100,num_hidden_layers=3,hidden_size=128,max_input_seq_length=0,dropout=0.5,norm_type=batch"

# DECODE_FILE="data/tests/raw/transcripts/8th - 9th Dec 1987.txt"

# problem: hansard_line_has_speaker
SPLIT_LINES_FILEPATH=""
SPLIT_LINES_OUTPATH=""
USR_DIR="hansardparser/plenaryparser/TxtParser/SpeakerParser/t2t_problem"
PROBLEM="hansard_line_has_speaker"
PREDICT_PROBLEM="hansard_line_has_speaker_predict"
TMP_DIR=$SPLIT_LINES_OUTPATH
MODEL="transformer_encoder"
HPARAMS_SET="transformer_tiny"
HPARAMS="batch_size=50,num_hidden_layers=2,hidden_size=64,learning_rate=0.001,max_input_seq_length=500"
DECODE_HPARAMS="batch_size=10"
DECODE_FILE="data/tests/raw/speaker_names.txt"

# problem: hansard_line_speaker_span
SPLIT_LINES_FILEPATH=""
SPLIT_LINES_OUTPATH=""
USR_DIR="hansardparser/plenaryparser/TxtParser/SpeakerParser/t2t_problem"
PROBLEM="hansard_line_speaker_span"
TMP_DIR=$SPLIT_LINES_OUTPATH
MODEL="aligned"
HPARAMS_SET="aligned_base"
HPARAMS="batch_size=50,num_hidden_layers=1,hidden_size=64,learning_rate=0.001,max_length=100"
DECODE_HPARAMS="batch_size=10,beam_size=1"
DECODE_FILE="data/tests/raw/speaker_names.txt"


# problem-invariant args.
DATA_DIR="data/generated/plenaryparser/t2t_data/$PROBLEM"  # $PREDICT_PROBLEM
TRAIN_DIR="experiments/plenaryparser/t2t_train/$PROBLEM/$MODEL/$HPARAMS_SET"
EXPORT_DIR="$TRAIN_DIR/export/Servo"
DECODE_DIR="data/generated/plenaryparser/t2t_decode/$PROBLEM"  # $PREDICT_PROBLEM
DECODE_OUTPUT_FILE=$DECODE_DIR/decode.txt
BUCKET="gs://hansardparser-models/$PROBLEM"
MODEL_NAME=$PREDICT_PROBLEM\_$MODEL\_$HPARAMS_SET  # $PREDICT_PROBLEM
MODEL_VERSION="v0"


# training args.
TRAIN_STEPS=20000
# EVAL_STEPS=100

# splits hand-labeled line type training data into train, dev, and test sets.
cd hansardparser/plenaryparser && 
    python -m build_training_set.split_lines -v 1 \
        --filepath ../../$SPLIT_LINES_FILEPATH \
        --outpath ../../$SPLIT_LINES_OUTPATH \
        --by_sitting &&
    cd -


# creates directories for t2t output.
mkdir -p $DATA_DIR $DECODE_DIR $TRAIN_DIR  # $TMP_DIR 


# generate serialized data.
t2t-datagen \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM


# train model.
t2t-trainer \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams=$HPARAMS \
    --hparams_set=$HPARAMS_SET \
    --output_dir=$TRAIN_DIR \
    --train_steps=$TRAIN_STEPS \
    --local_eval_frequency=1000 \
    --eval_throttle_seconds=120 \
    --eval_use_test_set false


# (optional) decode/predict examples.
t2t-decoder \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS_SET \
    --decode_hparams=$DECODE_HPARAMS \
    --hparams=$HPARAMS \
    --output_dir=$TRAIN_DIR \
    --decode_from_file="$DECODE_FILE" \
    --decode_to_file=$DECODE_OUTPUT_FILE


# export model.
t2t-exporter \
    --t2t_usr_dir=$USR_DIR \
    --model=$MODEL \
    --hparams_set=$HPARAMS_SET \
    --hparams=$HPARAMS \
    --problem=$PREDICT_PROBLEM \
    --output_dir=$TRAIN_DIR \
    --data_dir="."

# upload model for string prediction to google cloud storage.
LATEST_EXPORT=$(ls ${EXPORT_DIR} | tail -1)
mv $EXPORT_DIR/$LATEST_EXPORT $EXPORT_DIR/$LATEST_EXPORT"_str_predict"
gsutil cp -r $EXPORT_DIR/$LATEST_EXPORT"_str_predict" $BUCKET/$MODEL_NAME/$LATEST_EXPORT"_str_predict"
# gsutil cp -r "../data/generated/t2t_data" "gs://hansardparser-data"

# create model on ml-engine.
gcloud ml-engine models create $MODEL_NAME  --enable-logging --regions us-central1

# create new version of model on ml-engine.
gcloud ml-engine versions create $MODEL_VERSION \
    --model $MODEL_NAME \
    --runtime-version 1.6 \
    --python-version 3.5 \
    --origin $BUCKET/$MODEL_NAME/$LATEST_EXPORT"_str_predict"


# t2t-query-server \
#     --t2t_usr_dir=$USR_DIR \
#     --cloud_mlengine_model_name $MODEL_NAME \
#     --cloud_mlengine_model_version $MODEL_VERSION \
#     --problem $PROBLEM \
#     --data_dir $DATA_DIR


# (optional) submit a batch prediction using serialized data in a GCP bucket.
# JOB_NAME=$PROBLEM
# DATA_BUCKET="gs://hansardparser-data"
# gcloud ml-engine jobs submit prediction $JOB_NAME \
#     --model $MODEL_NAME \
#     --version $MODEL_VERSION \
#     --input-paths $DATA_BUCKET/t2t_data/hansard_line_type4/hansard_line_type4-train-00000-of-00009 \
#     --output-path $DATA_BUCKET/t2t_preds/hansard_line_type4/hansard_line_type4-train-00000-of-00009 \
#     --region us-central1 \
#     --data-format "TF_RECORD"


# (optional) retrieve predictions from ml-engine.
# TODO: write actual unit tests for online predictions for each model.
# INSTANCES="/Users/bnjmacdonald/Documents/current/projects/hansardparser/data/tests/raw/hansard_lines.txt"
# gcloud ml-engine predict --model $MODEL_NAME \
#     --version $MODEL_VERSION \
#     --text-instances $INSTANCES \
#     --verbosity "debug"


# update google cloud function.
# cd hansardparser/plenaryparser/TxtParser/LineLabeler && \
#     gcloud beta functions deploy predictHansardLineType4 --trigger-http && \
#     cd -
