#!/bin/bash

python -m hansardparser.plenaryparser.classify.tf.predict -v 2 \
   --outpath data/tests/generated/plenaryparser/classify/tf/preds \
   --documents data/tests/raw/hansard_lines.txt \
   --builder data/generated/plenaryparser/text2vec/builders/2018-07-14T170136 \
   --clf experiments/plenaryparser/line_classifier/classifier0
