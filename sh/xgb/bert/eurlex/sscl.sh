nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type sscl \
    --dataset_name eurlex \
    --max_length 512 &