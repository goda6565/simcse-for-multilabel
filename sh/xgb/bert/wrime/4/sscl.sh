nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type sscl \
    --dataset_name wrime \
    --output_dir outputs/bert/sscl/wrime/4 \
    --max_length 128 &