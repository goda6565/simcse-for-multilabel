nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type dscl \
    --dataset_name wrime \
    --output_dir outputs/bert/dscl/wrime/4 \
    --max_length 128 &