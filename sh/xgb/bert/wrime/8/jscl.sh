nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type jscl \
    --dataset_name wrime \
    --output_dir outputs/bert/jscl/wrime/8 \
    --max_length 128 &