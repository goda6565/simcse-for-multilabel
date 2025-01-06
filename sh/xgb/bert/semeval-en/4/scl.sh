nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type scl \
    --dataset_name semeval-en \
    --output_dir outputs/bert/scl/semeval-en/4 \
    --max_length 128 &
