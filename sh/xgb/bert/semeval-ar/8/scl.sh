nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type scl \
    --dataset_name semeval-ar \
    --output_dir outputs/bert/scl/semeval-ar/8 \
    --max_length 128 &
