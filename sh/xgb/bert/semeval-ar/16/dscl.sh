nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type dscl \
    --dataset_name semeval-ar \
    --output_dir outputs/bert/dscl/semeval-ar/16 \
    --max_length 128 &