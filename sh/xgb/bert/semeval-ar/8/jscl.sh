nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type jscl \
    --dataset_name semeval-ar \
    --output_dir outputs/bert/jscl/semeval-ar/8 \
    --max_length 128 &