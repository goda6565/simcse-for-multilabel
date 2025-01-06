nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type jscl \
    --dataset_name semeval-en \
    --output_dir outputs/bert/jscl/semeval-en/16 \
    --max_length 128 &