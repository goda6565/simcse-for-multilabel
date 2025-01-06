nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type jscl \
    --dataset_name semeval-es \
    --output_dir outputs/bert/jscl/semeval-es/4 \
    --max_length 128 &