nohup python -m xgb.experiments.main \
    --model_name l2v \
    --model_type dscl \
    --dataset_name semeval-ar \
    --output_dir outputs/l2v/dscl/semeval-ar/4 \
    --max_length 128 &