nohup python -m xgb.experiments.main \
    --model_name l2v \
    --model_type jscl \
    --dataset_name semeval-ar \
    --output_dir outputs/l2v/jscl/semeval-ar/16 \
    --max_length 128 &