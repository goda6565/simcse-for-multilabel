nohup python -m xgb.experiments.main \
    --model_name l2v \
    --model_type jscl \
    --dataset_name semeval-en \
    --output_dir outputs/l2v/jscl/semeval-en/16 \
    --max_length 128 &