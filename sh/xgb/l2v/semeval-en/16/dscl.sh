nohup python -m xgb.experiments.main \
    --model_name l2v \
    --model_type dscl \
    --dataset_name semeval-en \
    --output_dir outputs/l2v/dscl/semeval-en/16 \
    --max_length 128 &