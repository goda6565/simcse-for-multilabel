nohup python -m xgb.experiments.main \
    --model_name l2v \
    --model_type sscl \
    --dataset_name semeval-en \
    --output_dir outputs/l2v/sscl/semeval-en/8 \
    --max_length 128 &