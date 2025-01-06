nohup python -m l2v.experiments.run_dscl \
  --output_dir outputs/l2v/dscl/semeval-en/16 \
  --dataset_name semeval-en \
  --per_device_batch_size 16 \
  --learning_rate 1e-5 \
  --max_length 128 &