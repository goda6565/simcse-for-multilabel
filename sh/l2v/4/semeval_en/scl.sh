nohup python -m l2v.experiments.run_scl \
  --output_dir outputs/l2v/scl/semeval-en/4 \
  --dataset_name semeval-en \
  --per_device_batch_size 4 \
  --learning_rate 1e-5 \
  --max_length 128 &