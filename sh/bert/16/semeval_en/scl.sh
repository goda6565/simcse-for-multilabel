python -m bert.experiments.run_scl \
  --output_dir outputs/bert/scl/semeval-en/16 \
  --dataset_name semeval-en \
  --per_device_batch_size 16 \
  --learning_rate 1e-5 \
  --max_length 128 \