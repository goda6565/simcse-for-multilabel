python -m bert.experiments.run_jscl \
  --output_dir outputs/bert/jscl/semeval-en/8 \
  --dataset_name semeval-en \
  --per_device_batch_size 8 \
  --learning_rate 1e-5 \
  --max_length 128 \