python -m bert.experiments.run_jscl \
  --output_dir outputs/bert/jscl/semeval-ar/8 \
  --dataset_name semeval-ar \
  --per_device_batch_size 8 \
  --learning_rate 1e-5 \
  --max_length 128 \