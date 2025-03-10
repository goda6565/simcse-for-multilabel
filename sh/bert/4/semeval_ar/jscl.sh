python -m bert.experiments.run_jscl \
  --output_dir outputs/bert/jscl/semeval-ar/4 \
  --dataset_name semeval-ar \
  --per_device_batch_size 4 \
  --learning_rate 1e-5 \
  --max_length 128 \