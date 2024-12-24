python -m bert.experiments.run_dscl \
  --output_dir outputs/bert/dscl/semeval-ar \
  --dataset_name semeval-ar \
  --per_device_batch_size 32 \
  --learning_rate 1e-5 \
  --max_length 128 \