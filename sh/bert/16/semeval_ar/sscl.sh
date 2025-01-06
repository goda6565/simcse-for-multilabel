python -m bert.experiments.run_sscl \
  --output_dir outputs/bert/sscl/semeval-ar/16 \
  --dataset_name semeval-ar \
  --per_device_batch_size 16 \
  --learning_rate 1e-5 \
  --max_length 128 \