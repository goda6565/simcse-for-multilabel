python -m bert.experiments.run_sscl \
  --output_dir outputs/bert/sscl/semeval-en/4 \
  --dataset_name semeval-en \
  --per_device_batch_size 4 \
  --learning_rate 1e-5 \
  --max_length 128 \