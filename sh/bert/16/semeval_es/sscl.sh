python -m bert.experiments.run_sscl \
  --output_dir outputs/bert/sscl/semeval-es/16 \
  --dataset_name semeval-es \
  --per_device_batch_size 16 \
  --learning_rate 1e-5 \
  --max_length 128 \