python -m bert.experiments.run_sscl \
  --output_dir outputs/bert/sscl/semeval-es \
  --dataset_name semeval-es \
  --per_device_batch_size 32 \
  --learning_rate 1e-5 \
  --record_steps 30 \
  --max_length 128 \