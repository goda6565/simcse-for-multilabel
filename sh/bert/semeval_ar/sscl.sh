python -m bert.experiments.run_sscl \
  --output_dir outputs/bert/sscl/semeval-ar \
  --dataset_name semeval-ar \
  --per_device_batch_size 32 \
  --learning_rate 1e-5 \
  --record_steps 60 \
  --max_length 128 \