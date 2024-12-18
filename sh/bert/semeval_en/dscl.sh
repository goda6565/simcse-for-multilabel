python -m bert.experiments.run_dscl \
  --output_dir outputs/bert/dscl/semeval-en \
  --dataset_name semeval-en \
  --per_device_batch_size 32 \
  --learning_rate 1e-5 \
  --record_steps 60 \
  --max_length 128 \