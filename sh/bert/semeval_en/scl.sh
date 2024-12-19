python -m bert.experiments.run_scl \
  --output_dir outputs/bert/scl/semeval-en \
  --dataset_name semeval-en \
  --per_device_batch_size 32 \
  --learning_rate 1e-5 \
  --record_steps 60 \
  --max_length 128 \