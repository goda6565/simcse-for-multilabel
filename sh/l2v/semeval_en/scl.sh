python -m l2v.experiments.run_scl \
  --output_dir outputs/l2v/scl/semeval-en \
  --dataset_name semeval-en \
  --per_device_batch_size 16 \
  --learning_rate 1e-5 \
  --record_steps 60 \
  --max_length 128 \