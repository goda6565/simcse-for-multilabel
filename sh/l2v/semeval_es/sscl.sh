python -m l2v.experiments.run_sscl \
  --output_dir outputs/l2v/sscl/semeval-es \
  --dataset_name semeval-es \
  --per_device_batch_size 16 \
  --learning_rate 1e-5 \
  --record_steps 30 \
  --max_length 128 \