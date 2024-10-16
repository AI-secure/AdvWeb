export CUDA_VISIBLE_DEVICES=0

accelerate launch dpo/dpo_training.py \
    --model_name="mistral" \
    --no_task \
    --model_name_or_path="data/task_demo_-1_aug/together/new_models/mistral-7b-sft" \
    --model_dtype="bfloat16" \
    --output_dir="data/task_demo_-1_aug/together/new_models/mistral-7b-dpo" \
    --learning_rate=1e-4 \
    --max_steps 1000 \
    --max_prompt_length 3200 \
    --max_length 9500 \
    --per_device_train_batch_size 1