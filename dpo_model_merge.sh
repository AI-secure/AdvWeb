python dpo/merge_peft_adapter.py \
        --base_model_name="data/task_demo_-1_aug/together/new_models/mistral-7b-sft" \
        --adapter_model_name="data/task_demo_-1_aug/together/new_models/mistral-7b-dpo/checkpoint-300/" \
        --output_name="data/task_demo_-1_aug/together/new_models/mistral-7b-dpo-merged"