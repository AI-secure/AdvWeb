cd SeeAct/offline_experiments/gpt_4_standalone

python adv_gpt.py \
    --dataset data/task_demo_-1_aug/subset_test_data_aug/train.json \
    --output outputs/adv_gpt_demo.json

python ../../../dpo/build_agent_dataset.py \
    --prompt_file SeeAct/offline_experiments/gpt_4_standalone/prompt_no_task.txt \
    --log_file outputs/adv_gpt_demo.json \
    --dataset data/task_demo_-1_aug/subset_test_data_aug/train.json \
    --output_dir data/task_demo_-1_aug/together/data