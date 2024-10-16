cd SeeAct/offline_experiments/gpt_4_standalone

python adv_hf.py \
       --model_path "data/task_demo_-1_aug/together/new_models/mistral-7b-sft" \
       --test_dataset "data/task_demo_-1_aug/subset_test_data_aug/test.json" \
       --log_file "outputs/adv_hf_task_demo_dpo.json" 