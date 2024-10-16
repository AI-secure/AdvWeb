import sys
import json
import random
import numpy as np
import os
import re
import argparse
sys.path.append(os.getcwd())

from datasets import load_dataset, Dataset
from utils import HFAdvGenerator
from template_config.chat_template import get_chat_template


model_to_path = {
    'mistral': "mistralai/Mistral-7B-Instruct-v0.2",
    'llama': "meta-llama/Llama-2-7b-chat-hf",
}

model_to_template = {
    'mistral': "mistral-instruct",
    'llama': "llama-2-chat",
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(dataset_path):
    print(f'loading from {dataset_path}')
    with open(dataset_path) as f:
        return json.load(f)


def extract_multi_choice_text(text):
    pattern = r'(A\..*?None of the other options match the correct element)'
    match = re.search(pattern, text, re.DOTALL)
    assert match, "No multi-choice section found"
    return match.group(1)

# SFT dataset
def load_agent_data_together(model_name='mistral', no_task=True, seed=42, val_ratio=0.1):
    log_path = args.log_file
    print(f'log path: {log_path}')
    prompt_file = args.prompt_file
    print(f'prompt file: {prompt_file}')
    with open(log_path) as f:
        log_data = json.load(f)
    dataset = load_dataset(args.dataset)
    chat_template = model_to_template[model_name]
    model_path = model_to_path[model_name]
    adv_generater = HFAdvGenerator(model_path=model_path, prompt_file=prompt_file)

    chat_template = get_chat_template(chat_template)
    if chat_template is not None:
        adv_generater.tokenizer.chat_template = chat_template

    togther_dataset = []
    for data_id, logs in log_data.items():
        data_id = int(data_id)
        data = dataset[data_id]
        injected_prompt_2 = data['prompt_2_template'].replace(' {value} ', '')
        task = data['task'] + '.'
        input_prompt = adv_generater.get_prompt(task, extract_multi_choice_text(injected_prompt_2))
        success_adv_string_ids = []
        success_adv_strings = []
        for adv_string_id in range(10):
            current_logs = [item for item in logs if item['adv_string_id'] == adv_string_id]
            success_cnt = len([i for i in current_logs if i['attack_success']])
            if success_cnt >= 5:
                success_adv_string_ids.append(adv_string_id)
                success_adv_strings.append(current_logs[0]['adv_string'])
        print(f'data {data_id} has {len(success_adv_string_ids)} success adv strings')
        if len(success_adv_string_ids) > 0:
            fail_adv_strings = []
            for adv_string_id in range(10):
                if adv_string_id not in success_adv_string_ids:
                    current_logs = [item for item in logs if item['adv_string_id'] == adv_string_id]
                    fail_adv_strings.append(current_logs[0]['adv_string'])
            for chosen in success_adv_strings:
                prompt = input_prompt + ' ' + chosen
                togther_dataset.append({'text': prompt})

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = args.output_dir + '/sft_train.jsonl'
    with open(os.path.join(args.output_dir, output_file), 'w') as f:
        for line in togther_dataset:
            json.dump(line, f)
            f.write('\n')

# DPO dataset
def load_agent_data(model_name='mistral', no_task=True, seed=42, val_ratio=0.1):
    log_path = args.log_file
    prompt_file = args.prompt_file
    with open(log_path) as f:
        log_data = json.load(f)
    dataset = load_dataset(args.dataset)
    chat_template = model_to_template[model_name]
    model_path = model_to_path[model_name]
    adv_generater = HFAdvGenerator(model_path=model_path, prompt_file=prompt_file)

    chat_template = get_chat_template(chat_template)
    if chat_template is not None:
        adv_generater.tokenizer.chat_template = chat_template

    dpo_dataset = []
    for data_id, logs in log_data.items():
        data_id = int(data_id)
        data = dataset[data_id]
        injected_prompt_2 = data['prompt_2_template'].replace(' {value} ', '')
        task = data['task'] + '.'
        input_prompt = adv_generater.get_prompt(task, extract_multi_choice_text(injected_prompt_2))
        success_adv_string_ids = []
        success_adv_strings = []
        for adv_string_id in range(10):
            current_logs = [item for item in logs if item['adv_string_id'] == adv_string_id]
            success_cnt = len([i for i in current_logs if i['attack_success']])
            if success_cnt >= 5: 
                success_adv_string_ids.append(adv_string_id)
                success_adv_strings.append(current_logs[0]['adv_string'])
        print(f'data {data_id} has {len(success_adv_string_ids)} success adv strings')
        if len(success_adv_string_ids) > 0:
            fail_adv_strings = []
            for adv_string_id in range(10):
                if adv_string_id not in success_adv_string_ids:
                    current_logs = [item for item in logs if item['adv_string_id'] == adv_string_id]
                    fail_adv_strings.append(current_logs[0]['adv_string'])
            for chosen in success_adv_strings:
                for rejected in fail_adv_strings:
                    dpo_item = {
                        'prompt': input_prompt,
                        'chosen': chosen,
                        'rejected': rejected,
                    }
                    dpo_dataset.append(dpo_item)

    dpo_dataset = Dataset.from_list(dpo_dataset).train_test_split(test_size=val_ratio, shuffle=True, seed=seed)

    print(dpo_dataset)
    dpo_train_output = args.output_dir + '/dpo_train.jsonl'
    dpo_test_output = args.output_dir + '/dpo_test.jsonl'
    os.makedirs(args.output_dir, exist_ok=True)
    with open(dpo_train_output, 'w') as f:
        for line in dpo_dataset['train']:
            json.dump(line, f)
            f.write('\n')
    with open(dpo_test_output, 'w') as f:
        for line in dpo_dataset['test']:
            json.dump(line, f)
            f.write('\n')
    return dpo_dataset['train'], dpo_dataset['test']


if __name__ == '__main__':
    seed = 42
    set_seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_task', default=True, action='store_true', help='Remove task info')
    parser.add_argument('--model', default='mistral', help='Model name')
    parser.add_argument('--log_file', help='Log file path')
    parser.add_argument('--dataset', help='Dataset file path')
    parser.add_argument('--output_dir', help='Output file path')
    parser.add_argument('--prompt_file', help='Prompt file path')
    args = parser.parse_args()

    # sft dataset
    load_agent_data_together(no_task=True, seed=seed)

    # dpo dataset
    load_agent_data(no_task=True, seed=seed)


