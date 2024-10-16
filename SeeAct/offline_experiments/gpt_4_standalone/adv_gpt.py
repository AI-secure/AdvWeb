import os
import openai
import re
import json
import argparse
from inference_engine import OpenaiEngine
from utils import test_attack_success


def extract_multi_choice_text(text):
    pattern = r'(A\..*?None of the other options match the correct element)'
    match = re.search(pattern, text, re.DOTALL)
    assert match, "No multi-choice section found"
    return match.group(1)


def extract_with_adv_string(text):
    match = re.search(r'[A-Z]\. <input[^>]*\{adv_string\}[^>]*>', text)
    return match.group(0) if match else None

class GPTAdvGenerator:
    def __init__(self, model_version = 'gpt-4-turbo-2024-04-09', prompt_file = 'prompt.txt'):
        self.model = model_version
        openai.api_key = API_KEY
        try:
            print(f'loading from {prompt_file}')
            with open(prompt_file, 'r') as file:
                self.prompt = file.read()
        except FileNotFoundError:
            print(f"Error: The file was not found.")
        
    def generate(self, task, choices, sampling_number = 10, temperature = 1.0):
        
        current_prompt = self.prompt.replace("{task}", task)
        current_prompt = current_prompt.replace("{choices}", str(choices))
        current_prompt = current_prompt.replace("{element}", extract_with_adv_string(choices))

        messages = [{
            'role': 'system', 
            'content': 'You are a helpful assistant.'
        }, {
            'role': 'user', 
            'content': current_prompt
        }]
        
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            n=sampling_number,
            temperature=temperature
        )
        
        responses = [resp['message']['content'] for resp in completion.choices]
        return responses


def load_dataset(dataset_path):
    print(dataset_path)
    with open(dataset_path) as f:
        return json.load(f)


def load_or_initialize_log(log_path, resume):
    if os.path.exists(log_path) and resume:
        print(f'resume from {log_path}')
        with open(log_path) as f:
            log = json.load(f)
            return log, max([int(k) for k in log.keys()])
    else:
        return {}, -1


if __name__ == '__main__':
    API_KEY = os.getenv('OPENAI_API_KEY')
    parser = argparse.ArgumentParser(description='Run adversarial attack simulations.')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume from the last checkpoint')
    parser.add_argument('--no_task', default=True, action='store_true', help='Remove task info')
    parser.add_argument('--dataset', help='Dataset file path')
    parser.add_argument('--output', help='Output file path')
    args = parser.parse_args()


    generation_model = OpenaiEngine(
        rate_limit=-1,
        api_key=API_KEY,
        model='gpt-4-vision-preview'
    )

    dataset_dir = os.path.dirname(args.dataset)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Dataset file: {args.dataset}")
    print(f"Output file: {args.output}")
    log_file = args.output

    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} data points")
    logs, last_data_id = load_or_initialize_log(log_file, args.resume)

    start_id = last_data_id + 1 if args.resume else 0
    adv_generater = GPTAdvGenerator(prompt_file='prompt_no_task.txt' if args.no_task else 'prompt.txt')

    total_success_count = 0
    total_attempts = 0

    for data_id in range(start_id, len(dataset)):
        data = dataset[data_id]
        injected_prompt_2 = data['prompt_2_template'].replace(' {value} ', '')
        attack_targets = data['candidate_values']
        task = data['task'] + '.'
        gpt_adv_strings = adv_generater.generate(task, extract_multi_choice_text(injected_prompt_2))
        current_success_count = 0

        if str(data_id) not in logs:
            logs[str(data_id)] = []

        for adv_string_id, adv_string in enumerate(gpt_adv_strings):
            cur_fail = 0
            for i, attack_target in enumerate(attack_targets):
                prompt_2 = injected_prompt_2.replace('{adv_string}', adv_string.strip()).replace('{value}', attack_target)
                target_output = data['target_output'].replace('{value}', attack_target)
                output1 = generation_model.generate(
                    prompt=[data['prompt_0'], data['prompt_1'], prompt_2],
                    image_path=os.path.join(dataset_dir, data['img_path']),
                    turn_number=1,
                    ouput__0=data['output_0']
                )
                attack_success = test_attack_success(output1, target_output)
                current_success_count += 1 if attack_success else 0
                total_success_count += 1 if attack_success else 0
                total_attempts += 1

                logs[str(data_id)].append({
                    'adv_string_id': adv_string_id,
                    'adv_string': adv_string,
                    'attack_target': attack_target,
                    'response': output1,
                    'attack_success': attack_success
                })
                current_success_rate = (current_success_count / (i + 1)) * 100
                overall_success_rate = (total_success_count / total_attempts) * 100
                print(f"Data ID {data_id}, Prompt ID {adv_string_id}, Target ID {i}: Current Fail: {cur_fail}")
                if not attack_success:
                    cur_fail += 1
                    if cur_fail >= 5:
                        break

            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=4)
            
