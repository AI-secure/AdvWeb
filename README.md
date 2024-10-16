# AdvWeb: Controllable Black-box Attacks on VLM-powered Web Agents
<img src="https://github.com/yuanmengqi/AdvWeb1/blob/main/pipe_inference.png" alt="Image" width="700"/>
Code for our paper AdvWeb: Controllable Black-box Attacks on VLM-powered Web Agents

## Setup

Create virtual environment, for example with conda:
```
conda create -n AdvWeb python=3.12.2
conda activate AdvWeb
```

Install dependencies:
```
pip install -r requirements.txt
```

Clone this repository:
```
git clone https://github.com/yuanmengqi/AdvWeb.git
```

Set up OpenAI API key and other keys to the environment:  
(Our pipeline supports attacking various large language models such as GPT, Gemini, and Claude. Here, we take attacking GPT as an example.)

```
export OPENAI_API_KEY=<YOUR_KEY>
export HUGGING_FACE_HUB_TOKEN=<YOUR_KEY>
```

## Data 
We conduct experiments on the [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) dataset and test our approach against the state-of-the-art web agent framework, [SeeAct](https://osu-nlp-group.github.io/SeeAct/).

Download the source data [Multimodal-Mind2Web](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web/tree/main) from Hugging Face and store it in the path `data/Multimodal-Mind2Web/data/`.

Download the [Seeact Source Data](https://buckeyemailosu-my.sharepoint.com/personal/zheng_2372_buckeyemail_osu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzheng%5F2372%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2Fseeact%5Fsource%5Fdata&ga=1) and store it in the path `data/seeact_source_data/`.

## Run Demo
### Data Generation
#### Construct the training set and test set
Run the notebook `data_generation.ipynb` to filter data from the source dataset and construct the training set and test set.
#### Build datasets for SFT and DPO
Run `training_data_generation.sh` to test the quality of the data in the training set and construct datasets for SFT and DPO.  

After completing the Data Generation section, your file structure should look like this:
```
├──task_demo_-1_aug
    ├──attack_dataset.json
    ├──subset_test_data_aug
    │   ├── train.json
    │   ├── test.json
    │   ├── augmented_dataset.json
    │   ├── predictions
    │   │   ├── prediction-4api-augment-data.jsonl
    │   │   ├── augmented_dataset_correct.json
    │   │   └── prediction-4api-augment-data-correct.jsonl
    │   └── imgs
    │       └── f5da4b14-026d-4a10-ab89-f5720418f2b4_9016ffb6-7468-4495-ad07-756ac9f2af03.jpg
    └── together
        └── data
            └── sft_train_data.jsonl
```
### Model Training
#### SFT
We fine-tune the model by calling Together AI's API. The basic training process is as follows (for more instructions, please refer to the [Together AI docs](https://docs.together.ai/docs/fine-tuning-overview)):  
Set up Together AI API key:
```
export TOGETHER_API_KEY=<YOUR_KEY>
```
Upload training dataset:
```
together files upload "xxx.jsonl"
```
Train the SFT model:
```
together fine-tuning create \
  --training-file "file-xxx" \
  --model "mistralai/Mistral-7B-Instruct-v0.2" \
  --lora \
  --batch-size 16
```
Download the SFT model:
```
together fine-tuning download "ft-xxx"
```
You can store the SFT model in the path `data/task_demo_-1_aug/together/new_models/`.
#### DPO
Run `dpo_training.sh` to train the DPO model.  
Select the best training model based on the training curve, and run `dpo_model_merge.sh` to merge the model.
#### Evaluation
Run `evaluation.sh` to evaluate the SFT and DPO models.
## Citation
If you find this code useful, please cite our paper:

```
```

