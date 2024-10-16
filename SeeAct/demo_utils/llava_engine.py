# -*- coding: utf-8 -*-
# Copyright (c) 2024 OSU Natural Language Processing Group
#
# Licensed under the OpenRAIL-S License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.licenses.ai/ai-pubs-open-rails-vz1
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
# from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image

from .conversation import conv_templates, SeparatorStyle


class Engine:
    def __init__(self) -> None:
        pass

    def tokenize(self, input):
        return self.tokenizer(input)


class LLaVAEngine(Engine):
    def __init__(
            self,
            model_id="llava-hf/llava-v1.6-mistral-7b-hf",
            cache_dir="",
            stop=["\n"],
            temperature=0,
            **kwargs,
    ) -> None:
        """Init an LLaVA engine

        Args:
            model_id: model name, like "llava-hf/llava-1.5-7b-hf"
            cache_dir: path to cache model files
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            model (_type_, optional): Model family. Defaults to None.
        """
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.stop = stop
        self.temperature = temperature
        if 'mistral' in self.model_id:
            self.conv_template = conv_templates["mistral_instruct"]
        elif '34b' in self.model_id:
            self.conv_template = conv_templates["chatml_direct"]
        else:
            self.conv_template = conv_templates["llava_v1"]

        # Initialize prompt and image processor
        print(self.model_id)
        print(self.cache_dir)
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_id, device_map="auto",
                                                                       cache_dir=self.cache_dir)
        Engine.__init__(self, **kwargs)


    def generate(self, prompt: list = None, max_new_tokens=4096, temperature=None, image_path=None,
                 action_description=None, turn_number=0, **kwargs):
        system_prompt = prompt[0]
        action_generation_prompt = prompt[1]
        grounding_prompt = prompt[2]

        if turn_number == 0:
            raise Exception("Not implemented.")
        elif turn_number == 1:
            # Prepare LLaVA Prompt
            image = Image.open(image_path)

            if self.conv_template is not None:
                conv = self.conv_template.copy()
                conv.system = system_prompt
                conv.append_message(conv.roles[0], self.stop[0] + action_generation_prompt + self.stop[0])
                conv.append_message(conv.roles[1], action_description + self.stop[0])
                conv.append_message(conv.roles[0], '<image> ' + grounding_prompt + self.stop[0])
                conv.append_message(conv.roles[1], None)
                query = conv.get_prompt()  # Prompt generated from the selected template
            else:
                assert 'vicuna' in self.model_id, self.model_id
                query = "SYSTEM: " + system_prompt + self.stop[0] + "USER: " + self.stop[0] + action_generation_prompt + \
                        self.stop[0] + "ASSISTANT: " + action_description + self.stop[
                            0] + "USER: <image>" + grounding_prompt + self.stop[0] + "ASSISTANT: "


            prompts = [query]
            images = [image]
            inputs = self.processor(prompts, images, return_tensors="pt").to("cuda")

            # Conduct Inference
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_text = self.processor.batch_decode(output, skip_special_tokens=True)
            answer2 = generated_text[0].split('[/INST]')[-1]
            return answer2