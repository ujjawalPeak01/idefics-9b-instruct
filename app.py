import json
import numpy as np
import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor


class InferlessPythonModel:

    # Implement the Load function here for the model
    def initialize(self):
        checkpoint = "HuggingFaceM4/idefics-9b"
        self.model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to("cuda")
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    
    # Function to perform inference 
    def infer(self, inputs):
        prompts = [[inputs["prompts"]]]
        inputs = self.processor(prompts, return_tensors="pt").to("cuda")
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

        generated_ids = self.model.generate(**inputs, bad_words_ids=bad_words_ids, max_length=500)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        final_result = ""
        for i, t in enumerate(generated_text):
            print(f"{i}:\n{t}\n")
            final_result += f"{i}:\n{t}\n"

        return {"generated_text": final_result}

    # perform any cleanup activity here
    def finalize(self,args):
        self.model = None
        self.processor = None
        
