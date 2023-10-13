import json
import numpy as np
import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor


class InferlessPythonModel:

    # Implement the Load function here for the model
    def initialize(self):
        checkpoint = "HuggingFaceM4/idefics-9b"
        model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to("cuda")
        processor = AutoProcessor.from_pretrained(checkpoint)

    
    # Function to perform inference 
    def infer(self, inputs):
        prompts = [
                "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
                "In this picture from Asterix and Obelix, we can see"
            ],
        ]
        inputs = processor(prompts, return_tensors="pt").to("cuda")
        bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

        generated_ids = model.generate(**inputs, bad_words_ids=bad_words_ids, max_length=100)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        final_result = ""
        for i, t in enumerate(generated_text):
            print(f"{i}:\n{t}\n")
            final_result += f"{i}:\n{t}\n"

        return {"generated_text": final_result}

    # perform any cleanup activity here
    def finalize(self,args):
        self.model = None
        self.processor = None
        
