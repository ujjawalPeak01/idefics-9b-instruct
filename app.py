import json
import numpy as np
import torch
from transformers import pipeline


class InferlessPythonModel:

    # Implement the Load function here for the model
    def initialize(self):
        self.generator = pipeline("text-generation", model="HuggingFaceM4/idefics-9b-instruct", device=0)

    
    # Function to perform inference 
    def infer(self, inputs):
        prompt = inputs["prompt"]
        pipeline_output = self.generator(prompt, do_sample=True)
        generated_txt = pipeline_output[0]["generated_text"]
        print("Text: ", generated_txt)

        return {"generated_text": generated_txt}

    # perform any cleanup activity here
    def finalize(self,args):
        self.pipe = None
