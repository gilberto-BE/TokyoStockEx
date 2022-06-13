import torch
import pandas as pd
import numpy as np



class Predict:
    def __init__(self, path_to_model):
        self.path_to_model = path_to_model
    
    def load_model(self):
        model = torch.load(self.path_to_model)
        return model.eval()
