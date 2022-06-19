import torch
import pandas as pd
import numpy as np
from metrics import metrics


class Predict:
    def __init__(self, path_to_model):
        self.path_to_model = path_to_model
    
    def load_model(self):
        self.model = torch.load(self.path_to_model)
        return self.model.eval()


def run_pred_step(test_loader, model, x_cat=True, target=False):
    running_loss = 0.0
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # with torch.no_grad:
    for batch, data in enumerate(test_loader):
        x = data['num_features'].to(device)
        if target:
            y = data['target'].to(device)
        if x_cat is not None:
            x_cat = data['cat_features'].to(device)
            pred = model(x, x_cat).to(device)
        else:
            pred = model(x)
        print('Test predictions:')
        print(pred)
            
    #         loss = self.loss_fn(pred, y)
    #         running_loss += loss
    #     avg_loss = running_loss/(batch + 1)
    #     val_metrics = metrics(pred, y)
    # return pred, avg_loss, val_metrics