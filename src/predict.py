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


def run_val_step(self, valid_loader, model, x_cat=True):
    running_loss = 0.0
    model.eval()
    with torch.no_grad:
        for batch, data in enumerate(valid_loader):
            x = data['num_features'].to(self.device)
            y = data['target'].to(self.device)
            if x_cat is not None:
                x_cat = data['cat_features'].to(self.device)
                pred = model(x, x_cat).to(self.device)
            else:
                pred = model(x)
            loss = self.loss_fn(pred, y)
            running_loss += loss
        avg_loss = running_loss/(batch + 1)
        val_metrics = metrics(pred, y)
    return pred, avg_loss, val_metrics