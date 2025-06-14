import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EarlyStopping:
    def __init__(self, patience, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_rmse = float('inf')
        self.min_delta = 1e-6

    def __call__(self, rmse, model, path='best_model.pth'):
        score = -rmse  # 越小越好，所以取負

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(rmse, model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(rmse, model, path)
            self.counter = 0

    def save_checkpoint(self, rmse, model, path):
        '''儲存最佳模型'''
        torch.save(model.state_dict(), path)
        self.best_rmse = rmse
        # print(path)
        # if self.verbose:
        #     print(f'Saved model with RMSE: {rmse:.4f}')