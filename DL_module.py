# здесь будут реализованы DL модели для детекции точек разладки:
# ALACPD

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

class TAEnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, skip_size):
        super(TAEnet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.skip_size = skip_size
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, ::self.skip_size, :])
        return predictions

# Фиксация случайных значений при задании весов модели
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ALACPD:
    def __init__(self, input_dim, hidden_dim, output_dim, skip_sizes, ensemble_size, lr=0.001, w=6, horizon=4, beta=0.6, nCPD=3):
        self.models = [TAEnet(input_dim, hidden_dim, output_dim, skip_size) for skip_size in skip_sizes]
        self.criteria = [nn.MSELoss() for _ in range(ensemble_size)]
        self.optimizers = [optim.Adam(model.parameters(), lr=lr) for model in self.models]
        self.thresholds = [None for _ in range(ensemble_size)]
        self.losses = []
        self.w = w
        self.horizon = horizon
        self.beta = beta
        self.nCPD = nCPD
    
    def initialize(self, train_loader, ninit, einit):
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            for epoch in tqdm(range(einit), desc="Initializing Models"):
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = nn.MSELoss()(outputs, targets[:, ::model.skip_size, :])
                    loss.backward()
                    optimizer.step()
            self.calculate_threshold(model, train_loader, ninit)
    
    def calculate_threshold(self, model, train_loader, ninit):
        model.eval()
        losses = []
        with torch.no_grad():
            for inputs, targets in train_loader:
                outputs = model(inputs)
                loss = nn.MSELoss()(outputs, targets[:, ::model.skip_size, :])
                losses.append(loss.item())
        self.thresholds[self.models.index(model)] = np.mean(losses) + 2 * np.std(losses)


    def reinitialize_models(self, t, ninit, ereinit, series):
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            for epoch in range(ereinit):
                for i in range(max(0, t - ninit), t):
                    sample = torch.tensor(series[i], dtype=torch.float32).view(1, 1, -1)
                    optimizer.zero_grad()
                    output = model(sample)
                    loss = nn.MSELoss()(output, sample[:, ::model.skip_size, :])
                    loss.backward()
                    optimizer.step()
            self.calculate_threshold(model, torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                torch.tensor(series[max(0, t - ninit):t], dtype=torch.float32).view(-1, 1, 1), 
                torch.tensor(series[max(0, t - ninit):t], dtype=torch.float32).view(-1, 1, 1)
            ), batch_size=1), ninit)
            

    def train_online(self, series, ninit, einit, etrain, ereinit, threshold=0.5, anomaly_window=10, vol_window=21):
        data = torch.tensor(series, dtype=torch.float32).view(-1, 1, 1)
        dataset = torch.utils.data.TensorDataset(data, data)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        
        self.initialize(train_loader, ninit, einit)
        
        anomalous_samples = []
        change_points = []
        volatilities = []
        
        for t in tqdm(range(ninit, len(series)), desc="Online Training"):
            sample = data[t].view(1, 1, -1)
            sample_loss = 0
            
            for model, optimizer, threshold in zip(self.models, self.optimizers, self.thresholds):
                model.eval()
                with torch.no_grad():
                    output = model(sample)
                    loss = nn.MSELoss()(output, sample[:, ::model.skip_size, :])
                    sample_loss += loss.item()
            
            sample_loss /= len(self.models)
            volatilities.append(sample_loss)

            # усредняем волатильность за последние vol_window дней
            if len(volatilities) > vol_window:
                dynamic_threshold = np.mean(volatilities[-vol_window:]) + 2 * np.std(volatilities[-vol_window:])
            else:
                dynamic_threshold = threshold
            
            if sample_loss > dynamic_threshold:
                anomalous_samples.append(t)
                if len(anomalous_samples) >= self.nCPD and all(anomalous_samples[-1] - cp >= anomaly_window for cp in change_points):
                    confirmed_change_points = sum([1 for model in self.models if self.check_anomaly(model, sample)])
                    if confirmed_change_points >= self.beta * len(self.models):
                        change_points.append(t)
                        self.reinitialize_models(t, ninit, ereinit, series)
                    anomalous_samples = []
            else:
                anomalous_samples = []
                for model, optimizer in zip(self.models, self.optimizers):
                    model.train()
                    for _ in range(etrain):
                        optimizer.zero_grad()
                        output = model(sample)
                        loss = nn.MSELoss()(output, sample[:, ::model.skip_size, :])
                        loss.backward()
                        optimizer.step()
        
        change_points = self.filter_change_points(change_points)
        return change_points
    
    def check_anomaly(self, model, sample):
        with torch.no_grad():
            output = model(sample)
            loss = nn.MSELoss()(output, sample[:, ::model.skip_size, :])
        return loss.item() > self.thresholds[self.models.index(model)]
    
    def filter_change_points(self, change_points, min_length=10):
        filtered_points = []
        for i in range(len(change_points) - 1):
            if change_points[i+1] - change_points[i] > min_length:
                filtered_points.append(change_points[i])
        if change_points:
            filtered_points.append(change_points[-1])
        return filtered_points
    
    def plot_loss(self):
        plt.figure(figsize=(15, 3))
        plt.plot(self.losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.show()
    
    def plot_series_with_change_points(self, series, change_points, true_change_points=None):
        plt.figure(figsize=(15, 3))
        plt.plot(series, label='Series')
        for cp in change_points:
            plt.axvline(x=cp, color='r', linestyle='--', alpha = 0.8, label='Detected Change Point' if cp == change_points[0] else "")
        if true_change_points is not None:
            for tcp in true_change_points:
                plt.axvline(x=tcp, color='g', linestyle='-', label='True Change Point' if tcp == true_change_points[0] else "")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Series with Detected Change Points')
        plt.legend()
        plt.show()

    
    def calculate_metrics(self, detected_change_points, true_change_points, tolerance=5):
        delays = []
        for tcp in true_change_points:
            matching_dcp = [dcp for dcp in detected_change_points if abs(dcp - tcp) <= tolerance]
            if matching_dcp:
                delays.append(min(abs(dcp - tcp) for dcp in matching_dcp))
            else:
                delays.append(float('inf'))
        
        add = np.mean([d for d in delays if d != float('inf')])
        
        # Рассчитываем FDD
        fdd = len(detected_change_points)
        for dcp in detected_change_points:
            if not any(abs(dcp - tcp) <= tolerance for tcp in true_change_points):
                fdd = dcp
                break
        
        return {
            'Average Detection Delay (ADD)': add,
            'False Detection Delay (FDD)': fdd,
            'Undetected True Change Points': [tcp for tcp, delay in zip(true_change_points, delays) if delay == float('inf')]
        }
    
