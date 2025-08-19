import numpy as np
from typing import Tuple
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from utils import compute_entropy


seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def min_max_normalize(values):
    min_val, max_val = min(values), max(values)
    return [(v - min_val) / (max_val - min_val + 1e-10) for v in values]
    
def train_adaboost_b(net: nn.Module, trainloader: DataLoader, lr: float, epochs: int, rounds:int) -> Tuple[float, float, nn.Module]:
    try:
        dataset = trainloader.dataset
        batch_size = trainloader.batch_size
        sample_weights = torch.ones(len(dataset), device=DEVICE) / len(dataset)
        models, alphas = [], []
        weights_entropy, dists = [], []
        K = 10  # Number of classes for SAMME
        pre_err = 10000000
        for _ in range(3):
            label_weight_sum = {i: 0.0 for i in range(K)}
            for i in range(len(dataset)):
                data, label = dataset[i]  
                w = sample_weights[i].item()
                label = label.item() if isinstance(label, torch.Tensor) else label
                label_weight_sum[label] += w * len(dataset)
            entropy = compute_entropy(label_weight_sum)
            if _ == 0:
                loader = trainloader
            else:
                loader = DataLoader(dataset, batch_size=batch_size, sampler=WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True))
            model = type(net)().to(DEVICE); model.load_state_dict(net.state_dict())
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=1e-4, weight_decay=1e-5)
            criterion = nn.CrossEntropyLoss()
            model.train()
            for ep in range(epochs):
                for x, y in loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    loss = criterion(model(x), y)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

            model.eval()
            incorrect_flags = torch.zeros(len(dataset), device=DEVICE)
            with torch.no_grad():
                for idx in range(len(dataset)):
                    x, y = dataset[idx]
                    x = x.unsqueeze(0).to(DEVICE)
                    y = torch.tensor([y], device=DEVICE)
                    pred = model(x).argmax(dim=1)
                    incorrect_flags[idx] = (pred != y).float()

            weighted_err = (incorrect_flags * sample_weights).sum().item()
            weighted_err = min(max(weighted_err, 1e-10), 1 - 1e-10)
            alpha = np.log((1 - weighted_err) / weighted_err) + np.log(K - 1)

            pre_err = weighted_err
            alphas.append(alpha); models.append(model)
            weights_entropy.append(entropy)
            dists.append(label_weight_sum)
            # Update sample weights
            preds, labels = [], []
            with torch.no_grad():
                for x, y in DataLoader(dataset, batch_size=batch_size):
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    preds.append(model(x).argmax(1)); labels.append(y)
            incorrect_tensor = torch.tensor(incorrect_flags, device=DEVICE)
            sample_weights = sample_weights * torch.exp(0.5 * alpha * incorrect_tensor)
            sample_weights = sample_weights / sample_weights.sum()
            
        norm_alphas = min_max_normalize(alphas)
        norm_entropy = min_max_normalize(weights_entropy)
        scores = [(a * e)/(a + e) if (a + e) != 0 else 0 for a, e in zip(norm_alphas, norm_entropy)]
        idx = np.argmax(scores)
        best_model = models[idx]
        best_model.eval(); total_loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = best_model(x)
                total_loss += criterion(out, y).item() * len(x)
                correct += (out.argmax(1) == y).sum().item()

        loss = total_loss / len(dataset)
        acc = correct / len(dataset)
        return loss, acc, best_model

    except Exception as e:
        print(f"Error in train_adaboost_b: {e}")
        return 0.0, 0.0, net
