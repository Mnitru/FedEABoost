import numpy as np
from collections import Counter, OrderedDict
from typing import List, Dict, Tuple
import random
import math
import copy

import torch
from torch.distributions.dirichlet import Dirichlet
from torchvision.datasets import CIFAR10, EMNIST, FashionMNIST
import torchvision.transforms as transforms
import torch.nn as nn
from torch.optim import SGD
from scipy.spatial.distance import cosine
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.data.sampler import WeightedRandomSampler

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def renormalize(dist: torch.tensor, labels: List[int], label: int):
    idx = labels.index(label)
    dist[idx] = 0
    dist /= sum(dist)
    dist = torch.concat((dist[:idx], dist[idx+1:]))
    return dist


def load_data(dataset: str):
    if dataset == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = CIFAR10("data", train=True, download=True, transform=train_transform)
        testset = CIFAR10("data", train=False, download=True, transform=test_transform)
    
    elif dataset == "fmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        trainset = FashionMNIST(root='data', train=True, download=True, transform=transform)
        testset = FashionMNIST(root='data', train=True, download=True, transform=transform)
    return trainset, testset

def partition_data(trainset, num_clients: int, num_iids: int, alpha: float, beta: float):
    classes = trainset.classes
    num_classes = len(classes)
    class_indices = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(trainset):
        class_indices[label].append(idx)
    ids = [[] for _ in range(num_clients)]
    label_dist = []
    for client_idx in range(num_clients):
        concentration = torch.ones(num_classes) * alpha
        class_distribution = Dirichlet(concentration).sample().numpy()
        class_distribution = class_distribution / class_distribution.sum()
        for c in range(num_classes):
            np.random.shuffle(class_indices[c])
        client_size = len(trainset) // num_clients  
        remaining_samples = client_size
        temp_indices = []
        for c in range(num_classes):
            num_samples = int(np.round(class_distribution[c] * client_size))
            num_samples = min(num_samples, len(class_indices[c]), remaining_samples)
            if num_samples > 0:
                temp_indices.extend(class_indices[c][:num_samples])
                class_indices[c] = class_indices[c][num_samples:]
                remaining_samples -= num_samples
        if not temp_indices:
            available_classes = [c for c in range(num_classes) if class_indices[c]]
            if available_classes:
                c = np.random.choice(available_classes)
                temp_indices.append(class_indices[c][0])
                class_indices[c].pop(0)
                remaining_samples -= 1
        while remaining_samples > 0 and any(class_indices[c] for c in range(num_classes)):
            available_classes = [c for c in range(num_classes) if class_indices[c]]
            if not available_classes:
                break
            c = np.random.choice(available_classes)
            temp_indices.append(class_indices[c][0])
            class_indices[c].pop(0)
            remaining_samples -= 1
        ids[client_idx] = temp_indices
        counter = Counter([trainset[idx][1] for idx in ids[client_idx]])
        label_dist.append({classes[i]: counter.get(i, 0) for i in range(num_classes)})
    return ids, label_dist


def compute_entropy(counts: Dict):
    entropy = 0.0
    counts = list(counts.values())
    counts = [0 if value is None else value for value in counts]
    for value in counts:
        entropy += -value/sum(counts) * math.log(value/sum(counts), len(counts)) if value != 0 else 0
    return entropy


def train(net, trainloader, learning_rate: float, proximal_mu: float = None):
    device = next(net.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=learning_rate)
    net.train()
    running_loss, running_corrects = 0.0, 0
    global_params = copy.deepcopy(net).parameters()
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        if proximal_mu is not None:
            proximal_term = 0.0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)
            loss = criterion(outputs, labels) + (proximal_mu / 2) * proximal_term
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        predicted = torch.argmax(outputs, dim=1)
        running_loss += loss.item() * images.shape[0]
        running_corrects += torch.sum(predicted == labels).item()
    running_loss /= len(trainloader.sampler)
    accuracy = running_corrects / len(trainloader.sampler)
    return running_loss, accuracy

def train_adaboost(net, trainloader, learning_rate: float, proximal_mu: float = None, selection_method: str = "entropy_ensemble") -> Tuple[float, float]:
    device = next(net.parameters()).device
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    
    num_models = 10
    local_epochs = 10
    num_classes = 10
    dataset_size = len(trainloader.dataset)
    sample_weights = torch.ones(dataset_size, device=device) / dataset_size
    ensemble_models = []
    model_alphas = []
    model_errors = []
    
    global_params = copy.deepcopy(net).parameters()
    
    full_loader = DataLoader(trainloader.dataset, batch_size=trainloader.batch_size, 
                             shuffle=False, num_workers=trainloader.num_workers, 
                             pin_memory=(device.type == 'cuda'))
    
    for model_idx in range(num_models):
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        loader = DataLoader(trainloader.dataset, batch_size=trainloader.batch_size, 
                           sampler=sampler, num_workers=trainloader.num_workers, 
                           pin_memory=(device.type == 'cuda'))
        
        model = copy.deepcopy(net).to(device)
        if torch.cuda.device_count() > 1 and device.type == 'cuda':
            model = nn.DataParallel(model)
        
        optimizer = SGD(model.parameters(), lr=learning_rate)
        
        model.train()
        for epoch in range(local_epochs):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    if proximal_mu is not None:
                        proximal_term = 0.0
                        for local_weights, global_weights in zip(model.parameters(), global_params):
                            proximal_term += (local_weights - global_weights).norm(2)
                        loss += (proximal_mu / 2) * proximal_term
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        ensemble_models.append(model.module if isinstance(model, nn.DataParallel) else model)
        
        model.eval()
        predictions = []
        all_labels = []
        with torch.no_grad():
            for images, labels in full_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                predictions.append(predicted)
                all_labels.append(labels)
        predictions = torch.cat(predictions)
        all_labels = torch.cat(all_labels)
        
        weighted_error = 0.0
        for c in range(num_classes):
            class_mask = (all_labels == c)
            if class_mask.sum() > 0:
                class_error = ((predictions != all_labels) & class_mask).float().sum() / class_mask.sum()
                class_weight = sample_weights[class_mask].sum() / sample_weights.sum()
                weighted_error += class_weight * class_error
        weighted_error = max(weighted_error.item(), 0.01)
        model_errors.append(weighted_error)
        
        alpha = np.log((1 - weighted_error) / (weighted_error + 1e-10)) + np.log(num_classes - 1)
        model_alphas.append(alpha)
        
        sample_weights = torch.where(predictions != all_labels, sample_weights * np.exp(alpha), sample_weights)
        sample_weights = torch.clamp(sample_weights / (sample_weights.sum() + 1e-10), min=1e-5, max=5.0) * len(sample_weights)
        
        class_weights = {f'class_{c}': sample_weights[all_labels == c].sum().item() for c in range(num_classes)}
        entropy = compute_entropy(class_weights)
        max_entropy = np.log(num_classes)
        entropy_ratio = entropy / max_entropy
        target_entropy_ratio = 0.9
        
        if entropy_ratio < target_entropy_ratio:
            beta = 0.3 * (entropy_ratio / target_entropy_ratio)
            for c in range(num_classes):
                class_mask = (all_labels == c)
                if class_mask.sum() > 0:
                    sample_weights[class_mask] = sample_weights[class_mask] ** beta
            sample_weights = sample_weights / (sample_weights.sum() + 1e-10) * len(sample_weights)
            sample_weights = torch.clamp(sample_weights, min=1e-5, max=5.0)
    
    # Đánh giá và in thông tin mô hình
    model_entropies = []
    model_losses = []
    print(f"\nĐánh giá mô hình trong client:")
    for idx, model in enumerate(ensemble_models):
        model.eval()
        total_entropy = 0.0
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for images, labels in full_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    outputs = model(images)
                    probs = F.softmax(outputs, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                    loss = criterion(outputs, labels)
                total_entropy += entropy.sum().item()
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
        avg_entropy = total_entropy / total_samples
        avg_loss = total_loss / total_samples
        print(f"Mô hình {idx + 1}: Loss = {avg_loss:.4f}, Entropy = {avg_entropy:.4f}, Alpha = {model_alphas[idx]:.4f}, Weighted Error = {model_errors[idx]:.4f}")
        model_entropies.append(avg_entropy)
        model_losses.append(avg_loss)
    
    # Chọn hoặc tổng hợp mô hình
    if selection_method == "best_error":
        best_idx = np.argmin(model_errors)
        client_model = ensemble_models[best_idx]
        running_loss = model_losses[best_idx]
        print(f"Chọn mô hình {best_idx + 1} với Weighted Error thấp nhất: {model_errors[best_idx]:.4f}")
    elif selection_method == "entropy_ensemble":
        valid_models = [model for idx, model in enumerate(ensemble_models) if model_losses[idx] <= 2.0]
        valid_alphas = [model_alphas[idx] for idx, loss in enumerate(model_losses) if loss <= 2.0]
        valid_entropies = [model_entropies[idx] for idx, loss in enumerate(model_losses) if loss <= 2.0]
        valid_losses = [model_losses[idx] for idx, loss in enumerate(model_losses) if loss <= 2.0]
        
        if not valid_models:
            print("Không có mô hình hợp lệ, chọn mô hình đầu tiên")
            valid_models = [ensemble_models[0]]
            valid_alphas = [model_alphas[0]]
            valid_entropies = [model_entropies[0]]
            valid_losses = [model_losses[0]]
        
        client_model = copy.deepcopy(net).to(device)
        model_weights = [alpha / (entropy + loss + 1e-10) for alpha, entropy, loss in zip(valid_alphas, valid_entropies, valid_losses)]
        total_weight = sum(model_weights)
        normalized_weights = [w / total_weight for w in model_weights]
        
        client_weights = {k: torch.zeros_like(v, device=device) for k, v in client_model.state_dict().items()}
        for k in client_weights.keys():
            if client_weights[k].dtype == torch.float:
                model_params = torch.stack([model.state_dict()[k].to(device) for model in valid_models])
                weight_shape = [-1] + [1] * (model_params.dim() - 1)
                weighted_params = model_params * torch.tensor(normalized_weights, device=device).view(*weight_shape)
                client_weights[k] = torch.sum(weighted_params, dim=0)
            else:
                client_weights[k].copy_(valid_models[0].state_dict()[k].to(device))
        
        with torch.no_grad():
            client_model.load_state_dict(client_weights)
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    # Đánh giá mô hình cuối
    client_model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in full_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                outputs = client_model(images)
                loss = criterion(outputs, labels)
            predicted = torch.argmax(outputs, dim=1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(predicted == labels).item()
            total_samples += images.size(0)
    
    running_loss /= total_samples
    accuracy = running_corrects / total_samples
    print(f"Mô hình cuối: Loss = {running_loss:.4f}, Độ chính xác = {accuracy:.4f}")
    
    return running_loss, accuracy

def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    corrects, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(next(net.parameters()).device)
            outputs = net(images)
            predicted = torch.argmax(outputs, dim=1)
            loss += criterion(outputs, labels).item() * images.shape[0]
            corrects += torch.sum(predicted == labels).item()
    loss /= len(testloader.sampler)
    accuracy = corrects / len(testloader.sampler)
    return loss, accuracy

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict)
   

