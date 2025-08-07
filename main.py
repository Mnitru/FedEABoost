import torch
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import flwr as fl
from flwr.common import ndarrays_to_parameters
from typing import Dict, Tuple, List
from fedalg import FedAvg, FedProx
from model import CNN4, LeNet, ResNet18
import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

# Loading the data

NUM_CLIENTS = 20
BATCH_SIZE = 64

trainset, testset = utils.load_data("fmnist")

ids, label_dist = partition_data(trainset, num_clients=20, num_iids=0, alpha=0.5, beta=0.0)

for i in range(NUM_CLIENTS):
    print(f"Client {i+1}: {dist[i]}")

entropies = [utils.compute_entropy(dist[i]) for i in range(NUM_CLIENTS)]

trainloaders = []
valloaders = []
val_length = [int(len(testset)/NUM_CLIENTS)] * NUM_CLIENTS
valsets = random_split(testset, val_length)
for i in range(NUM_CLIENTS):
    trainloaders.append(DataLoader(trainset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(ids[i])))
    valloaders.append(DataLoader(valsets[i], batch_size=BATCH_SIZE))


# Defining Flower client

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, net: torch.nn.Module, trainloader: torch.utils.data.DataLoader, valloader: torch.utils.data.DataLoader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config: Dict) -> List:
        return utils.get_parameters(self.net)

    def fit(self, parameters: List, config: Dict) -> Tuple[List, int, Dict]:
        utils.set_parameters(self.net, parameters)
        learning_rate = config.get("learning_rate", 0.01)
        proximal_mu = config.get("proximal_mu", None)
        strategy = config.get("strategy", "fedavg")

        if strategy == "fedavg":
            loss, accuracy = utils.train(self.net, self.trainloader, learning_rate=learning_rate)
        elif strategy == "fedprox":
            loss, accuracy = utils.train(self.net, self.trainloader, learning_rate=learning_rate, proximal_mu=proximal_mu)
        elif strategy == "adaboost_best_error":
            loss, accuracy = utils.train_adaboost(self.net, self.trainloader, learning_rate=learning_rate, 
                                                  proximal_mu=proximal_mu, selection_method="best_error")
        elif strategy == "adaboost_entropy_ensemble":
            loss, accuracy = utils.train_adaboost(self.net, self.trainloader, learning_rate=learning_rate, 
                                                  proximal_mu=proximal_mu, selection_method="entropy_ensemble")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return utils.get_parameters(self.net), len(self.trainloader.sampler), {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "id": self.cid
        }

    def evaluate(self, parameters: List, config: Dict) -> Tuple[float, int, Dict]:
        utils.set_parameters(self.net, parameters)
        loss, accuracy = utils.test(self.net, self.valloader)
        return float(loss), len(self.valloader.sampler), {"accuracy": float(accuracy), "loss": float(loss)}

def client_fn(cid: str) -> FlowerClient:
    net = utils.LeNet().to(DEVICE)  
    trainloader = utils.trainloaders[int(cid)]  
    valloader = utils.valloaders[int(cid)]  
    return FlowerClient(cid, net, trainloader, valloader)


# Training

NUM_ROUNDS = 200
current_parameters = ndarrays_to_parameters(utils.get_parameters(CNN4()))
client_resources = {"num_cpus": 1, "num_gpus": 0.1} if DEVICE.type == "cuda" else {"num_cpus": 1, "num_gpus": 0.0}

fl.simulation.start_simulation(
    client_fn = client_fn,
    num_clients = NUM_CLIENTS,
    config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy = FedAvg(num_rounds=NUM_ROUNDS, 
                        num_clients=NUM_CLIENTS,
                        current_parameters=current_parameters, 
                        ),
    client_resources = client_resources
)
