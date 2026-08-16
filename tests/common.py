from __future__ import annotations

import os
import pathlib

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from cs336_systems.custom_dataset import DecoderDataset
from cs336_systems.utils import *

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"


def validate_ddp_net_equivalence(net):
    # Helper to validate synchronization of nets across ranks.
    net_module_states = list(net.module.state_dict().values())
    # Check that all tensors in module's state_dict() are equal.
    for t in net_module_states:
        tensor_list = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, t)
        for tensor in tensor_list:
            assert torch.allclose(tensor, t)


class _FC2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 50, bias=True)
        self.fc.bias.requires_grad = False

    def forward(self, x):
        x = self.fc(x)
        return x


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = _FC2()
        self.fc3 = nn.Linear(50, 10, bias=False)
        self.relu = nn.ReLU()
        self.embedding = nn.Embedding(37, 10)
        self.no_grad_fixed_param = nn.Parameter(torch.tensor([2.0, 2.0]), requires_grad=False)
        self.ln = nn.LayerNorm(10)
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        """
        originally x type is torch.int64
        after embedding it is float32 by default
        """
        embedded_tokens = self.embedding(x)
        print(embedded_tokens.dtype) # float32 by default
        x = self.relu(self.fc1(embedded_tokens))
        print(f"first ff layer {x.dtype}")
        x = self.ln(x)
        print(f"layer norm {x.dtype}")
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ToyModelWithTiedWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 10, bias=False)
        self.fc4 = nn.Linear(10, 50, bias=False)
        self.fc5 = nn.Linear(50, 10, bias=False)
        self.fc4.weight = self.fc2.weight
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def _setup_process_group(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    # https://discuss.pytorch.org/t/should-local-rank-be-equal-to-torch-cuda-current-device/150873/2
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError("Unable to find CUDA devices.")
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def _cleanup_process_group():
    # Synchronize before we destroy the process group
    dist.barrier()
    dist.destroy_process_group()

def read_from(file):
    """
    :param file:
    :return: A list of the lines in the file, each exactly 20 characters long
    """
    all_lines = []
    for line in open(file):
        all_lines.append(line[:-1]) # eat the \n
    print("%i lines read in" % len(all_lines))
    return all_lines

if __name__=='__main__':
    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # For Mac M1/M2/M3 chips
    else:
        device = torch.device("cpu")
    print(f"Device Type: {device.type}")

    model = ToyModel().to(device)
    print(model.parameters)
    x = torch.tensor([0]*10,dtype=torch.float32)
    y = torch.tensor([0]*10,dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    number = [chr(ord('0') + i) for i in range(0, 10)]
    vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' '] + number
    vocab_index = Indexer()
    for char in vocab:
        vocab_index.add_and_get_index(char)
    print(repr(vocab_index))

    char_to_idx = {
        char: vocab_index.index_of(char)
        for char in vocab
    }
    dataset = read_from("cs336_systems/evaluation_train.txt")
    dataset = DecoderDataset(dataset, char_to_idx, max_length=11)

    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        persistent_workers=4 > 0
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for name, p in model.named_parameters():
            print(name, p.dtype)
    """
    no_grad_fixed_param torch.float32
    fc1.weight torch.float32
    fc2.fc.weight torch.float32
    fc2.fc.bias torch.float32
    fc3.weight torch.float32
    embedding.weight torch.float32
    """

    """
    auto cast on mps

    model layers:
    first ff layer torch.float16
    layer norm torch.float32
    logits torch.float16
    loss torch.float32

    gradients
    no_grad_fixed_param grad is None
    fc1.weight, data type: torch.float32
    fc2.fc.weight, data type: torch.float32
    fc2.fc.bias grad is None
    fc3.weight, data type: torch.float32
    embedding.weight, data type: torch.float32
    ln.weight, data type: torch.float32
    ln.bias, data type: torch.float32
    linear.weight grad is None
    linear.bias grad is None
    """

    for idx, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        with torch.autocast(device_type="mps", dtype=torch.float16):
            """
            PyTorch chooses lower precision for operations that are 
            usually safe and faster in FP16, such as matrix multiplications 
            and convolutions. It keeps or promotes some operations to FP32 
            when FP16 would be unstable.
            """
            logits = model(input)
            print(f"logits {logits.dtype}")
            loss = loss_fn(logits, target)
            print(f"loss {loss.dtype}")
        if torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}, data type: {param.grad.dtype}")
                else:
                    print(name, "grad is None")

            optimizer.step()

