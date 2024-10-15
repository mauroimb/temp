import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.nn as nn

# Initialize distributed training environment
def setup_distributed(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def train(rank, world_size):
    print(f"Running on rank {rank}.")
    
    setup_distributed(rank, world_size)
    
    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # Dummy input
    inputs = torch.randn(20, 10).to(rank)
    targets = torch.randn(20, 10).to(rank)
    
    for epoch in range(10):  # Training loop
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    cleanup()

if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"].split("-")[-1])  # Extract the rank from pod name (e.g., distrib-sts-0 -> rank 0)

    train(rank, world_size)
