import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm  # Import tqdm for the progress bar

# == Model Definitions ==

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out

class ResCapsNet(nn.Module):
    def __init__(self, num_res_blocks=3, primary_caps_dim=8, primary_caps_channels=16,
                 digit_caps_dim=16, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.res_blocks = nn.Sequential(*[ResidualBlock(32) for _ in range(num_res_blocks)])
        self.primary_capsules = nn.Conv2d(32, primary_caps_channels * primary_caps_dim,
                                          kernel_size=9, stride=8)
        self.primary_caps_dim = primary_caps_dim
        self.primary_caps_channels = primary_caps_channels
        self.num_primary_caps = primary_caps_channels * 9 * 9
        self.num_classes = num_classes
        self.digit_caps_dim = digit_caps_dim
        self.W = nn.Parameter(
            0.01 * torch.randn(1, self.num_primary_caps, num_classes,
                               digit_caps_dim, primary_caps_dim)
        )

    def squash(self, s, dim=-1):
        s_norm_sq = (s ** 2).sum(dim=dim, keepdim=True)
        s_norm    = torch.sqrt(s_norm_sq + 1e-9)
        v = (s_norm_sq / (1.0 + s_norm_sq)) * (s / s_norm)
        return v

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.res_blocks(x)
        x = self.primary_capsules(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.primary_caps_channels, self.primary_caps_dim, 9, 9)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(batch_size, -1, self.primary_caps_dim)

        u = x.unsqueeze(2)
        W = self.W.repeat(batch_size, 1, 1, 1, 1)
        u_hat = torch.matmul(W, u.unsqueeze(-1)).squeeze(-1)
        b = torch.zeros(batch_size, self.num_primary_caps, self.num_classes, device=x.device)
        for r in range(3):
            c = F.softmax(b, dim=2).unsqueeze(-1)
            s = (c * u_hat).sum(dim=1)
            v = self.squash(s, dim=-1)
            if r < 2:
                agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1)
                b = b + agreement
        lengths = torch.sqrt((v ** 2).sum(dim=-1) + 1e-9)
        return v, lengths

# Margin loss
def margin_loss(lengths, labels, m_pos=0.9, m_neg=0.1, lambda_=0.5):
    batch_size = lengths.size(0)
    T = torch.eye(lengths.size(1), device=lengths.device).index_select(dim=0, index=labels)
    L_present = T * F.relu(m_pos - lengths).pow(2)
    L_absent  = (1 - T) * F.relu(lengths - m_neg).pow(2)
    loss = L_present + lambda_ * L_absent
    return loss.sum(dim=1).mean()


def setup_ddp(rank: int, world_size: int):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    init_method = f"tcp://{master_addr}:{master_port}"
    dist.init_process_group(
        backend='nccl',
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )


def train(rank, world_size, epochs=1, batch_size=64):
    setup_ddp(rank, world_size)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')
    print(f"[Rank {rank}/{world_size} | Local {local_rank}] starting on {device}")

    transform = transforms.Compose([
        transforms.Resize((78, 78), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler  = DistributedSampler(test_dataset,  num_replicas=world_size, rank=rank, shuffle=False)

    # Increase num_workers and prefetch_factor for better data loading performance
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, prefetch_factor=2)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, prefetch_factor=2)

    device = torch.device(f"cuda:{local_rank}")
    model = ResCapsNet().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Mixed precision setup
    scaler = GradScaler()

    best_acc = 0.0
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(1, epochs+1):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = total_correct = num_samples = 0

        print(f"[Rank {rank}] Epoch {epoch} started")

        # Progress bar for the training loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} - Train", dynamic_ncols=True)

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixed precision
            with autocast():
                _, outputs = model(images)
                loss = margin_loss(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            num_samples += images.size(0)

            # Update the progress bar with training metrics
            pbar.set_postfix(loss=total_loss/num_samples, acc=100.0 * total_correct/num_samples)

        loss_tensor    = torch.tensor(total_loss, device=device)
        correct_tensor = torch.tensor(total_correct, device=device)
        dist.reduce(loss_tensor,    dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(correct_tensor, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            train_loss = loss_tensor.item() / len(train_sampler.dataset)
            train_acc  = 100.0 * correct_tensor.item() / len(train_sampler.dataset)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

        model.eval()
        total_loss = total_correct = num_samples = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                with autocast():
                    _, outputs = model(images)
                    loss = margin_loss(outputs, labels)

                total_loss += loss.item() * images.size(0)
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                num_samples += images.size(0)

        loss_tensor    = torch.tensor(total_loss, device=device)
        correct_tensor = torch.tensor(total_correct, device=device)
        dist.reduce(loss_tensor,    dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(correct_tensor, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            test_loss = loss_tensor.item() / len(test_sampler.dataset)
            test_acc  = 100.0 * correct_tensor.item() / len(test_sampler.dataset)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.module.state_dict(), "best_rescaps_model.pth")

            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                  f"Train Acc={train_acc:.2f}%, Test Loss={test_loss:.4f}, "
                  f"Test Acc={test_acc:.2f}%")

    if rank == 0:
        dist.destroy_process_group()

if __name__ == "__main__":
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train(rank, world_size)
