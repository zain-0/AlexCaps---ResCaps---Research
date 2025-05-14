#!/usr/bin/env python3
"""
Distributed training script for AlexCapsNet using PyTorch DDP.
- Supports multi-GPU, multi-node training.
- Records and saves loss/accuracy per epoch, confusion matrix, ROC curve.
- Verbose status updates for monitoring.
- Uses mixed precision for improved performance.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR

# Parse arguments
parser = argparse.ArgumentParser(description="Distributed training for AlexCapsNet")
parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
parser.add_argument("--batch-size", type=int, default=128, help="batch size per GPU")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--data-path", type=str, default="./data", help="path to dataset")
parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)),
                    help="local rank for distributed training")
args = parser.parse_args()

# Set device based on local rank
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")

# Initialize process group for DDP
dist.init_process_group(backend="gloo", init_method="env://")
world_size = dist.get_world_size()
rank = dist.get_rank()

# ====== VERBOSE STATUS ======
if rank == 0:
    print(f"[RANK {rank}] ====== Starting Distributed AlexCapsNet Training ======")
    print(f"[RANK {rank}] Arguments: {args}")
print(f"[RANK {rank}] Using device: {device}")
print(f"[RANK {rank}] Initialized DDP - World Size: {world_size}, Rank: {rank}")

# Seed for reproducibility
torch.manual_seed(args.seed + rank)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(args.seed + rank)

# Define Capsule and AlexCapsNet models
class CapsuleLayer(nn.Module):
    def __init__(self, in_capsules, in_dim, out_capsules, out_dim, num_routing=3):
        super(CapsuleLayer, self).__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_routing = num_routing
        self.W = nn.Parameter(torch.randn(1, in_capsules, out_capsules, out_dim, in_dim) * 0.01)

    def squash(self, s, eps=1e-9):
        s_norm_sq = (s ** 2).sum(dim=-1, keepdim=True)
        return (s_norm_sq / (1 + s_norm_sq)) * (s / torch.sqrt(s_norm_sq + eps))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(-1)
        u_hat = torch.matmul(self.W, x).squeeze(-1)
        b_ij = torch.zeros(batch_size, self.in_capsules, self.out_capsules, device=x.device)
        for r in range(self.num_routing):
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(-1)
            s_j = (c_ij * u_hat).sum(dim=1)
            v_j = self.squash(s_j)
            if r < self.num_routing - 1:
                b_ij = b_ij + (u_hat * v_j.unsqueeze(1)).sum(dim=-1)
        return v_j

class AlexCapsNet(nn.Module):
    def __init__(self):
        super(AlexCapsNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.primary_caps = nn.Conv2d(256, 32 * 8, kernel_size=1, stride=1)
        self.digit_caps = CapsuleLayer(in_capsules=128, in_dim=8, out_capsules=10, out_dim=16, num_routing=3)

    def forward(self, x):
        x = self.features(x)
        x = self.primary_caps(x)
        b = x.size(0)
        x = x.view(b, 32, 8, x.size(2), x.size(3))
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(b, -1, 8)
        v = self.digit_caps(x)
        lengths = torch.norm(v, dim=-1)
        return lengths

model = AlexCapsNet().to(device)
model = DDP(model, device_ids=[args.local_rank])

# Optimizer with LARS (Layer-wise Adaptive Rate Scaling) for better performance with large batch sizes
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=2, gamma=0.7)  # Learning rate scheduler

# Enable mixed precision training
scaler = GradScaler()

def margin_loss(lengths, labels, m_plus=0.9, m_minus=0.1, lambda_val=0.5):
    one_hot = torch.eye(10, device=lengths.device).index_select(dim=0, index=labels)
    loss_pos = one_hot * F.relu(m_plus - lengths).pow(2)
    loss_neg = (1.0 - one_hot) * F.relu(lengths - m_minus).pow(2)
    return (loss_pos + lambda_val * loss_neg).sum(dim=1).mean()

# Data transforms
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Datasets
train_dataset = torchvision.datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform_test)

train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, pin_memory=True, num_workers=8)

print(f"[RANK {rank}] Datasets and DataLoaders prepared.")

train_losses, train_accs = [], []
test_losses, test_accs = [], []

if rank == 0:
    print(f"[RANK {rank}] Starting training for {args.epochs} epochs...")

from tqdm import tqdm  # Import tqdm

# Training loop with progress bar
accumulation_steps = 4  # Accumulate gradients over 4 steps to simulate larger batch size
for epoch in range(args.epochs):
    print(f"[RANK {rank}] === Epoch {epoch+1}/{args.epochs} ===")
    model.train()
    train_sampler.set_epoch(epoch)
    epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

    # Create the progress bar for this epoch
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch", position=0, leave=True) as pbar:
        optimizer.zero_grad()
        for step, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            # Mixed precision
            with autocast():
                output = model(data)
                loss = margin_loss(output, target)

            # Accumulate gradients
            scaler.scale(loss).backward()
            
            # Update gradients every 'accumulation_steps' steps
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            epoch_correct += (preds == target).sum().item()
            epoch_total += data.size(0)

            # Update the progress bar with the current loss and accuracy
            pbar.set_postfix(loss=epoch_loss / epoch_total, accuracy=epoch_correct / epoch_total)

    # Aggregate metrics from all processes
    loss_tensor = torch.tensor(epoch_loss, dtype=torch.float64, device=device)
    correct_tensor = torch.tensor(epoch_correct, dtype=torch.float64, device=device)
    total_tensor = torch.tensor(epoch_total, dtype=torch.float64, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    avg_loss = loss_tensor.item() / total_tensor.item()
    avg_acc = correct_tensor.item() / total_tensor.item()
    if rank == 0:
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)

    # Validation
    print(f"[RANK {rank}] Validation for epoch {epoch+1}...")
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = margin_loss(output, target)
            val_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            val_correct += (preds == target).sum().item()
            val_total += data.size(0)

    val_loss_tensor = torch.tensor(val_loss, dtype=torch.float64, device=device)
    val_correct_tensor = torch.tensor(val_correct, dtype=torch.float64, device=device)
    val_total_tensor = torch.tensor(val_total, dtype=torch.float64, device=device)
    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_total_tensor, op=dist.ReduceOp.SUM)
    val_avg_loss = val_loss_tensor.item() / val_total_tensor.item()
    val_avg_acc = val_correct_tensor.item() / val_total_tensor.item()
    if rank == 0:
        test_losses.append(val_avg_loss)
        test_accs.append(val_avg_acc)
        print(f"[RANK {rank}] Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Val Loss: {val_avg_loss:.4f}, Val Acc: {val_avg_acc:.4f}")

    # Update the learning rate scheduler
    scheduler.step()

# Final evaluation and saving metrics
if rank == 0:
    print(f"[RANK {rank}] Training complete. Saving results...")
    model.eval()
    local_preds, local_labels, local_probs = [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            local_probs.extend(probs.cpu().numpy())
            local_preds.extend(preds.cpu().numpy())
            local_labels.extend(target.cpu().numpy())

    gathered_preds, gathered_labels, gathered_probs = [None]*world_size, [None]*world_size, [None]*world_size
    dist.all_gather_object(gathered_preds, local_preds)
    dist.all_gather_object(gathered_labels, local_labels)
    dist.all_gather_object(gathered_probs, local_probs)

    import csv
    all_preds = np.array([p for sub in gathered_preds for p in sub])
    all_labels = np.array([t for sub in gathered_labels for t in sub])
    all_probs = np.vstack([elem for sub in gathered_probs for elem in sub])

    with open("metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(len(train_losses)):
            writer.writerow([i+1, train_losses[i], train_accs[i], test_losses[i], test_accs[i]])

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close(fig)

    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(all_labels, classes=list(range(10)))
    fpr, tpr, thresholds = roc_curve(y_true_bin.ravel(), all_probs.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.close()

    print(f"[RANK {rank}] Metrics, confusion matrix, and ROC curve saved.")
