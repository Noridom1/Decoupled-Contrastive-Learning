import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoImageProcessor
from dcl import DCL, DCLW
from dataset import SimCLRPairDataset
from model import DINOv2SimCLR
from utils.checkpoint import *
from tqdm import tqdm

def train(args):
    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Dataset & Processor
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    dataset = SimCLRPairDataset(args.data_dir, processor_name=args.model_name)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 3. Backbone + Projection Head Model
    backbone = AutoModel.from_pretrained(args.model_name)
    for param in backbone.parameters():
        param.requires_grad = False

    model = DINOv2SimCLR(backbone=backbone, feature_dim=args.feature_dim).to(device)

    # 4. Optimizer
    optimizer = torch.optim.Adam(model.g.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 5. Contrastive Loss
    criterion = DCLW(sigma=args.sigma, temperature=args.temp) if args.loss == "dclw" else DCL(temperature=args.temp)

    # 6. Load Checkpoint (if given)
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, args.checkpoint, device)

    # 7. Training Loop
    model.train()
    for epoch in range(start_epoch, args.epochs):
        total_loss = 0.0
        for img1, img2 in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            img1, img2 = img1.to(device), img2.to(device)

            _, z1 = model(img1)
            _, z2 = model(img2)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"📘 Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

        # Save checkpoint
        if args.save_path:
            os.makedirs(args.save_path, exist_ok=True)
            save_path = os.path.join(args.save_path, f"latest.pth")
            save_checkpoint(model, optimizer, epoch, save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder containing original and augmented folders")
    parser.add_argument("--model_name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--loss", choices=["dcl", "dclw"], default="dcl")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint to resume from")

    args = parser.parse_args()
    train(args)
