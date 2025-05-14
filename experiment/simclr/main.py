import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, dataset
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.stl10 import STL10
from tqdm import tqdm

from experiment.simclr import utils
from experiment.simclr.model import Model
from experiment.simclr.dataset import *


# train for one epoch to learn unique features
from loss import DCL
from loss.dcl import DCLW


def train(net, data_loader, train_optimizer, args, epoch, device):
    net = net.to(device)
    net.train()
    
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2 in train_bar:
        pos_1, pos_2 = pos_1.to(device, non_blocking=True), pos_2.to(device, non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        if args.loss == 'dcl':
            l = DCL(temperature=args.temperature).to(device)
            loss = l(out_1, out_2) + l(out_2, out_1)
        elif args.loss == 'dclw':
            l = DCLW(temperature=args.temperature)
            loss = l(out_1, out_2) + l(out_2, out_1)
        elif args.loss == 'ce':
            out = torch.cat([out_1, out_2], dim=0)
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size, device=device)).bool()
            sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size, -1)

            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.temperature)
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, args.epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, args, epoch):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=args.k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / args.temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * args.k, args.c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, args.c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, args.epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


def select_dataset(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        return utils.create_pair_dataset(CIFAR10)
    if dataset_name == 'cifar100':
        return utils.create_pair_dataset(CIFAR100)
    if dataset_name == 'stl10':
        return utils.create_pair_dataset(STL10)
    raise ValueError("Invalid dataset name")


def run_train():
    import argparse
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import os
    import pandas as pd

    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--loss', default='ce', type=str, help='loss function')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name: cifar10, cifar100, stl10')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')

    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, total_epochs = args.batch_size, args.epochs
    data_root = args.dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = PairedImageDataset(root=data_root, transform=train_transform)
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)

    model = Model(feature_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    results = {'train_loss': []}
    save_name_pre = f'{feature_dim}_{temperature}_{k}_{batch_size}_{total_epochs}_{args.loss}'
    os.makedirs('results', exist_ok=True)
    checkpoint_path = f'results/{save_name_pre}_checkpoint.pth'

    # Resume logic
    start_epoch = 1
    best_loss = 1e9
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(model.state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        results = checkpoint.get('results', {'train_loss': []})

        # move optimizer tensors to the right device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # Training loop
    for epoch in range(start_epoch, total_epochs + 1):
        train_loss = train(model, train_loader, optimizer, args, epoch, device)
        results['train_loss'].append(train_loss)

        # Save CSV statistics
        df = pd.DataFrame(data=results, index=range(1, epoch + 1))
        df.to_csv(f'results/{save_name_pre}_statistics.csv', index_label='epoch')

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), f'results/{save_name_pre}_model.pth')

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'results': results
        }
        torch.save(checkpoint, checkpoint_path)