import torch


def save_checkpoint(model, optimizer, epoch, save_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'save_loss': save_loss
    }, path)
    print(f"✅ Saved checkpoint at {path}")

def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    save_loss = checkpoint['save_loss']

    # Move optimizer tensors to device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    start_epoch = checkpoint['epoch'] + 1
    print(f"🔁 Resumed from checkpoint: {path}, starting at epoch {start_epoch}")
    return model, optimizer, start_epoch, save_loss

