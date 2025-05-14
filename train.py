from experiment.simclr.main import run_train

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()  # Optional, but good for safety
    run_train()
