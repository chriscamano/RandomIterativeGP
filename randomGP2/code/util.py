import torch
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
import psutil
import os
from data.get_uci import ElevatorsDataset, ProteinDataset, BikeDataset, CTSlicesDataset,Kin40KDataset,KeggDirectedDataset,KeggUndirectedDataset,PoleteleDataset
from torch.utils.data import Dataset, DataLoader, random_split

def memory_dump():
    # Get the virtual memory details
    vm = psutil.virtual_memory()
    
    # Get the swap memory details
    sm = psutil.swap_memory()
    
    print("===== Virtual Memory =====")
    print(f"Total: {get_size(vm.total)}")
    print(f"Available: {get_size(vm.available)}")
    print(f"Used: {get_size(vm.used)}")
    print(f"Percentage: {vm.percent}%")
    
    print("\n===== Swap Memory =====")
    print(f"Total: {get_size(sm.total)}")
    print(f"Free: {get_size(sm.free)}")
    print(f"Used: {get_size(sm.used)}")
    print(f"Percentage: {sm.percent}%")

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor
    return f"{bytes:.2f}P{suffix}"

def fetch_uci_dataset(target_dataset: str, data_path=None, train_frac=4/9, val_frac=3/9, batch_size=1024):
    """
    Fetches a UCI dataset, splits it into train, validation, and test sets, and returns the data.

    Args:
        target_dataset (str): The name of the dataset to load.
        data_path (str): Path to the dataset file. If None, defaults to a specific path for 'elevators'.
        train_frac (float): Fraction of data to use for training.
        val_frac (float): Fraction of data to use for validation.
        batch_size (int): Batch size for data loading.

    Returns:
        Tuple[torch.Tensor]: Train data, train labels, test data, test labels.
    """
    # Dataset map for future extensions
    dataset_map = {
        "protein": ProteinDataset,
        "bike": BikeDataset,
        "elevators": ElevatorsDataset,
        "slice": CTSlicesDataset,
        "kin40k": Kin40KDataset,
        "keggdirected": KeggDirectedDataset,
        "keggundirected": KeggUndirectedDataset,
        "pol": PoleteleDataset,
    }

    # If data_path is not provided, default to the specific path for 'elevators'
    if data_path is None:
        data_path = r"C:\Users\fredw\chris\Research\softki\data\uci_datasets\uci_datasets\elevators\data.csv"

    # Check if the target dataset is valid
    if target_dataset.lower() not in dataset_map:
        raise ValueError(f"Unknown dataset '{target_dataset}'. Available options: {list(dataset_map.keys())}")

    # Load the dataset
    dataset_class = dataset_map[target_dataset.lower()]
    dataset = dataset_class(data_path)

    # Split the dataset
    train_size = int(len(dataset) * train_frac)
    val_size = int(len(dataset) * val_frac)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Ensure dimension attributes are copied
    train_dataset.dim = dataset.dim
    val_dataset.dim = dataset.dim
    test_dataset.dim = dataset.dim

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # Extract data from DataLoaders
    train_data, train_labels = next(iter(train_loader))
    test_data, test_labels = next(iter(test_loader))
    print("Dataset loaded")
    return train_data, train_labels, test_data, test_labels


# Define data generation function
def generate_data(n=40, test_n=100, noise_std=0.1, device='cuda:0', dtype=torch.float64):
    X_train = torch.linspace(-3, 3, n, dtype=dtype, device=device).unsqueeze(-1)
    y_train = torch.sin(X_train * 2) + torch.cos(3 * X_train) + noise_std * torch.randn_like(X_train)
    
    X_test = torch.linspace(-3, 3, test_n, dtype=dtype, device=device).unsqueeze(-1)
    y_test = torch.sin(X_test * 2) + torch.cos(3 * X_test)  # Precomputed test labels
    
    return X_train, y_train.squeeze(), X_test, y_test.squeeze()

# Define model training function
def train(model, train_x, train_y, test_x, test_y, training_iterations=100, lr=0.1):
    # optimizer = torch.optim.Adam([
    #     {'params': model.kernel.parameters()}, 
    #     {'params': model.likelihood.parameters()}
    # ], lr=lr)
    optimizer = torch.optim.Adam([
        {'params': model.kernel.parameters()}, 
        {'params': [model.noise.u]}  # Use raw_value instead of noise()
    ], lr=lr)
    # lr_sched = lambda epoch: 1.0  # Modify this function to change learning rate dynamically
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_sched)

    runtime_log, mll_loss_log, test_rmse_log = [], [], []
    for i in tqdm(range(training_iterations)):
        start_time = time.time()
        optimizer.zero_grad()
        
        model.fit(train_x, train_y)
        loss = model.compute_mll(train_y)
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        optimizer.step()
        # scheduler.step()  # Update the learning rate

        # print(loss)
        mean, covar = model.predict(test_x)
        total_time = time.time() - start_time
        runtime_log.append(total_time)
        mll_loss_log.append(-loss.item())
        
        test_rmse = (torch.mean(torch.abs(mean - test_y))).item()
        test_rmse_log.append(test_rmse)
        if (i + 1) % 20 == 0:
            print(f'Iter {i+1}/{training_iterations}, Loss: {loss.item():.4f}')
    
    return model, runtime_log, mll_loss_log, test_rmse_log, mean, covar

def eval(model, test_x, test_y):
    mean, covar = model.predict(test_x)
    test_rmse = (torch.mean(torch.abs(mean - test_y))).item()
    return mean, covar, test_rmse



def plot_gpr_results(train_x, train_y, test_x, test_y, GP_mean, std, runtime_log=None, mll_loss_log=None, test_rmse_log=None):
    GP_mean=GP_mean.detach().cpu().numpy()
    std=std.detach().cpu().numpy()
    # Create the main GPR plot
    plt.figure(figsize=(10, 5), dpi=400)
    
    # Plot GPR mean prediction
    plt.plot(test_x.cpu().numpy(), GP_mean, color='navy', lw=2, label=r'GPR mean $\widehat{m}$')
    
    # Plot uncertainty region
    plt.fill_between(test_x.cpu().numpy().ravel(), 
                    GP_mean - 1.96*std, 
                    GP_mean + 1.96*std, 
                    color='navy', 
                    alpha=0.2, 
                    label='Uncertainty')
    
    # Plot training points
    plt.scatter(train_x.cpu().numpy(), 
               train_y.cpu().numpy(),
               s=40, 
               color='darkorange', 
               edgecolors='black', 
               label='Noisy Observations', 
               zorder=3)
    
    # Plot true function
    plt.plot(test_x.detach().cpu().numpy(), 
            test_y.detach().cpu().numpy(), 
            color='green', 
            alpha=0.6, 
            label="True Function", 
            zorder=2)
    
    # Customize appearance
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    
    # Move the legend above the plot
    plt.legend(fontsize=14, 
              loc='upper center', 
              bbox_to_anchor=(0.5, 1.15),
              ncol=4, 
              fancybox=True, 
              shadow=False, 
              framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent legend cutoff
    
    # If training metrics are provided, create a second plot
    if any(x is not None for x in [runtime_log, mll_loss_log, test_rmse_log]):
        plt.figure(figsize=(15, 5), dpi=400)
        if mll_loss_log is not None:
            plt.subplot(1, 2, 1)
            plt.plot(mll_loss_log, color='blue', lw=2)
            plt.title('Negative Log Likelihood Loss', fontsize=14)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.grid(True, alpha=0.3)
        
        if test_rmse_log is not None:
            plt.subplot(1, 2, 2)
            plt.plot(test_rmse_log, color='red', lw=2)
            plt.title('Test RMSE', fontsize=14)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('RMSE', fontsize=12)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    plt.show()

