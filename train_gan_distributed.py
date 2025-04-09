import os
import datetime
import logging
import torch
import torch.nn as nn
import torch.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

from datasets.audio_dataset import AudioDataset
from encoder import DenseEncoder
from decoder import DenseDecoder
from critic import BasicCritic
from utils import ssim

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_distributed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_for_distributed(rank, world_size, master_addr, master_port):
    """
    Initialize distributed training
    Args:
        rank: Unique ID of each node (0 to world_size-1)
        world_size: Total number of nodes
        master_addr: IP address of master node
        master_port: Port number for communication
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)

    # Initialize process group
    dist.init_process_group(
        backend="nccl", # Use NCCL backend for GPU training
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank
    )

    logger.info(f"Initialized process group: rank {rank}/{world_size}")

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def get_data_loader(dataset, batch_size, rank, world_size):
    """Create distributed data loader"""
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

def train_distributed(local_rank, world_size, node_rank, nodes, master_addr,
                     master_port, batch_size=10, epochs=10, dataset_path='datasets/processed'):
    """
    Main training function for distributed training
    Args:
        local_rank: GPU ID on current node
        world_size: Total number of processes
        node_rank: Rank of current node
        nodes: Total number of nodes
        master_addr: IP address of master node
        master_port: Port number for communication
        batch_size: Batch size per GPU
        epochs: Number of training epochs
        dataset_path: Path to dataset (must be same on all nodes)
    """
    # Calculate global rank
    global_rank = node_rank * torch.cuda.device_count() + local_rank

    # Initialize distributed process group
    setup_for_distributed(global_rank, world_size, master_addr, master_port)

    # Set device
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # Load dataset from local storage
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise RuntimeError(f"Dataset not found at {dataset_path}. Please prepare dataset on all nodes.")

    train_dataset = AudioDataset(train_path, normalize=True, data_type='music')
    test_dataset = AudioDataset(test_path, normalize=True, data_type='music')

    # Initialize models
    data_depth = 2
    hidden_size = 96

    encoder = DenseEncoder(data_depth, hidden_size).to(device)
    decoder = DenseDecoder(data_depth, hidden_size).to(device)
    critic = BasicCritic(hidden_size).to(device)

    # Wrap models with DDP
    encoder = DDP(encoder, device_ids=[local_rank])
    decoder = DDP(decoder, device_ids=[local_rank])
    critic = DDP(critic, device_ids=[local_rank])

    # Optimizers
    learning_rate = 0.0002
    cr_optimizer = Adam(critic.parameters(), lr=learning_rate)
    en_de_optimizer = Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=learning_rate)

    # Initialize datasets and dataloaders
    train_loader = get_data_loader(train_dataset, batch_size, global_rank, world_size)
    test_loader = get_data_loader(test_dataset, batch_size, global_rank, world_size)

    # Tensorboard writer (only for rank 0)
    writer = None
    if global_rank == 0:
        writer = SummaryWriter(f"./logs/distributed")

    # Initialize metrics dictionary
    metrics = {field: [] for field in [
        'val.encoder_mse', 'val.decoder_loss', 'val.decoder_acc',
        'val.cover_score', 'val.generated_score', 'val.ssim',
        'val.psnr', 'val.bpp', 'train.encoder_mse',
        'train.decoder_loss', 'train.decoder_acc',
        'train.cover_score', 'train.generated_score',
    ]}

    # Training loop
    scaler = amp.GradScaler()

    for ep in range(epochs):
        train_loader.sampler.set_epoch(ep)

        if global_rank == 0:
            logger.info(f"Epoch {ep+1}")

        # Training steps similar to original script but with distributed considerations
        for cover, *rest in train_loader:
            with amp.autocast(device_type=device.type):
                cover = cover.to(device)
                N, _, H, W = cover.size()

                # Training logic here (similar to original script)
                # ...

        # Validation steps (only on rank 0)
        if global_rank == 0:
            # Validation logic here (similar to original script)
            # ...

            # Save checkpoint
            save_model(encoder.module, decoder.module, critic.module,
                      en_de_optimizer, cr_optimizer, metrics, ep)

    if writer is not None:
        writer.close()
    cleanup()

if __name__ == "__main__":
    # Configuration for distributed training
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=1,
                        help='Number of nodes/machines')
    parser.add_argument('--node-rank', type=int, default=0,
                        help='Rank of current node (0 to nodes-1)')
    parser.add_argument('--master-addr', type=str, default='localhost',
                        help='IP address of master node')
    parser.add_argument('--master-port', type=int, default=12355,
                        help='Port for master node')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dataset-path', type=str, default='datasets/processed',
                        help='Path to dataset directory')
    args = parser.parse_args()

    # Calculate world size (total processes = nodes * gpus per node)
    n_gpus_per_node = torch.cuda.device_count()
    world_size = args.nodes * n_gpus_per_node

    # Spawn processes on current node
    mp.spawn(
        train_distributed,
        args=(world_size, args.node_rank, args.nodes,
              args.master_addr, args.master_port,
              args.batch_size, args.epochs, args.dataset_path),
        nprocs=n_gpus_per_node,
        join=True
    )
