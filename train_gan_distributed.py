import os
import datetime
import logging
import time
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


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    print(f"Setting up distributed training with rank {rank} and world size {world_size}")

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method='file:///tmp/distributed_training')


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

def save_model(encoder, decoder, critic, en_de_optimizer, cr_optimizer, metrics, ep, save_dir='models/distributed'):
    """Save model checkpoints for distributed training
    Note: This should only be called from rank 0 process
    """
    os.makedirs(save_dir, exist_ok=True)
    now = datetime.datetime.now()

    # Get the average validation accuracy across all processes
    if 'val.decoder_acc' in metrics and len(metrics['val.decoder_acc']) > 0:
        cover_score = metrics['val.decoder_acc'][-1]
    else:
        cover_score = 0.0

    name = f"distributed_{encoder.__class__.__name__}_{decoder.__class__.__name__}_{cover_score:.3f}_{now.strftime('%Y-%m-%d_%Hh%Mm%S')}.pt"
    fname = os.path.join(save_dir, name)

    states = {
        'state_dict_encoder': encoder.state_dict(),
        'state_dict_decoder': decoder.state_dict(),
        'state_dict_critic': critic.state_dict(),
        'en_de_optimizer': en_de_optimizer.state_dict(),
        'cr_optimizer': cr_optimizer.state_dict(),
        'metrics': metrics,
        'train_epoch': ep,
        'date': now.strftime("%Y-%m-%d_%H:%M:%S"),
    }
    torch.save(states, fname)
    logger.info(f"Saved checkpoint to {fname}")

    # Save latest model for easy loading
    latest_path = os.path.join(save_dir, "latest_model.pt")
    torch.save(states, latest_path)

def load_model(encoder, decoder, critic, en_de_optimizer, cr_optimizer, path, device):
    """Load saved model checkpoint for distributed training"""
    logger.info(f"Loading checkpoint from {path}")

    loc = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(path, map_location=loc)

    encoder.load_state_dict(checkpoint['state_dict_encoder'])
    decoder.load_state_dict(checkpoint['state_dict_decoder'])
    critic.load_state_dict(checkpoint['state_dict_critic'])

    # Only load optimizer states on rank 0
    if dist.get_rank() == 0:
        en_de_optimizer.load_state_dict(checkpoint['en_de_optimizer'])
        cr_optimizer.load_state_dict(checkpoint['cr_optimizer'])

    return checkpoint['metrics'], checkpoint['train_epoch'], checkpoint['date']

def train_distributed(local_rank, world_size, node_rank, nodes, master_addr, master_port,
                     batch_size=10, epochs=10, dataset_path='datasets/processed',
                     load_checkpoint=None, learning_rate=0.0002, data_depth=2,
                     hidden_size=96, weight_encoder_mse=100):
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
        load_checkpoint: Path to checkpoint file to resume training
        learning_rate: Learning rate for optimizers
        data_depth: Data depth for encoder/decoder
        hidden_size: Hidden size for models
        weight_encoder_mse: Weight for encoder MSE loss
    """
    setup(node_rank, world_size)

    # Calculate global rank
    global_rank = node_rank * torch.cuda.device_count() + local_rank

    print("cuda:", local_rank)

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
    encoder = DenseEncoder(data_depth, hidden_size).to(device)
    decoder = DenseDecoder(data_depth, hidden_size).to(device)
    critic = BasicCritic(hidden_size).to(device)

    # Wrap models with DDP
    encoder = DDP(encoder, device_ids=[local_rank])
    decoder = DDP(decoder, device_ids=[local_rank])
    critic = DDP(critic, device_ids=[local_rank])

    # Optimizers
    cr_optimizer = Adam(critic.parameters(), lr=learning_rate)
    en_de_optimizer = Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=learning_rate)

    # Initialize datasets and dataloaders
    train_loader = get_data_loader(train_dataset, batch_size, global_rank, world_size)
    test_loader = get_data_loader(test_dataset, batch_size, global_rank, world_size)

    # Tensorboard writer (only for rank 0)
    writer = None
    if global_rank == 0:
        writer = SummaryWriter(f"./logs/distributed")

    # Load checkpoint if specified
    if load_checkpoint and os.path.exists(load_checkpoint):
        metrics, start_epoch, date = load_model(
            encoder, decoder, critic,
            en_de_optimizer, cr_optimizer,
            load_checkpoint, local_rank
        )
        if global_rank == 0:
            logger.info(f"Resumed training from checkpoint saved at {date}")
    else:
        start_epoch = 0
        metrics = {field: [] for field in [
            'val.encoder_mse', 'val.decoder_loss', 'val.decoder_acc',
            'val.cover_score', 'val.generated_score', 'val.ssim',
            'val.psnr', 'val.bpp', 'train.encoder_mse',
            'train.decoder_loss', 'train.decoder_acc',
            'train.cover_score', 'train.generated_score',
        ]}

    # Training loop
    scaler = amp.GradScaler()
    iter_train_critic = 0
    iter_train_enc_dec = 0
    iter_valid = 0

    for ep in range(start_epoch, epochs):
        train_loader.sampler.set_epoch(ep)

        if global_rank == 0:
            logger.info(f"Epoch {ep+1}")

        # Train critic
        for cover, *rest in train_loader:
            iter_train_critic += 1

            with amp.autocast(device_type=device.type):
                cover = cover.to(device)
                N, _, H, W = cover.size()

                # Generate payload and encoded image
                payload = torch.zeros((N, data_depth, H, W), device=device).random_(0, 2)
                generated = encoder(cover, payload)

                # Get critic scores
                cover_score = torch.mean(critic(cover))
                generated_score = torch.mean(critic(generated))

                # Update critic
                cr_optimizer.zero_grad()
                scaler.scale(cover_score - generated_score).backward(retain_graph=False)
                scaler.step(cr_optimizer)
                scaler.update()

            # Clamp critic weights
            with torch.no_grad():
                for p in critic.parameters():
                    p.data.clamp_(-0.1, 0.1)

            if global_rank == 0 and writer is not None:
                writer.add_scalar('cover_score/train', cover_score.item(), iter_train_critic)
                writer.add_scalar('generated_score/train', generated_score.item(), iter_train_critic)
                metrics['train.cover_score'].append(cover_score.item())
                metrics['train.generated_score'].append(generated_score.item())

        # Train encoder-decoder
        for cover, *rest in train_loader:
            iter_train_enc_dec += 1

            with amp.autocast(device_type=device.type):
                cover = cover.to(device)
                N, _, H, W = cover.size()

                payload = torch.zeros((N, data_depth, H, W), device=device).random_(0, 2)
                generated = encoder(cover, payload)
                decoded = decoder(generated)

                # Calculate losses
                encoder_mse = nn.functional.mse_loss(generated, cover)
                decoder_ce = nn.functional.binary_cross_entropy_with_logits(decoded, payload)
                decoder_mse = nn.functional.mse_loss(decoded, payload)
                decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
                generated_score = torch.mean(critic(generated))

                decoder_loss = decoder_ce + 0.5 * decoder_mse - 0.1 * generated_score
                loss = weight_encoder_mse * encoder_mse + decoder_loss

                # Update encoder-decoder
                en_de_optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(en_de_optimizer)
                scaler.update()

            if global_rank == 0 and writer is not None:
                writer.add_scalar('encoder_mse/train', encoder_mse.item(), iter_train_enc_dec)
                writer.add_scalar('decoder_loss/train', decoder_loss.item(), iter_train_enc_dec)
                writer.add_scalar('decoder_acc/train', decoder_acc.item(), iter_train_enc_dec)
                metrics['train.encoder_mse'].append(encoder_mse.item())
                metrics['train.decoder_loss'].append(decoder_loss.item())
                metrics['train.decoder_acc'].append(decoder_acc.item())

        # Validation (only on rank 0)
        if global_rank == 0:
            for cover, *rest in test_loader:
                iter_valid += 1

                with torch.no_grad(), amp.autocast(device_type=device.type):
                    cover = cover.to(device)
                    N, _, H, W = cover.size()

                    payload = torch.zeros((N, data_depth, H, W), device=device).random_(0, 2)
                    generated = encoder(cover, payload)
                    decoded = decoder(generated)

                    # Calculate metrics
                    encoder_mse = nn.functional.mse_loss(generated, cover)
                    decoder_loss = nn.functional.binary_cross_entropy_with_logits(decoded, payload)
                    decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
                    generated_score = torch.mean(critic(generated))
                    cover_score = torch.mean(critic(cover))

                    ssim_val = ssim(cover, generated)
                    psnr = 10 * torch.log10(4 / encoder_mse)
                    bpp = data_depth * (2 * decoder_acc.item() - 1)

                    if writer is not None:
                        # Log validation metrics
                        writer.add_scalar('encoder_mse/test', encoder_mse.item(), iter_valid)
                        writer.add_scalar('decoder_loss/test', decoder_loss.item(), iter_valid)
                        writer.add_scalar('decoder_acc/test', decoder_acc.item(), iter_valid)
                        writer.add_scalar('cover_score/test', cover_score.item(), iter_valid)
                        writer.add_scalar('generated_score/test', generated_score.item(), iter_valid)
                        writer.add_scalar('ssim/test', ssim_val.item(), iter_valid)
                        writer.add_scalar('psnr/test', psnr.item(), iter_valid)
                        writer.add_scalar('bpp/test', bpp, iter_valid)

                        metrics['val.encoder_mse'].append(encoder_mse.item())
                        metrics['val.decoder_loss'].append(decoder_loss.item())
                        metrics['val.decoder_acc'].append(decoder_acc.item())
                        metrics['val.cover_score'].append(cover_score.item())
                        metrics['val.generated_score'].append(generated_score.item())
                        metrics['val.ssim'].append(ssim_val.item())
                        metrics['val.psnr'].append(psnr.item())
                        metrics['val.bpp'].append(bpp)

            # Save checkpoint on rank 0
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
    parser.add_argument('--load-checkpoint', type=str, default=None,
                       help='Path to checkpoint file to resume training')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout in seconds for node connections')
    parser.add_argument('--retry-interval', type=int, default=5,
                       help='Seconds to wait between connection retries')
    parser.add_argument('--learning-rate', type=float, default=0.0002,
                       help='Learning rate for optimizers')
    parser.add_argument('--data-depth', type=int, default=2,
                       help='Data depth for encoder/decoder')
    parser.add_argument('--hidden-size', type=int, default=96,
                       help='Hidden size for models')
    parser.add_argument('--weight-encoder-mse', type=float, default=100.0,
                       help='Weight for encoder MSE loss')
    args = parser.parse_args()

    # Calculate world size (total processes = nodes * gpus per node)
    n_gpus_per_node = torch.cuda.device_count()
    world_size = args.nodes * n_gpus_per_node

    print(f"World size: {world_size}")
    print(f"Number of GPUs per node: {n_gpus_per_node}")
    print(f"Master address: {args.master_addr}")
    print(f"Master port: {args.master_port}")

    # Spawn processes on current node
    mp.spawn(
        train_distributed,
        args=(world_size, args.node_rank, args.nodes,
              args.master_addr, args.master_port,
              args.batch_size, args.epochs, args.dataset_path,
              args.load_checkpoint, args.learning_rate,
              args.data_depth, args.hidden_size,
              args.weight_encoder_mse),
        nprocs=n_gpus_per_node,
        join=True
    )
