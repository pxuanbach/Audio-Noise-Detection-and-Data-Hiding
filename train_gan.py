import logging
import numpy as np
import torch
import torch.nn as nn
import torch.amp as amp
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import os
import gc
from torchvision import transforms
# from datasets.audio_dataset import AudioDataset
from datasets import AudioToImageFolder
from encoder import DenseEncoder, ImprovedEncoder
from decoder import DenseDecoder, ImprovedDecoder
from critic import BasicCritic, ImprovedCritic
from utils import ssim
from utils.audio_to_stft import audio_to_stft

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def safe_add_histogram(writer: SummaryWriter, tag, value, step):
    """Safely add histogram to tensorboard, handling empty tensors and None gradients"""
    try:
        if value is not None and value.nelement() > 0:
            writer.add_histogram(tag, value.data.cpu().numpy(), step)
    except Exception as e:
        logger.warning(f"Failed to add histogram for {tag}: {str(e)}")

def safe_add_image(writer, tag, image_tensor, step):
    """Safely add image to tensorboard, handling batched tensors"""
    try:
        # Take first image if batched
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]

        # Ensure tensor is in right format
        if image_tensor.shape[0] == 1:  # If single channel
            image_tensor = image_tensor.repeat(3, 1, 1)  # Convert to 3 channels

        writer.add_image(tag, image_tensor, step, dataformats='CHW')
    except Exception as e:
        logger.warning(f"Failed to add image for {tag}: {str(e)}")

def save_model(encoder, decoder, critic, en_de_optimizer, cr_optimizer, metrics, ep, save_dir='models'):
    """Save model checkpoints"""
    now = datetime.datetime.now()
    cover_score = metrics['val.decoder_acc'][-1]
    name = f"{encoder.__class__.__name__}_{decoder.__class__.__name__}_{cover_score:.3f}_{now.strftime('%Y-%m-%d_%Hh%Mm%S')}.dat"
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, name)

    states = {
        'state_dict_critic': critic.state_dict(),
        'state_dict_encoder': encoder.state_dict(),
        'state_dict_decoder': decoder.state_dict(),
        'en_de_optimizer': en_de_optimizer.state_dict(),
        'cr_optimizer': cr_optimizer.state_dict(),
        'metrics': metrics,
        'train_epoch': ep,
        'date': now.strftime("%Y-%m-%d_%H:%M:%S"),
    }
    torch.save(states, fname)

def load_model(encoder, decoder, critic, en_de_optimizer, cr_optimizer, path):
    """Load saved model checkpoint"""
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

    critic.load_state_dict(checkpoint['state_dict_critic'])
    encoder.load_state_dict(checkpoint['state_dict_encoder'])
    decoder.load_state_dict(checkpoint['state_dict_decoder'])
    en_de_optimizer.load_state_dict(checkpoint['en_de_optimizer'])
    cr_optimizer.load_state_dict(checkpoint['cr_optimizer'])

    return checkpoint['metrics'], checkpoint['train_epoch'], checkpoint['date']


def fit_gan(encoder, decoder, critic, en_de_optimizer, cr_optimizer, metrics, train_loader, valid_loader,
           epochs=32, device: torch.device = torch.device('cpu'), load_model_path=None, writer=None, save_dir='models',):
    """Train the GAN model

    Args:
        encoder: The encoder model
        decoder: The decoder model
        critic: The critic model
        en_de_optimizer: Optimizer for encoder and decoder
        cr_optimizer: Optimizer for critic
        metrics: Dictionary to store training and validation metrics
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        epochs: Number of epochs to train
        device: Device to use for training ('cuda' or 'cpu')
        load_model_path: Path to load a pretrained model
        writer: Tensorboard SummaryWriter
        use_gradient_checkpointing: If True, enables gradient checkpointing to reduce memory usage
                                  at the cost of increased computation time
    """
    if writer is None:
        writer = SummaryWriter("./logs")
        logger.info(f'Tensorboard logs stored in: {writer.get_logdir()}')

    iter_train_critic = 0
    iter_train_enc_dec = 0
    iter_valid = 0

    # Load pretrained model if specified
    if load_model_path:
        metrics, start_epoch, date = load_model(encoder, decoder, critic, en_de_optimizer, cr_optimizer, load_model_path)
        logger.info(f"Loaded model from {date}")
    else:
        start_epoch = 0

    for ep in range(start_epoch, epochs):
        logger.info(f"Epoch {ep+1}")

        # Train critic
        for cover, *rest in tqdm(train_loader, desc='Training critic'):
            iter_train_critic += 1
            gc.collect()

            cover = cover.to(device)
            N, _, H, W = cover.size()

            # Generate payload and encoded image
            payload = torch.zeros((N, encoder.data_depth, H, W), device=device).random_(0, 2)
            generated = encoder.forward(cover, payload)

            # Get critic scores
            cover_score = torch.mean(critic.forward(cover))
            generated_score = torch.mean(critic.forward(generated))

            # Update critic
            cr_optimizer.zero_grad()
            (cover_score - generated_score).backward(retain_graph=False)
            cr_optimizer.step()

            for p in critic.parameters():
                p.data.clamp_(-0.1, 0.1)

            # Log metrics
            writer.add_scalar('cover_score/train', cover_score.item(), iter_train_critic)
            writer.add_scalar('generated_score/train', generated_score.item(), iter_train_critic)
            metrics['train.cover_score'].append(cover_score.item())
            metrics['train.generated_score'].append(generated_score.item())

            # Log critic gradients
            for tag, value in critic.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram(
                    'critic/' + tag,
                    value.data.cpu().numpy(),
                    iter_train_critic
                )
                writer.add_histogram(
                    'critic/' + tag + '/grad',
                    value.grad.data.cpu().numpy(),
                    iter_train_critic
                )
                # safe_add_histogram(writer, 'critic/'+tag, value, iter_train_critic)
                # if value.grad is not None:
                #     safe_add_histogram(writer, 'critic/'+tag+'/grad', value.grad, iter_train_critic)

        # Train encoder-decoder
        for cover, *rest in tqdm(train_loader, desc='Training encoder-decoder'):
            iter_train_enc_dec += 1
            gc.collect()

            cover = cover.to(device)
            N, _, H, W = cover.size()

            # Generate payload and encoded image
            payload = torch.zeros((N, encoder.data_depth, H, W), device=device).random_(0, 2)
            generated = encoder.forward(cover, payload)
            decoded = decoder.forward(generated)

            # Calculate losses
            encoder_mse = nn.functional.mse_loss(generated, cover)
            decoder_loss = nn.functional.binary_cross_entropy_with_logits(decoded, payload)
            decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
            generated_score = torch.mean(critic.forward(generated))


            # Update encoder-decoder
            en_de_optimizer.zero_grad()
            (100 * encoder_mse + decoder_loss + generated_score).backward()
            en_de_optimizer.step()

            # Log metrics
            writer.add_scalar('encoder_mse/train', encoder_mse.item(), iter_train_enc_dec)
            writer.add_scalar('decoder_loss/train', decoder_loss.item(), iter_train_enc_dec)
            writer.add_scalar('decoder_acc/train', decoder_acc.item(), iter_train_enc_dec)

            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            metrics['train.decoder_acc'].append(decoder_acc.item())

            for tag, value in encoder.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram(
                    'encoder/'+ tag,
                    value.data.cpu().numpy(),
                    iter_train_enc_dec
                )
                writer.add_histogram(
                    'encoder/'+ tag + '/grad',
                    value.grad.data.cpu().numpy(),
                    iter_train_enc_dec
                )

            for tag, value in decoder.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram(
                    'decoder/' + tag,
                    value.data.cpu().numpy(),
                    iter_train_enc_dec
                )
                writer.add_histogram(
                    'decoder/' + tag + '/grad',
                    value.grad.data.cpu().numpy(),
                    iter_train_enc_dec
                )

        # Validation
        for cover, *rest in tqdm(valid_loader, desc='Validation'):
            iter_valid += 1
            gc.collect()

            cover = cover.to(device)
            N, _, H, W = cover.size()

            payload = torch.zeros((N, encoder.data_depth, H, W), device=device).random_(0, 2)
            generated = encoder.forward(cover, payload)
            decoded = decoder.forward(generated)

            # Calculate metrics
            encoder_mse = nn.functional.mse_loss(generated, cover)
            decoder_loss = nn.functional.binary_cross_entropy_with_logits(decoded, payload)
            decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
            generated_score = torch.mean(critic.forward(generated))
            cover_score = torch.mean(critic.forward(cover))

            ssim_val = ssim(cover, generated)
            psnr = 10 * torch.log10(4 / encoder_mse)
            bpp = encoder.data_depth * (2 * decoder_acc.item() - 1)

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

            # Log sample images
            # if iter_valid % 100 == 0:
            #     safe_add_image(writer, 'cover/test', cover, ep)
            #     safe_add_image(writer, 'generated/test', generated, ep)
            #     safe_add_image(writer, 'payload/test', cover-generated, ep)
            writer.add_image('cover/test', torch.tensor(np.abs(cover[:,0,:,:].cpu().detach().numpy()+1j*cover[:,1,:,:].cpu().detach().numpy())).unsqueeze(0), ep, dataformats='CNHW')
            writer.add_image('generated/test', torch.tensor(np.abs(generated[:,0,:,:].cpu().detach().numpy()+1j*generated[:,1,:,:].cpu().detach().numpy())).unsqueeze(0), ep, dataformats='CNHW')
            writer.add_image('payload/test', torch.tensor(np.abs((cover[:,0,:,:].cpu().detach().numpy()+1j*cover[:,1,:,:].cpu().detach().numpy())-(generated[:,0,:,:].cpu().detach().numpy()+1j*generated[:,1,:,:].cpu().detach().numpy()))).unsqueeze(0), ep, dataformats='CNHW')

        logger.info(f'encoder_mse: {encoder_mse:.3f} - decoder_loss: {decoder_loss:.3f} - decoder_acc: {decoder_acc:.3f} - '
                    f'cover_score: {cover_score:.3f} - generated_score: {generated_score:.3f} - '
                    f'ssim: {ssim_val:.3f} - psnr: {psnr:.3f} - bpp: {bpp:.3f}')

        # Save checkpoint
        save_model(encoder, decoder, critic, en_de_optimizer, cr_optimizer, metrics, ep, save_dir)

    # Add model graphs to tensorboard
    writer.add_graph(encoder, (cover, payload))
    writer.add_graph(decoder, generated)
    writer.add_graph(critic, generated)
    writer.close()

if __name__ == '__main__':
    config = {
        'batch_size': 4,
        'epochs': 32,
        'learning_rate': 0.0005,
        'channels_size': 2,
        'data_depth': 2,
        'hidden_size': 32,
    }

    data_depth = config['data_depth']
    hidden_size = config['hidden_size']
    channels_size = config['channels_size']
    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = config['learning_rate']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # encoder = DenseEncoder(data_depth, hidden_size, channels_size).to(device)
    encoder = ImprovedEncoder(data_depth, hidden_size, channels_size).to(device)
    # decoder = DenseDecoder(data_depth, hidden_size, channels_size).to(device)
    decoder = ImprovedDecoder(data_depth, hidden_size, channels_size).to(device)
    # critic = BasicCritic(hidden_size, channels_size).to(device)
    critic = ImprovedCritic(hidden_size, channels_size).to(device)

    # Optimizers
    cr_optimizer = Adam(critic.parameters(), lr=learning_rate)
    en_de_optimizer = Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=learning_rate)

    # Metrics tracking
    METRIC_FIELDS = [
        'val.encoder_mse', 'val.decoder_loss', 'val.decoder_acc',
        'val.cover_score', 'val.generated_score', 'val.ssim',
        'val.psnr', 'val.bpp', 'train.encoder_mse',
        'train.decoder_loss', 'train.decoder_acc',
        'train.cover_score', 'train.generated_score',
    ]
    metrics = {field: [] for field in METRIC_FIELDS}

    # Training settings
    load_model_path = None # 'results/model/previous_model.dat' to resume training
    writer = SummaryWriter("./logs")

    #region new dataset machanism
    data_dir="D:/Backup/FSDKaggle2018"
    transform = transforms.Compose([transforms.Lambda(lambda wav: audio_to_stft(wav, 320))])
    train_set = AudioToImageFolder(data_dir, transform=transform)
    part_train_set = torch.utils.data.random_split(train_set, [1000, len(train_set)-1000])[0]
    train_loader = torch.utils.data.DataLoader(part_train_set, batch_size=batch_size, shuffle=True,)

    test_set = AudioToImageFolder(data_dir, transform=transform)
    part_test_set = torch.utils.data.random_split(test_set, [150, len(test_set)-150])[0]
    test_loader = torch.utils.data.DataLoader(part_test_set, batch_size=4, shuffle=True)
    #endregion


    # Start training
    fit_gan(encoder, decoder, critic, en_de_optimizer, cr_optimizer, metrics,
            train_loader, test_loader, epochs=epochs, device=device,
            load_model_path=load_model_path, writer=writer,
            save_dir=f"models/gan_{hidden_size}_{data_depth}_{channels_size}_epochs_{epochs}",)
