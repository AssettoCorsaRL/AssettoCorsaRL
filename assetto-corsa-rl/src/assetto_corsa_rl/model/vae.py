# inspired by https://github.com/timoklein/car_racer

import torch
from torch import nn
import torch.nn.functional as F
import lightning as pl
from typing import Tuple, Optional
import lpips
import sys
import os


class ConvBlock(nn.Module):
    """Convolutional building block: Conv2d -> BatchNorm2d -> LeakyReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        padding: int = 1,
        slope: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DeConvBlock(nn.Module):
    """Deconvolutional building block: ConvTranspose2d -> BatchNorm2d -> LeakyReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        padding: int = 1,
        slope: float = 0.2,
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.deconv(x)))


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm for stable training."""

    def __init__(
        self,
        channels: int,
        groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.dropout(x)
        x = self.conv2(x)
        return x + residual


class SelfAttention(nn.Module):
    """Multi-head self-attention for capturing long-range dependencies."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # attention: (B, heads, head_dim, HW) @ (B, heads, HW, head_dim) -> (B, heads, HW, HW)
        q = q.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)
        k = k.permute(0, 1, 2, 3)  # (B, heads, head_dim, HW)
        v = v.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)

        attn = torch.matmul(q, k) * self.scale  # (B, heads, HW, HW)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, heads, HW, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)

        return out + residual


class DownBlock(nn.Module):
    """Encoder block: downsample + residual blocks + optional attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        for _ in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(out_channels, dropout=dropout))
            if use_attention:
                self.attn_blocks.append(SelfAttention(out_channels))
            else:
                self.attn_blocks.append(nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        for res, attn in zip(self.res_blocks, self.attn_blocks):
            x = res(x)
            x = attn(x)
        return x


class UpBlock(nn.Module):
    """Decoder block: upsample + residual blocks + optional attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        for _ in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(out_channels, dropout=dropout))
            if use_attention:
                self.attn_blocks.append(SelfAttention(out_channels))
            else:
                self.attn_blocks.append(nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        for res, attn in zip(self.res_blocks, self.attn_blocks):
            x = res(x)
            x = attn(x)
        return x


class MiddleBlock(nn.Module):
    """Middle block with attention for bottleneck processing."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.res1 = ResidualBlock(channels, dropout=dropout)
        self.attn = SelfAttention(channels)
        self.res2 = ResidualBlock(channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class ConvVAE(pl.LightningModule):
    """Convolutional VAE implemented as a PyTorch LightningModule.

    - Inputs: RGB images of shape (B, C, H, W) where H, W are specified by im_shape
    - Latent: `z_dim`-dimensional Gaussian
    - Reconstruction: output sigmoid-ed RGB image in same shape as input

    Main hyperparameters: `z_dim`, `lr`, `beta` (KL multiplier), `im_shape`
    """

    def __init__(
        self,
        z_dim: int = 32,
        lr: float = 1e-3,
        beta: float = 1.0,
        in_channels: int = 3,
        warmup_steps: int = 500,
        im_shape: Tuple[int, int] = (64, 64),
        # New architecture parameters
        base_channels: int = 32,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (10,),
        dropout: float = 0.1,
        mse_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.z_dim = z_dim
        self.lr = lr
        self.beta = beta
        self.in_channels = in_channels
        self.warmup_steps = warmup_steps
        self.im_shape = im_shape  # (H, W)
        self.base_channels = base_channels
        self.mse_weight = mse_weight

        self.lpips = lpips.LPIPS(net="vgg")
        # freeze LPIPS network
        for param in self.lpips.parameters():
            param.requires_grad = False

        channels = [base_channels * m for m in channel_multipliers]

        self.encoder_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        current_res = im_shape[0]
        in_ch = base_channels

        for i, out_ch in enumerate(channels):
            use_attn = current_res in attention_resolutions
            self.encoder_blocks.append(
                DownBlock(
                    in_ch,
                    out_ch,
                    num_res_blocks=num_res_blocks,
                    use_attention=use_attn,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
            current_res //= 2

        self.middle = MiddleBlock(channels[-1], dropout=dropout)

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, im_shape[0], im_shape[1])
            dummy_output = self._encode_features(dummy_input)
            self.encoder_out_shape = dummy_output.shape[1:]  # (C, H, W) after encoder
            C_enc, H_enc, W_enc = self.encoder_out_shape

        latent_channels = z_dim
        self.to_latent = nn.Conv2d(C_enc, latent_channels, kernel_size=1)
        self.from_latent = nn.Conv2d(latent_channels, C_enc, kernel_size=1)

        self.decoder_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))

        for i in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i + 1]
            current_res = im_shape[0] // (2 ** (len(channels) - i - 1))
            use_attn = current_res in attention_resolutions
            self.decoder_blocks.append(
                UpBlock(
                    in_ch,
                    out_ch,
                    num_res_blocks=num_res_blocks,
                    use_attention=use_attn,
                    dropout=dropout,
                )
            )

        self.decoder_out = nn.Sequential(
            UpBlock(
                reversed_channels[-1],
                base_channels,
                num_res_blocks=num_res_blocks,
                use_attention=False,
                dropout=dropout,
            ),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        self.final_resize = nn.Upsample(size=im_shape, mode="bilinear", align_corners=False)

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features through encoder without computing latent params."""
        x = self.encoder_in(x)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.middle(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        h = self._encode_features(x)
        z = self.to_latent(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)

        for block in self.decoder_blocks:
            h = block(h)

        out = self.decoder_out(h)

        out = self.final_resize(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reconstruction."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def _loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # scale from [0, 1] to [-1, 1]
        recon_scaled = recon * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0
        lpips_loss = self.lpips(recon_scaled, target_scaled).mean()

        # Add MSE loss if weight > 0
        if self.mse_weight > 0:
            mse_loss = F.mse_loss(recon, target, reduction="mean")
            loss = lpips_loss + self.mse_weight * mse_loss
        else:
            loss = lpips_loss

        return loss

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        recon = self.forward(x)
        target = x[:, -3:, :, :] if x.dim() == 4 and x.size(1) != recon.size(1) else x

        loss = self._loss(recon, target)

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/lr",
            self.optimizers().param_groups[0]["lr"],
            prog_bar=True,
            on_step=True,
        )

        if self.mse_weight > 0:
            recon_scaled = recon * 2.0 - 1.0
            target_scaled = target * 2.0 - 1.0
            lpips_loss = self.lpips(recon_scaled, target_scaled).mean()
            mse_loss = F.mse_loss(recon, target, reduction="mean")
            self.log("train/lpips", lpips_loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/mse", mse_loss, on_step=True, on_epoch=True, sync_dist=True)

        # Log images every 500 steps
        if self.global_step > 0 and self.global_step % 500 == 0:
            self.log_images(target, recon)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        recon = self.forward(x)
        target = x[:, -3:, :, :] if x.dim() == 4 and x.size(1) != recon.size(1) else x
        loss = self._loss(recon, target)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)

        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @torch.no_grad()
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """Sample from the prior and decode to image space."""
        _, H_enc, W_enc = self.encoder_out_shape
        z = torch.randn(num_samples, self.z_dim, H_enc, W_enc, device=self.device)
        return self.decode(z)

    def log_images(self, x: torch.Tensor, recon: torch.Tensor):
        """Log comparison images to WandB."""
        if not hasattr(self.logger, "experiment"):
            return

        try:
            import wandb
            import torchvision

            imgs = torch.cat([x[:4], recon[:4]], dim=0)
            grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True)

            self.logger.experiment.log(
                {
                    "recon/compare": wandb.Image(grid),
                },
                step=self.global_step,
            )
        except Exception as e:
            print(f"Failed to log images: {e}")


class VAEEncoder(nn.Module):
    """Wrapper to use VAE encoder components as a feature extractor."""

    def __init__(self, vae):
        super().__init__()
        self.encoder_in = vae.encoder_in
        self.encoder_blocks = vae.encoder_blocks
        self.middle = vae.middle

    def forward(self, x):
        x = self.encoder_in(x)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.middle(x)
        return x


def load_vae_encoder(
    checkpoint_path: str,
    device,
    in_channels: int = 4,
    trainable: bool = True,
    verbose: bool = True,
) -> Tuple[nn.Module, int]:
    """Load a VAE encoder from checkpoint with automatic parameter detection.

    Args:
        checkpoint_path: Path to VAE checkpoint file
        device: Device to load the encoder on
        in_channels: Number of input channels (e.g., 4 for stacked frames)
        trainable: Whether to make encoder parameters trainable
        verbose: Whether to print loading information

    Returns:
        Tuple of (encoder_module, output_size) where encoder_module handles
        channel adaptation if needed and output_size is the flattened feature size
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        hparams = checkpoint.get("hyper_parameters", {})
        state_dict = checkpoint.get("state_dict", checkpoint)

        if not hparams:
            if "to_latent.weight" in state_dict:
                z_dim = state_dict["to_latent.weight"].shape[0]
            else:
                z_dim = 1024

            if "encoder_in.weight" in state_dict:
                vae_in_channels = state_dict["encoder_in.weight"].shape[1]
            else:
                vae_in_channels = 3
        else:
            z_dim = hparams.get("z_dim", 1024)
            vae_in_channels = hparams.get("in_channels", 3)
    else:
        z_dim = 1024
        vae_in_channels = 3
        state_dict = checkpoint

    old_stdout = sys.stdout
    if not verbose:
        sys.stdout = open(os.devnull, "w")

    try:
        vae = ConvVAE(z_dim=z_dim, in_channels=vae_in_channels)
    finally:
        if not verbose:
            sys.stdout.close()
            sys.stdout = old_stdout

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        vae.load_state_dict(checkpoint["state_dict"])
    else:
        vae.load_state_dict(checkpoint)

    vae = vae.to(device)

    vae_encoder = VAEEncoder(vae)
    for param in vae_encoder.parameters():
        param.requires_grad = trainable

    # handle channel mismatch with adapter if needed
    needs_adapter = vae_in_channels != in_channels
    if needs_adapter and verbose:
        print(
            f"VAE: {vae_in_channels}ch→adapter→{in_channels}ch, z_dim={z_dim}, {'trainable' if trainable else 'frozen'}"
        )
        encoder_module = nn.Sequential(
            nn.Conv2d(
                in_channels,
                vae_in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                device=device,
            ),
            vae_encoder,
            nn.Flatten(start_dim=1),
        )
    else:
        if verbose:
            print(
                f"VAE: z_dim={z_dim}, in_channels={vae_in_channels}, {'trainable' if trainable else 'frozen'}"
            )
        encoder_module = nn.Sequential(
            vae_encoder,
            nn.Flatten(start_dim=1),
        )

    output_size = 256 * 4 * 4

    return encoder_module, output_size


def main() -> None:
    from torchinfo import summary

    vae = ConvVAE(z_dim=128, im_shape=(84, 84))

    summary(
        vae,
        input_size=(1, 3, 84, 84),
        col_names=["input_size", "output_size", "num_params"],
    )


if __name__ == "__main__":
    main()
