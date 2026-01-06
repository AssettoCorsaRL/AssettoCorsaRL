# inspired by https://github.com/timoklein/car_racer

import torch
from torch import nn
import torch.nn.functional as F
import lightning as pl
from typing import Tuple
import lpips


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


class ConvVAE(pl.LightningModule):
    """Convolutional VAE implemented as a PyTorch LightningModule.

    - Inputs: RGB images of shape (B, 3, 64, 64)
    - Latent: `z_dim`-dimensional Gaussian
    - Reconstruction: output sigmoid-ed RGB image in same shape as input

    Main hyperparameters: `z_dim`, `lr`, `beta` (KL multiplier)
    """

    def __init__(
        self,
        z_dim: int = 32,
        lr: float = 1e-3,
        beta: float = 1.0,
        in_channels: int = 3,
        warmup_steps: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.z_dim = z_dim
        self.lr = lr
        self.beta = beta
        self.in_channels = in_channels
        self.warmup_steps = warmup_steps

        # LPIPS perceptual loss (uses VGG by default)
        self.lpips = lpips.LPIPS(net="vgg")
        # Freeze LPIPS network (we don't train it)
        for param in self.lpips.parameters():
            param.requires_grad = False

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                self.in_channels, 32, kernel_size=4, stride=2, padding=1
            ),  # 32x32x32
            nn.LeakyReLU(0.2),
            ConvBlock(32, 64, 4, stride=2, padding=1),  # 64x16x16
            ConvBlock(64, 128, 4, stride=2, padding=1),  # 128x8x8
            ConvBlock(128, 256, 4, stride=2, padding=1),  # 256x4x4
        )

        # latent heads
        self.flat_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flat_dim, z_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, z_dim)
        self.fc_dec = nn.Linear(z_dim, self.flat_dim)

        # decoder
        # We start from (B, 256, 4, 4)
        self.decoder = nn.Sequential(
            DeConvBlock(256, 128, 4, stride=2, padding=1),  # 128x8x8
            DeConvBlock(128, 64, 4, stride=2, padding=1),  # 64x16x16
            DeConvBlock(64, 32, 4, stride=2, padding=1),  # 32x32x32
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 3x64x64
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mu and logvar for input batch.

        Accepts either (B, C, H, W) or single-sample (C, H, W); single samples will
        be expanded to a batch of size 1.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, 256, 4, 4)
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def _loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = target.size(0)

        # Scale from [0, 1] to [-1, 1]
        recon_scaled = recon * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0
        lpips_loss = self.lpips(recon_scaled, target_scaled).mean()

        # Combined reconstruction loss: MSE + perceptual
        recon_loss = lpips_loss

        # KL divergence between posterior and standard normal (per-sample average)
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        loss = recon_loss + self.beta * kl
        return loss, recon_loss, kl

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        recon, mu, logvar = self.forward(x)
        target = x[:, -3:, :, :] if x.dim() == 4 and x.size(1) != recon.size(1) else x

        # Only use LPIPS loss for reconstruction
        recon_scaled = recon * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0
        recon_loss = self.lpips(recon_scaled, target_scaled).mean()

        # KL divergence
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        loss = recon_loss + self.beta * kl

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/recon", recon_loss, on_step=False, on_epoch=True)
        self.log("train/kl", kl, on_step=False, on_epoch=True)
        self.log(
            "train/lr",
            self.optimizers().param_groups[0]["lr"],
            prog_bar=True,
            on_step=True,
        )

        if self.global_step > 0 and self.global_step % 500 == 0:
            self.log_images(target, recon)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        recon, mu, logvar = self.forward(x)
        target = x[:, -3:, :, :] if x.dim() == 4 and x.size(1) != recon.size(1) else x
        loss, recon_loss, kl = self._loss(recon, target, mu, logvar)

        # Log metrics
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/recon", recon_loss, sync_dist=True)
        self.log("val/kl", kl, sync_dist=True)
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
        z = torch.randn(num_samples, self.z_dim, device=self.device)
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
