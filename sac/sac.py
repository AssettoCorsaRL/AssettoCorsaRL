import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from sac.utils import soft_update, hard_update
from sac.model import GaussianPolicy, QNetwork, DeterministicPolicy
import logging
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Set the device globally if a GPU is available."""


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
        )
        # For CarRacing 96x96: conv output is 64x8x8 -> 4096
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),  # Stabilize encoder output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W); normalize if coming in 0-255
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        z = self.conv(x)
        z = self.head(z)
        return z


class SAC(object):
    """
    SAC (Soft actor critic) algorithm.
    Objects contain the Q networks and optimizers, as well as functions to
    safe and load models, selecting actions and perform updates for the objective functions.


    ## Parameters:

    - **num_inputs** *(int)*: dimension of input (In this case number of variables of the latent representation)
    - **action_space**: action space of environment (E.g. for car racer: Box(3,) which means that the action space has 3 actions that are continuous.)
    - **args**: namespace with needed arguments (such as discount factor, used policy and temperature parameter)

    """

    def __init__(
        self,
        action_space,
        policy: str = "Gaussian",
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 0.0001,
        alpha: float = 0.2,
        automatic_temperature_tuning: bool = False,
        batch_size: int = 256,
        hidden_size: int = 256,
        target_update_interval: int = 1,
        input_dim: int = 32,
        device=None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr

        self.policy_type = policy
        self.target_update_interval = target_update_interval
        self.automatic_temperature_tuning = automatic_temperature_tuning

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.bs = batch_size

        # Shared conv encoder
        self.encoder = ConvEncoder(in_channels=3, latent_dim=input_dim).to(self.device)
        # Target encoder for stable value estimation (soft-updated like critic_target)
        self.encoder_target = ConvEncoder(in_channels=3, latent_dim=input_dim).to(
            self.device
        )
        hard_update(self.encoder_target, self.encoder)
        # Freeze target encoder (only updated via soft_update)
        for param in self.encoder_target.parameters():
            param.requires_grad = False

        self.critic = QNetwork(input_dim, action_space.shape[0], hidden_size).to(
            self.device
        )
        self.critic_optim = Adam(
            list(self.critic.parameters()) + list(self.encoder.parameters()), lr=self.lr
        )

        self.critic_target = QNetwork(input_dim, action_space.shape[0], hidden_size).to(
            self.device
        )
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            if self.automatic_temperature_tuning:
                # Use action dimension (not product) for a milder entropy target
                self.target_entropy = -float(action_space.shape[0])
                # Initialize log_alpha so initial alpha == provided alpha
                self.log_alpha = torch.log(
                    torch.tensor(self.alpha, device=self.device)
                ).detach()
                self.log_alpha.requires_grad_(True)
                self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
                # sync self.alpha to the param value
                self.alpha = self.log_alpha.exp().item()

            self.policy = GaussianPolicy(
                input_dim, action_space.shape[0], hidden_size
            ).to(self.device)
            # Include encoder parameters in policy optimizer so policy gradients update encoder
            self.policy_optim = Adam(
                list(self.policy.parameters()) + list(self.encoder.parameters()),
                lr=self.lr,
            )

        else:
            self.alpha = 0
            self.automatic_temperature_tuning = False
            self.policy = DeterministicPolicy(
                input_dim, action_space.shape[0], hidden_size
            ).to(self.device)
            self.policy_optim = Adam(
                list(self.policy.parameters()) + list(self.encoder.parameters()),
                lr=self.lr,
            )

        # Print network architectures on init
        print("Encoder:\n", self.encoder)
        print("Encoder target:\n", self.encoder_target)
        print("Critic network:\n", self.critic)
        print("Critic target network:\n", self.critic_target)
        print("Policy network:\n", self.policy)

        total_params = 0
        for module in (self.encoder, self.policy, self.critic):
            for p in module.parameters():
                if p.requires_grad:
                    total_params += p.numel()
        if (
            hasattr(self, "log_alpha")
            and isinstance(self.log_alpha, torch.Tensor)
            and self.log_alpha.requires_grad
        ):
            total_params += self.log_alpha.numel()
        print(f"Total trainable parameters: {total_params:,}")

        settings = (
            f"INITIALIZING SAC ALGORITHM WITH {self.policy_type} POLICY"
            f"\nRunning on: {self.device}"
            f"\nSettings: Automatic Temperature tuning = {self.automatic_temperature_tuning}, Update Interval = {self.target_update_interval}"
            f"\nParameters: Learning rate = {self.lr}, Batch Size = {self.bs} Gamma = {self.gamma}, Tau = {self.tau}, Alpha = {self.alpha}"
            f"\nArchitecture: Input dimension = {self.input_dim}, Hidden layer dimension = {self.hidden_size}"
            "\n--------------------------"
        )

        print(settings)

    def _encode(
        self, obs_batch: torch.Tensor, use_target: bool = False
    ) -> torch.Tensor:
        # obs_batch expected (B, H, W, C) or (B, C, H, W)
        if obs_batch.dim() == 4 and obs_batch.shape[1] != 3:
            obs_batch = obs_batch.permute(0, 3, 1, 2)
        encoder = self.encoder_target if use_target else self.encoder
        return encoder(obs_batch)

    def select_action(self, state, eval=False):
        # state: np.array (H, W, C) or (C, H, W)
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        if state.dim() == 3:
            state = state.unsqueeze(0)
        z = self._encode(state.to(self.device))
        if not eval:
            action, _, _ = self.policy.sample(z)
        else:
            _, _, action = self.policy.sample(z)
        action = action.detach().cpu().numpy()
        return action[0]

    def update_parameters(
        self,
        memory,
        batch_size,
        updates,
        batch=None,
        weights=None,
        idxs=None,
    ):
        # if a batch was pre-sampled (PER), use it; otherwise sample here
        if batch is None:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = (
                memory.sample(batch_size)
            )
            weights = None
            idxs = None
        else:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = (
                batch
            )

        state_batch_t = torch.as_tensor(
            state_batch, dtype=torch.float32, device=self.device
        )
        next_state_batch_t = torch.as_tensor(
            next_state_batch, dtype=torch.float32, device=self.device
        )
        # Encode states - encoder trained through critic
        state_latent = self._encode(state_batch_t)
        # Use target encoder for next states (more stable value estimates)
        with torch.no_grad():
            next_state_latent = self._encode(next_state_batch_t, use_target=True)

        action_batch = torch.as_tensor(
            action_batch, dtype=torch.float32, device=self.device
        )
        reward_batch = torch.as_tensor(
            reward_batch, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        mask_batch = torch.as_tensor(
            mask_batch, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        if weights is not None:
            weights_t = torch.as_tensor(
                weights, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
            weights_t = torch.clamp(weights_t, max=2.0)
        else:
            weights_t = None

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_latent
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_latent, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_latent, action_batch)
        td_error1 = next_q_value.detach() - qf1
        td_error2 = next_q_value.detach() - qf2

        if weights_t is not None:
            qf1_loss = (weights_t * td_error1.pow(2)).mean()
            qf2_loss = (weights_t * td_error2.pow(2)).mean()
        else:
            qf1_loss = F.mse_loss(qf1, next_q_value.detach())
            qf2_loss = F.mse_loss(qf2, next_q_value.detach())

        critic_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        # capture gradient norm for critic+encoder
        critic_grad_norm = clip_grad_norm_(
            list(self.critic.parameters()) + list(self.encoder.parameters()), 5.0
        )
        self.critic_optim.step()

        # Re-encode state for policy (allows encoder to learn from policy gradients too)
        state_latent_pi = self._encode(state_batch_t)
        pi, log_pi, _ = self.policy.sample(state_latent_pi)
        # Detach encoder output for critic evaluation to avoid double-backprop through encoder
        qf1_pi, qf2_pi = self.critic(state_latent_pi.detach(), pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        # capture gradient norm for policy+encoder
        policy_grad_norm = clip_grad_norm_(
            list(self.policy.parameters()) + list(self.encoder.parameters()), 5.0
        )
        self.policy_optim.step()

        if self.automatic_temperature_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)
            alpha_tlogs = torch.tensor(self.alpha, device=self.device)

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.encoder_target, self.encoder, self.tau)

        if idxs is not None and hasattr(memory, "update_priorities"):
            prios = (td_error1.detach().abs() + td_error2.detach().abs()) * 0.5
            prios = prios.squeeze(1).cpu().numpy()
            memory.update_priorities(idxs, prios)
        # Diagnostics for logging
        try:
            mean_log_pi = log_pi.mean().item()
        except Exception:
            mean_log_pi = float("nan")
        try:
            mean_min_qf_pi = min_qf_pi.mean().item()
        except Exception:
            mean_min_qf_pi = float("nan")
        try:
            mean_qf1 = qf1.mean().item()
            mean_qf2 = qf2.mean().item()
        except Exception:
            mean_qf1 = float("nan")
            mean_qf2 = float("nan")
        td_error_mean = (
            ((td_error1.detach().abs() + td_error2.detach().abs()) * 0.5).mean().item()
        )

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
            mean_log_pi,
            mean_min_qf_pi,
            mean_qf1,
            mean_qf2,
            td_error_mean,
            float(critic_grad_norm),
            float(policy_grad_norm),
        )

    # Save model parameters
    def save_model(
        self,
        env_name: str,
        identifier: str,
        suffix: str = ".pt",
        actor_path=None,
        critic_path=None,
        encoder_path=None,
    ):
        path = Path("models/")
        path.mkdir(exist_ok=True)

        if actor_path is None:
            actor_path = (path / f"sac_actor_{env_name}_{identifier}").with_suffix(
                suffix
            )
        if critic_path is None:
            critic_path = (path / f"sac_critic_{env_name}_{identifier}").with_suffix(
                suffix
            )
        if encoder_path is None:
            encoder_path = (path / f"sac_encoder_{env_name}_{identifier}").with_suffix(
                suffix
            )
        print(f"Saving models to {actor_path}, {critic_path} and {encoder_path}")
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.encoder.state_dict(), encoder_path)

    # Load model parameters
    """
    Insert your description here.  
    
    ## Parameters:  
    
    - **param1** *(type)*:  
    
    ## Input:  
    
    - **Input 1** *(shapes)*:  
    
    ## Output:  
    
    - **Output 1** *(shapes)*:  
    """

    def load_model(self, actor_path, critic_path, encoder_path=None):
        print(
            f"Loading models from {actor_path}, {critic_path}"
            + (f" and {encoder_path}" if encoder_path else "")
        )
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path, map_location=DEVICE))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location=DEVICE))
            hard_update(self.critic_target, self.critic)
        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
            hard_update(self.encoder_target, self.encoder)
