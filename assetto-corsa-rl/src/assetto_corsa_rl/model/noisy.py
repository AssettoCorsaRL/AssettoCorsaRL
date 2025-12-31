# NOTE: This file (or portions) is derived from:
# https://github.com/thomashirtz/noisy-networks/blob/main/noisynetworks.py
# The original repository does NOT include a license, so legally you do not have
# permission to copy, modify, or redistribute the code without the author's consent.
# This implementation is substantially the same as that upstream file.
# If you distribute or reuse this code, contact the original author for permission.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractNoisyLayer(nn.Module, ABC):
    """
    An abstract layer that introduces noise into the neural network's parameters,
    simulating a form of regularization and potentially enhancing generalization.

    This class serves as a base for specific implementations that apply noise in
    various ways to the layer's weights and biases.

    Attributes:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        sigma (float): The standard deviation of the noise distribution.
        mu_bias (torch.nn.Parameter): The mean values of the bias parameters.
        mu_weight (torch.nn.Parameter): The mean values of the weight parameters.
        sigma_bias (torch.nn.Parameter): The standard deviations of the bias noise.
        sigma_weight (torch.nn.Parameter): The standard deviations of the weight noise.
        cached_bias (torch.Tensor or None): Cached bias values to avoid recomputation.
        cached_weight (torch.Tensor or None): Cached weight values to avoid recomputation.

    Methods:
        forward(x, sample_noise=True): Performs the forward pass of the layer.
        register_noise_buffers(): Abstract method to register noise-related buffers.
        sample_noise(): Abstract method to sample and apply noise to the parameters.
        parameter_initialization(): Abstract method for initializing parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: float,
    ):
        super().__init__()

        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")
        if not (sigma >= 0):
            raise ValueError(f"sigma must be non-negative, got {sigma}")

        self.sigma = sigma
        self.in_features = in_features
        self.out_features = out_features

        self.mu_bias = nn.Parameter(torch.zeros(out_features))
        self.mu_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.zeros(out_features))
        self.sigma_weight = nn.Parameter(torch.zeros(out_features, in_features))

        self.cached_bias = None
        self.cached_weight = None

        self.register_noise_buffers()
        self.parameter_initialization()
        self.sample_noise()

    def forward(self, x: torch.Tensor, sample_noise: bool = True) -> torch.Tensor:
        """
        Computes the forward pass of the layer, optionally applying noise to the parameters.

        Args:
            x (torch.Tensor): The input tensor.
            sample_noise (bool): If True, samples and applies noise to the parameters before
                                 the forward pass. Default is True.

        Returns:
            torch.Tensor: The output of the layer after applying the linear transformation
                          and optionally the noise.
        """
        if self.training:
            if sample_noise:
                self.sample_noise()
            return nn.functional.linear(x, weight=self.weight, bias=self.bias)
        else:
            return nn.functional.linear(x, weight=self.mu_weight, bias=self.mu_bias)

    @abstractmethod
    def register_noise_buffers(self) -> None:
        """
        Abstract method to register noise-related buffers in subclasses.
        Subclasses should implement this method to register buffers that
        hold the noise terms for weights and biases.
        """

    @abstractmethod
    def _calculate_bound(self) -> float:
        """
        Calculates the initialization bound for parameters based on the specific noise model
        and layer configuration. This bound is used to initialize the layer's parameters
        uniformly within a range, ensuring a proper scale of weights and biases at the
        beginning of training.

        Returns:
            float: The calculated bound value for parameter initialization.
        """

    @property
    @abstractmethod
    def weight(self) -> torch.Tensor:
        """
        Abstract property to get the current weights with noise applied. This property
        should dynamically compute the noisy weights based on the current state of
        the noise parameters and the base weight parameters.

        Returns:
            torch.Tensor: The current weights with noise applied.
        """

    @property
    @abstractmethod
    def bias(self) -> torch.Tensor:
        """
        Abstract property to get the current biases with noise applied.
        Subclasses should implement this property to return the noisy biases.

        Returns:
            torch.Tensor: The biases with noise applied.
        """

    @abstractmethod
    def sample_noise(self) -> None:
        """
        Abstract method to sample noise and apply it to the parameters.
        Subclasses should implement this method to sample noise according
        to their specific noise model and apply it to the weights and biases.
        """

    @abstractmethod
    def parameter_initialization(self) -> None:
        """
        Abstract method for initializing the layer's parameters.
        Subclasses should implement this method to initialize the mean
        and standard deviation of the weights and biases.
        """


class FactorisedNoisyLayer(AbstractNoisyLayer):
    """
    A layer that applies factorised Gaussian noise to parameters, reducing
    the computational complexity compared to independent noise application.
    This approach is particularly useful in environments where the noise
    needs to be efficiently generated and applied across many parameters.

    Inherits from AbstractNoisyLayer and implements methods for factorised noise
    application, parameter initialization, and noise buffer registration.

    Additional Attributes:
        epsilon_input (torch.Tensor): Input-side noise for factorised noise generation.
        epsilon_output (torch.Tensor): Output-side noise for factorised noise generation.
    """

    epsilon_input: torch.Tensor
    epsilon_output: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: float = 0.5,
    ):
        super().__init__(
            in_features=in_features, out_features=out_features, sigma=sigma
        )

    def register_noise_buffers(self) -> None:
        """
        Registers factorised noise buffers, creating empty tensors for input and
        output noise components with dimensions matching input and output features.
        """
        self.register_buffer(name="epsilon_input", tensor=torch.empty(self.in_features))
        self.register_buffer(
            name="epsilon_output", tensor=torch.empty(self.out_features)
        )

    def _calculate_bound(self) -> float:
        """
        Determines the initialization bound for the FactorisedNoisyLayer based on the inverse
        square root of the number of input features. This approach to determining the bound
        takes advantage of the factorised noise model's efficiency and aims to balance the
        variance of the outputs relative to the variance of the inputs. Ensuring that the
        initialization of weights does not saturate the neurons and allows for stable
        gradients during the initial phases of training.

        Returns:
            float: The calculated bound for initializing the layer's parameters, optimized
            for the factorised noise model to encourage stability and efficiency in
            parameter updates during the onset of learning.
        """
        return self.in_features ** (-0.5)

    @property
    def weight(self) -> torch.Tensor:
        """
        Computes and returns the noisy weights by applying factorised noise to
        the mean weight values through an outer product of input and output noises.

        Returns:
            torch.Tensor: The noisy weights computed using factorised noise.
        """
        if self.cached_weight is None:
            self.cached_weight = (
                self.sigma_weight * torch.ger(self.epsilon_output, self.epsilon_input)
                + self.mu_weight
            )
        return self.cached_weight

    @property
    def bias(self) -> torch.Tensor:
        """
        Computes and returns the noisy biases by applying output-side factorised noise
        to the mean bias values.

        Returns:
            torch.Tensor: The noisy biases computed using output-side factorised noise.
        """
        if self.cached_bias is None:
            self.cached_bias = self.sigma_bias * self.epsilon_output + self.mu_bias
        return self.cached_bias

    def sample_noise(self) -> None:
        """
        Samples factorised noise for both inputs and outputs using a standard normal
        distribution and applies a transformation to achieve the desired noise distribution.
        Resets cached weights and biases to ensure fresh computation with new noise.
        """
        with torch.no_grad():
            epsilon_input = torch.randn(
                self.in_features, device=self.epsilon_input.device
            )
            epsilon_output = torch.randn(
                self.out_features, device=self.epsilon_output.device
            )
            self.epsilon_input.copy_(
                epsilon_input.sign() * torch.sqrt(torch.abs(epsilon_input))
            )
            self.epsilon_output.copy_(
                epsilon_output.sign() * torch.sqrt(torch.abs(epsilon_output))
            )
        self.cached_weight = None
        self.cached_bias = None

    def parameter_initialization(self) -> None:
        """
        Initializes the parameters of the layer by setting the standard deviation of the
        noise and uniformly initializing the mean values within a bound derived from the
        inverse square root of the input features.
        """
        bound = self._calculate_bound()
        self.sigma_bias.data.fill_(self.sigma * bound)
        self.sigma_weight.data.fill_(self.sigma * bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.mu_weight.data.uniform_(-bound, bound)


class NoisyLazyLinear(nn.Module):
    """A lazily-initialized wrapper that creates a FactorisedNoisyLayer on first forward.

    Useful as a drop-in replacement for `nn.LazyLinear` in architectures that want
    factorised noisy layers for exploration.
    """

    def __init__(self, out_features: int, sigma: float = 0.5, device=None):
        super().__init__()
        if out_features <= 0:
            raise ValueError("out_features must be positive")
        if not (sigma >= 0):
            raise ValueError("sigma must be non-negative")
        self.out_features = out_features
        self.sigma = sigma
        self.device = device
        self._layer = None

    def forward(self, x: torch.Tensor):
        # Lazily create the underlying noisy layer using the observed input size
        if self._layer is None:
            in_features = x.shape[-1]
            self._layer = FactorisedNoisyLayer(
                in_features, self.out_features, sigma=self.sigma
            ).to(self.device if self.device is not None else x.device)
        return self._layer(x)

    def sample_noise(self) -> None:
        """Manually sample new noise for the underlying noisy layer (if created)."""
        if self._layer is not None:
            self._layer.sample_noise()
