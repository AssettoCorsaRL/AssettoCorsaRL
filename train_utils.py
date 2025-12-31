import torch
import torch.nn.functional as F
from tensordict import TensorDict


def reduce_value_to_batch(x, batch_size):
    """Reduce a value-shaped tensor or TensorDict to a (batch_size, 1) tensor.

    See original docstring in the repo.
    """
    try:
        v = x.get("value") if hasattr(x, "get") else x

        if v.shape[0] == batch_size:
            if v.ndim == 1:
                return v.view(-1, 1)
            return v.flatten(1).mean(dim=1, keepdim=True)

        return v.view(batch_size, -1).mean(dim=1, keepdim=True)
    except Exception:
        return None


def pack_pixels(x):
    """Pack pixel tensors to a compact integer format.

    - If input values are in [0, 1], pack to `torch.uint8` in [0, 255].
    """
    if not isinstance(x, torch.Tensor):
        return x
    return (x.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()


def unpack_pixels(x):
    """Inverse of `pack_pixels`.

    Converts integer-packed pixels back to floating-point in [0,1] (uint8) or [-1,1] (int8).
    """
    if not isinstance(x, torch.Tensor):
        return x

    if x.dtype == torch.uint8:
        return x.to(torch.float32) / 255.0
    if x.dtype == torch.int8:
        return x.to(torch.float32) / 127.0

    # Default: convert to float32 assuming already in [0,1]
    return x.to(torch.float32)


def sample_random_actions(num_envs, device=None):
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    steer = torch.empty(num_envs, 1, device=device).uniform_(-1, 1)
    gas = torch.empty(num_envs, 1, device=device).uniform_(0, 1)
    brake = torch.empty(num_envs, 1, device=device).uniform_(0, 1)
    return torch.cat([steer, gas, brake], dim=-1)


def sample_random_action(n=1, dev=None):
    return sample_random_actions(n, device=dev)


def get_inner(td):
    return td["next"] if "next" in td.keys() else td


def extract_reward_and_done(td, num_envs, device):
    td = get_inner(td)
    if "reward" in td.keys():
        rewards = td["reward"].view(num_envs).to(device)
    elif "rewards" in td.keys():
        rewards = td["rewards"].view(num_envs).to(device)
    else:
        raise KeyError(f"Unexpected TensorDict structure. Keys: {td.keys()}")

    dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
    dones |= td["done"].view(num_envs).to(device).to(torch.bool)
    dones |= td["terminated"].view(num_envs).to(device).to(torch.bool)
    dones |= td["truncated"].view(num_envs).to(device).to(torch.bool)
    return rewards, dones


def expand_actions_for_envs(actions, target_batch):
    if isinstance(target_batch, (tuple, list, torch.Size)) and len(target_batch) > 1:
        extra = target_batch[1:]
        new_shape = (actions.shape[0],) + (1,) * len(extra) + (actions.shape[1],)
        expand_shape = (actions.shape[0],) + tuple(extra) + (actions.shape[1],)
        return actions.view(new_shape).expand(expand_shape)
    return actions


def add_transition(rb, i, pixels, next_pixels, action, reward, done):
    packed_pixels = pack_pixels(pixels[i])
    packed_next = pack_pixels(next_pixels[i])
    action_cpu = action[i].to(torch.float32).cpu()
    reward_cpu = reward[i].unsqueeze(0).cpu()
    done_cpu = done[i].unsqueeze(0).cpu()
    transition = TensorDict(
        {
            "pixels": packed_pixels,
            "action": action_cpu,
            "reward": reward_cpu,
            "next_pixels": packed_next,
            "done": done_cpu,
        },
        batch_size=[],
    )
    rb.add(transition)


def fix_action_shape(a, batch_size, action_dim=None):
    if not isinstance(a, torch.Tensor):
        return a
    if a.ndim == 1:
        a = a.view(batch_size, -1)
    elif a.ndim > 2:
        a = a.view(batch_size, -1)
    if action_dim is None:
        return a
    L = a.shape[1]
    if L == action_dim:
        return a
    if L % action_dim == 0:
        return a.view(batch_size, L // action_dim, action_dim).mean(dim=1)
    return a[:, :action_dim]
