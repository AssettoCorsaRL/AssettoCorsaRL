"""Load a pretrained SAC model and run episodes to evaluate/visualize it.

Usage:
    Play 5 episodes with rendering (human window):
    python assetto-corsa-rl\scripts\load.py --model pretrained_model.pt --episodes 5 --render

    Play 3 episodes on CPU without render:
    python assetto-corsa-rl\scripts\load.py --model pretrained_model.pt --episodes 3 --device cpu

Options:
    --deterministic  Use the actor mean (deterministic) instead of sampling actions
"""

import argparse
import sys
import os
from pathlib import Path
import torch
from tensordict import TensorDict

try:
    from assetto_corsa_rl.env import create_gym_env  # type: ignore
    from assetto_corsa_rl.model.sac import SACPolicy, get_device  # type: ignore
except Exception:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from assetto_corsa_rl.env import create_gym_env  # type: ignore
    from assetto_corsa_rl.model.sac import SACPolicy, get_device  # type: ignore


def load_checkpoint(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return ckpt


def deterministic_action_from_actor(actor, pixels):
    """Compute a deterministic action from the actor by using the mean (loc).

    We use tanh(loc) and map to action bounds if available in actor.distribution_kwargs.
    Falls back to tanh(loc) if bounds aren't present.
    """
    td_in = TensorDict({"pixels": pixels}, batch_size=pixels.shape[0])
    params = actor.module(td_in)
    loc = params["loc"]  #  [B, action_dim]
    raw = torch.tanh(loc)

    try:
        dist_kwargs = getattr(actor, "distribution_kwargs", None)
        if dist_kwargs is None:
            dist_kwargs = getattr(actor, "_distribution_kwargs", None)
        if dist_kwargs is not None and "low" in dist_kwargs and "high" in dist_kwargs:
            low = torch.as_tensor(
                dist_kwargs["low"], device=raw.device, dtype=raw.dtype
            )
            high = torch.as_tensor(
                dist_kwargs["high"], device=raw.device, dtype=raw.dtype
            )

            low = low.unsqueeze(0).expand_as(raw)
            high = high.unsqueeze(0).expand_as(raw)
            action = low + (raw + 1.0) * 0.5 * (high - low)
            return action
    except Exception:
        pass

    return raw


def play(
    model_path,
    device=None,
    episodes=5,
    max_steps=1000,
    render=False,
    deterministic=False,
    seed=None,
):
    device = torch.device(device) if device else get_device()
    print(f"Using device: {device}")

    render_mode = "human" if render else None
    env = create_gym_env(device=device, num_envs=1, render_mode=render_mode)

    policy = SACPolicy(env=env, device=device)
    modules = policy.modules()
    actor = modules["actor"]
    value = modules["value"]
    q1 = modules["q1"]
    q2 = modules["q2"]

    with torch.no_grad():
        td = env.reset()
        sample_pixels = td["pixels"][:1].to(device)
        if sample_pixels.ndim == 3:
            sample_pixels = sample_pixels.unsqueeze(0)
        if sample_pixels.shape[1] != 4:
            print("Using dummy pixels tensor (1,4,84,84) to initialize lazy layers")
            sample_pixels = torch.zeros(
                1, 4, 84, 84, device=device, dtype=sample_pixels.dtype
            )

        sample_action = torch.zeros(1, env.action_spec.shape[-1], device=device)

        def _has_uninitialized_params(mod):
            for p in mod.parameters():
                try:
                    if p.numel() == 0:
                        return True
                except Exception:
                    return True
            return False

        try:
            actor(TensorDict({"pixels": sample_pixels}, batch_size=1))
        except Exception as e:
            print("actor forward (probabilistic wrapper) failed during init:", e)
        try:
            value(sample_pixels)
        except Exception as e:
            print("value forward failed during init:", e)
        try:
            q1(sample_pixels, sample_action)
        except Exception as e:
            print("q1 forward failed during init:", e)
        try:
            q2(sample_pixels, sample_action)
        except Exception as e:
            print("q2 forward failed during init:", e)

        retries = 3
        for i in range(retries):
            needs = []
            if _has_uninitialized_params(actor):
                needs.append("actor")
                print(
                    "Actor has uninitialized params; running alternative module forward"
                )
                try:
                    actor.module(TensorDict({"pixels": sample_pixels}, batch_size=1))
                except Exception as e:
                    print("actor.module forward failed:", e)
            if _has_uninitialized_params(value):
                needs.append("value")
                print("Value has uninitialized params; running value forward")
                try:
                    value(sample_pixels)
                except Exception as e:
                    print("Value forward retry failed:", e)
                try:
                    if hasattr(value, "module"):
                        value.module(sample_pixels)
                except Exception as e:
                    print("value.module forward failed:", e)
            if _has_uninitialized_params(q1):
                needs.append("q1")
                print("Q1 has uninitialized params; running q1 forward")
                try:
                    q1(sample_pixels, sample_action)
                except Exception as e:
                    print("Q1 forward retry failed:", e)
                try:
                    if hasattr(q1, "module"):
                        q1.module(sample_pixels, sample_action)
                except Exception as e:
                    print("q1.module forward failed:", e)
            if _has_uninitialized_params(q2):
                needs.append("q2")
                print("Q2 has uninitialized params; running q2 forward")
                try:
                    q2(sample_pixels, sample_action)
                except Exception as e:
                    print("Q2 forward retry failed:", e)
                try:
                    if hasattr(q2, "module"):
                        q2.module(sample_pixels, sample_action)
                except Exception as e:
                    print("q2.module forward failed:", e)

            if not needs:
                break

            sample_pixels = torch.randn(
                1, 4, 84, 84, device=device, dtype=sample_pixels.dtype
            )

        if (
            _has_uninitialized_params(actor)
            or _has_uninitialized_params(value)
            or _has_uninitialized_params(q1)
            or _has_uninitialized_params(q2)
        ):
            print(
                "Warning: Some lazy parameters remain uninitialized after retries; load_state_dict may fail."
            )

    checkpoint = load_checkpoint(model_path, device)
    try:
        if "actor_state" in checkpoint:
            actor.load_state_dict(checkpoint["actor_state"])
            print("Loaded actor state")
        if "q1_state" in checkpoint:
            q1.load_state_dict(checkpoint["q1_state"])
            print("Loaded q1 state")
        if "q2_state" in checkpoint:
            q2.load_state_dict(checkpoint["q2_state"])
            print("Loaded q2 state")
        if "value_state" in checkpoint:
            value.load_state_dict(checkpoint["value_state"])
            print("Loaded value state")
    except Exception as e:
        print(f"Warning: failed to load some states: {e}")

    actor.eval()
    value.eval()
    q1.eval()
    q2.eval()

    if seed is not None:
        torch.manual_seed(seed)

    episode_returns = []

    for ep in range(episodes):
        td = env.reset()
        inner = td["next"] if "next" in td.keys() else td
        total_reward = 0.0
        steps = 0

        while steps < max_steps:
            pixels = inner["pixels"][:1].to(device)

            with torch.no_grad():
                if deterministic:
                    action = deterministic_action_from_actor(actor, pixels)
                else:
                    actor_td = TensorDict({"pixels": pixels}, batch_size=1)
                    out = actor(actor_td)
                    action = out.get("action")
                    if action is None:
                        action = deterministic_action_from_actor(actor, pixels)

            if action.ndim == 1:
                action = action.unsqueeze(0)

            action = action.to(device).float()

            action_td = TensorDict({"action": action}, batch_size=[1])
            print(f"Step {steps}: action = {action.cpu().numpy().flatten()}")

            next_td = env.step(action_td)
            inner_next = next_td["next"] if "next" in next_td.keys() else next_td

            r = 0.0
            if "reward" in inner_next.keys():
                r = float(inner_next["reward"].item())
            elif "rewards" in inner_next.keys():
                r = float(inner_next["rewards"].item())

            total_reward += r
            steps += 1

            inner = inner_next

            if render:
                try:
                    env.render()
                except Exception:
                    pass

            done = False
            for key in ("done", "terminated", "truncated"):
                if key in inner.keys():
                    done = done or bool(inner[key].item())
            if done:
                break

        episode_returns.append(total_reward)
        print(
            f"Episode {ep+1}/{episodes} finished: steps={steps}, reward={total_reward:.2f}"
        )

    print(
        f"Average reward over {len(episode_returns)} episodes: {sum(episode_returns)/len(episode_returns):.2f}"
    )

    try:
        env.close()
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a pretrained SAC model and play episodes"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to pretrained model (.pt)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (e.g., cuda:0, cpu)"
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to play"
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Max steps per episode"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render environment to screen"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic (mean) actions"
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    play(
        model_path=args.model,
        device=args.device,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        deterministic=args.deterministic,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
