<div align="center">
<img src="./assets/on_track.gif" alt="Human Player">
<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
        <h1>Assetto Corsa RL</h1>
    </summary>
  </ul>
</div>
<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
        <h2> A reinforcement learning project for driving in Assetto Corsa. </h2>
    </summary>
  </ul>
</div>
</div>

> [!TIP]
> Check out the wandb for the latest runs:

---
<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
        <h2>🚀 Roadmap</h2>   
    </summary>
  </ul>
</div>


1. Train a Soft Actor-Critic (SAC) agent on a simplified 2D environment: OpenAI’s CarRacing-v3.
2. Adapt and port the trained agent to the Assetto Corsa racing simulator.
3. Implement full race simulations using AI-driven race strategies.

---

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
        <h2>⚙️ Configuration </h2>
    </summary>
  </ul>
</div>

- `configs/env_config.yaml` — environment hyperparameters (observation size, frame stacking, num envs, etc.)
- `configs/model_config.yaml` — model and training hyperparameters (learning rates, replay buffer size, etc.)
`configs/model_config.yaml` — model and training hyperparameters (learning rates, replay buffer size, etc.)

Training code merges values from both files and fills missing keys from `SACConfig` defaults.

---
<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
        <h2>🛠️ Installation</h2>
    </summary>
  </ul>
</div>

- Coming soon.

---
<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
        <h2>📁 Checkpoints & Experiments </h2>
    </summary>
  </ul>
</div>

Checkpoints will generate in  `models/` (e.g., `sac_checkpoint_100000.pt`). Experiment logs and artifacts are stored in `wandb/` run directories.

---
<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
        <h2>Contributing </h2>
    </summary>
  </ul>
</div>


Contributions are welcome! Suggested workflow:

1. Fork the repo and create a feature branch
2. Add tests for new behavior
3. Open a pull request and describe the change


---

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
        <h2>Contact</h2>
    </summary>
  </ul>
</div>

If you have questions or want to collaborate, open an issue or reach out via the project's issue tracker.


