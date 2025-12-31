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

---

## 🚀 Features

- Train SAC agents with configurable environments and models
- Config-driven experiments using YAML files (`configs/env_config.yaml`, `configs/model_config.yaml`)
- Logging & tracking with Weights & Biases (wandb)
- Reproducible experiments and model checkpoints (see `models/`)

---


## ⚙️ Configuration

- `configs/env_config.yaml` — environment hyperparameters (observation size, frame stacking, num envs, etc.)
- `configs/model_config.yaml` — model and training hyperparameters (learning rates, replay buffer size, etc.)

Training code merges values from both files and fills missing keys from `SACConfig` defaults.

---

## 📁 Checkpoints & Experiments

Checkpoints will generate in  `models/` (e.g., `sac_checkpoint_100000.pt`). Experiment logs and artifacts are stored in `wandb/` run directories.

---

## Contributing

Contributions are welcome! Suggested workflow:

1. Fork the repo and create a feature branch
2. Add tests for new behavior
3. Open a pull request and describe the change


---

## Contact

If you have questions or want to collaborate, open an issue or reach out via the project's issue tracker.


