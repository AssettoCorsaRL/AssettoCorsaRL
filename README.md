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

A reinforcement learning agent (using the SAC algorithm) learns to drive a Formula 1 car around the Monaco GP circuit in Assetto Corsa. The agent controls steering, acceleration, and braking by interacting with the track, receiving feedback, and improving over time.

> [!IMPORTANT]
> Visit the [Documentation](https://assettocorsarl.github.io/AssettoCorsaRL-DOCS/docs)
>
> Check out our [PyPi package](https://pypi.org/project/assetto-corsa-rl/)

---
<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
        <h2>🚀 Roadmap</h2>   
    </summary>
  </ul>
</div>


- [x] 1. Train a Soft Actor-Critic (SAC) agent on a simplified 2D environment: OpenAI’s CarRacing-v3.
-  [ ] 2. Adapt and port the trained agent to the Assetto Corsa racing simulator.
- [ ] 3. Implement full race simulations using AI-driven race strategies.

---

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
        <h2>⚙️ Configuration </h2>
    </summary>
  </ul>
</div>

**For Car-Racing:**

- `configs/car-racing/env_config.yaml` — environment hyperparameters (observation size, frame stacking, num envs, etc.)
- `configs/car-racing/model_config.yaml` — model and training hyperparameters (learning rates, replay buffer size, etc.)
`configs/car-racing/model_config.yaml` — model and training hyperparameters (learning rates, replay buffer size, etc.)

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


