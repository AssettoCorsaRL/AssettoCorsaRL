# AssettoCorsaRL

Project experimenting with training a reinforcement-learning (RL) agent to drive around the Monaco circuit.  The long-term goal is to train with Assetto Corsa; currently we iterate on a simpler 2D environment (`CarRacing-v3`) to develop algorithms, data pipelines, and perception modules.

**Status:** Prototype. Primary experiments use Soft Actor-Critic (SAC) on OpenAI Gym's `CarRacing` (2D) before moving to Assetto Corsa.

**Key points**
- Reinforcement learning algorithm: Soft Actor-Critic (SAC)
- Short-term env: `CarRacing` (2D, Gym / Box2D)
- Long-term env: Assetto Corsa (3D simulator) — goal track: Monaco
- Perception: autoencoder experiments live in `perception/`

Quickstart
----------

Coming SOON!

Repository layout
-----------------

- `train_sac.py` — training entry point for SAC experiments
- `load_sac.py` — load and test a saved model
- `RLHF_sac.py` — record and pretrain the SAC model - **EXPIREMENTAL**
- `sac/` — SAC algorithm, networks, replay memory and utils
- `perception/` — autoencoders, data generation and utils for observation processing
- `models/` — saved model checkpoints (.pt)
- `runs/` — experiment logs and run directories

Training notes
--------------

- We start by training on Gym's `CarRacing` to iterate quickly and validate training pipelines.
- Use standard SAC hyperparameters as a baseline; tune reward shaping, frame-stacking, and perception as needed.
- Save periodic checkpoints into `models/` and log runs under `runs/`.

Transition plan: CarRacing → Assetto Corsa (Monaco)
--------------------------------------------------

1. Finalize RL pipeline and stable policies on `CarRacing`.
2. Replace observation stack with the perception encoder from `perception/` so the agent consumes learned latent representations.
3. Integrate Assetto Corsa: capture frames from Assetto Corsa, adapt preprocessing to match perception encoder, and validate domain transfer.
4. Start training/evaluation on a simplified Assetto Corsa track asset (Monaco) with progressive difficulty.

Contributing
------------

Open issues or create PRs for: clearly reproducible training commands, missing dependency pins, or scripts to capture Assetto Corsa frames.

