class Config:
    def __init__(self):
        self.episodes = 2000
        self.max_t = 1000
        self.batch_size = 128
        self.buffer_size = 200_000
        self.lr = 1e-4
        self.seed = 0
        # self.eps_start = 1.0
        # self.eps_end = 0.01
        # self.eps_decay = 0.995
        self.layer_size = 256
        self.n_step = 3
        self.tau = 1e-3
        self.gamma = 0.99
        self.n = 64
        self.n_envs = 4
