import torch
import torch.nn as nn
import torch.multiprocessing as mp


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(x)


def rollout_worker(shared_actor, worker_id, steps=5):
    # worker uses shared actor directly on GPU — no weight copy needed
    obs = torch.randn(1, 8).cuda()
    for i in range(steps):
        with torch.no_grad():
            action = shared_actor(obs)
        print(f"[Worker {worker_id}] step {i}, action mean: {action.mean():.4f}")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    actor = TinyActor().cuda()
    actor.share_memory()  # all params now in shared CUDA memory

    workers = [mp.Process(target=rollout_worker, args=(actor, i)) for i in range(2)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    print("Parent actor still intact:", actor.fc.weight.shape)
