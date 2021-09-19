import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt


import datetime
import time
from collections import deque, namedtuple
from itertools import count
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

epochs = 200
episodes_per_epoch = 64
policy_hidden = 32
value_hidden = 32
n_policy_updates = 16
n_value_updates = 16
gamma = 0.99
epsilon = 0.1
value_lr = 1e-3
policy_lr = 3e-4
max_traj_length = 1000

class Trajectory:
    def __init__(self, observations=[], actions=[], rewards=[], dones=[], logits=[],):
        self.obs = observations
        self.a = actions
        self.r = rewards
        self.d = dones
        self.logits = logits
        self.len = 0

    def add(self, obs: torch.Tensor, a: torch.Tensor, r: torch.Tensor, d: torch.Tensor, logits: torch.Tensor,):
        self.obs.append(obs)
        self.a.append(a)
        self.r.append(r)
        self.d.append(d)
        self.logits.append(logits)
        self.len += 1

    def disc_r(self, gamma, normalize=False):
        disc_rewards = []
        r = 0.0
        for reward in self.r[::-1]:
            r = reward + gamma * r
            disc_rewards.insert(0, r)
        disc_rewards = torch.tensor(disc_rewards, device=device, dtype=dtype)
        if normalize:
            disc_rewards = (disc_rewards - disc_rewards.mean()) / (disc_rewards.std() + np.finfo(np.float32).eps)
        return disc_rewards

    def __len__(self):
        return self.len


class PPO:
    def __init__(self, env, policy_hidden=policy_hidden, value_hidden=value_hidden):
        self.env = env
        if self.env.unwrapped.spec is not None:
            self.env_name = self.env.unwrapped.spec.id
        else:
            self.env_name = self.env.unwrapped.__class__.__name__

        self.policy = (
            nn.Sequential(
                nn.Linear(env.observation_space.shape[0], policy_hidden),
                nn.Dropout(p=0.6),
                nn.ReLU(),
                nn.Linear(policy_hidden, env.action_space.n),
            )
            .to(device)
            .to(dtype)
        )

        self.value = (
            nn.Sequential(
                nn.Linear(env.observation_space.shape[0], value_hidden),
                nn.Dropout(p=0.6),
                nn.ReLU(),
                nn.Linear(value_hidden, 1),
            )
            .to(device)
            .to(dtype)
        )

    def update(self, batch, hp, policy_optim, value_optim, writer):
        obs = [torch.stack(traj.obs)[:-1] for traj in batch]
        disc_r = [traj.disc_r(hp["gamma"], normalize=True) for traj in batch]
        a = [torch.stack(traj.a) for traj in batch]
        with torch.no_grad():
            v = [self.value(o) for o in obs]
            adv = [disc_r[i] - v[i] for i in range(len(batch))]
            old_logits = [torch.stack(traj.logits) for traj in batch]
            old_logprobs = [-F.cross_entropy(old_logits[i], a[i]) for i in range(len(batch))]

        for j in range(hp["n_policy_updates"]):
            policy_loss = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)
            for i, traj in enumerate(batch):
                curr_logits = self.policy(obs[i])
                curr_logprobs = -F.cross_entropy(curr_logits, a[i])
                ratio = torch.exp(curr_logprobs - old_logprobs[i])
                clipped_ratio = torch.clamp(ratio, 1 - hp["epsilon"], 1 + hp["epsilon"])
                policy_loss = (policy_loss + torch.min(ratio * adv[i], clipped_ratio * adv[i]).mean())

            policy_loss = policy_loss / len(batch)
            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

        for j in range(hp["n_value_updates"]):
            value_loss = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)
            for i in range(len(batch)):
                v = self.value(obs[i]).view(-1)
                value_loss = value_loss + F.mse_loss(v, disc_r[i])
            value_loss = value_loss / len(batch)
            value_optim.zero_grad()
            value_loss.backward()
            value_optim.step()

        return policy_loss.item(), value_loss.item()

    def train(self, epochs=epochs, episodes_per_epoch=episodes_per_epoch, n_value_updates=n_value_updates, n_policy_updates=n_policy_updates,
     value_lr=value_lr, policy_lr=policy_lr, gamma=gamma, epsilon=epsilon, max_traj_length=max_traj_length,
      log_dir="./logs/", RENDER=False, PLOT_REWARDS=True, VERBOSE=False, TENSORBOARD_LOG=True):
        hp = locals()
        start_time = datetime.datetime.now()
        print(f"Start time: {start_time:%d-%m-%Y %H:%M:%S}" f"\nTraining model on {self.env_name} | " f"Observation Space: {self.env.observation_space} | " 
        f"Action Space: {self.env.action_space}\n" f"Hyperparameters: \n{hp}\n")
        
        log_path = Path(log_dir).joinpath(f"{start_time:%d%m%Y%H%M%S}")
        log_path.mkdir(parents=True, exist_ok=False)

        if TENSORBOARD_LOG:
            writer = SummaryWriter(log_path)
            writer.add_text("hyperparameters", f"{hp}")
        else:
            writer = None

        self.policy.train()
        self.value.train()

        value_optim = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        policy_optim = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)

        rewards = []
        e = 0

        for epoch in range(epochs):

            epoch_rewards = []
            batch = []

            # Sample trajectories
            for _ in range(episodes_per_epoch):
                # initialise tracking variables
                obs = self.env.reset()
                obs = torch.tensor(obs, device=device, dtype=dtype)
                traj = Trajectory([obs], [], [], [], [])
                d = False
                e += 1

                # run for single trajectory
                for i in range(max_traj_length):
                    if RENDER and (
                        e == 0 or (e % ((epochs * episodes_per_epoch) / 10)) == 0
                    ):
                        self.env.render()

                    a_logits = self.policy(obs)
                    a = torch.distributions.Categorical(logits=a_logits).sample()

                    obs, r, d, _ = self.env.step(a.item())

                    obs = torch.tensor(obs, device=device, dtype=dtype)
                    r = torch.tensor(r, device=device, dtype=dtype)
                    traj.add(obs, a, r, d, a_logits)

                    if d:
                        break

                epoch_rewards.append(sum(traj.r))
                batch.append(traj)

            # Update value and policy
            p_loss, v_loss = self.update(
                batch, hp, policy_optim, value_optim, writer
            )

            # Log rewards and losses
            epoch_rewards = torch.tensor(epoch_rewards, device=device, dtype=dtype)
            avg_episode_reward = torch.mean(epoch_rewards[-episodes_per_epoch:])
            avg_episode_reward.detach().cpu().numpy()
            rewards.append(avg_episode_reward)
            if writer is not None:
                writer.add_scalar("policy_loss", p_loss, epoch)
                writer.add_scalar("value_loss", v_loss, epoch)
                writer.add_scalar("rewards", avg_episode_reward, epoch)

            if VERBOSE and (epoch == 0 or ((epoch + 1) % (epochs / 10)) == 0):
                print(
                    f"Epoch {epoch+1}: Average Episodic Reward = {avg_episode_reward:.2f} |"
                    f" Value Loss = {p_loss:.2f} |"
                    f" Policy Loss = {v_loss:.2f}"
                )

        
        self.env.close()
        print(f"\nTraining Completed in {(datetime.datetime.now() - start_time).seconds} seconds")
        self.save(log_path.joinpath(f"{self.__class__.__name__}_{self.env_name}.pt"))
        if PLOT_REWARDS:
            plt.plot(rewards)
            plt.savefig(log_path.joinpath(f"{self.__class__.__name__}_{self.env_name}_reward_plot.png"))

    def save(self, path):
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_state_dict": self.value.state_dict(),
            },
            path,
        )
        print(f"\nSaved model parameters to {path}")

    def load(self, path=None):
        if path is None:
            path = f"./models/{self.__class__.__name__}_{self.env_name}.pt"
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value.load_state_dict(checkpoint["value_state_dict"])
        print(f"\nLoaded model parameters from {path}")

    def eval(self, episodes, render=False):

        print(f"\nEvaluating model for {episodes} episodes ...\n")
        start_time = datetime.datetime.now()
        self.policy.eval()
        rewards = []

        for episode in range(episodes):

            observation = self.env.reset()
            observation = torch.tensor(observation, device=device, dtype=dtype)
            done = False
            episode_rewards = []

            while not done:
                if render:
                    self.env.render()

                logits = self.policy(observation)
                action = torch.distributions.Categorical(logits=logits).sample()
                next_observation, reward, done, _ = self.env.step(action.item())
                episode_rewards.append(float(reward))
                next_observation = torch.tensor(next_observation, device=device, dtype=dtype)
                observation = next_observation

            total_episode_reward = sum(episode_rewards)
            rewards.append(total_episode_reward)
            print(f"Episode {episode+1}: Total Episode Reward = {total_episode_reward:.2f}")
            rewards.append(total_episode_reward)

        self.env.close()
        print(f"\nAverage Reward for an episode = {np.mean(rewards):.2f}")
        print(f"Evaluation Completed in {(datetime.datetime.now() - start_time).seconds} seconds")


if __name__ == "__main__":

    import gym

    env = gym.make("CartPole-v1")
    model = PPO(env)
    model.train(VERBOSE=True, PLOT_REWARDS=True, TENSORBOARD_LOG=True)
    model.eval(10)