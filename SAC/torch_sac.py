import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import traceback
from collections import deque, namedtuple
# from datetime import datetime, time
import time
from itertools import count
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

eps = 1e-4
log_max = 2
log_min = -20

timesteps = 20000
test_episodes = None
max_traj_length = 1000
batch_size = 128
max_buffer_size = timesteps // 2
update_after = 500
update_every = 1
select_after = 1000
q_hidden = 64
policy_hidden= 64
policy_lr = 1e-3
q_lr = 1e-3
temp_lr = 3e-4
discounting = 0.99
polyack = 0.995
grad_norm_clip = 20
verbose = False
save_frequency = 10

Transition = namedtuple("Transition", ("observation", "action", "reward", "next_observation", "done"))

class Actor(nn.Module):
    def __init__(self, s_dim, hidden_dim, a_dim, upper_lim, lower_lim):
        super(Actor, self).__init__()
        self.scale_factor = upper_lim - lower_lim
        self.range_midpoint = (upper_lim + lower_lim) / 2
        self.common_model = (
            nn.Sequential(
                nn.Linear(s_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            .to(device)
            .to(dtype)
        )
        self.mean_head = nn.Linear(hidden_dim, a_dim).to(device).to(dtype)
        self.std_dev_head = nn.Linear(hidden_dim, a_dim).to(device).to(dtype)

    def forward(self, x):
        x = self.common_model(x)
        mean = self.mean_head(x)
        log_std_dev = self.std_dev_head(x)
        log_std_dev = torch.clamp(log_std_dev, min=log_min, max=log_max)
        return mean, log_std_dev

    def sample_action(self, s, get_logprob=True, deterministic=False):
        mean, log_std_dev = self.forward(s)
        std_dev = torch.exp(log_std_dev)
        action_dist = torch.distributions.Normal(mean, std_dev)

        if deterministic:
            action = mean
        else:
            action = action_dist.rsample()

        if get_logprob:
            logprob = action_dist.log_prob(action)
            logprob -= 2 * (np.log(2) - logprob - nn.functional.softplus(-2 * logprob))
            logprob = logprob.sum(axis=-1, keepdim=True)

        else:
            logprob = None

        action = torch.tanh(action) * self.scale_factor + self.range_midpoint

        return action, logprob

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_dim):
        super(Critic, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.model = (
            nn.Sequential(
                nn.Linear(s_dim + a_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            .to(device)
            .to(dtype)
        )

    def forward(self, s, a):
        x = torch.cat([s.view(-1, self.s_dim), a.view(-1, self.a_dim)], dim=1)
        return self.model(x)


class SAC:
    def __init__(self, env, q_hidden_dim=q_hidden, policy_hidden_dim=policy_hidden):
        self.env = env
        if self.env.unwrapped.spec is not None:
            self.env_name = env.unwrapped.spec.id
        else:
            self.env_name = env.unwrapped.__class__.__name__

        self.action_shape = env.action_space.shape
        self.observation_shape = env.observation_space.shape
        self.action_min = env.action_space.low[0]
        self.action_max = env.action_space.high[0]

        Q1 = Critic(self.observation_shape[0], self.action_shape[0], q_hidden_dim)
        Q2 = Critic(self.observation_shape[0], self.action_shape[0], q_hidden_dim)
        self.Qs = [Q1, Q2]

        target_Q1 = copy.deepcopy(Q1)
        target_Q2 = copy.deepcopy(Q2)
        self.target_Qs = [target_Q1, target_Q2]

        for target_Q in self.target_Qs:
            for p in target_Q.parameters():
                p.requires_grad = False

        self.policy = Actor(self.observation_shape[0], policy_hidden_dim, self.action_shape[0], self.action_max, self.action_min)

        self.entropy_temp_log = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)
        self.entropy_temp = torch.exp(self.entropy_temp_log).item()
        self.target_entropy = -1 * np.prod(self.action_shape)

    def select_action(self, observation, select_after=0, step_count=0, deterministic=False):
        if step_count < select_after:
            action = torch.tensor(self.env.action_space.sample(), device=device, dtype=dtype).unsqueeze(0)
        else:
            with torch.no_grad():
                action, _ = self.policy.sample_action(observation, get_logprob=False, deterministic=deterministic)

        return action

    def update(self, replay_buffer, batch_size, q_optimizers, policy_optimizer, gamma, polyak_const, entropy_temp_optimizer, grad_norm_clip):
        sample = Transition(*[torch.cat(i) for i in [*zip(*random.sample(replay_buffer, batch_size))]])
        
        with torch.no_grad():
            next_action, logprob = self.policy.sample_action(sample.next_observation)
            next_state_q_val = torch.min(*[Q(sample.next_observation, next_action) for Q in self.target_Qs])
            target_q_vals = sample.reward + gamma * (next_state_q_val - self.entropy_temp * logprob) * (~sample.done)

        q_loss = 0.0

        for Q, target_Q, q_optimizer in zip(self.Qs, self.target_Qs, q_optimizers):
            q_vals = Q(sample.observation, sample.action)

            q_optimizer.zero_grad()
            curr_q_loss = nn.MSELoss()(target_q_vals, q_vals)
            curr_q_loss.backward()
            nn.utils.clip_grad_norm_(Q.parameters(), grad_norm_clip)
            q_optimizer.step()
            q_loss += float(curr_q_loss)

        for Q in self.Qs:
            for p in Q.parameters():
                p.requires_grad = False

        policy_optimizer.zero_grad()
        action, logprob = self.policy.sample_action(sample.observation)
        min_q_val = torch.min(*[Q(sample.observation, action) for Q in self.Qs])
        policy_loss = -1 * torch.mean(min_q_val - self.entropy_temp * logprob)
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), grad_norm_clip)
        policy_optimizer.step()

        for Q in self.Qs:
            for p in Q.parameters():
                p.requires_grad = True

        # Tune entropy regularization temperature
        # entropy_temp_optimizer.zero_grad()
        # entropy_temp_loss = -1 * torch.mean(
        #     self.entropy_temp_log * (logprob + self.target_entropy).detach()
        # )
        # entropy_temp_loss.backward()
        # entropy_temp_optimizer.step()
        # torch.exp(self.entropy_temp_log).item()
        self.entropy_temp = 0.02

        for Q, target_Q in zip(self.Qs, self.target_Qs):
            with torch.no_grad():
                for p_target, p in zip(target_Q.parameters(), Q.parameters()):
                    p_target.data.mul_(polyak_const)
                    p_target.data.add_((1 - polyak_const) * p.data)

        return q_loss, policy_loss.item()

    def train(self, q_lr=q_lr, policy_lr=policy_lr, temp_lr=temp_lr, max_buffer_size=max_buffer_size, select_after=select_after, batch_size=batch_size,
             discounting=discounting, polyack=polyack, update_after=update_after, update_every=update_every, grad_norm_clip=grad_norm_clip,
             save_frequency=save_frequency, RENDER=False, VERBOSE=False, PLOT_REWARDS=False):

        self.policy.train()
        for Q in self.Qs:
            Q.train()
        
        q_optimizers = [torch.optim.Adam(Q.parameters(), lr=q_lr) for Q in self.Qs]
        policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        entropy_temp_optimizer = torch.optim.Adam([self.entropy_temp_log], lr=temp_lr)
        replay_buffer = deque(maxlen=max_buffer_size)
        rewards = []
        step_count = 0

        start_time = time.time()
        for episode in count():
            observation = self.env.reset()
            observation = torch.tensor(observation, device=device, dtype=dtype).unsqueeze(0)
            done = torch.tensor([False], device=device, dtype=torch.bool).unsqueeze(0)
            episode_rewards = []

            for _ in range(max_traj_length):
                if RENDER:
                    self.env.render()
                step_count += 1
                action = self.select_action(observation, select_after=select_after, step_count=0, deterministic=False)
                next_observation, reward, done, _ = self.env.step(action[0])
                episode_rewards.append(float(reward))
                next_observation = torch.tensor(next_observation, device=device, dtype=dtype).unsqueeze(0)
                reward = torch.tensor([reward], device=device, dtype=dtype).unsqueeze(0)
                done = torch.tensor([done], device=device, dtype=torch.bool).unsqueeze(0)
                
                transition = Transition(observation, action, reward, next_observation, done)
                replay_buffer.append(transition)
                observation = next_observation

                if (step_count > update_after and step_count % update_every == 0):
                    q_loss, policy_loss = self.update(replay_buffer, batch_size, q_optimizers, policy_optimizer, discounting, polyack, entropy_temp_optimizer, grad_norm_clip)

                
                if save_frequency is not None:
                    if (step_count >= update_after and step_count % (timesteps // save_frequency) == 0):
                        self.save()

                if done or step_count == timesteps:
                    break
            if step_count == timesteps:
                break

            total_episode_reward = sum(episode_rewards)
            rewards.append(total_episode_reward)

            if VERBOSE:
                print(f"Episode {episode+1} (Step Count = {step_count}) | Reward = {total_episode_reward:.2f} | ", end="")

                if step_count >= update_after: #  and step_count % (timesteps // save_frequency) == 0:
                    print(f" Q Loss = {q_loss:.2f} | Policy Loss = {policy_loss:.2f} | {self.entropy_temp}")

                else:
                    print("Collecting Experience")

        self.env.close()
        print(f"\nTraining Completed in {(time.time() - start_time):.2f} seconds")
        if PLOT_REWARDS:
            plt.plot(rewards)
            plt.title(f"Training {self.__class__.__name__} on {self.env_name}")
            plt.xlabel("Episodes")
            plt.ylabel("Rewards")
            plt.savefig(f"./plots/{self.__class__.__name__}_{self.env_name}_reward_plot.png")


    def save(self, path=None):
        if path is None:
            path = f"./models/{self.__class__.__name__}_{self.env_name}.pt"
        torch.save({"q1_state_dict": self.Qs[0].state_dict(), "q2_state_dict": self.Qs[1].state_dict(), "policy_state_dict": self.policy.state_dict()}, path)
        print(f"\nSaved model parameters to {path}")

    def load(self, path=None):
        if path is None:
            path = f"./models/{self.__class__.__name__}_{self.env_name}.pt"
        checkpoint = torch.load(path)
        self.Qs[0].load_state_dict(checkpoint["q1_state_dict"])
        self.Qs[1].load_state_dict(checkpoint["q2_state_dict"])
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"\nLoaded model parameters from {path}")

    def eval(self, episodes, RENDER=False, deterministic=True):
        print(f"\nEvaluating model for {episodes} episodes ...\n")

        start_time = time.time()
        self.policy.eval()
        rewards = []

        for episode in range(episodes):
            observation = self.env.reset()
            observation = torch.tensor(observation, device=device, dtype=dtype)
            done = False
            episode_rewards = []

            while not done:
                if RENDER:
                    self.env.render()

                action = self.select_action(observation, deterministic=deterministic)
                next_observation, reward, done, _ = self.env.step(action.detach())
                episode_rewards.append(float(reward))
                next_observation = torch.tensor(next_observation, device=device, dtype=dtype)
                observation = next_observation

            total_episode_reward = sum(episode_rewards)
            rewards.append(total_episode_reward)
            print(
                f"Episode {episode+1}: Total Episode Reward = {total_episode_reward:.2f}"
            )
            rewards.append(total_episode_reward)

        self.env.close()
        print(f"\nAverage Reward for an episode = {np.mean(rewards):.2f}")
        print(f"Evaluation Completed in {(time.time() - start_time):.2f} seconds")

if __name__ == "__main__":
    from pybullet_envs import bullet
    env = bullet.racecarGymEnv.RacecarGymEnv(renders=False, isDiscrete=False)

    model = SAC(env)
    model.train(VERBOSE=True, PLOT_REWARDS=True, save_frequency=10)
    model.save()
    model.eval(10, RENDER=True)