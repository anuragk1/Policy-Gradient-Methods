import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from itertools import count
import copy
import random
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

timesteps = 10000
q_hidden = 64
policy_hidden = 64
q_lr = 3 * 1e-3
policy_lr = 3 * 1e-3
gamma = 0.95
batch_size = 128
polyak = 0.995
noise_std = 0.2
max_buffer_size = timesteps // 2
update_after = timesteps // 10
update_interval = 1
select_after = timesteps // 5
max_traj_length = 1000

Transition = namedtuple("Transition", ("observation", "action", "reward", "next_observation", "done"))

class Actor(nn.Module):
    def __init__(self, s_dim, hidden_dim, a_dim, lim):
        super(Actor, self).__init__()
        self.lim = lim
        self.model = (
            nn.Sequential(
                nn.Linear(s_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, a_dim),
                nn.Tanh(),
            )
            .to(device)
            .to(dtype)
        )

    def forward(self, a):
        return self.model(a) * self.lim

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
                nn.Linear(hidden_dim, 1),
            )
            .to(device)
            .to(dtype)
        )

    def forward(self, s, a):
        x = torch.cat([s.view(-1, self.s_dim), a.view(-1, self.a_dim)], dim=1)
        return self.model(x)

class DDPG:
    def __init__(self, env, q_hidden_dim=q_hidden, policy_hidden_dim=policy_hidden):
        self.env = env
        if self.env.unwrapped.spec is not None:
            self.env_name = self.env.unwrapped.spec.id
        else:
            self.env_name = self.env.unwrapped.__class__.__name__

        self.action_shape = env.action_space.shape
        self.observation_shape = env.observation_space.shape
        self.action_min = env.action_space.low[0]
        self.action_max = env.action_space.high[0]

        self.Q = Critic(self.observation_shape[0], self.action_shape[0], q_hidden_dim)
        self.target_Q = copy.deepcopy(self.Q)
        for p in self.target_Q.parameters():
            p.requires_grad = False

        self.policy = Actor(self.observation_shape[0], policy_hidden_dim, self.action_shape[0], self.action_max,)
        self.target_policy = copy.deepcopy(self.policy)
        for p in self.target_policy.parameters():
            p.requires_grad = False

    def select_action(self, observations, noise=0, select_after=0, step_count=0):
        if step_count < select_after:
            action = torch.tensor(self.env.action_space.sample(), device=device, dtype=dtype).unsqueeze(0)
        else:
            with torch.no_grad():
                noisy_action = self.policy(observations) + noise * torch.randn(size=self.action_shape, device=device, dtype=dtype)
                action = torch.clamp(noisy_action, self.action_min, self.action_max)

        return action

    def update(self, replay_buffer, batch_size, q_optimizer, policy_optimizer, gamma, polyak):
        sample = Transition(*[torch.cat(i) for i in [*zip(*random.sample(replay_buffer, batch_size))]])

        with torch.no_grad():
            target_q_vals = sample.reward + gamma * self.target_Q(sample.next_observation, self.target_policy(sample.next_observation)) * (~sample.done)

        q_vals = self.Q(sample.observation, sample.action)

        q_optimizer.zero_grad()
        q_loss = nn.MSELoss()(target_q_vals, q_vals)
        q_loss.backward()
        q_optimizer.step()

        for p in self.Q.parameters():
            p.requires_grad = False

        policy_optimizer.zero_grad()
        policy_loss = -1 * torch.mean(self.Q(sample.observation, self.policy(sample.observation)))
        policy_loss.backward()
        policy_optimizer.step()

        for p in self.Q.parameters():
            p.requires_grad = True

        with torch.no_grad():
            for p_target, p in zip(self.target_Q.parameters(), self.Q.parameters()):
                p_target.data = (
                    polyak * p_target.data + (1 - polyak) * p.data
                )
            for p_target, p in zip(
                self.target_policy.parameters(), self.policy.parameters()
            ):
                p_target.data = (
                    polyak * p_target.data + (1 - polyak) * p.data
                )

        return q_loss.item(), policy_loss.item()

    def train(self, timesteps=timesteps, q_lr=q_lr, p_lr=policy_lr, gamma=gamma, batch_size=batch_size, polyak_const=polyak, noise=noise_std, max_buffer_size=max_buffer_size,
     update_after=update_after, update_every=update_interval, select_after=select_after, max_traj_length=max_traj_length, SAVE_FREQUENCY=None, RENDER=False, VERBOSE=False, PLOT_REWARDS=False,):
        hp = locals()
        print(f"\nTraining model on {self.env_name} | " f"Observation Space: {self.env.observation_space} | " f"Action Space: {self.env.action_space}")

        start_time = time.time()
        self.Q.train()
        self.policy.train()
        q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=q_lr)
        policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=p_lr)
        replay_buffer = deque(maxlen=max_buffer_size)
        rewards = []
        step_count = 0

        for episode in count():
            observation = self.env.reset()
            observation = torch.tensor(observation, device=device, dtype=dtype).unsqueeze(0)
            done = torch.tensor([False], device=device, dtype=torch.bool).unsqueeze(0)
            episode_rewards = []

            for _ in range(max_traj_length):
                step_count += 1
                if RENDER:
                    self.env.render()

                action = self.select_action(observation, noise, select_after, step_count)
                # print(action.item())
                action = action.cpu().numpy()
                # print(type(action))
                next_observation, reward, done, _ = self.env.step(action[0])  # for racecarenv self.env.step(action.item()[0][0])
                action = torch.tensor(action, device=device, dtype=dtype)
                episode_rewards.append(float(reward))
                next_observation = torch.tensor(next_observation, device=device, dtype=dtype).unsqueeze(0)
                reward = torch.tensor([reward], device=device, dtype=dtype).unsqueeze(0)
                done = torch.tensor([done], device=device, dtype=torch.bool).unsqueeze(0)

                transition = Transition(
                    observation, action, reward, next_observation, done
                )
                replay_buffer.append(transition)
                observation = next_observation

                if step_count >= update_after and step_count % update_every == 0:
                    # print(replay_buffer)
                    q_loss, policy_loss = self.update(replay_buffer, batch_size, q_optimizer, policy_optimizer, gamma, polyak_const)

                if SAVE_FREQUENCY is not None:
                    if (step_count >= update_after and step_count % (timesteps // SAVE_FREQUENCY) == 0):
                        self.save()

                if done or step_count == timesteps:
                    break

            if step_count == timesteps:
                break

            total_episode_reward = sum(episode_rewards)
            rewards.append(total_episode_reward)

            if VERBOSE:
                print(f"Episode {episode+1}: Step Count = {step_count} | Reward = {total_episode_reward:.2f} | ", end="",)
                if step_count >= update_after:
                    print(f" Q Loss = {q_loss:.2f} | Policy Loss = {policy_loss:.2f}")

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
        torch.save({"q_state_dict": self.Q.state_dict(), "policy_state_dict": self.policy.state_dict()}, path)
        print(f"\nSaved model parameters to {path}")

    def load(self, path=None):
        if path is None:
            path = f"./models/{self.__class__.__name__}_{self.env_name}.pt"
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint["q_state_dict"])
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"\nLoaded model parameters from {path}")

    def eval(self, episodes, RENDER=False):
        print(f"\nEvaluating model for {episodes} episodes ...\n")
        start_time = time.time()
        self.Q.eval()
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

                action = self.select_action(observation)
                action = action.cpu().numpy()
                next_observation, reward, done, _ = self.env.step(action)#self.env.step(action.detach())
                episode_rewards.append(float(reward))
                next_observation = torch.tensor(next_observation, device=device, dtype=dtype)
                observation = next_observation

            total_episode_reward = sum(episode_rewards)
            rewards.append(total_episode_reward)

            print(f"Episode {episode+1}: Total Episode Reward = {total_episode_reward:.2f}")
            rewards.append(total_episode_reward)

        self.env.close()
        print(f"\nAverage Reward for an episode = {np.mean(rewards):.2f}")
        print(f"Evaluation Completed in {(time.time() - start_time):.2f} seconds")

if __name__ == "__main__":
    import gym
    env = gym.make("LunarLanderContinuous-v2")

    # from pybullet_envs import bullet
    # env = bullet.racecarGymEnv.RacecarGymEnv(renders=False, isDiscrete=False)

    model = DDPG(env)
    model.train(VERBOSE=True, PLOT_REWARDS=True, SAVE_FREQUENCY=10)
    model.save()
    # model.load()
    model.eval(50, RENDER=False)
