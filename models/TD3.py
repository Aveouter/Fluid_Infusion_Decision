import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import random
from collections import deque

# Actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

# Critic
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = torch.relu(self.l1(torch.cat([state, action], 1)))
        q = torch.relu(self.l2(q))
        return self.l3(q)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.buffer = deque(maxlen=int(max_size))

    def add(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))
        return state, action, next_state, reward, done

# TD3 Agent
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(state_dim, action_dim).cuda()
        self.critic2 = Critic(state_dim, action_dim).cuda()
        self.critic1_target = Critic(state_dim, action_dim).cuda()
        self.critic2_target = Critic(state_dim, action_dim).cuda()
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=1e-3)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=1e-3)

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        for it in range(iterations):
            state, action, next_state, reward, done = replay_buffer.sample(batch_size)

            state = torch.FloatTensor(state).cuda()
            action = torch.FloatTensor(action).cuda()
            next_state = torch.FloatTensor(next_state).cuda()
            reward = torch.FloatTensor(reward).cuda()
            done = torch.FloatTensor(done).cuda()

            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q value
            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q2 = self.critic2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * discount * target_Q

            # Optimize Critic1
            current_Q1 = self.critic1(state, action)
            loss_Q1 = nn.MSELoss()(current_Q1, target_Q.detach())
            self.critic1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic1_optimizer.step()

            # Optimize Critic2
            current_Q2 = self.critic2(state, action)
            loss_Q2 = nn.MSELoss()(current_Q2, target_Q.detach())
            self.critic2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic2_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic1(state, self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Main training loop
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    td3 = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    episodes = 100
    exploration_noise = 0.1
    batch_size = 100

    for episode in range(episodes):
        state, done = env.reset(), False
        episode_reward = 0
        while not done:
            action = td3.select_action(np.array(state))
            action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(-max_action, max_action)

            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, float(done))

            state = next_state
            episode_reward += reward

            if len(replay_buffer.buffer) > batch_size:
                td3.train(replay_buffer, iterations=1, batch_size=batch_size)

        print(f"Episode: {episode}, Reward: {episode_reward:.2f}")
