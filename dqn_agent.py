import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# 1. Define the Neural Network Architecture
class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        # Input layer takes the 4 state variables from your gym env
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        # Output layer predicts rewards for the 2 actions (Red vs Green)
        self.out = nn.Linear(128, action_dim)
        
    def forward(self, x):
        # Using Rectified Linear Units (ReLU) for non-linear processing
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# 2. Define the Agent Brain and Memory Replay Logic
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters for Deep Q-Learning
        self.gamma = 0.95            # Discount factor (importance of future rewards)
        self.epsilon = 1.0           # Starting exploration rate (100% random guesses at first)
        self.epsilon_min = 0.01      # Minimum exploration rate
        self.epsilon_decay = 0.995   # How fast the agent stops guessing and starts trust-building
        self.lr = 0.001              # Adam optimizer learning rate
        self.batch_size = 64
        self.memory = deque(maxlen=50000) # Memory replay buffer size
        
        # Policy and Target Networks (Standard Double/Stable DQN setup)
        self.policy_net = DQNNetwork(state_dim, action_dim)
        self.target_net = DQNNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        # Epsilon-Greedy selection: Explore random choices vs Exploit known strategies
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Convert state vector to a PyTorch tensor for prediction
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state_t).argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        # Remember what happened so we can review experience logs later
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        # Only start learning once the memory has enough data samples
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a random batch of past events from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Cast tracking parameters into PyTorch Tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Get expected Q-values from our active policy network
        current_q = self.policy_net(states).gather(1, actions)
        
        # Calculate target Q-values using the stable target network
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
            
        # Run Backpropagation to update weights based on Mean Squared Error Loss
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Slowly decay exploration as the agent gains more experience
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self):
        # Sync weights from training loop
        self.target_net.load_state_dict(self.policy_net.state_dict())
