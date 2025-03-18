import socket
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.DEBUG,  # Set the logging level to DEBUG
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
HOST = '127.0.0.1'
PORT = 5000
CBR_TARGET = 0.65
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
EPSILON = 0.1  # Exploration rate (though SAC doesn't use epsilon like Q-learning)

# Simplified state discretization
POWER_BINS = [5, 15, 25, 30]
BEACON_BINS = [1, 5, 10, 20]
CBR_BINS = [0.0, 0.3, 0.6, 1.0]

# SAC-specific constants
BATCH_SIZE = 64
BUFFER_SIZE = 100000
TAU = 0.005  # Target network update rate

# Experience Replay Buffer for SAC
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        idx = np.random.choice(len(self.buffer), BATCH_SIZE)
        batch = [self.buffer[i] for i in idx]
        return zip(*batch)

    def size(self):
        return len(self.buffer)

# Define Actor and Critic Networks for SAC
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)  # Output size is 4 for 4 discrete actions

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)  # Output raw scores (logits)
        return logits  # No activation function (softmax will be applied during action selection)

class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SAC:
    def __init__(self, state_dim, action_dim, gamma=0.99, learning_rate=LEARNING_RATE):
        self.actor = Actor(state_dim, action_dim).float()
        self.critic1 = Critic(state_dim, action_dim).float()
        self.critic2 = Critic(state_dim, action_dim).float()
        self.target_critic1 = Critic(state_dim, action_dim).float()
        self.target_critic2 = Critic(state_dim, action_dim).float()
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.gamma = gamma
        self.tau = TAU
        self.entropy_coef = 0.2  # Regularization term

        self.action_map = {
            0: "increase beacon rate",
            1: "decrease beacon rate",
            2: "increase power transmission",
            3: "decrease power transmission"
        }
    
    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.actor(state_tensor)  # Get raw logits
        prob_dist = torch.softmax(logits, dim=-1)  # Apply softmax to get probabilities
        logger.debug(f"Logits: {logits}, Probabilities: {prob_dist}")  # Log logits and probabilities
        action_int = torch.multinomial(prob_dist, 1).item()  # Sample action based on probabilities
        action_str = self.action_map[action_int]  # Map the integer action to the corresponding string action
        return action_str


    def update(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            logger.debug("Replay buffer too small to update.")
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        actions = np.array(actions)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Critic loss
        next_action = self.actor(next_states)
        next_q1 = self.target_critic1(next_states, next_action)
        next_q2 = self.target_critic2(next_states, next_action)
        next_q = torch.min(next_q1, next_q2)
        target_q = rewards + self.gamma * (1 - dones) * next_q

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = torch.mean((q1 - target_q) ** 2)
        critic2_loss = torch.mean((q2 - target_q) ** 2)

        # Actor loss (maximize the Q-values)
        action = self.actor(states)
        q1 = self.critic1(states, action)
        q2 = self.critic2(states, action)
        min_q = torch.min(q1, q2)
        actor_loss = -torch.mean(min_q - self.entropy_coef * action)
        
        logger.debug(f"Critic1 Loss: {critic1_loss.item()}, Critic2 Loss: {critic2_loss.item()}, Actor Loss: {actor_loss.item()}")

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward(retain_graph=True)
        self.critic2_optimizer.step()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft target network update
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)

        logger.debug("Network update complete.")
        
    def _soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

# QLearningServer for multi-agent SAC environment communication
class SACServer:
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((HOST, PORT))
        self.server.listen(1)
        self.agent = SAC(3, 1)  # Example dimensions: 3 for state (power, beacon, cbr), 1 for action (increase or decrease)
    
    def handle_client(self, conn):
        while True:
            data = conn.recv(1024)
            if not data:
                logger.info("No data received, closing connection.")
                break
            try:
                state = json.loads(data.decode())
                logger.debug(f"Received state from client: {state}")
                current_power = state['power_transmission']
                current_beacon = state['beacon_rate']
                current_cbr = state['channel_busy_rate']
            
                # Select action
                action = self.agent.select_action([current_power, current_beacon, current_cbr])
                logger.debug(f"Selected Action: {action}")
            
                # Implement action
                if action == "increase beacon rate":
                    new_beacon = min(20, current_beacon + 1)
                    new_power = current_power
                elif action == "decrease beacon rate":
                    new_beacon = max(1, current_beacon - 1)
                    new_power = current_power
                elif action == "increase power tx":
                    new_power = min(30, current_power + 1)
                    new_beacon = current_beacon
                elif action == "decrease power tx":
                    new_power = max(5, current_power - 1)
                    new_beacon = current_beacon
            
                # Reward calculation
                reward = -abs(current_cbr - CBR_TARGET)
            
                # Store experience and update SAC
                self.agent.replay_buffer.push(
                    [current_power, current_beacon, current_cbr],
                    action,
                    reward,
                    [new_power, new_beacon, current_cbr],
                    done=False
                )
                self.agent.update()

                # Send action to client
                response = {
                    'power_transmission': new_power,
                    'beacon_rate': new_beacon,
                    'reward': reward
                }
                conn.send(json.dumps(response).encode())
                logger.debug(f"Sent response to client: {response}")
            except Exception as e:
                logger.error(f"Error: {e}")
                break

    def start(self):
        while True:
            logger.info(f"Server listening on {HOST}:{PORT}")
            conn, addr = self.server.accept()
            logger.info(f"Connected: {addr}")
            self.handle_client(conn)
            conn.close()

if __name__ == "__main__":
    server = SACServer()
    server.start()
