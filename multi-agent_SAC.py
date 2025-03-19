import socket
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from collections import deque

# Constants
HOST = '127.0.0.1'
PORT = 5000
CBR_TARGET = 0.65
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 100000
TAU = 0.005  # Target network update rate

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SAC-specific constants
POWER_BINS = [5, 15, 25, 30]
BEACON_BINS = [1, 5, 10, 20]
CBR_BINS = [0.0, 0.3, 0.6, 1.0]

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

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

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
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).float()
        self.critic1 = Critic(state_dim, action_dim).float()
        self.critic2 = Critic(state_dim, action_dim).float()
        self.target_critic1 = Critic(state_dim, action_dim).float()
        self.target_critic2 = Critic(state_dim, action_dim).float()
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LEARNING_RATE)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.gamma = DISCOUNT_FACTOR
        self.tau = TAU
        self.entropy_coef = 0.2  # Regularization term

        self.action_map = {0: "increase beacon rate", 1: "decrease beacon rate", 
                           2: "increase power transmission", 3: "decrease power transmission"}

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.actor(state_tensor)
        prob_dist = torch.softmax(logits, dim=-1)
        action_int = torch.multinomial(prob_dist, 1).item()
        action_str = self.action_map[action_int]
        logger.debug(f"Selected action: {action_str}")
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

        next_action = self.actor(next_states)
        next_q1 = self.target_critic1(next_states, next_action)
        next_q2 = self.target_critic2(next_states, next_action)
        next_q = torch.min(next_q1, next_q2)
        target_q = rewards + self.gamma * (1 - dones) * next_q

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = torch.mean((q1 - target_q) ** 2)
        critic2_loss = torch.mean((q2 - target_q) ** 2)

        action = self.actor(states)
        q1 = self.critic1(states, action)
        q2 = self.critic2(states, action)
        min_q = torch.min(q1, q2)
        actor_loss = -torch.mean(min_q - self.entropy_coef * action)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward(retain_graph=True)
        self.critic2_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)

    def _soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

class SACServer:
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((HOST, PORT))
        self.server.listen(1)
        self.agent = SAC(4, 1)  # State dimensions: 4 (power, beacon, cbr, snr), Action dimension: 1

    def handle_client(self, conn):
        while True:
            try:
                data = conn.recv(1024)
                if not data:
                    logger.info("No data received, closing connection.")
                    break
                logger.debug(f"Received data: {data.decode()}")
                
                state = json.loads(data.decode())
                current_power = state['transmissionPower']
                current_beacon = state['beaconRate']
                current_cbr = state['CBR']
                current_snr = state['SNR']
                
                # Select action
                action = self.agent.select_action([current_power, current_beacon, current_cbr, current_snr])

                # Implement action
                if action == "increase beacon rate":
                    new_beacon = min(20, current_beacon + 1)
                    new_power = current_power
                elif action == "decrease beacon rate":
                    new_beacon = max(1, current_beacon - 1)
                    new_power = current_power
                elif action == "increase power transmission":
                    new_power = min(30, current_power + 1)
                    new_beacon = current_beacon
                elif action == "decrease power transmission":
                    new_power = max(5, current_power - 1)
                    new_beacon = current_beacon

                # Reward calculation (use CBR and SNR for reward)
                reward = -abs(current_cbr - CBR_TARGET) - abs(current_snr - CBR_TARGET)

                # Store experience and update SAC
                self.agent.replay_buffer.push(
                    [current_power, current_beacon, current_cbr, current_snr],
                    action,
                    reward,
                    [new_power, new_beacon, current_cbr, current_snr],
                    done=False
                )
                self.agent.update()

                # Send action and MCS (without using it for learning)
                response = {
                    'transmissionPower': new_power,
                    'beaconRate': new_beacon,
                    'MCS': state.get('MCS', 'N/A'),  # Return MCS as received
                }
                logger.debug(f"Sent response: {response}")
                conn.send(json.dumps(response).encode())
            except Exception as e:
                logger.error(f"Error while handling client data: {e}")
                break

    def start(self):
        try:
            while True:
                logger.info(f"Server listening on {HOST}:{PORT}")
                conn, addr = self.server.accept()
                logger.info(f"Connected to {addr}")
                self.handle_client(conn)
                conn.close()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user, shutting down...")
            self.server.close()
        except Exception as e:
            logger.error(f"Error while running server: {e}")
            self.server.close()

if __name__ == "__main__":
    server = SACServer()
    server.start()
