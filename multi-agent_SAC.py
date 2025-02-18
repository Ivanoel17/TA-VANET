import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces, Env
from maddpg import MADDPG
from collections import namedtuple
 
# Ensure SUMO_HOME is set for the SUMO traffic simulation tool
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = "/usr/share/sumo"
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
 
import traci  # TraCI control of SUMO
 
# Define the VANET Environment for Cooperative Multi-Agent Learning
class VANETEnv(Env):
    def __init__(self, car_logs, num_agents=3, ieee_standard='802.11p', bandwidth=20):
        super(VANETEnv, self).__init__()
        
        self.num_agents = num_agents
        self.ieee_standard = ieee_standard
        self.bandwidth = bandwidth
        
        # Action space: [Power, Data rate, Beacon rate] for each agent
        self.action_space = [spaces.Box(low=np.array([-3, -3, -3]), high=np.array([3, 3, 3]), dtype=np.float32) for _ in range(self.num_agents)]
        # Observation space: [Power, Data rate, Beacon rate, Vehicle density, SNR] for each agent
        self.observation_space = [spaces.Box(low=np.array([1, 1, 1, 0.001, 0]), high=np.array([30, 30, 20, 0.8, 100]), dtype=np.float32) for _ in range(self.num_agents)]
        
        self.car_logs = car_logs  # Logs to initialize the state
        self.state = None  # Current state of the environment
        self.CBR_target = 0.6  # Target CBR value
        self.SNR_target = 15.0  # Target SNR value
        
    def reset(self):
        # Randomly initialize the state for each agent
        self.state = [self._get_initial_state() for _ in range(self.num_agents)]
        return self.state
    
    def step(self, actions):
        rewards = []
        next_states = []
        
        for i, action in enumerate(actions):
            power, data_rate, beacon_rate = self.state[i][:3]
            vehicle_density = self.state[i][3]
            
            # Apply action adjustments
            new_power = np.clip(power + action[0], 1, 30)
            new_data_rate = np.clip(data_rate + action[1], 1, 30)
            new_beacon_rate = np.clip(beacon_rate + action[2], 1, 20)
            
            # Calculate new SNR considering the impact of neighboring vehicles
            new_snr = self.calculate_snr(new_power, new_data_rate, vehicle_density)
            
            # Calculate CBR based on updated parameters
            new_cbr = self.calculate_cbr(new_power, new_data_rate, new_beacon_rate, vehicle_density)
            
            # Update the state
            next_state = np.array([new_power, new_data_rate, new_beacon_rate, vehicle_density, new_snr])
            next_states.append(next_state)
            
            # Calculate the reward based on CBR and SNR
            reward = self.reward_function(new_cbr, new_snr)
            rewards.append(reward)
        
        # Set the new state
        self.state = next_states
        
        # In a simple scenario, assume the episode ends after each step
        done = True  
        return next_states, rewards, [done] * self.num_agents, {}
 
    def _get_initial_state(self):
        # Initialize the state from car logs
        log = np.random.choice(self.car_logs)
        return np.array([log['power_transmission'], log['data_rate'], log['beacon_rate'], log['vehicle_density'], log['snr']])
    
    def calculate_snr(self, power, data_rate, vehicle_density):
        # Placeholder: SNR calculation considering interference from neighbors
        interference = vehicle_density * np.random.uniform(0.5, 1.5)  # Simulate interference effect
        snr = power / (interference + 1e-9)  # Simple SNR model
        return snr
    
    def calculate_cbr(self, power, data_rate, beacon_rate, vehicle_density):
        # Placeholder: CBR calculation considering current state
        return np.random.uniform(0.4, 0.8)  # Simulated CBR for demonstration
    
    def reward_function(self, cbr, snr):
        # Reward function that penalizes deviations from target CBR and SNR
        cbr_deviation = abs(cbr - self.CBR_target)
        snr_deviation = abs(snr - self.SNR_target)
        
        # Combine penalties for both CBR and SNR
        reward = - (cbr_deviation + snr_deviation)
        return reward
 
# Define the MADDPG Agent and Training Process
class MultiAgentTrainer:
    def __init__(self, env, num_agents=3, gamma=0.95, tau=0.01, lr=0.01):
        self.env = env
        self.maddpg = MADDPG(num_agents, env.observation_space, env.action_space, gamma=gamma, tau=tau, lr=lr)
        
    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = np.zeros(self.env.num_agents)
            
            done = False
            while not done:
                actions = self.maddpg.select_action(state)
                next_state, rewards, done, _ = self.env.step(actions)
                
                self.maddpg.memory.push(state, actions, rewards, next_state, done)
                self.maddpg.update()
                
                state = next_state
                total_reward += rewards
            
            print(f"Episode {episode} - Total Reward: {total_reward}")
            if episode % 100 == 0:
                self.maddpg.save_checkpoint()
 
    def evaluate(self):
        state = self.env.reset()
        done = False
        total_reward = np.zeros(self.env.num_agents)
        
        while not done:
            actions = self.maddpg.select_action(state, noise=False)
            next_state, rewards, done, _ = self.env.step(actions)
            state = next_state
            total_reward += rewards
        
        print(f"Evaluation - Total Reward: {total_reward}")
 
# Main function to train and evaluate the MADDPG agents
def main():
    car_logs = [{'power_transmission': 10, 'data_rate': 20, 'beacon_rate': 5, 'vehicle_density': 0.5, 'snr': 20}] * 100  # Dummy logs
    env = VANETEnv(car_logs)
    trainer = MultiAgentTrainer(env)
    trainer.train()
    trainer.evaluate()
 
if __name__ == "__main__":
    main()

