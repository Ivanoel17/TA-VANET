import sys
import socket  # Networking interface
import threading  # Thread-based parallelism
import random  # Random number generation
import numpy as np  # Numerical operations on arrays
import json  # JSON encoding and decoding
import gym  # OpenAI Gym for creating RL environments
from gym import spaces  # Spaces module to define action and observation spaces
import os  # Operating system interfaces
from mn_wifi.net import Mininet_wifi  # Mininet-WiFi network creation
from mn_wifi.link import wmediumd  # Link type for Mininet-WiFi
from mn_wifi.wmediumdConnector import interference  # Interference model for Mininet-WiFi
 
# Ensure SUMO_HOME is set, which is needed to control the SUMO traffic simulation tool
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = "/usr/share/sumo"  # Set SUMO_HOME to a default path if not already set
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))  # Add SUMO tools to the system path
 
import traci  # Import traci to control SUMO from Python
 
# Define beta value (used in vehicle density adjustment formula)
beta = 3  # You can adjust this value to control the sensitivity of vehicle density to power changes
 
# Define a custom OpenAI Gym environment for Q-Learning in VANETs (Vehicular Ad-Hoc Networks)
class VANETEnv(gym.Env):
    def __init__(self, car_logs, alpha, gamma, epsilon, pi_b, pi_p_prime, pi_p2, kb):
        super(VANETEnv, self).__init__()  # Initialize the parent class (gym.Env)
 
        # Define the action space and observation space for the RL agent
        self.action_space = spaces.Discrete(4)  # Actions: increase/decrease beacon rate or power transmission
        self.observation_space = spaces.Tuple((
            spaces.Discrete(20),  # Beacon rate
            spaces.Discrete(30),  # Power transmission
            spaces.Discrete(100)  # Vehicle density
        ))
 
        # Q-Learning parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = 1000
 
        # Weights for the reward function
        self.pi_b = pi_b
        self.pi_p_prime = pi_p_prime
        self.pi_p2 = pi_p2
        self.kb = kb
 
        # Initialize the Q-table with zeros
        self.Q = np.zeros((20, 30, 4))
 
        # Store car logs for initializing states
        self.car_logs = car_logs
        self.current_state = None
 
    def reset(self):
        log = random.choice(self.car_logs)
        self.current_state = (log['beacon_rate'], log['power_transmission'], log['vehicle_density'])
        return self.current_state
 
    def step(self, action):
        b, p, rho = self.current_state
        delta_b, delta_p = self.action_to_delta(action)
 
        new_b = max(1, min(b + delta_b, 20))
        new_p = max(1, min(p + delta_p, 30))
 
        # Adjust vehicle density based on the new power transmission
        rho = rho * ((p / new_p) ** (1 / beta))
        self.current_state = (new_b, new_p, rho)
 
        # Retrieve CBR value from logs
        cbr = self.current_state[2]  # Assuming the CBR is stored in the vehicle density for simplicity
        reward = self.reward_function(cbr, new_p, p)
 
        done = True  # End the episode after one action in this simplified case
        return self.current_state, reward, done, {}
 
    def action_to_delta(self, action):
        # Action map: 0 = increase beacon, 1 = decrease beacon, 2 = increase power, 3 = decrease power
        if action == 0:
            return 1, 0
        elif action == 1:
            return -1, 0
        elif action == 2:
            return 0, 3
        elif action == 3:
            return 0, -3
 
    def reward_function(self, cbr, p, p_prime):
        def H(x):
            return 1 if x >= 0 else 0
 
        def g(x, k):
            return x * (H(x) - 2 * H(x - k))
 
        # Reward function based on CBR and power differences
        reward = (self.pi_b * g(cbr, self.kb)) - (self.pi_p_prime * abs(p - p_prime)) - (self.pi_p2 * g(p_prime, 20))
        return reward
 
    def select_action(self, state):
        b_idx, p_idx, _ = state[0] - 1, state[1] - 1, state[2]
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.Q[b_idx, p_idx])
 
    def optimize_cbr(self, car_log):
        self.current_state = (car_log['beacon_rate'], car_log['power_transmission'], car_log['vehicle_density'])
        best_action = self.select_action(self.current_state)
        delta_b, delta_p = self.action_to_delta(best_action)
        new_beacon_rate = max(1, min(self.current_state[0] + delta_b, 20))
        new_power_transmission = max(1, min(self.current_state[1] + delta_p, 30))
        return new_beacon_rate, new_power_transmission
 
    def log_action(self, car_id, new_beacon_rate, new_power_transmission):
        print(f"[INFO] Car {car_id}: Adjusted Beacon Rate to {new_beacon_rate}, Power Transmission to {new_power_transmission}")
 
# Define a server class to handle incoming car logs and apply Q-learning
class Server:
    def __init__(self, host='localhost', port=12345, alpha=0.1, gamma=0.9, epsilon=0.1, pi_b=75, pi_p_prime=5, pi_p2=20, kb=0.6, net=None):
        self.host = host
        self.port = port
        self.car_logs = []
        self.net = net  # Initialize with the Mininet network object
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.pi_b = pi_b
        self.pi_p_prime = pi_p_prime
        self.pi_p2 = pi_p2
        self.kb = kb
 
    def start_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}...")
 
        while True:
            client_socket, client_address = server_socket.accept()
            print(f"Connection from {client_address}")
            threading.Thread(target=self.handle_client, args=(client_socket,)).start()
 
    def handle_client(self, client_socket):
        try:
            data = client_socket.recv(4096)
            if not data:
                return
            decoded_data = json.loads(data.decode('utf-8'))
            self.car_logs.append(decoded_data)
            self.process_car_log(decoded_data)
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()
 
    def process_car_log(self, car_log):
        env = VANETEnv([car_log], self.alpha, self.gamma, self.epsilon, self.pi_b, self.pi_p_prime, self.pi_p2, self.kb)
        new_beacon_rate, new_power_transmission = env.optimize_cbr(car_log)
        env.log_action(car_log['car_id'], new_beacon_rate, new_power_transmission)
        self.apply_optimized_settings(car_log['car_id'], new_beacon_rate, new_power_transmission)
 
    def apply_optimized_settings(self, car_id, beacon_rate, power_transmission):
        if not self.net:
            print("[ERROR] Network object is not initialized in apply_optimized_settings.")
            return
 
        try:
            car = self.net.getNodeByName(car_id)
            if car:
                car.setParam("beacon_rate", beacon_rate)
                car.setParam("txpower", power_transmission)
                traci.vehicle.setParameter(car_id, "beaconRate", str(beacon_rate))
                traci.vehicle.setParameter(car_id, "txpower", str(power_transmission))
                print(f"Applied settings to {car_id}: Beacon Rate = {beacon_rate}, Power Transmission = {power_transmission}")
            else:
                print(f"[ERROR] Car with ID {car_id} not found in the network.")
        except Exception as e:
            print(f"Error applying settings to {car_id}: {e}")
 
def main():
    # Initialize the Mininet-WiFi network object
    net = Mininet_wifi(link=wmediumd, wmediumd_mode=interference)
 
    # Start the network
    net.start()
 
    # Parameter configuration for the server
    host = 'localhost'
    port = 9999
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    pi_b = 75
    pi_p_prime = 5
    pi_p2 = 20
    kb = 0.6
 
    # Create and start the server, passing the `net` object
    server = Server(
        host=host,
        port=port,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        pi_b=pi_b,
        pi_p_prime=pi_p_prime,
        pi_p2=pi_p2,
        kb=kb,
        net=net  # Pass the Mininet-WiFi network object
    )
    server.start_server()
 
    # Cleanup network when done
    net.stop()
 
if __name__ == "__main__":
    main()

