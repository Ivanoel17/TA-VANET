import socket  # Networking interface
import threading  # Thread-based parallelism
import random  # Random number generation
import numpy as np  # Numerical operations on arrays
import json  # JSON encoding and decoding
import gym  # OpenAI Gym for creating RL environments
from gym import spaces  # Spaces module to define action and observation spaces
 
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
 
        # Q-Learning hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = 1000
        self.num_steps = 10
        self.convergence_threshold = 0.01
 
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
 
    def select_action(self):
        b_idx, p_idx, _ = self.current_state[0] - 1, self.current_state[1] - 1, self.current_state[2]
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.Q[b_idx, p_idx])
        
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

    def compute_neighbors(self, rho, p, new_p, beta):
        return rho * ((p / new_p) ** (1 / beta))
    
    def reward_function(self, cbr, p, p_prime):
        def H(x):
            return 1 if x >= 0 else 0
 
        def g(x, k):
            return x * (H(x) - 2 * H(x - k))
 
        # Reward function based on CBR and power differences
        reward = (self.pi_b * g(cbr, self.kb)) - (self.pi_p_prime * abs(p - p_prime)) - (self.pi_p2 * g(p_prime, 20))
        return reward
    
    def update_q_table(self, action, reward, next_state):
         max_future_q = np.max(self.Q[next_state])
         current_q = self.Q[self.current_state][action]
         self.Q[self.current_state][action] = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
 
    def step(self):
        action = self.select_action()
        b, p, rho = self.current_state
        
        delta_b, delta_p = self.action_to_delta(action)
 
        new_b = max(1, min(b + delta_b, 20))
        new_p = max(1, min(p + delta_p, 30))
 
        # Adjust vehicle density based on the new power transmission
        new_rho = self.compute_neighbors(rho, p, new_p, beta)
        next_state = (new_b, new_p, new_rho)
 
        # Retrieve CBR value from logs
        cbr = new_rho  # Assuming the CBR is stored in the vehicle density for simplicity
        reward = self.reward_function(cbr, new_p, p)
 
        self.update_q_table(action, reward, next_state)
        self.current_state = next_state
        return new_b, new_p
 
    def train(self, car_log):
        for episode in range(self.num_episodes):
            self.current_state = (car_log['beacon_rate'], car_log['power_transmission'], car_log['vehicle_density'])
            steps = 0
            while steps < self.num_steps:
                old_q_values = np.copy(self.Q)
                new_beacon_rate, new_transmission_power = self.step()
                steps += 1
                if np.max(np.abs(self.Q - old_q_values)) < self.convergence_threshold:
                    break  # Stop episode when Q-values converge
        return new_beacon_rate, new_transmission_power
 
    def log_action(self, car_id, new_beacon_rate, new_power_transmission):
        print(f"[INFO] Car {car_id}: Adjusted Beacon Rate to {new_beacon_rate}, Power Transmission to {new_power_transmission}")
 
# Define a server class to handle incoming car logs and apply Q-learning
class Server:
    def __init__(self, host='localhost', port=5000, alpha=0.1, gamma=0.9, epsilon=0.1, pi_b=75, pi_p_prime=5, pi_p2=20, kb=0.6):
        self.host = host
        self.port = port
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.pi_b = pi_b
        self.pi_p_prime = pi_p_prime
        self.pi_p2 = pi_p2
        self.kb = kb
 
    def start_server(self):
        """Start the Q-learning server."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}...")
 
        while True:
            client_socket, client_address = server_socket.accept()
            print(f"Connection from {client_address}")
            threading.Thread(target=self.handle_client, args=(client_socket,)).start()
 
    def handle_client(self, client_socket):
        """Handle incoming client requests."""
        while True:  # Keep the connection open
            data = client_socket.recv(4096)
            if not data:  # Client disconnected
                break
            decoded_data = json.loads(data.decode('utf-8'))
            response = self.process_car_log(decoded_data)
            client_socket.sendall(response.encode('utf-8'))
        client_socket.close()
 
    def process_car_log(self, car_log):
        """Process incoming car data and start an episode."""
        env = VANETEnv([car_log], self.alpha, self.gamma, self.epsilon, self.pi_b, self.pi_p_prime, self.pi_p2, self.kb)
        new_beacon_rate, new_power_transmission = env.train(car_log)
        env.log_action(car_log['car_id'], new_beacon_rate, new_power_transmission)
        
        # Send response back to client
        response = json.dumps({
            "car_id": car_log['car_id'],
            "beacon_rate": new_beacon_rate,
            "power_transmission": new_power_transmission
        })
        return response
 
 
def main():
    # Parameter configuration for the server
    host = 'localhost'
    port = 5000
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
    )
    server.start_server()
 
if __name__ == "__main__":
    main()
