import sys
import os
import math
import time
import subprocess
import csv
import socket
from threading import Thread
from scapy.all import sniff, Dot11
import re
from scipy.special import gamma
 
# Ensure SUMO_HOME is set correctly for accessing SUMO tools
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = "/usr/share/sumo"
 
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
 
import pytz
import datetime
import traci
from mininet.log import setLogLevel, info
from mn_wifi.cli import CLI
from mn_wifi.net import Mininet_wifi
from mn_wifi.sumo.runner import sumo
from mn_wifi.link import wmediumd, mesh
from mn_wifi.wmediumdConnector import interference
from mn_wifi.node import Car
import json
# MCS tables for different IEEE 802.11 standards, including 802.11bd
# These tables map modulation and coding scheme (MCS) indexes to modulation types and data rates.
mcs_tables = {
    '802.11a': [
        ('BPSK', 6.0), ('BPSK', 9.0),
        ('QPSK', 12.0), ('QPSK', 18.0),
        ('16-QAM', 24.0), ('16-QAM', 36.0),
        ('64-QAM', 48.0), ('64-QAM', 54.0)
    ],
    '802.11b': [
        ('DSSS', 1.0), ('DSSS', 2.0),
        ('CCK', 5.5), ('CCK', 11.0)
    ],
    '802.11g': [
        ('BPSK', 6.0), ('BPSK', 9.0),
        ('QPSK', 12.0), ('QPSK', 18.0),
        ('16-QAM', 24.0), ('16-QAM', 36.0),
        ('64-QAM', 48.0), ('64-QAM', 54.0)
    ],
    '802.11n': [
        (0, 'BPSK', 6.5, 13.5),
        (1, 'QPSK', 13.0, 27.0),
        (2, 'QPSK', 19.5, 40.5),
        (3, '16-QAM', 26.0, 54.0),
        (4, '16-QAM', 39.0, 81.0),
        (5, '64-QAM', 52.0, 108.0),
        (6, '64-QAM', 58.5, 121.5),
        (7, '64-QAM', 65.0, 135.0)
    ],
    '802.11ac': [
        (0, 'BPSK', 6.5, 13.5, 29.3, 58.5),
        (1, 'QPSK', 13.0, 27.0, 58.5, 117.0),
        (2, 'QPSK', 19.5, 40.5, 87.8, 175.5),
        (3, '16-QAM', 26.0, 54.0, 117.0, 234.0),
        (4, '16-QAM', 39.0, 81.0, 175.5, 351.0),
        (5, '64-QAM', 52.0, 108.0, 234.0, 468.0),
        (6, '64-QAM', 58.5, 121.5, 263.3, 526.5),
        (7, '64-QAM', 65.0, 135.0, 292.5, 585.0),
        (8, '256-QAM', 78.0, 162.0, 351.0, 702.0),
        (9, '256-QAM', None, None, 390.0, 780.0)
    ],
    '802.11ax': [
        (0, 'BPSK', 7.3, 15.0, 30.0, 60.0),
        (1, 'QPSK', 14.6, 30.0, 60.0, 120.0),
        (2, 'QPSK', 21.9, 45.0, 90.0, 180.0),
        (3, '16-QAM', 29.2, 60.0, 120.0, 240.0),
        (4, '16-QAM', 43.9, 90.0, 180.0, 360.0),
        (5, '64-QAM', 58.5, 120.0, 240.0, 480.0),
        (6, '64-QAM', 65.8, 135.0, 270.0, 540.0),
        (7, '64-QAM', 73.1, 150.0, 300.0, 600.0),
        (8, '256-QAM', 87.8, 180.0, 360.0, 720.0),
        (9, '256-QAM', 97.5, 200.0, 400.0, 800.0)
    ],
    '802.11p': [
        ('BPSK', 3.0), ('BPSK', 6.0),
        ('QPSK', 12.0), ('QPSK', 18.0),
        ('16-QAM', 24.0), ('16-QAM', 36.0),
        ('64-QAM', 48.0), ('64-QAM', 54.0)
    ],
    '802.11bd': [
        # MCS table for 802.11bd
        (0, 'BPSK', 7.0, 14.0),  # MCS 0
        (1, 'QPSK', 14.0, 28.0),  # MCS 1
        (2, 'QPSK', 21.0, 42.0),  # MCS 2
        (3, '16-QAM', 28.0, 56.0),  # MCS 3
        (4, '16-QAM', 42.0, 84.0),  # MCS 4
        (5, '64-QAM', 56.0, 112.0),  # MCS 5
        (6, '64-QAM', 63.0, 126.0),  # MCS 6
        (7, '64-QAM', 70.0, 140.0),  # MCS 7
        (8, '256-QAM', 84.0, 168.0),  # MCS 8
        (9, '256-QAM', 105.0, 210.0)  # MCS 9
    ]
}
 
def interface_exists(interface):
    """Check if a network interface exists."""
    try:
        subprocess.check_output(['ip', 'link', 'show', interface], stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        return False
 
def get_wireless_info(interface, retries=3, delay=2):
    """Get wireless information from an interface using the iw command."""
    for attempt in range(retries):
        if interface_exists(interface):
            try:
                result = subprocess.check_output(['iw', 'dev', interface, 'link'], text=True)
                return result
            except subprocess.CalledProcessError as e:
                print(f"Attempt {attempt+1}: Error accessing interface {interface}: {e}")
                time.sleep(delay)
        else:
            print(f"Attempt {attempt+1}: Interface {interface} does not exist.")
            time.sleep(delay)
    return None
 
def parse_wireless_standard(iw_output):
    """Parse the wireless standard from iw command output."""
    standard = None
    if "phy802.11n" in iw_output:
        standard = '802.11n'
    if "phy802.11a" in iw_output:
        standard = '802.11a'
    if "phy802.11b" in iw_output:
        standard = '802.11b'
    if "phy802.11g" in iw_output:
        standard = '802.11g'
    if "phy802.11ac" in iw_output:
        standard = '802.11ac'
    if "phy802.11ax" in iw_output:
        standard = '802.11ax'
    if "phy802.11p" in iw_output:
        standard = '802.11p'
    elif "phy802.11bd" in iw_output:
        standard = '802.11bd'
    return standard
 
def parse_mcs_index(iw_output):
    """Parse the MCS index from iw command output."""
    mcs_index = None
    match = re.search(r'MCS index:\s+(\d+)', iw_output)
    if match:
        mcs_index = int(match.group(1))
    else:
        mcs_index = 7  # Default MCS index if not found
    return mcs_index
 
def get_modulation_and_datarate(interface, ieee_standard, ht_cap):
    """Determine modulation type and data rate based on MCS index and bandwidth."""
    mcs_table = mcs_tables.get(ieee_standard)
    if not mcs_table:
        print(f"No MCS table found for the defined standard {ieee_standard}.")
        return 'N/A', 'N/A'
 
    iw_output = get_wireless_info(interface)
    if not iw_output:
        print(f"Unable to get wireless information for {interface}.")
        return 'N/A', 'N/A'
 
    mcs_index = parse_mcs_index(iw_output)
    modulation = None
    datarate = None
 
    # Determine bandwidth based on ht_cap
    bandwidth = 40 if 'HT40' in ht_cap else 20
 
    if ieee_standard in mcs_tables:
        for entry in mcs_table:
            if entry[0] == mcs_index:
                datarate = entry[3] if bandwidth == 40 and len(entry) > 3 else entry[2]
                modulation = entry[1]
                break
    else:
        modulation, datarate = mcs_table[mcs_index]  # Fixed rates for legacy standards
 
    return modulation, datarate if modulation and datarate else 'N/A', 'N/A'
 
def getdatetime():
    """Get current date and time formatted in Asia/Jakarta timezone."""
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    currentDT = utc_now.astimezone(pytz.timezone("Asia/Jakarta"))
    DATIME = currentDT.strftime("%Y-%m-%d %H:%M:%S")
    return DATIME
 
def calculate_subnet_mask(number_of_nodes):
    """Calculate subnet mask based on the number of nodes."""
    number_of_ips_needed = number_of_nodes + 2
    bits_needed_for_hosts = math.ceil(math.log2(number_of_ips_needed))
    subnet_mask = 32 - bits_needed_for_hosts
    return subnet_mask
 
def calculate_communication_range(tx_power_dbm, frequency_mhz=5180, threshold_dbm=-80):
    """
    Calculate the communication range based on the transmission power using the Free Space Path Loss (FSPL) model.
    :param tx_power_dbm: Transmission power in dBm.
    :param frequency_mhz: Frequency in MHz (default is 2400 MHz for 2.4 GHz Wi-Fi).
    :param threshold_dbm: Receive sensitivity threshold in dBm (default is -90 dBm).
    :return: Communication range in meters.
    """
    tx_power_watts = 10 ** (tx_power_dbm / 10) / 1000
    threshold_watts = 10 ** (threshold_dbm / 10) / 1000
    lambda_m = 300 / frequency_mhz
    #range_m = (lambda_m / (4 * math.pi)) * math.sqrt(tx_power_watts / threshold_watts)
    range_m = (lambda_m / (4 * math.pi)) * (tx_power_watts / threshold_watts) ** (1 / (2 * 2.8))
    return range_m
 
 
def calculate_communication_range_nakagami(tx_power_dbm, frequency_mhz=5000, threshold_dbm=-80, beta=2.8, m=1):
    """
    Calculate the communication range using the Nakagami-m fading model and path loss.
    :return: Communication range in meters.
    """
    try:
        tx_power_watts = 10 ** (tx_power_dbm / 10) / 1000
        threshold_watts = 10 ** (threshold_dbm / 10) / 1000
        wavelength = 300 / frequency_mhz
        A = ((4 * math.pi) / wavelength) ** 2
        r_cs = (gamma(m + 1 / beta) / (gamma(m) * (threshold_watts * A * (m / tx_power_watts)))) ** (1 / beta)
        
        if r_cs < 0 or math.isnan(r_cs):
            raise ValueError("Calculated communication range is invalid.")
        return r_cs
    except Exception as e:
        print(f"Error calculating communication range: {e}")
        return 0  # Return 0 if the calculation fails
 
 
 
def is_within_reach(car1, car2):
    """Check if two cars are within the communication range of each other based on transmission power."""
    x1, y1 = car1.params.get('position', (0, 0))
    x2, y2 = car2.params.get('position', (0, 0))
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
 
    tx_power_car1 = car1.wintfs[0].txpower
    tx_power_car2 = car2.wintfs[0].txpower
 
    range_car1 = calculate_communication_range_nakagami(tx_power_car1)
    range_car2 = calculate_communication_range_nakagami(tx_power_car2)
 
    return distance <= range_car1 and distance <= range_car2
 
def start_tcpdump(pcap_file):
    """Start tcpdump to capture packets."""
    cmd = ["tcpdump", "-i", "any", "-w", pcap_file]
    info(f"Starting tcpdump with output file {pcap_file}\n")
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return process
 
def stop_tcpdump(process):
    """Stop tcpdump process."""
    if process:
        process.terminate()
 
def calculate_r_cs(S, p, beta, m, wavelength):
    """Calculate communication range using Rician Shadowing Model."""
    A = ((4 * math.pi) / wavelength) ** 2
    r_cs = (gamma(m + 1/beta) / (gamma(m) * (S * A * (m/p)))) ** (1/beta)
    return r_cs
 
def calculate_CBR(n, b, C=1200):
    """Calculate the Channel Busy Ratio (CBR)."""
    return (n + 1) * b / C
 
class CustomCar(Car):
    def __init__(self, name, **params):
        super().__init__(name, **params)
        self.neighbors_info = {}
        self.modulation = None
        self.data_rate = None
        self.beacon_rate = None
 
    def update_neighbor_info(self, neighbor_id, info):
        """Update the neighbor information for a specific car."""
        self.neighbors_info[neighbor_id] = info
 
def find_neighbors_position(car, net, check_range):
    """Find neighbors of a car based on position."""
    neighbors = []
    for target_car in net.cars:
        if car != target_car and (not check_range or is_within_reach(car, target_car)):
            neighbors.append(target_car.name)
    return neighbors
 
def find_neighbors_beacon(car, net, check_range):
    """Find neighbors of a car based on beacon replies."""
    neighbors = []
    for target_car in net.cars:
        if car != target_car and (not check_range or is_within_reach(car, target_car)):
            neighbors.append(target_car.name)
    return neighbors
 
def send_beacon(car, beacon_rate, net, check_range, range_detection_method, ieee_standard):
    """Send beacon packets at the specified rate with range checking."""
    if range_detection_method == 'position':
        neighbors = find_neighbors_position(car, net, check_range)
    elif range_detection_method == 'beacon':
        neighbors = find_neighbors_beacon(car, net, check_range)
    
    modulation, data_rate = get_modulation_and_datarate(car.wintfs[0].name, ieee_standard, car.wintfs[0].ht_cap)
    car.modulation = modulation
    car.data_rate = data_rate
    car.beacon_rate = beacon_rate
    
    beacon_info = {
        'timestamp': getdatetime(),
        'car_id': car.name,
        'position': car.params.get('position', (0, 0)),
        'speed': car.params.get('speed', 0),
        'power_transmission': car.wintfs[0].txpower,
        'modulation': modulation,
        'data_rate': data_rate,
        'beacon_rate': beacon_rate,
        'channel_busy_rate': car.cbr
    }
    
    # Broadcast beacon to neighbors
    for neighbor in neighbors:
        target_car = net.getNodeByName(neighbor)
        target_car.update_neighbor_info(car.name, beacon_info)
 
def receive_beacon(car, beacon_info):
    """Receive beacon information from a neighbor car."""
    neighbor_id = beacon_info['car_id']
    car.update_neighbor_info(neighbor_id, beacon_info)
 
def ping_flood(source, target_ip):
    """Send a flood of ping packets to a target IP."""
    cmd = ["ping", "-i", "0.01", target_ip]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode == 0:
        info(f"Successfully sent ping flood from {source.name} to {target_ip}")
 
def flood_all(net, flood_type, rate_mbps=1, unicast=True, check_range=True):
    """Perform flooding attack between cars."""
    def flood_car(car1, car2):
        target_ip = car2.params['ip']
        if flood_type == 'ping':
            ping_flood(car1, target_ip)
    
    threads = []
    for car1 in net.cars:
        for car2 in net.cars:
            if car1 != car2 and (not check_range or is_within_reach(car1, car2)):
                thread = Thread(target=flood_car, args=(car1, car2))
                threads.append(thread)
                thread.start()
 
    for thread in threads:
        thread.join()
 
def setup_socket_connection(server_ip, server_port):
    """Setup a socket connection to the server for logging."""
    socket_client = None
    try:
        socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket_client.connect((server_ip, server_port))
    except Exception as e:
        print(f"[ERROR] Unable to connect to socket server: {str(e)}")
        socket_client = None
    return socket_client
 
def log_data(timestamp, car, net, log_options, csv_writer=None, socket_client=None):
    """Log data for each car. Supports multiple logging options."""
    car_data = {
        'timestamp': timestamp,
        'car_id': car.name,
        'position': car.params.get('position', (0, 0)),
        'speed': car.params.get('speed', 0),
        'power_transmission': car.wintfs[0].txpower,
        'modulation': car.modulation,
        'data_rate': car.data_rate,
        'beacon_rate': car.beacon_rate,
        'channel_busy_rate': car.cbr,
        'vehicle_density': len(find_neighbors_position(car, net, check_range)) + 1  # Include vehicle density
    }
 
    # Log to CSV if selected
    if 'csv' in log_options and csv_writer:
        csv_writer.writerow(car_data)
 
    # Log to stdout if selected
    if 'stdout' in log_options:
        print(car_data)
 
    # Log to socket if selected
    if 'socket' in log_options and socket_client:
        socket_client.sendall(json.dumps(car_data).encode('utf-8'))
 
    for neighbor_id, neighbor_info in car.neighbors_info.items():
        neighbor_data = {
            'timestamp': timestamp,
            'car_id': car.name,
            'neighbor_id': neighbor_id,
            'position': neighbor_info['position'],
            'speed': neighbor_info['speed'],
            'power_transmission': neighbor_info['power_transmission'],
            'modulation': neighbor_info['modulation'],
            'data_rate': neighbor_info['data_rate'],
            'beacon_rate': neighbor_info['beacon_rate'],
            'channel_busy_rate': neighbor_info['channel_busy_rate']
        }
 
        # Log neighbor data to the selected options
        if 'csv' in log_options and csv_writer:
            csv_writer.writerow(neighbor_data)
        if 'stdout' in log_options:
            print(neighbor_data)
        if 'socket' in log_options and socket_client:
            socket_client.sendall(json.dumps(neighbor_data).encode('utf-8'))
 
 
def data_logging(net, flood_type, data_collection_interval=1, duration=60, rate_mbps=1, 
                 dump_packets=True, log_options=None, unicast=True, check_range=True, 
                 enable_beacon=True, beacon_rate=10, enable_cbr=False, range_detection_method='position', 
                 ieee_standard='802.11ac', run_flood=True, socket_client=None):
    """Log data of the network and perform flooding attacks."""
    start_time = time.time()
 
    pcap_file = f"{os.getcwd()}/traffic.pcap" if dump_packets else None
    tcpdump_process = start_tcpdump(pcap_file) if dump_packets else None
 
    csv_file = None
    csv_writer = None
    if log_options and 'csv' in log_options:
        csv_file = open(f"{os.getcwd()}/network_data.csv", mode='w', newline='')
        fieldnames = [
            'timestamp', 'car_id', 'position', 'speed', 'power_transmission',
            'modulation', 'data_rate', 'beacon_rate', 'channel_busy_rate',
            'neighbor_id', 'vehicle_density'
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
 
 
    while time.time() - start_time < duration:
        traci.simulationStep()
        
        timestamp = getdatetime()
        vehicles = traci.vehicle.getIDList()
        
        for vehicle_id in vehicles:
            vehicle_id_str = f'car{int(float(vehicle_id)) + 1}'
            
            try:
                car = net.getNodeByName(vehicle_id_str)
            except KeyError:
                continue
 
            # Set position and speed for each car
            x, y = traci.vehicle.getPosition(vehicle_id)
            car.params['position'] = (x, y)
            car.params['speed'] = traci.vehicle.getSpeed(vehicle_id)
 
            power_transmission = car.wintfs[0].txpower
            modulation, data_rate = get_modulation_and_datarate(car.wintfs[0].name, ieee_standard, car.wintfs[0].ht_cap)
            
            # Define the constants
            S = 1e-10  # Sensitivity threshold in Watts
            beta = 3   # Path loss exponent
            m = 1      # Shape parameter
            wavelength = 0.125  # Carrier wavelength in meters (e.g., 2.4 GHz corresponds to 0.125 meters)
            rho = len(find_neighbors_position(car, net, check_range)) + 1  # Number of sensed neighbors including the vehicle itself
 
            r_cs = calculate_r_cs(S, power_transmission, beta, m, wavelength)
            C = 1200  # Channel capacity is constant at 1200 beacons per second
            cbr = calculate_CBR(rho, beacon_rate, C) if enable_cbr else "N/A"
            
            car.cbr = cbr  # Store CBR in the car object
            
            if enable_beacon:
                send_beacon(car, beacon_rate, net, check_range, range_detection_method, ieee_standard)
            
            log_data(timestamp, car, net, log_options, csv_writer, socket_client)  # Pass net as an argument
        
        time.sleep(data_collection_interval)
        
        if run_flood:
            flood_all(net, flood_type, rate_mbps, unicast, check_range)
 
    if dump_packets:
        stop_tcpdump(tcpdump_process)
 
    if log_options and 'csv' in log_options and csv_file:
        csv_file.close()
 
    traci.close()
    net.stop()
    subprocess.run(["mn", "-c"])
    sys.exit(0)
 
 
def topology(num_cars, sumo_config_file, flood_type, duration, rate_mbps, unicast=True,
             check_range=True, dump_packets=True, log_options=None, enable_beacon=True, 
             beacon_rate=10, enable_cbr=False, range_detection_method='beacon', 
             ieee_standard='802.11ac', run_flood=True, server_ip=None, server_port=None):
    """Set up the network topology and start the simulation."""
    net = Mininet_wifi(link=wmediumd, wmediumd_mode=interference)
 
    socket_client = None
    if log_options and 'socket' in log_options and server_ip and server_port:
        socket_client = setup_socket_connection(server_ip, server_port)
 
    info("*** Creating nodes\n")
    for id in range(num_cars):
        #car = net.addCar(f'car{id + 1}', cls=CustomCar, wlans=2, encrypt=['wpa2', ''])  # Two interfaces per car
        car = net.addCar(f'car{id + 1}', cls=CustomCar, encrypt=['wpa2', ''])  # One interfaces per car
    info("*** Configuring Propagation Model\n")
    net.setPropagationModel(model="logDistance", exp=2.8)
 
    info("*** Configuring nodes\n")
    net.configureNodes()
 
    info("*** Creating Links\n")
    for car in net.cars:
        print(f"Configuring mesh with freq={5180}, channel={38}, ht_cap={car.wintfs[0].ht_cap}")
       
        #net.addLink(car, intf=car.wintfs[0].name, mode='ac', cls=mesh, ssid='mesh-ssid', channel=1, ht_cap='HT40+')
        #net.addLink(car, intf=car.wintfs[1].name, mode='ac', cls=mesh, ssid='mesh-ssid1', channel=6, ht_cap='HT40+')
        #net.addLink(car, intf=car.wintfs[0].name, mode='ac', cls=mesh, ssid='mesh-ssid', channel=38, freq=5180, ht_cap='HT40+')
        net.addLink(car, intf=car.wintfs[0].name, mode='ac', cls=mesh, ssid='mesh-ssid', channel=38, freq=5180)
        #net.addLink(car, intf=car.wintfs[1].name, mode='ac', cls=mesh, ssid='mesh-ssid', channel=102, freq=5180, ht_cap='HT40+')
 
 
    info("*** Starting network\n")
    net.build()
    time.sleep(5)  # Add a delay to ensure interfaces are properly initialized
 
    subnet_mask = calculate_subnet_mask(num_cars)
    base_ip = 192 * 256**3 + 168 * 256**2
 
    for id, car in enumerate(net.cars):
        ip = base_ip + id + 1
        ip_str = f'{(ip // (256**3)) % 256}.{(ip // (256**2)) % 256}.{(ip // 256) % 256}.{ip % 256}'
        car.setIP(f'{ip_str}/{subnet_mask}', intf=f'{car.wintfs[0].name}')
        car.setIP(f'{ip_str}/{subnet_mask}', intf=f'{car.wintfs[1].name}')
        info(f"Car {car.name} assigned IP {ip_str}/{subnet_mask}\n")
 
    info("*** Running simulation\n")
    
    sumoCmd = ["sumo-gui", "-c", sumo_config_file, "--start", "--delay", "500"]
    sumo_process = subprocess.Popen(sumoCmd)
    time.sleep(2)  # Give SUMO time to start
    traci.start(["sumo", "-c", sumo_config_file], port=8813)
    traci.setOrder(1)
    
    data_logging(net, flood_type, duration=duration, rate_mbps=rate_mbps, 
                 dump_packets=dump_packets, log_options=log_options, unicast=unicast, check_range=check_range,
                 enable_beacon=enable_beacon, beacon_rate=beacon_rate, enable_cbr=enable_cbr, 
                 range_detection_method=range_detection_method, ieee_standard=ieee_standard, 
                 run_flood=run_flood, socket_client=socket_client)
 
    info("*** Stopping network\n")
    sumo_process.terminate()
 
    if socket_client:
        socket_client.close()
 
    # Enter Mininet CLI
    info("*** Running CLI\n")
    CLI(net)
 
if __name__ == '__main__':
    setLogLevel('info')
    num_cars = 40  # Default number of cars
    flood_type = 'ping'  # Choose from 'syn', 'ack', 'udp', or 'ping'
    duration = 600  # Duration of the simulation in seconds
    rate_mbps = 1  # Rate of flooding attack in Mbps
    sumo_config_file = 'rdmc.sumocfg'  # SUMO configuration file
 
    # Configuration flags
    unicast = True  # True for unicast, False for broadcast
    check_range = True  # True to check range, False to skip range check
    dump_packets = True  # True to dump packets, False to skip
    log_options = ['stdout', 'csv', 'socket']  # Multiple logging options allowed
    server_ip = '127.0.0.1'  # IP address of the server for socket logging
    server_port = 9999  # Port of the server for socket logging
    enable_beacon = True  # True to enable beacon communication, False to disable
    beacon_rate = 10  # Beacon rate in Hz
    enable_cbr = True  # True to calculate channel busy rate, False to skip
    range_detection_method = 'beacon'  # Choose from 'position' or 'beacon'
    ieee_standard = '802.11ac'  # Define the IEEE 802.11 standard used
    run_flood = True  # Choose whether to run the flooding function
 
    topology(num_cars, sumo_config_file, flood_type, duration, rate_mbps, unicast, check_range, dump_packets, 
             log_options, enable_beacon, beacon_rate, enable_cbr, range_detection_method, ieee_standard, 
             run_flood, server_ip, server_port)
 

