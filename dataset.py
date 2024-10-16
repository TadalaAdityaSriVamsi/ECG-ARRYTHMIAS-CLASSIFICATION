import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of records
num_records = 100

# Generating random feature values
data = {
    'NetworkX_Topology': np.random.choice(['Star', 'Mesh', 'Ring', 'Tree'], num_records),  # Simulating network topology types
    'OVC_Enabled': np.random.choice([0, 1], num_records),  # Binary indicator for OVC enabled or not
    'Switch_Type': np.random.choice(['Layer2', 'Layer3'], num_records),  # Simulating switch types
    'Router_Count': np.random.randint(1, 10, num_records),  # Number of routers
    'Host_Count': np.random.randint(5, 100, num_records),  # Number of hosts
    'Protocol_TCP': np.random.randint(100, 10000, num_records),  # TCP traffic volume
    'Protocol_UDP': np.random.randint(100, 5000, num_records),  # UDP traffic volume
    'Protocol_ICMP': np.random.randint(10, 1000, num_records),  # ICMP traffic volume
    'Latency_ms': np.random.randint(1, 100, num_records),  # Latency in milliseconds
    'Packet_Loss_%': np.random.randint(0, 10, num_records),  # Packet loss percentage
    'Eavesdropping': np.random.choice([0, 1], num_records)  # Label: 0 for no eavesdropping, 1 for eavesdropping
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Save the dataset to a CSV file
df.to_csv('network_dataset.csv', index=False)

print("\nDataset created successfully and saved to 'network_dataset.csv'")
