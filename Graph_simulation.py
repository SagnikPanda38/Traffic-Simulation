# Graph-Based Traffic Simulator for 8-Block City (Based on User Map)
# ---------------------------------------------------------------
# Each block is a node, roads are directed edges, center is a roundabout node
# This simulator plugs directly into YOLO + RL later

import random
import networkx as nx
import numpy as np

# -----------------------------
# 1. Create City Graph
# -----------------------------

G = nx.DiGraph()

blocks = [
    "Technology", "Administrative", "Residential", "Industrial",
    "Logistics", "Financial", "Research", "Commercial"
]

CENTER = "Roundabout"

# Add nodes
G.add_node(CENTER)
for b in blocks:
    G.add_node(b)

# Directed edges based on traffic flow in map
for b in blocks:
    G.add_edge(b, CENTER)      # incoming traffic
    G.add_edge(CENTER, b)      # outgoing traffic

# -----------------------------
# 2. Traffic State per Road
# -----------------------------

traffic_state = {b: random.randint(5, 25) for b in blocks}

# -----------------------------
# 3. Signal Control (One Direction Green)
# -----------------------------

def apply_signal(green_block, state, clearance_rate=5):
    """
    green_block: block given green signal
    state: dict of vehicle counts
    clearance_rate: vehicles cleared per step
    """
    cleared = min(clearance_rate, state[green_block])
    state[green_block] -= cleared

    # congestion increases slightly on others
    for b in state:
        if b != green_block:
            state[b] += random.randint(0, 2)

    return cleared

# -----------------------------
# 4. Reward Function
# -----------------------------

def compute_reward(state, cleared):
    total_wait = sum(state.values())
    return cleared * 2 - total_wait * 0.1

# -----------------------------
# 5. Simulation Step
# -----------------------------

def step(action_block, state):
    cleared = apply_signal(action_block, state)
    reward = compute_reward(state, cleared)
    return state, reward

# -----------------------------
# 6. Run Demo Simulation
# -----------------------------

print("Initial Traffic State:")
print(traffic_state)

for t in range(10):
    action = max(traffic_state, key=traffic_state.get)  # greedy baseline
    traffic_state, reward = step(action, traffic_state)

    print(f"\nTime Step {t+1}")
    print("Green Signal:", action)
    print("Traffic:", traffic_state)
    print("Reward:", round(reward, 2))

# -----------------------------
# This simulator is RL-ready:
# - State  = traffic_state.values()
# - Action = block index (0–7)
# - Reward = congestion reduction
# -----------------------------
