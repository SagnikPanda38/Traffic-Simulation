import streamlit as st
import numpy as np
import torch
import pandas as pd
import time
from traffic_gym_env import EmpiricalTrafficEnv
from dqn_agent import DQNAgent

st.set_page_config(page_title="AI Traffic Simulator", layout="wide")

st.title("🚦 Deep Reinforcement Learning Urban Traffic Optimizer")
st.subheader("Powered by PyTorch DQN & Gymnasium")

# Sidebar Configuration Controls
st.sidebar.header("Simulation Settings")
max_vehicles = st.sidebar.slider("Total Vehicles in Fleet", 50, 1000, 500)
max_active = st.sidebar.slider("Max Concurrent Active Vehicles", 5, 30, 12)
use_ai = st.sidebar.toggle("Activate PyTorch DQN Brain", value=True)

# 1. Initialize State Dim Constraints
state_dim = 4
action_dim = 2

@st.cache_resource
def load_agent():
    # Helper to cache agent neural net so it doesn't reload on every webpage click
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    # Check if we have trained weights available
    if os.path.exists("traffic_dqn_weights.pth"):
        agent.policy_net.load_state_dict(torch.load("traffic_dqn_weights.pth"))
        agent.epsilon = 0.0  # Set exploration to zero for evaluation demo mode
    return agent

import os
agent = load_agent()

# Layout Placeholders for Live UI
col1, col2, col3 = st.columns(3)
metric_completed = col1.metric("Completed Vehicles", "0")
metric_active = col2.metric("Active Vehicles En Route", "0")
metric_reward = col3.metric("Running Optimization Reward", "0.0")

chart_placeholder = st.empty()
status_text = st.empty()

if st.button("🏁 Launch Live Simulation Run"):
    # 2. Instantiate headless environment parameters
    env = EmpiricalTrafficEnv(baseline_csv_path="traffic_simulation_data.csv", render_mode=None)
    env.TOTAL_VEHICLES = max_vehicles
    env.MAX_ACTIVE = max_active
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    history_rewards = []
    history_completed = []
    steps = 0
    
    # Run the simulation loop interactively
    while not (terminated or truncated):
        if use_ai:
            action = agent.select_action(obs)
        else:
            # Fallback baseline: purely random cycle switching
            action = env.action_space.sample() if steps % 20 == 0 else (action if 'action' in locals() else 0)
            
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = next_obs
        steps += 1
        
        # Track statistics over the timeline
        history_rewards.append(reward)
        history_completed.append(info['completed_vehicles'])
        
        # Update dashboard elements periodically to avoid slowing down browser
        if steps % 15 == 0:
            metric_completed.metric("Completed Vehicles", f"{info['completed_vehicles']} / {max_vehicles}")
            metric_active.metric("Active Vehicles En Route", f"{info['active_vehicles']}")
            metric_reward.metric("Current Instantaneous Reward", f"{reward:+.2f}")
            
            # Draw real-time metrics charts
            df_metrics = pd.DataFrame({
                "Steps Timeline": range(len(history_rewards)),
                "Optimization Delay Control Delta": history_rewards
            }).set_index("Steps Timeline")
            
            chart_placeholder.line_chart(df_metrics)
            time.sleep(0.01)  # Brief pause to mimic execution pacing
            
    status_text.success(f"Simulation completed successfully after {steps} execution intervals!")
    env.close()
  
