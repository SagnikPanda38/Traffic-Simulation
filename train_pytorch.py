import os
import torch
import numpy as np
import pygame
from traffic_gym_env import EmpiricalTrafficEnv
from dqn_agent import DQNAgent  # Uses the PyTorch agent class we built earlier

def main():
    # 1. Initialize the environment 
    # Change render_mode to None during fast training, use "human" to watch it live
    env = EmpiricalTrafficEnv(baseline_csv_path="traffic_simulation_data.csv", render_mode=None)
    
    # 2. Extract state and action space dimensions dynamically
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 3. Instantiate our PyTorch DQN Agent
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Training Parameters
    MAX_EPISODES = 150
    TARGET_UPDATE_INTERVAL = 5  # Sync target network every 5 episodes
    
    print(f"==================================================")
    print(f"🚀 Starting PyTorch DRL Traffic Optimization Engine")
    print(f"State Dimensions: {state_dim} | Action Dimensions: {action_dim}")
    print(f"==================================================")
    
    for episode in range(MAX_EPISODES):
        state, info = env.reset()
        episode_reward = 0.0
        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated):
            # Pygame event handling to keep window responsive if rendering is turned on
            if env.render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Training forcefully aborted by user.")
                        env.close()
                        return

            # 4. Agent predicts the best action based on current traffic state
            action = agent.select_action(state)
            
            # 5. Environment processes the action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 6. Save performance metrics to memory replay buffer
            agent.store_transition(state, action, reward, next_state, terminated or truncated)
            
            # 7. Run a PyTorch optimization step to update NN weights
            agent.train_step()
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
        # Periodically sync target weights to stabilize Q-value targets
        if episode % TARGET_UPDATE_INTERVAL == 0:
            agent.update_target_network()
            
        # Logging printout metrics
        print(f"Ep: {episode+1:03d}/{MAX_EPISODES} | "
              f"Steps: {step_count} | "
              f"Net Reward: {episode_reward:+.2f} | "
              f"Completed Cars: {info['completed_vehicles']} | "
              f"Exploration Rate (ε): {agent.epsilon:.3f}")

    # 8. Save the final trained model weights
    model_filename = "traffic_dqn_weights.pth"
    torch.save(agent.policy_net.state_dict(), model_filename)
    print(f"\nTraining Complete! Neural Network weights saved to '{model_filename}'")
    env.close()

if __name__ == "__main__":
    main()
