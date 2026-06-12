import pygame
from traffic_gym_env import EmpiricalTrafficEnv

# 1. Instantiate environment with rendering enabled
# Note: Ensure "traffic_simulation_data.csv" is in your root directory!
env = EmpiricalTrafficEnv(baseline_csv_path="traffic_simulation_data.csv", render_mode="human")

# 2. Reset the system according to Gymnasium standard specification
obs, info = env.reset()
print("Initial Vector Observation Vector Spatially Validated:", obs)

terminated = False
truncated = False
step_counter = 0

print("Starting environment verification loop...")

while not (terminated or truncated):
    # 3. CRITICAL PYGAME FIX: Process window event window queues to prevent OS freezing
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            terminated = True
            
    # 4. ACTION SELECTION: Instead of complete chaotic random sampling every millisecond,
    # let's let it sample a steady phase, or toggle it gently so you can watch physics step.
    if step_counter % 20 == 0:
        action = env.action_space.sample()
    else:
        # Keep the previous action until the next block cycle interval
        action = action if 'action' in locals() else 1 
    
    # Process environmental step parameters
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    step_counter += 1
    if step_counter % 100 == 0:
        print(f"Step: {step_counter} | Active Rewards: {reward:.2f} | Completed Units: {info['completed_vehicles']}")

print(f"\nPipeline Evaluation Complete after {step_counter} steps.")
print("Environmental state terminated gracefully.")

# Shutdown Pygame instance safely
env.close()
