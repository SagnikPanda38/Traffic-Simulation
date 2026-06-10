from traffic_gym_env import EmpiricalTrafficEnv

# 1. Instantiate environment with rendering enabled
env = EmpiricalTrafficEnv(baseline_csv_path="traffic_simulation_data.csv", render_mode="human")

# 2. Reset the system according to Gymnasium standard specification
obs, info = env.reset()
print("Initial Vector Observation Vector Spatially Validated:", obs)

terminated = False
truncated = False
step_counter = 0

while not (terminated or truncated):
    # Sample a random exploratory action from the Gymnasium action space
    action = env.action_space.sample()
    
    # Process environmental step parameters
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    step_counter += 1
    if step_counter % 100 == 0:
        print(f"Step: {step_counter} | Active Rewards: {reward:.2f} | Completed Units: {info['completed_vehicles']}")

print("Pipeline Evaluation Complete. Environmental state terminated gracefully.")
env.close()