import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("rl_pygame_simulation_data.csv")
plt.style.use("default")

avg_wait = df.groupby("timestep")["waiting"].mean()

plt.figure()
plt.plot(avg_wait)
plt.xlabel("Timestep")
plt.ylabel("Average Waiting Time")
plt.title("Average Waiting Time Over Time")
plt.show()

