import pygame
import math
import random
import time
import csv

# =============================
# LOAD PREVIOUS SIMULATION DATA
# =============================

previous_routes = {}

with open("random_traffic_simulation_data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["Source_Block"], row["Destination_Block"])
        previous_routes.setdefault(key, []).append(
            float(row["Traffic_Light_Waiting_Time"])
        )

# Average historical waiting time per route
avg_previous_wait = {
    k: sum(v) / len(v) for k, v in previous_routes.items()
}

routes = list(avg_previous_wait.keys())

# =============================
# CONFIG
# =============================

TOTAL_VEHICLES = 500
MAX_ACTIVE = 12
LIGHT_CYCLE = 30
VEHICLE_SPEED = 2
EPSILON = 0.7

# =============================
# CITY MAP
# =============================

CENTER = (450, 450)

blocks = {
    "Technology": (450, 120),
    "Administrative": (700, 250),
    "Residential": (780, 450),
    "Industrial": (700, 650),
    "Logistics": (450, 780),
    "Financial": (200, 650),
    "Research": (120, 450),
    "Commercial": (200, 250)
}

# =============================
# PYGAME SETUP
# =============================

pygame.init()
screen = pygame.display.set_mode((900, 900))
pygame.display.set_caption("RL Traffic Control (Empirical Reward)")
clock = pygame.time.Clock()

# =============================
# TRAFFIC LIGHTS
# =============================

traffic_lights = {
    f"{b}_{d}": {
        "state": random.choice([0, 1]),
        "last_switch": time.time() - LIGHT_CYCLE
    }
    for b in blocks for d in ["IN", "OUT"]
}

def can_switch(light):
    return time.time() - traffic_lights[light]["last_switch"] >= LIGHT_CYCLE

# =============================
# RL AGENT (ON-POLICY)
# =============================

class TrafficAgent:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.policy = {}

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        return self.policy.get(state, random.choice([0, 1]))

    def learn(self, state, action, reward):
        if reward > 0:
            self.policy[state] = action

agent = TrafficAgent(EPSILON)

# =============================
# VEHICLE
# =============================

class Vehicle:
    def __init__(self, vid, source, destination):
        self.id = vid
        self.source = source
        self.destination = destination
        self.pos = list(blocks[source])
        self.stage = "IN"
        self.spawn_time = time.time()
        self.wait_time = 0
        self.waiting = False
        self.wait_start = None

    def light_key(self):
        return f"{self.source}_IN" if self.stage == "IN" else f"{self.destination}_OUT"

    def move(self):
        target = CENTER if self.stage == "IN" else blocks[self.destination]
        light = self.light_key()

        state = (
            self.source,
            self.destination,
            self.stage,
            traffic_lights[light]["state"]
        )

        action = agent.act(state)

        if action != traffic_lights[light]["state"] and can_switch(light):
            traffic_lights[light]["state"] = action
            traffic_lights[light]["last_switch"] = time.time()

        if traffic_lights[light]["state"] == 0:
            if not self.waiting:
                self.waiting = True
                self.wait_start = time.time()
            return False

        if self.waiting:
            self.wait_time += time.time() - self.wait_start
            self.waiting = False

        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        dist = math.hypot(dx, dy)

        if dist < 3:
            if self.stage == "IN":
                self.stage = "OUT"
            else:
                return True
        else:
            self.pos[0] += VEHICLE_SPEED * dx / dist
            self.pos[1] += VEHICLE_SPEED * dy / dist

        return False

    def draw(self):
        pygame.draw.circle(
            screen, (0, 100, 255),
            (int(self.pos[0]), int(self.pos[1])), 5
        )

# =============================
# DRAW MAP
# =============================

def draw_city():
    screen.fill((255, 255, 255))

    for pos in blocks.values():
        pygame.draw.line(screen, (0, 0, 0), pos, CENTER, 3)

    pygame.draw.circle(screen, (0, 160, 0), CENTER, 28, 3)

    for pos in blocks.values():
        pygame.draw.circle(screen, (0, 0, 0), pos, 10, 2)

    for k, v in traffic_lights.items():
        block = k.split("_")[0]
        x, y = blocks[block]
        color = (0, 200, 0) if v["state"] else (255, 0, 0)
        offset = 16 if "IN" in k else -16
        pygame.draw.circle(screen, color, (x + offset, y + offset), 5)

# =============================
# SIMULATION LOOP
# =============================

vehicles = []
completed = 0
vehicle_id = 1
results = []
total_reward = 0

running = True
while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if completed + len(vehicles) < TOTAL_VEHICLES and len(vehicles) < MAX_ACTIVE:
        src, dst = random.choice(routes)
        vehicles.append(Vehicle(vehicle_id, src, dst))
        vehicle_id += 1

    draw_city()

    for v in vehicles[:]:
        finished = v.move()
        v.draw()

        if finished:
            total_time = time.time() - v.spawn_time
            historical_wait = avg_previous_wait[(v.source, v.destination)]

            reward = historical_wait - v.wait_time
            reward = max(-30, min(30, reward))
            total_reward += reward

            agent.learn(
                (v.source, v.destination, v.stage, 1),
                1,
                reward
            )

            results.append([
                v.id,
                v.source,
                v.destination,
                round(v.wait_time, 2),
                round(historical_wait, 2),
                round(total_time, 2),
                round(reward, 2)
            ])

            vehicles.remove(v)
            completed += 1

    pygame.display.flip()

    if completed >= TOTAL_VEHICLES:
        running = False

pygame.quit()

# =============================
# SAVE RESULTS
# =============================

with open("rl_empirical_comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Vehicle_ID",
        "Source",
        "Destination",
        "RL_Waiting_Time",
        "Historical_Waiting_Time",
        "Total_Travel_Time",
        "Reward"
    ])
    writer.writerows(results)

print("Simulation completed")
print("Total reward earned by agent:", round(total_reward, 2))
