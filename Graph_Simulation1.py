# Pygame-Based Real-Time Traffic Simulation with RANDOM DATA GENERATION
# ------------------------------------------------------------------
# NO preloaded CSV is used.
# Vehicles are generated randomly during runtime.
# Data for 500 vehicles is collected and saved AFTER simulation.

import pygame
import math
import random
import time
import csv

# -----------------------------
# Constants
# -----------------------------

LIGHT_CYCLE = 30          # seconds
VEHICLE_SPEED = 2        # same for all vehicles
MAX_ACTIVE_VEHICLES = 12
TOTAL_VEHICLES = 500

# -----------------------------
# Pygame Setup
# -----------------------------

pygame.init()
WIDTH, HEIGHT = 900, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("City Traffic Simulation (Random Data Collection)")
clock = pygame.time.Clock()

# -----------------------------
# City Layout
# -----------------------------

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

block_names = list(blocks.keys())

# -----------------------------
# Traffic Lights (IN & OUT)
# -----------------------------

traffic_lights = {}
for b in block_names:
    traffic_lights[f"{b}_IN"] = random.choice(["RED", "GREEN"])
    traffic_lights[f"{b}_OUT"] = random.choice(["RED", "GREEN"])

last_light_switch = time.time()


def update_lights():
    global last_light_switch
    if time.time() - last_light_switch >= LIGHT_CYCLE:
        last_light_switch = time.time()
        for k in traffic_lights:
            traffic_lights[k] = random.choice(["RED", "GREEN"])

        # Ensure mixed states
        if all(v == "RED" for v in traffic_lights.values()):
            traffic_lights[random.choice(list(traffic_lights.keys()))] = "GREEN"
        if all(v == "GREEN" for v in traffic_lights.values()):
            traffic_lights[random.choice(list(traffic_lights.keys()))] = "RED"

# -----------------------------
# Vehicle Class
# -----------------------------

class Vehicle:
    def __init__(self, vid):
        self.id = vid
        self.source, self.destination = random.sample(block_names, 2)
        self.pos = list(blocks[self.source])
        self.stage = "IN"  # IN -> OUT
        self.spawn_time = time.time()
        self.wait_time = 0
        self.last_wait_check = time.time()

    def move(self):
        target = CENTER if self.stage == "IN" else blocks[self.destination]
        light_key = (
            f"{self.source}_IN" if self.stage == "IN" else f"{self.destination}_OUT"
        )

        # Stop at red light
        if traffic_lights[light_key] == "RED":
            self.wait_time += time.time() - self.last_wait_check
            self.last_wait_check = time.time()
            return False

        self.last_wait_check = time.time()

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
        pygame.draw.circle(screen, (0, 100, 255), (int(self.pos[0]), int(self.pos[1])), 5)

# -----------------------------
# Drawing
# -----------------------------

def draw_city():
    screen.fill((255, 255, 255))

    for pos in blocks.values():
        pygame.draw.line(screen, (0, 0, 0), pos, CENTER, 3)

    pygame.draw.circle(screen, (0, 180, 0), CENTER, 30, 3)

    for pos in blocks.values():
        pygame.draw.circle(screen, (0, 0, 0), pos, 10, 2)

    # Traffic lights
    for k, state in traffic_lights.items():
        block = k.split("_")[0]
        bx, by = blocks[block]
        color = (255, 0, 0) if state == "RED" else (0, 200, 0)
        offset = 18 if "IN" in k else -18
        pygame.draw.circle(screen, color, (bx + offset, by + offset), 5)

# -----------------------------
# Simulation Loop + Data Capture
# -----------------------------

vehicles = []
completed = 0
collected_data = []
next_vehicle_id = 1
running = True

while running:
    clock.tick(60)
    update_lights()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Spawn vehicles randomly
    if completed + len(vehicles) < TOTAL_VEHICLES and len(vehicles) < MAX_ACTIVE_VEHICLES:
        vehicles.append(Vehicle(next_vehicle_id))
        next_vehicle_id += 1

    draw_city()

    for v in vehicles[:]:
        finished = v.move()
        v.draw()
        if finished:
            total_time = time.time() - v.spawn_time
            collected_data.append([
                v.id,
                v.source,
                v.destination,
                round(v.wait_time, 2),
                round(total_time, 2)
            ])
            vehicles.remove(v)
            completed += 1

    pygame.display.flip()

    if completed >= TOTAL_VEHICLES:
        running = False

pygame.quit()

# -----------------------------
# Save Dataset
# -----------------------------

with open("random_traffic_simulation_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Vehicle_ID",
        "Source_Block",
        "Destination_Block",
        "Traffic_Light_Waiting_Time",
        "Total_Travel_Time"
    ])
    writer.writerows(collected_data)

print("Simulation complete. Data saved to random_traffic_simulation_data.csv")
