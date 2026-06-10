# Real-Time Traffic Signal Simulation using Pygame
# -------------------------------------------------
# Simulates AI-controlled traffic lights based on vehicle density

import pygame
import random
import time

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Traffic Optimization Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)

# Clock
clock = pygame.time.Clock()

# Vehicle settings
vehicle_width = 30
vehicle_height = 15
vehicles = []

# Traffic light state
green_time = 40  # seconds
red_time = 20
light_state = "RED"
last_switch = time.time()

# Font
font = pygame.font.SysFont(None, 32)

# AI-based signal timing logic
def ai_signal_timer(vehicle_count):
    if vehicle_count < 5:
        return 20
    elif vehicle_count < 10:
        return 40
    else:
        return 60

# Spawn vehicles
def spawn_vehicle():
    y = random.randint(250, 280)
    vehicles.append(pygame.Rect(0, y, vehicle_width, vehicle_height))

# Main loop
running = True
spawn_timer = 0

while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Spawn vehicles randomly
    spawn_timer += 1
    if spawn_timer > 40:
        spawn_vehicle()
        spawn_timer = 0

    # Draw road
    pygame.draw.rect(screen, GRAY, (0, 240, WIDTH, 120))

    # Count vehicles waiting
    vehicle_count = len(vehicles)

    # AI decides signal timing
    green_time = ai_signal_timer(vehicle_count)

    # Traffic light logic
    current_time = time.time()
    if light_state == "GREEN" and current_time - last_switch > green_time:
        light_state = "RED"
        last_switch = current_time
    elif light_state == "RED" and current_time - last_switch > red_time:
        light_state = "GREEN"
        last_switch = current_time

    # Draw traffic light
    light_color = GREEN if light_state == "GREEN" else RED
    pygame.draw.circle(screen, light_color, (700, 150), 20)

    # Move vehicles
    for v in vehicles[:]:
        if light_state == "GREEN" or v.x < 600:
            v.x += 3
        if v.x > WIDTH:
            vehicles.remove(v)
        pygame.draw.rect(screen, BLACK, v)

    # Display info
    info = font.render(f"Vehicles: {vehicle_count} | Signal: {light_state} | Green Time: {green_time}s", True, BLACK)
    screen.blit(info, (20, 20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
