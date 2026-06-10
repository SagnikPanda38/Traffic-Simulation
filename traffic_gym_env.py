import numpy as np
import math
import random
import time
import csv
import gymnasium as gym
from gymnasium import spaces

class EmpiricalTrafficEnv(gym.Env):
    """
    A high-grade, custom Gymnasium environment for urban traffic optimization.
    Features: Empirical reward shaping, 30s hardware constraints, and fully 
    decoupled vector observation spaces.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, baseline_csv_path="previous_simulation.csv", render_mode=None):
        super(EmpiricalTrafficEnv, self).__init__()
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # 1. ENVIRONMENT CONFIG & GEOMETRY
        self.TOTAL_VEHICLES = 500
        self.MAX_ACTIVE = 12
        self.LIGHT_CYCLE = 30.0  # Seconds
        self.VEHICLE_SPEED = 2.0
        self.CENTER = (450, 450)
        
        self.blocks = {
            "Technology": (450, 120), "Administrative": (700, 250),
            "Residential": (780, 450), "Industrial": (700, 650),
            "Logistics": (450, 780), "Financial": (200, 650),
            "Research": (120, 450), "Commercial": (200, 250)
        }
        self.block_names = list(self.blocks.keys())
        
        # 2. LOAD EMPIRICAL BASELINE DATA
        self.avg_previous_wait = self._load_baseline_data(baseline_csv_path)
        self.routes = list(self.avg_previous_wait.keys())

        # 3. DEFINE GYMNASIUM SPACES
        # Action Space: 0 = Force RED, 1 = Force GREEN for the active light
        self.action_space = spaces.Discrete(2)
        
        # Observation Space (Continuous Vector Box):
        # [Source_Index, Destination_Index, Stage_Index (0=IN, 1=OUT), Light_State (0=R, 1=G)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([len(self.blocks)-1, len(self.blocks)-1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

    def _load_baseline_data(self, path):
        previous_routes = {}
        try:
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                
                # Check for your exact headers
                src_col = "Source_Block"
                dst_col = "Destination_Block"
                wait_col = "Traffic_Light_Waiting_Time"

                # Verify if these columns actually exist in the file
                if reader.fieldnames and src_col not in reader.fieldnames:
                    print(f"⚠️ Warning: Found headers {reader.fieldnames} instead of {src_col}. Running fallback matching...")
                    # Fallback to the flexible logic if headers don't match perfectly
                    headers = [field.strip().lower() for field in reader.fieldnames]
                    src_col = reader.fieldnames[headers.index("source_block")] if "source_block" in headers else reader.fieldnames[0]
                    dst_col = reader.fieldnames[headers.index("destination_block")] if "destination_block" in headers else reader.fieldnames[1]
                    wait_col = reader.fieldnames[headers.index("traffic_light_waiting_time")] if "traffic_light_waiting_time" in headers else reader.fieldnames[2]

                for row in reader:
                    key = (row[src_col], row[dst_col])
                    previous_routes.setdefault(key, []).append(float(row[wait_col]))
                    
            return {k: sum(v) / len(v) for k, v in previous_routes.items()}
            
        except (FileNotFoundError, ValueError, KeyError, IndexError) as e:
            print(f"⚠️ Baseline load failed ({e}). Defaulting to standard 30s benchmark.")
            return {(src, dst): 30.0 for src in self.block_names for dst in self.block_names if src != dst}
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize Core Simulation State Vars
        self.traffic_lights = {
            f"{b}_{d}": {"state": random.choice([0, 1]), "last_switch": time.time() - self.LIGHT_CYCLE}
            for b in self.block_names for d in ["IN", "OUT"]
        }
        
        self.vehicles = []
        self.completed_count = 0
        self.next_vehicle_id = 1
        self.total_reward_accumulated = 0.0
        
        # Spawn initial batch
        self._manage_vehicle_spawns()
        
        # Return first observation + info dict
        return self._get_observation(), {}

    def step(self, action):
        if len(self.vehicles) == 0:
            # Handle edge case where no vehicles exist
            return self._get_observation(), 0.0, True, False, {}

        # 1. APPLY ACTION TO CURRENT ACTIVE VEHICLE'S INTERSECTION
        active_vehicle = self.vehicles[0]
        light_key = active_vehicle.light_key()
        
        current_time = time.time()
        if action != self.traffic_lights[light_key]["state"]:
            if current_time - self.traffic_lights[light_key]["last_switch"] >= self.LIGHT_CYCLE:
                self.traffic_lights[light_key]["state"] = action
                self.traffic_lights[light_key]["last_switch"] = current_time

        # 2. RUN SIMULATION PHYSICS FOR ONE TIMESTEP
        step_reward = 0.0
        vehicles_to_remove = []
        
        for v in self.vehicles:
            finished = v.update_physics(self.traffic_lights, self.CENTER, self.blocks, self.VEHICLE_SPEED)
            if finished:
                # Calculate scientific empirical reward structure
                historical_wait = self.avg_previous_wait.get((v.source, v.destination), 30.0)
                reward = historical_wait - v.wait_time
                reward = max(-30.0, min(30.0, reward)) # Clipping for gradient stability
                
                step_reward += reward
                self.completed_count += 1
                vehicles_to_remove.append(v)
                
        for v in vehicles_to_remove:
            self.vehicles.remove(v)

        # Spawn new vehicles if there's room
        self._manage_vehicle_spawns()

        # self.total_reward_accumulated += step_reward

        # 3. CONSTRUCT GYMNASIUM RETURN TUPLE
        next_obs = self._get_observation()
        terminated = self.completed_count >= self.TOTAL_VEHICLES
        truncated = False # Can implement step timeouts here if needed
        info = {"completed_vehicles": self.completed_count, "active_vehicles": len(self.vehicles)}

        if self.render_mode == "human":
            self.render()

        return next_obs, step_reward, terminated, truncated, info

    def _get_observation(self):
        if not self.vehicles:
            return np.array([0, 0, 0, 0], dtype=np.float32)
        
        # Frame active vehicle context as flat numeric vector
        v = self.vehicles[0]
        src_idx = self.block_names.index(v.source)
        dst_idx = self.block_names.index(v.destination)
        stage_idx = 0 if v.stage == "IN" else 1
        light_state = self.traffic_lights[v.light_key()]["state"]
        
        return np.array([src_idx, dst_idx, stage_idx, light_state], dtype=np.float32)

    def _manage_vehicle_spawns(self):
        while (self.completed_count + len(self.vehicles) < self.TOTAL_VEHICLES) and (len(self.vehicles) < self.MAX_ACTIVE):
            src, dst = random.choice(self.routes if self.routes else [random.sample(self.block_names, 2)])
            self.vehicles.append(GymVehicle(self.next_vehicle_id, src, dst, self.blocks[src]))
            self.next_vehicle_id += 1

    def render(self):
        if self.render_mode is None:
            return
        
        import pygame
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((900, 900))
                pygame.display.set_caption("Production RL Traffic Env")
                self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))
        
        # Draw physical layout infrastructure
        for pos in self.blocks.values():
            pygame.draw.line(self.screen, (180, 180, 180), pos, self.CENTER, 3)
        pygame.draw.circle(self.screen, (0, 160, 0), self.CENTER, 28, 3)

        for name, pos in self.blocks.items():
            pygame.draw.circle(self.screen, (50, 50, 50), pos, 12, 2)

        # Draw interactive traffic signals
        for k, v in self.traffic_lights.items():
            block = k.split("_")[0]
            x, y = self.blocks[block]
            color = (0, 200, 0) if v["state"] else (255, 0, 0)
            offset = 18 if "IN" in k else -18
            pygame.draw.circle(self.screen, color, (x + offset, y + offset), 6)

        # Draw active vehicle entities
        for v in self.vehicles:
            pygame.draw.circle(self.screen, (0, 100, 255), (int(v.pos[0]), int(v.pos[1])), 6)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is None:
            import pygame
            pygame.quit()


class GymVehicle:
    """Helper data object to monitor kinematics and performance states."""
    def __init__(self, vid, source, destination, spawn_pos):
        self.id = vid
        self.source = source
        self.destination = destination
        self.pos = list(spawn_pos)
        self.stage = "IN"
        self.spawn_time = time.time()
        self.wait_time = 0.0
        self.waiting = False
        self.wait_start = None

    def light_key(self):
        return f"{self.source}_IN" if self.stage == "IN" else f"{self.destination}_OUT"

    def update_physics(self, traffic_lights, center, blocks, speed):
        light = self.light_key()
        target = center if self.stage == "IN" else blocks[self.destination]

        # Light checking physics
        if traffic_lights[light]["state"] == 0:  # RED
            if not self.waiting:
                self.waiting = True
                self.wait_start = time.time()
            return False

        if self.waiting:
            self.wait_time += time.time() - self.wait_start
            self.waiting = False

        dx, dy = target[0] - self.pos[0], target[1] - self.pos[1]
        dist = math.hypot(dx, dy)

        if dist < 4:
            if self.stage == "IN":
                self.stage = "OUT"
            else:
                return True  # Reached endpoint
        else:
            self.pos[0] += speed * dx / dist
            self.pos[1] += speed * dy / dist
        return False