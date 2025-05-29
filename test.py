import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time

# Game Constants
WIDTH, HEIGHT = 288, 512
PIPE_WIDTH, PIPE_GAP = 52, 100
BIRD_RADIUS = 10
BIRD_X = 50
GRAVITY = 1
JUMP_VELOCITY = -9
FPS = 30

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# DQN Model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # jump or not
        )

    def forward(self, x):
        return self.fc(x)
    


clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
# # Game Environment
class FlappyEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.bird_y = HEIGHT // 2
        self.bird_vel = 0
        self.pipe_x = WIDTH
        self.pipe_gap_y = random.randint(100, 400)
        self.score = 0
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.bird_y / HEIGHT,
            self.bird_vel / 10.0,
            (self.pipe_x - BIRD_X) / WIDTH,
            self.pipe_gap_y / HEIGHT
        ], dtype=np.float32)

    def step(self, action):
        # Apply jump
        if action == 1:
            self.bird_vel = JUMP_VELOCITY

        self.bird_vel += GRAVITY
        self.bird_y += self.bird_vel
        self.pipe_x -= 4

        # Pipe passed
        if self.pipe_x < -50:
            self.pipe_x = WIDTH
            self.pipe_gap_y = random.randint(100, 400)
            self.score += 1

        done = self.bird_y < 0 or self.bird_y > HEIGHT
        hit_pipe = (self.pipe_x < BIRD_X + 25 < self.pipe_x + 50) and not (self.pipe_gap_y - PIPE_GAP // 2 < self.bird_y < self.pipe_gap_y + PIPE_GAP // 2)
        if hit_pipe:
            done = True

        reward = 1.0
        if done:
            reward = -100.0

        return self._get_state(), reward, done

    def render(self):
        screen.fill((135, 206, 250))  # sky blue
        # Pipe top
        pygame.draw.rect(screen, (0, 255, 0), (self.pipe_x, 0, PIPE_WIDTH, self.pipe_gap_y - PIPE_GAP // 2))
        # Pipe bottom
        pygame.draw.rect(screen, (0, 255, 0), (self.pipe_x, self.pipe_gap_y + PIPE_GAP // 2, PIPE_WIDTH, HEIGHT))
        # Bird
        pygame.draw.circle(screen, (255, 255, 0), (60, int(self.bird_y)), BIRD_RADIUS)
        # Score
        screen.blit(font.render(f"Score: {self.score}", True, (0, 0, 0)), (10, 10))
        pygame.display.flip()

# # Training Hyperparameters
EPISODES = 300
GAMMA = 0.99
LR = 0.001
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000

# # DQN Setup
env = FlappyEnv()
model = DQN(4, 2)
optimizer = optim.Adam(model.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)
epsilon = EPSILON_START

def train_step():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states      = torch.tensor(states, dtype=torch.float32)
    actions     = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards     = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones       = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = model(states).gather(1, actions)
    next_q = model(next_states).max(1)[0].unsqueeze(1).detach()
    target_q = rewards + (GAMMA * next_q * (1 - dones))

    loss = nn.MSELoss()(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# # Main Training Loop
# for episode in range(EPISODES):
#     state = env.reset()
#     total_reward = 0
#     done = False

#     while not done:
#         if random.random() < epsilon:
#             action = random.randint(0, 1)
#         else:
#             with torch.no_grad():
#                 action = model(torch.tensor(state).unsqueeze(0)).argmax().item()

#         next_state, reward, done = env.step(action)
#         memory.append((state, action, reward, next_state, done))
#         state = next_state
#         total_reward += reward

#         train_step()

#     epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
#     print(f"Episode {episode+1}, Score: {total_reward}, Epsilon: {epsilon:.3f}")

# pygame.quit()

# Training loop with rendering
for ep in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                action = model(torch.tensor(state).unsqueeze(0)).argmax().item()

        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        train_step()
        env.render()
        # clock.tick(FPS)

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    print(f"Episode {ep+1}, Score: {env.score}, Epsilon: {epsilon:.3f}")

pygame.quit()
