import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import time

# --- 1. Simplified Flappy Bird Environment ---
# This is a very basic simulation, not a graphical game.
# State: (bird_y, bird_velocity, next_pipe_x, next_pipe_gap_y)
# Actions: 0 (do nothing), 1 (flap)
class FlappyBirdEnv:
    def __init__(self):
        self.pipe_interval = 20  # How often new pipes appear
        self.pipe_width = 3     # Width of the pipe
        self.gap_size = 5       # Size of the gap in the pipe
        self.max_y = 20         # Max height for the bird/pipes
        self.min_y = 0          # Min height for the bird/pipes
        self.reset()

    def reset(self):
        self.bird_y = self.max_y // 2
        self.bird_velocity = 0
        self.pipes = deque([(self.max_y, random.randint(3, self.max_y - 3 - self.gap_size)) for _ in range(3)]) # (x_pos, gap_y) - placeholder
        self.pipes_x_pos = deque([self.pipe_interval * (i+1) for i in range(3)]) # X positions for pipes
        self.score = 0
        self.frames_since_last_pipe = 0
        return self._get_state()

    def _get_state(self):
        # Find the next pipe
        next_pipe_x = -1
        next_pipe_gap_y = -1

        for i, pipe_x in enumerate(self.pipes_x_pos):
            if pipe_x - self.pipe_width < 0: # This pipe has passed
                continue
            else: # This is our next pipe
                next_pipe_x = pipe_x
                next_pipe_gap_y = self.pipes[i][1] # gap_y is the top of the gap
                break

        # Normalize state features (simplistic normalization)
        norm_bird_y = self.bird_y / self.max_y
        norm_bird_velocity = (self.bird_velocity + 5) / 10 # Assuming velocity range -5 to 5
        norm_next_pipe_x = next_pipe_x / (self.pipe_interval * 3) # Max x based on initial pipes
        norm_next_pipe_gap_y = next_pipe_gap_y / self.max_y

        return np.array([norm_bird_y, norm_bird_velocity, norm_next_pipe_x, norm_next_pipe_gap_y], dtype=np.float32)

    def step(self, action):
        reward = 0.1 # Small reward for staying alive
        done = False

        # Apply gravity
        self.bird_velocity += -1 if action == 0 else 3 # 0: gravity, 1: flap
        self.bird_velocity = max(-5, min(5, self.bird_velocity)) # Clamp velocity

        self.bird_y += self.bird_velocity
        self.bird_y = max(self.min_y, min(self.max_y, self.bird_y)) # Clamp bird position

        # Move pipes
        for i in range(len(self.pipes_x_pos)):
            self.pipes_x_pos[i] -= 1 # Pipes move left

        # Check for passing pipes or new pipes
        if self.frames_since_last_pipe >= self.pipe_interval:
            self.pipes_x_pos.append(self.max_y) # New pipe appears far right
            self.pipes.append((self.max_y, random.randint(3, self.max_y - 3 - self.gap_size)))
            self.frames_since_last_pipe = 0
        else:
            self.frames_since_last_pipe += 1

        # Remove passed pipes
        if self.pipes_x_pos[0] < -self.pipe_width:
            self.pipes_x_pos.popleft()
            self.pipes.popleft()

        # Check collision with pipes or ground/ceiling
        if self.bird_y <= self.min_y or self.bird_y >= self.max_y:
            done = True
            reward = -10 # Big penalty for hitting ceiling/ground

        # Check collision with pipes
        for i, pipe_x in enumerate(self.pipes_x_pos):
            if pipe_x <= 0 and pipe_x > -self.pipe_width: # Bird is within the x-range of the pipe
                gap_y = self.pipes[i][1]
                if not (self.bird_y > gap_y and self.bird_y < gap_y + self.gap_size):
                    done = True
                    reward = -10 # Big penalty for hitting pipe
                else:
                    # Successfully passed a pipe
                    if self.bird_y > gap_y and self.bird_y < gap_y + self.gap_size:
                        if self.bird_velocity > 0: # Ensure we are past the pipe
                            reward = 5 # Reward for passing a pipe
                            self.score += 1


        if done:
            reward = -10 # Reinforce negative outcome immediately

        new_state = self._get_state()
        return new_state, reward, done, {} # Last dict for info (not used here)


# --- 2. Neural Network (Q-Network) ---
class DQNAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNAgent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        return self.net(state)

# --- 3. Experience Replay Memory ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)),
                torch.LongTensor(np.array(actions)),
                torch.FloatTensor(np.array(rewards)),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(np.array(dones)))

    def __len__(self):
        return len(self.buffer)

# --- Hyperparameters ---
STATE_DIM = 4      # (bird_y, bird_velocity, next_pipe_x, next_pipe_gap_y)
ACTION_DIM = 2     # 0 (do nothing), 1 (flap)
BUFFER_CAPACITY = 10000
BATCH_SIZE = 64
GAMMA = 0.99       # Discount factor
EPS_START = 1.0    # Initial exploration rate
EPS_END = 0.01     # Final exploration rate
EPS_DECAY = 0.0005 # Rate of exploration decay
LR = 0.001         # Learning rate
TARGET_UPDATE = 10 # How often to update the target network
NUM_EPISODES = 5000

# Set device
# device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")
device = "cpu"
print(f"Using device: {device}")

# --- Initialize components ---
env = FlappyBirdEnv()
policy_net = DQNAgent(STATE_DIM, ACTION_DIM).to(device)

if os.path.exists("flappy_bird_dqn.pth"):
    policy_net.load_state_dict(torch.load("flappy_bird_dqn.pth", map_location=device))
    print("Loaded existing model from flappy_bird_dqn.pth")

target_net = DQNAgent(STATE_DIM, ACTION_DIM).to(device)
target_net.load_state_dict(policy_net.state_dict()) # Copy weights
target_net.eval() # Target net is for inference, not training

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
criterion = nn.MSELoss() # Or Huber Loss
replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

steps_done = 0

# --- Training Loop ---
def optimize_model():
    if len(replay_buffer) < BATCH_SIZE:
        return # Not enough samples to train

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states = states.to(device)
    actions = actions.to(device).unsqueeze(1) # Add a dimension for gather
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # Compute Q(s_t, a) - the Q-value for the action taken
    # We use .gather(1, actions) to select the Q-value for the action that was actually taken
    state_action_values = policy_net(states).gather(1, actions)

    # Compute V(s_{t+1}) for all next states, using the target network
    # max(1)[0] returns the maximum Q-value along dimension 1 (actions)
    # .detach() because we don't want to backpropagate through the target network
    next_state_values = target_net(next_states).max(1)[0].detach()

    # Compute the expected Q values (TD Target)
    # If the episode is done (dones == True), then the next state value is 0
    expected_state_action_values = rewards + (GAMMA * next_state_values * (1 - dones))

    # Compute Huber loss
    # For a more robust loss, you could use nn.SmoothL1Loss (Huber loss)
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Optional: Clip gradients to prevent exploding gradients
    # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 100)
    optimizer.step()

# Main training loop
print("Starting training...")
start_time = time.time()

for episode in range(NUM_EPISODES):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # Add batch dimension
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy action selection
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        np.exp(-1. * steps_done * EPS_DECAY)
        steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                # policy_net(state) gives Q-values for all actions
                # .max(1)[1] returns the index of the max Q-value
                action = policy_net(state).max(1)[1].view(1, 1).item()
        else:
            action = env.action_space.sample() if hasattr(env, 'action_space') else random.randrange(ACTION_DIM)

        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # Convert to tensors
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32, device=device)
        done_tensor = torch.tensor([float(done)], dtype=torch.float32, device=device)

        # Store the transition in replay memory
        replay_buffer.push(state.cpu().squeeze(0).numpy(), action, reward.cpu().item(), next_state.cpu().squeeze(0).numpy(), done_tensor.cpu().item())

        # Move to the next state
        state = next_state

        # Optimize the model
        optimize_model()

        # Update the target network periodically
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    if (episode + 1) % 100 == 0:
        elapsed_time = time.time() - start_time
        print(f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {eps_threshold:.2f}, Time: {elapsed_time:.2f}s")
        # You could add evaluation steps here to see how well the agent performs without exploration

print("\nTraining complete!")
print(f"Total training time: {time.time() - start_time:.2f} seconds")

# --- (Optional) Test the trained agent ---
print("\nTesting the trained agent (no exploration)...")
num_test_episodes = 5
total_scores = []

for _ in range(num_test_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    episode_score = 0
    while not done:
        with torch.no_grad(): # No gradient calculation during testing
            action = policy_net(state).max(1)[1].view(1, 1).item()
        
        next_state, reward, done, _ = env.step(action)
        state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_score += reward # Accumulate reward as a proxy for score
    total_scores.append(episode_score)
    print(f"Test Episode Score: {episode_score:.2f}")

print(f"Average Test Score over {num_test_episodes} episodes: {np.mean(total_scores):.2f}")

# You can save your model
torch.save(policy_net.state_dict(), "flappy_bird_dqn.pth")
print("Model saved to flappy_bird_dqn.pth")