# poker_agents.py
import torch
import torch.nn as nn
import torch.optim as optim
from poker_model import ActorCriticModel
from poker_game import PlayerAction
from collections import namedtuple, deque
import random

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = 'cpu'  # Uncomment to force CPU

class PokerAgent:
    def __init__(self, state_size, action_size, gamma, learning_rate, entropy_coeff=0.01, value_loss_coeff=0.5, load_model=False, load_path=None):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff

        self.model = ActorCriticModel(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=10000)  # Experience replay buffer

        self.load_model = load_model
        if self.load_model:
            self.load(load_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def get_action(self, state, epsilon, valid_actions=None):
        """
        Get an action from the agent.
        Args:
            state: Current game state
            epsilon: Exploration rate
            valid_actions: List of currently valid PlayerActions
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            action_probs, _ = self.model(state_tensor)
        self.model.train()

        # Create action map
        action_map = {
            0: PlayerAction.CHECK,
            1: PlayerAction.CALL,
            2: PlayerAction.FOLD,
            3: PlayerAction.RAISE,
            4: PlayerAction.ALL_IN
        }
        
        # Create reverse map for masking
        reverse_map = {v: k for k, v in action_map.items()}

        # Get the preferred action before masking
        preferred_action_idx = torch.argmax(action_probs).item()
        preferred_action = action_map[preferred_action_idx]
        
        # Check if preferred action is invalid and assign penalty
        penalty_reward = 0
        if preferred_action not in valid_actions:
            penalty_reward = -30  # Penalty for preferring invalid action
        
        # Create action mask based on valid actions
        if valid_actions:
            # Create a mask of zeros
            mask = torch.zeros_like(action_probs)
            # Set 1s for valid actions
            for action in valid_actions:
                mask[0, reverse_map[action]] = 1
            # Apply mask to probabilities
            action_probs = action_probs * mask
            # Renormalize probabilities
            action_probs = action_probs / (action_probs.sum() + 1e-10)

        if random.random() < epsilon:  # Exploration
            action = random.choice(valid_actions)
        else: # Exploitation
            valid_indices = [reverse_map[a] for a in valid_actions]
            valid_probs = action_probs[0, valid_indices]
            action = action_map[valid_indices[torch.argmax(valid_probs).item()]]
                
        return action, penalty_reward

    def remember(self, state, action, reward, next_state, done):
        # Convert PlayerAction enum to numerical action for training
        action_map = {
            PlayerAction.CHECK: 0,
            PlayerAction.CALL: 1,
            PlayerAction.FOLD: 2,
            PlayerAction.RAISE: 3,
            PlayerAction.ALL_IN: 4
        }
        numerical_action = action_map[action]
        self.memory.append((state, numerical_action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < 128:  # Minimum batch size
            return

        batch = random.sample(self.memory, 128)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Get current policy and value predictions
        action_probs, state_values = self.model(states)
        state_values = state_values.squeeze(-1)

        # Compute next state values
        with torch.no_grad():
            _, next_state_values = self.model(next_states)
            next_state_values = next_state_values.squeeze(-1)

        # Compute TD targets
        td_targets = rewards + self.gamma * next_state_values * (1 - dones)
        advantages = td_targets - state_values

        # Policy loss
        selected_action_probs = action_probs[torch.arange(len(actions)), actions]
        policy_loss = -torch.mean(torch.log(selected_action_probs + 1e-10) * advantages.detach())

        # Value loss
        value_loss = torch.mean((state_values - td_targets.detach()) ** 2)

        # Entropy loss (for exploration)
        entropy_loss = -torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1))

        # Total loss
        total_loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
