# poker_agents.py
import torch
import torch.nn as nn
import torch.optim as optim
from poker_model import ActorCriticModel
from poker_game import PlayerAction
from collections import namedtuple, deque
import random
import numpy as np

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = 'cpu'  # Uncomment to unforce CPU

class PokerAgent:
    def __init__(self, state_size, action_size, gamma, learning_rate, entropy_coeff=0.01, value_loss_coeff=0.5, load_model=False, load_path=None):
        self.device = device
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

        self.old_action_probs = None  # For tracking KL divergence

        self.is_human = False
        self.name = 'unknown_agent'

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def get_action(self, state, epsilon, valid_actions):
        # Convert valid PlayerActions to numerical indices
        action_map = {
            PlayerAction.FOLD: 0,
            PlayerAction.CHECK: 1,
            PlayerAction.CALL: 2,
            PlayerAction.RAISE: 3,
            PlayerAction.ALL_IN: 4
        }
        valid_indices = [action_map[a] for a in valid_actions]
        
        # Create action mask tensor with batch dimension
        action_mask = torch.zeros((1, self.action_size), device=self.device)
        for idx in valid_indices:
            action_mask[0, idx] = 1

        if np.random.random() < epsilon:
            # Random exploration with valid actions only
            chosen_index = np.random.choice(valid_indices)
            # Convert numerical index back to PlayerAction
            reverse_action_map = {v: k for k, v in action_map.items()}
            return reverse_action_map[chosen_index]
        else:
            # Convert state to tensor and add batch dimension
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action probabilities from model
            self.model.eval()
            with torch.no_grad():
                action_probs, _ = self.model(state_tensor)
            
            # Apply action mask and renormalize
            masked_probs = action_probs * action_mask
            if masked_probs.sum().item() == 0:  # Handle all-zero probabilities
                chosen_index = np.random.choice(valid_indices)
            else:
                # Normalize and convert to numpy
                masked_probs = masked_probs / masked_probs.sum()
                chosen_index = torch.argmax(masked_probs).item()
            
            # Verify the chosen action is valid
            if chosen_index not in valid_indices:
                # Fallback to random valid action with penalty
                penalty = -0.2  # Strong penalty for invalid prediction
                chosen_index = np.random.choice(valid_indices)
            
            # Convert numerical index back to PlayerAction
            reverse_action_map = {v: k for k, v in action_map.items()}
            return reverse_action_map[chosen_index]

    def remember(self, state, action, reward, next_state, done):        
        # Convert PlayerAction enum to numerical action for training
        action_map = {
            PlayerAction.FOLD: 0,
            PlayerAction.CHECK: 1,
            PlayerAction.CALL: 2,
            PlayerAction.RAISE: 3,
            PlayerAction.ALL_IN: 4,
            None: 0
        }
        numerical_action = action_map[action] if action is not None else action_map[None]
        self.memory.append((state, numerical_action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < 128:  # Minimum batch size
            return {'loss': 0, 'entropy': 0}

        batch = random.sample(self.memory, 128)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Handle terminal states (next_states is None)
        next_states_tensor = torch.zeros_like(states).to(device)  # Placeholder for terminal states
        for i, next_state in enumerate(next_states):
            if next_state is not None:
                next_states_tensor[i] = torch.FloatTensor(next_state).to(device)

        # Get current policy and value predictions
        action_probs, state_values = self.model(states)
        state_values = state_values.squeeze(-1)

        # Calculate KL divergence if we have old probabilities
        approx_kl = 0
        if self.old_action_probs is not None:
            approx_kl = torch.mean(
                torch.sum(self.old_action_probs * torch.log(self.old_action_probs / (action_probs + 1e-10)), dim=1)
            ).item()
        self.old_action_probs = action_probs.detach()

        # Compute next state values
        with torch.no_grad():
            _, next_state_values = self.model(next_states_tensor)
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

        # Compute standard deviation of advantages
        advantages_std = advantages.std().item()

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Return metrics
        metrics = {
            'approx_kl': approx_kl,
            'entropy_loss': entropy_loss.item(),
            'value_loss': value_loss.item(),
            'std': advantages_std,
            'learning_rate': self.learning_rate,
            'loss': total_loss.item()
        }
        return metrics
