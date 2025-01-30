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
    def __init__(self, state_size, action_sizes, gamma, learning_rate, entropy_coeff=0.01, value_loss_coeff=0.5, load_model=False):
        self.state_size = state_size
        self.action_sizes = action_sizes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff

        self.model = ActorCriticModel(state_size, action_sizes).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=10000)  # Experience replay buffer

        self.load_model = load_model
        if self.load_model:
            self.load()

    def load(self):
        self.model.load_state_dict(torch.load('saved_models/poker_agent.pth'))

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
            action_probs_grouped, _ = self.model(state_tensor)
        self.model.train()

        # Since we only have one action group (check, call, fold, raise)
        action_probs = action_probs_grouped[0]

        # Create action map
        action_map = {
            0: PlayerAction.CHECK,
            1: PlayerAction.CALL,
            2: PlayerAction.FOLD,
            3: PlayerAction.RAISE
        }
        
        # Create reverse map for masking
        reverse_map = {v: k for k, v in action_map.items()}
        
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
            if valid_actions:
                action = action_map[random.choice([reverse_map[a] for a in valid_actions])]
            else:
                action = action_map[random.randint(0, action_probs.size(1) - 1)]
        else:
            if valid_actions:
                valid_indices = [reverse_map[a] for a in valid_actions]
                valid_probs = action_probs[0, valid_indices]
                action = action_map[valid_indices[torch.argmax(valid_probs).item()]]
            else:
                action = action_map[torch.argmax(action_probs, dim=1).item()]
                
        return action

    def remember(self, state, action, reward, next_state, done):
        # Convert PlayerAction enum to numerical action for training
        action_map = {
            PlayerAction.CHECK: 0,
            PlayerAction.CALL: 1,
            PlayerAction.FOLD: 2,
            PlayerAction.RAISE: 3
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
        action_probs_grouped, state_values = self.model(states)
        action_probs = action_probs_grouped[0]  # We only have one action group

        # Compute next state values
        with torch.no_grad():
            _, next_state_values = self.model(next_states)
            next_state_values = next_state_values.squeeze(-1)

        # Compute TD targets
        td_targets = rewards + self.gamma * next_state_values * (1 - dones)
        advantages = td_targets - state_values.squeeze(-1)

        # Policy loss
        selected_action_probs = action_probs[range(len(actions)), actions]
        policy_loss = -torch.mean(torch.log(selected_action_probs + 1e-10) * advantages.detach())

        # Value loss
        value_loss = torch.mean((state_values.squeeze(-1) - td_targets.detach()) ** 2)

        # Entropy loss (for exploration)
        entropy_loss = -torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1))

        # Total loss
        total_loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
