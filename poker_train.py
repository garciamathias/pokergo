# poker_train.py
import numpy as np
import random as rd
import pygame
import torch
import time
from poker_agents import PokerAgent
from poker_game import PokerGame
import matplotlib.pyplot as plt

# Hyperparameters
EPISODES = 400
GAMMA = 0.9985
ALPHA = 0.001
GLOBAL_N = 11
EPS_DECAY = 0.98
STATE_SIZE = 23
RENDERING = True
def set_seed(seed=42):
    rd.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to run a single episode
def run_episode(agent_list, epsilon, rendering, episode, render_every):
    env = PokerGame()
    env.reset()
    done = False
    reward_list = np.zeros(len(agent_list))

    while not done:
        for agent in agent_list:
            action = agent.get_action(env.get_state(), epsilon)
            env.step(action)

    for agent in agent_list:
        agent.train_model()

    return reward_list

# Main Training Loop
def main_training_loop(agent_list, episodes, rendering, render_every = 10):
    for episode in range(episodes):
        epsilon = np.clip(0.5 * EPS_DECAY ** episode, 0.01, 0.5)
        
        reward_list = run_episode(agent_list, epsilon, rendering, episode, render_every)
        
        for i, reward in enumerate(reward_list):
            print(f"Agent {i+1} reward: {reward:.2f}")

        # Save the trained models every 50 episodes
        if episode % 50 == 49:
            for agent in agent_list:
                torch.save(agent.model.state_dict(), f"saved_models/poker_agent_{agent.name}_epoch_{episode+1}.pth")
