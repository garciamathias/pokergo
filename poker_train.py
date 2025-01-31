# poker_train.py
import os
import numpy as np
import random as rd
import pygame
import torch
import time
from poker_agents import PokerAgent
from poker_game import PokerGame, GamePhase, PlayerAction
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend Agg qui ne nécessite pas de GUI
import matplotlib.pyplot as plt
from collections import deque
import threading
import subprocess
from visualization import TrainingVisualizer, plot_rewards, plot_winning_stats

# Hyperparameters
EPISODES = 10000
GAMMA = 0.9985
ALPHA = 0.003
EPS_DECAY = 0.9998
STATE_SIZE = 32
RENDERING = False
FPS = 1
WINDOW_SIZE = 50  # Pour la moyenne mobile
PLOT_UPDATE_INTERVAL = 10  # Mettre à jour les graphiques tous les X épisodes
SAVE_INTERVAL = 500 # Sauvegarder les graphiques tous les X épisodes

def configure_git():
    """Configure Git to use merge strategy for pulls"""
    try:
        # Configure Git to use merge strategy
        subprocess.run(['git', 'config', 'pull.rebase', 'false'], check=True)
        print("Git configured successfully to use merge strategy")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not configure Git: {e}")

def set_seed(seed=42):
    rd.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_state_size():
    """
    Calculate the state size for 3 players
    """
    # Base state components
    state_size = (
        2 +  # 2 hole cards
        5 +  # 5 community cards
        1 +  # current phase
        1 +  # round number
        1 +  # current bet
        3 +  # stack sizes (3 players)
        3 +  # current bets (3 players)
        3 +  # activity status (3 players)
        3 +  # relative positions (3 players)
        5 +  # available actions
        3    # last actions (3 players)
    )
    return state_size

# Function to run a single episode
def run_episode(agent_list, epsilon, rendering, episode, render_every):
    """
    Run a single episode of the poker game.
    
    Returns:
        tuple: (reward_list, winning_list, actions_taken) containing final rewards, win status, and actions taken
    """
    env = PokerGame()
    env.reset()
    
    # Track cumulative rewards for each player
    cumulative_rewards = [0] * len(agent_list)
    # Store initial stacks for reward calculation
    initial_stacks = [player.stack for player in env.players]
    
    # Ajouter une liste pour stocker les actions
    actions_taken = []
    
    # Continue until the hand is over
    while not env.current_phase == GamePhase.SHOWDOWN:
        current_player = env.players[env.current_player_idx]
        current_agent = agent_list[env.current_player_idx]
        
        # Get state and valid actions
        state = env.get_state()
        valid_actions = [a for a in PlayerAction if env.action_buttons[a].enabled]
        
        # Get action from agent and handle penalty reward
        action, penalty_reward = current_agent.get_action(state, epsilon, valid_actions)
        cumulative_rewards[env.current_player_idx] += penalty_reward
        
        # Take action and get next state and reward
        next_state, reward = env.step(action)
        
        # Update cumulative rewards
        cumulative_rewards[env.current_player_idx] += reward
        
        # Store experience
        current_agent.remember(state, action, reward + penalty_reward, next_state, 
                             env.current_phase == GamePhase.SHOWDOWN)
        
        # Stocker l'action
        actions_taken.append(action)
        
        # If all players have folded except one, end the episode
        active_players = sum(1 for p in env.players if p.is_active)
        if active_players == 1:
            break
            
        # Handle rendering if enabled
        if rendering and (episode % render_every == 0):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return cumulative_rewards, [0] * len(agent_list), []
            env._draw()
            pygame.display.flip()
            env.clock.tick(FPS)
    
    # Calculate final rewards based on stack changes
    final_stacks = [player.stack for player in env.players]
    stack_changes = [(final - initial) / env.big_blind for final, initial in zip(final_stacks, initial_stacks)]
    
    # Combine stack changes with cumulative rewards
    final_rewards = [r + s for r, s in zip(cumulative_rewards, stack_changes)]
    
    # Calculate winners (players with highest reward are winners)
    max_reward = max(final_rewards)
    winning_list = [1 if reward == max_reward else 0 for reward in final_rewards]
    
    # Train all agents
    for agent in agent_list:
        agent.train_model()
    
    # Show final state if rendering
    if rendering and (episode % render_every == 0):
        env._draw()
        pygame.display.flip()
        pygame.time.wait(1000)

    return final_rewards, winning_list, actions_taken

# Main Training Loop
def main_training_loop(agent_list, episodes, rendering, render_every):
    # Initialize histories
    rewards_history = {}
    winning_history = {}
    
    # Créer le visualiseur
    visualizer = TrainingVisualizer(SAVE_INTERVAL)
    
    try:
        # Créer les dossiers nécessaires
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        if not os.path.exists('viz_pdf'):
            os.makedirs('viz_pdf')
            
        for episode in range(episodes):
            epsilon = np.clip(0.5 * EPS_DECAY ** episode, 0.01, 0.5)
            
            reward_list, winning_list, actions_taken = run_episode(agent_list, epsilon, rendering, episode, render_every)
            
            # Update histories
            rewards_history = update_rewards_history(rewards_history, reward_list, agent_list)
            winning_history = update_winning_history(winning_history, winning_list, agent_list)
            
            # Mettre à jour les graphiques tous les X épisodes
            if episode % PLOT_UPDATE_INTERVAL == 0:
                visualizer.update_plots(episode, reward_list, winning_list, actions_taken)
            
            # Print episode information
            print(f"\nEpisode [{episode + 1}/{episodes}]")
            print(f"Randomness: {epsilon*100:.3f}%")
            for i, reward in enumerate(reward_list):
                print(f"Agent {i+1} reward: {reward:.2f}")

            # Save the trained models and final plots
            if episode == episodes - 1:
                print("\nSaving models...")
                for agent in agent_list:
                    torch.save(agent.model.state_dict(), 
                             f"saved_models/poker_agent_{agent.name}_epoch_{episode+1}.pth")
                print("Models saved successfully!")
                
                # Force save the final visualization
                plt.figure(1)
                plot_rewards(rewards_history, window_size=50, save_path="viz_pdf/poker_rewards.jpg")
                plot_winning_stats(winning_history, save_path="viz_pdf/poker_wins.jpg")
                visualizer.save_counter = visualizer.save_interval  # Force an update
                visualizer.update_plots(episode, reward_list, winning_list, actions_taken)
                print("Visualization plots saved successfully!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Sauvegarder les graphiques même en cas d'interruption
        plot_rewards(rewards_history, window_size=50, save_path="viz_pdf/poker_rewards.jpg")
        plot_winning_stats(winning_history, save_path="viz_pdf/poker_wins.jpg")
        print("Visualization plots saved successfully!")
    finally:
        if rendering:
            pygame.quit()
        print("\nTraining completed")

def update_rewards_history(rewards_history: dict, reward_list: list, agent_list: list) -> dict:
    """
    Met à jour l'historique des récompenses pour chaque agent
    """
    for i, agent in enumerate(agent_list):
        if agent.name not in rewards_history:
            rewards_history[agent.name] = []
        rewards_history[agent.name].append(reward_list[i])
    return rewards_history

def update_winning_history(winning_history: dict, winning_list: list, agent_list: list) -> dict:
    """
    Met à jour l'historique des victoires pour chaque agent
    """
    for i, agent in enumerate(agent_list):
        if agent.name not in winning_history:
            winning_history[agent.name] = []
        winning_history[agent.name].append(winning_list[i])
    return winning_history
