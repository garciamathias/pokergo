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

# Hyperparameters
EPISODES = 3000
GAMMA = 0.9985
ALPHA = 0.003
EPS_DECAY = 0.9998
STATE_SIZE = 31
RENDERING = False
FPS = 1
WINDOW_SIZE = 50  # Pour la moyenne mobile
PLOT_UPDATE_INTERVAL = 10  # Mettre à jour les graphiques tous les X épisodes

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

class TrainingVisualizer:
    def __init__(self):
        # Créer trois sous-graphiques
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Configurer les graphiques
        self.ax1.set_title('Récompense moyenne par Agent')
        self.ax2.set_title('Taux de Victoire par Agent')
        self.ax3.set_title('Distribution des Actions par Agent')
        
        # Initialiser les données
        self.window_size = 50
        self.rewards_data = {f"Agent {i+1}": [] for i in range(3)}
        self.wins_data = {f"Agent {i+1}": [] for i in range(3)}
        self.action_data = {f"Agent {i+1}": {
            'FOLD': [], 'CHECK': [], 'CALL': [], 'RAISE': [], 'ALL_IN': []
        } for i in range(3)}
        self.episodes = []
        
        # Couleurs pour les agents et actions
        self.colors = ['red', 'green', 'blue']
        self.action_colors = {
            'FOLD': '#FF9999',    # Rouge clair
            'CHECK': '#99FF99',   # Vert clair
            'CALL': '#9999FF',    # Bleu clair
            'RAISE': '#FFFF99',   # Jaune
            'ALL_IN': '#FF99FF'   # Rose
        }
        
        # Configurer les axes
        self.ax1.set_xlabel('Épisodes')
        self.ax1.set_ylabel('Récompense moyenne')
        self.ax2.set_xlabel('Épisodes')
        self.ax2.set_ylabel('Taux de victoire (%)')
        self.ax3.set_xlabel('Épisodes')
        self.ax3.set_ylabel('Proportion des actions (%)')
        
        # Ajouter les grilles
        self.ax1.grid(True)
        self.ax2.grid(True)
        self.ax3.grid(True)
        
        # Initialiser les lignes pour les récompenses et victoires
        self.reward_lines = {}
        self.win_lines = {}
        for i, agent_name in enumerate(self.rewards_data.keys()):
            self.reward_lines[agent_name], = self.ax1.plot([], [], 
                                                         label=agent_name, 
                                                         color=self.colors[i])
            self.win_lines[agent_name], = self.ax2.plot([], [], 
                                                      label=agent_name, 
                                                      color=self.colors[i])
        
        self.ax1.legend()
        self.ax2.legend()
        
        # Add a counter for file saves
        self.save_counter = 0
        self.save_interval = 100  # Save every 100 updates instead of every update
        
    def update_action_distribution(self, agent_name, action):
        """Met à jour la distribution des actions pour un agent"""
        action_name = action.name if hasattr(action, 'name') else action
        for act in self.action_data[agent_name].keys():
            self.action_data[agent_name][act].append(1 if act == action_name else 0)
    
    def plot_action_distribution(self):
        """Trace la distribution des actions pour chaque agent"""
        self.ax3.clear()
        self.ax3.set_title('Distribution des Actions par Agent')
        self.ax3.grid(True)
        
        bar_width = 0.25
        agent_positions = np.arange(len(self.rewards_data))
        
        for action_idx, action_name in enumerate(self.action_data['Agent 1'].keys()):
            action_percentages = []
            for agent_name in self.rewards_data.keys():
                action_counts = self.action_data[agent_name][action_name]
                if action_counts:
                    percentage = sum(action_counts[-self.window_size:]) / min(len(action_counts), self.window_size) * 100
                else:
                    percentage = 0
                action_percentages.append(percentage)
            
            self.ax3.bar(agent_positions + action_idx * bar_width, 
                        action_percentages,
                        bar_width,
                        label=action_name,
                        color=self.action_colors[action_name])
        
        self.ax3.set_xticks(agent_positions + bar_width * 2)
        self.ax3.set_xticklabels(self.rewards_data.keys())
        self.ax3.legend()
        self.ax3.set_ylim(0, 100)
        
    def update_plots(self, episode, rewards, wins, actions):
        # Mettre à jour les données
        self.episodes.append(episode)
        
        # Mettre à jour les récompenses et victoires
        for i, agent_name in enumerate(self.rewards_data.keys()):
            self.rewards_data[agent_name].append(rewards[i])
            self.wins_data[agent_name].append(wins[i])
            if actions and i < len(actions):
                self.update_action_distribution(agent_name, actions[i])
            
            # Calculer la moyenne mobile des récompenses
            if len(self.rewards_data[agent_name]) >= self.window_size:
                moving_avg = [
                    sum(self.rewards_data[agent_name][max(0, j-self.window_size):j])/min(j, self.window_size)
                    for j in range(1, len(self.rewards_data[agent_name])+1)
                ]
                self.reward_lines[agent_name].set_data(self.episodes, moving_avg)
            
            # Calculer le taux de victoire cumulé
            if len(self.wins_data[agent_name]) > 0:
                win_rates = [
                    sum(self.wins_data[agent_name][:j+1])/(j+1) * 100
                    for j in range(len(self.wins_data[agent_name]))
                ]
                self.win_lines[agent_name].set_data(self.episodes, win_rates)
        
        # Mettre à jour la distribution des actions
        self.plot_action_distribution()
        
        # Ajuster les limites des axes
        for ax in [self.ax1, self.ax2]:
            ax.relim()
            ax.autoscale_view()
        
        # Ajuster les limites de l'axe y pour le taux de victoire
        self.ax2.set_ylim([-5, 105])
        
        # Increment save counter and save file less frequently
        self.save_counter += 1
        if self.save_counter >= self.save_interval:
            # Ajuster la mise en page et sauvegarder
            plt.tight_layout()
            self.fig.savefig('viz_pdf/training_progress.jpg')
            self.save_counter = 0

# Main Training Loop
def main_training_loop(agent_list, episodes, rendering, render_every):
    # Initialize histories
    rewards_history = {}
    winning_history = {}
    
    # Créer le visualiseur
    visualizer = TrainingVisualizer()
    
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

def plot_rewards(rewards_history: dict, window_size: int = 50, save_path: str = "viz_pdf/poker_rewards.jpg"):
    """
    Trace l'évolution des récompenses moyennes pour chaque agent
    """
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'blue']
    
    for (agent_name, rewards), color in zip(rewards_history.items(), colors):
        # Calculer la moyenne mobile
        if len(rewards) >= window_size:
            moving_avg = [
                sum(rewards[i:i+window_size])/window_size 
                for i in range(len(rewards)-window_size+1)
            ]
            episodes = range(window_size-1, len(rewards))
            plt.plot(episodes, moving_avg, label=agent_name, color=color)
    
    plt.title(f'Récompenses moyennes (fenêtre de {window_size} épisodes)')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense moyenne')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_winning_stats(winning_history: dict, window_size: int = 50, save_path: str = "viz_pdf/poker_wins.jpg"):
    """
    Trace l'évolution du taux de victoire pour chaque agent
    """
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'blue']
    
    for (agent_name, wins), color in zip(winning_history.items(), colors):
        # Calculer le taux de victoire moyen
        if len(wins) >= window_size:
            win_rate = [
                sum(wins[i:i+window_size])/window_size * 100 
                for i in range(len(wins)-window_size+1)
            ]
            episodes = range(window_size-1, len(wins))
            plt.plot(episodes, win_rate, label=agent_name, color=color)
    
    plt.title(f'Taux de victoire (fenêtre de {window_size} épisodes)')
    plt.xlabel('Épisode')
    plt.ylabel('Taux de victoire (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
