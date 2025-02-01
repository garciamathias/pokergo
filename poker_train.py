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

WINDOW_SIZE = 50  
PLOT_UPDATE_INTERVAL = 10  
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

set_seed(42)

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
        3 +  # last actions (3 players)
        1    # hand strength
    )
    return state_size

# Function to run a single episode
def run_episode(agent_list, epsilon, rendering, episode, render_every):
    # Initialiser l'environnement de jeu
    env = PokerGame()
    env.reset()
    
    # Initialiser les récompenses cumulatives pour chaque joueur
    cumulative_rewards = [0] * len(agent_list)
    # Stocker les stacks initiaux pour calculer les changements à la fin
    initial_stacks = [player.stack for player in env.players]
    # Listes pour stocker les actions prises et les forces des mains
    actions_taken = []
    hand_strengths = [] 

    # Boucle principale du jeu jusqu'à ce que la phase de showdown soit atteinte
    while not env.current_phase == GamePhase.SHOWDOWN:
        # Récupérer le joueur actuel et l'agent correspondant
        current_player = env.players[env.current_player_idx]
        current_agent = agent_list[env.current_player_idx]
        
        # Obtenir l'état actuel du jeu et les actions valides
        state = env.get_state()
        valid_actions = [a for a in PlayerAction if env.action_buttons[a].enabled]

        # Calculer la force de la main actuelle
        strength = env._evaluate_hand_strength(current_player)
        hand_strengths.append(strength)
        
        # Obtenir l'action choisie par l'agent et la pénalité associée (si l'agent choisit une action invalide, action_chosen est choisie aléatoirement parmi les actions valides et recevra une pénalité)
        action_chosen, penalty_reward = current_agent.get_action(state, epsilon, valid_actions)
        cumulative_rewards[env.current_player_idx] += penalty_reward
        
        # Exécuter l'action et obtenir le nouvel état et la récompense
        next_state, reward = env.step(action_chosen)
        cumulative_rewards[env.current_player_idx] += reward
        
        # Stocker l'expérience dans la mémoire de l'agent
        current_agent.remember(state, action_chosen, reward + penalty_reward, next_state, 
                             env.current_phase == GamePhase.SHOWDOWN)
        actions_taken.append(action_chosen)
        
        # Vérifier si un seul joueur est actif (les autres ont abandonné)
        active_players = sum(1 for p in env.players if p.is_active)
        if active_players == 1:
            break
            
        # Gérer l'affichage si le rendering est activé
        if rendering and (episode % render_every == 0):
            env._draw()
            pygame.display.flip()
            env.clock.tick(FPS)
    
    # Calculer les récompenses finales en fonction des changements de stack et du statut de victoire
    final_stacks = [player.stack for player in env.players]
    stack_changes = [(final - initial) / env.big_blind for final, initial in zip(final_stacks, initial_stacks)]
    
    # Déterminer les gagnants (joueurs avec le stack le plus élevé)
    max_stack = max(final_stacks)
    winning_list = [1 if stack == max_stack else 0 for stack in final_stacks]
    final_rewards = [r + s for r, s in zip(cumulative_rewards, stack_changes)]
    
    # Ajouter l'état terminal et la récompense finale pour chaque joueur
    for i, agent in enumerate(agent_list):
        env.current_player_idx = i
        terminal_state = env.get_state()
        is_winner = winning_list[i]
        # Attribuer une récompense finale basée sur la victoire/défaite et le changement de stack
        final_reward = 5.0 if is_winner else -2.5
        final_reward += stack_changes[i]  # Inclure l'impact du changement de stack
        agent.remember(terminal_state, None, final_reward, None, True)

    print("final_rewards: ", final_rewards)
    
    # Entraîner tous les agents avec les expériences mises à jour
    for agent in agent_list:
        agent.train_model()
    
    # Afficher l'état final si le rendu est activé
    if rendering and (episode % render_every == 0):
        env._draw()
        pygame.display.flip()
        pygame.time.wait(1000)

    return final_rewards, winning_list, actions_taken, hand_strengths

# Main Training Loop
def main_training_loop(agent_list, episodes, rendering, render_every):
    # Initialize histories
    rewards_history = {}
    winning_history = {}
    
    # Create the visualizer
    visualizer = TrainingVisualizer(SAVE_INTERVAL)
    
    try:
        for episode in range(episodes):
            # Decay epsilon
            epsilon = np.clip(0.5 * EPS_DECAY ** episode, 0.01, 0.5)
            
            # Run episode and get results
            reward_list, winning_list, actions_taken, hand_strengths = run_episode(
                agent_list, epsilon, rendering, episode, render_every
            )
            
            # Update histories
            rewards_history = update_rewards_history(rewards_history, reward_list, agent_list)
            winning_history = update_winning_history(winning_history, winning_list, agent_list)
            
            # Update visualizer
            if episode % PLOT_UPDATE_INTERVAL == 0:
                visualizer.update_plots(episode, reward_list, winning_list, actions_taken, hand_strengths)
            
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
            visualizer.save_counter = visualizer.save_interval  # Force an update
            visualizer.update_plots(episode, reward_list, winning_list, actions_taken, hand_strengths)
            plot_rewards(rewards_history, window_size=50, save_path="viz_pdf/poker_rewards.jpg")
            plot_winning_stats(winning_history, save_path="viz_pdf/poker_wins.jpg")
            print("Visualization plots saved successfully!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
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
