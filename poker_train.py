# poker_train.py
import numpy as np
import random as rd
import pygame
import torch
import time
from poker_agents import PokerAgent
from poker_game import PokerGame, GamePhase, PlayerAction
import matplotlib.pyplot as plt
from vizualization import plot_rewards, update_rewards_history

# Hyperparameters
EPISODES = 10000
GAMMA = 0.9985
ALPHA = 0.003
EPS_DECAY = 0.995
STATE_SIZE = 44
RENDERING = False

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
    
    # Continue until the hand is over
    while not env.current_phase == GamePhase.SHOWDOWN:
        current_player = env.players[env.current_player_idx]
        current_agent = agent_list[env.current_player_idx]
        
        # Get state and valid actions
        state = env.get_state()
        valid_actions = [a for a in PlayerAction if env.action_buttons[a].enabled]
        
        # Get action from agent with valid actions
        action = current_agent.get_action(state, epsilon, valid_actions)
        
        # Take action and get next state
        next_state, reward = env.step(action)
        
        # Store experience
        current_agent.remember(state, action, reward, next_state, False)
        
        # If all players have folded except one, end the episode
        active_players = sum(1 for p in env.players if p.is_active)
        if active_players == 1:
            break
            
        # Render the game if rendering is enabled
        if rendering and (episode % render_every == 0):
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return reward_list
            
            # Draw the game state
            env._draw()
            pygame.display.flip()
            
            # Control frame rate
            env.clock.tick(30)  # 30 FPS
    
    # Calculate final rewards based on stack changes
    for i, player in enumerate(env.players):
        reward_list[i] = player.stack - 400
    
    # Train all agents
    for agent in agent_list:
        agent.train_model()
        
    # If rendering, show the final state briefly
    if rendering and (episode % render_every == 0):
        env._draw()
        pygame.display.flip()
        pygame.time.wait(1000)  # Wait 1 second to show final state

    return reward_list

# Main Training Loop
def main_training_loop(agent_list, episodes, rendering, render_every = 10):
    # Initialize rewards history
    rewards_history = {}
    
    try:
        for episode in range(episodes):
            epsilon = np.clip(0.5 * EPS_DECAY ** episode, 0.01, 0.5)
            
            reward_list = run_episode(agent_list, epsilon, rendering, episode, render_every)
            
            # Update rewards history
            rewards_history = update_rewards_history(rewards_history, reward_list, agent_list)
            
            # Print episode information
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"Epsilon: {epsilon:.3f}")
            for i, reward in enumerate(reward_list):
                print(f"Agent {i+1} reward: {reward:.2f}")

            # Save the trained models every 50 episodes
            if episode == EPISODES - 1:
                print("\nSaving models...")
                for agent in agent_list:
                    torch.save(agent.model.state_dict(), f"saved_models/poker_agent_{agent.name}_epoch_{episode+1}.pth")
                print("Models saved successfully!")
                
                # Plot and save rewards visualization
                plot_rewards(rewards_history, window_size=50, save_path="viz_pdf/poker_rewards.jpg")
                print("Rewards plot saved successfully!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Plot rewards even if training was interrupted
        plot_rewards(rewards_history, window_size=50, save_path="viz_pdf/poker_rewards.jpg")
        print("Rewards plot saved successfully!")
    finally:
        if rendering:
            pygame.quit()
        print("\nTraining completed")
