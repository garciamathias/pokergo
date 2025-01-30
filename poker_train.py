# poker_train.py
import numpy as np
import random as rd
import pygame
import torch
import time
from poker_agents import PokerAgent
from poker_game import PokerGame, GamePhase, PlayerAction
import matplotlib.pyplot as plt
from vizualization import plot_wins, update_wins_history

# Hyperparameters
EPISODES = 1000
GAMMA = 0.9985
ALPHA = 0.003
EPS_DECAY = 0.9998
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
    
    # Continue until the hand is over
    while not env.current_phase == GamePhase.SHOWDOWN:
        current_player = env.players[env.current_player_idx]
        current_agent = agent_list[env.current_player_idx]
        
        state = env.get_state()
        valid_actions = [a for a in PlayerAction if env.action_buttons[a].enabled]
        
        action = current_agent.get_action(state, epsilon, valid_actions)
        next_state, reward = env.step(action)
        
        current_agent.remember(state, action, reward, next_state, False)
        
        active_players = sum(1 for p in env.players if p.is_active)
        if active_players == 1:
            break
            
        if rendering and (episode % render_every == 0):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
            
            env._draw()
            pygame.display.flip()
            env.clock.tick(30)
    
    # Get winner name
    active_players = [p for p in env.players if p.is_active]
    if len(active_players) == 1:
        winner_name = active_players[0].name
    else:
        # Evaluate hands and find winner
        player_hands = [(player, env.evaluate_hand(player)) for player in active_players]
        player_hands.sort(key=lambda x: (x[1][0].value, x[1][1]), reverse=True)
        winner_name = player_hands[0][0].name
    
    # Train all agents
    for agent in agent_list:
        agent.train_model()
        
    if rendering and (episode % render_every == 0):
        env._draw()
        pygame.display.flip()
        pygame.time.wait(1000)

    return winner_name

# Main Training Loop
def main_training_loop(agent_list, episodes, rendering, render_every = 10):
    # Initialize wins history
    wins_history = {}
    
    try:
        for episode in range(episodes):
            epsilon = np.clip(0.5 * EPS_DECAY ** episode, 0.01, 0.5)
            
            winner_name = run_episode(agent_list, epsilon, rendering, episode, render_every)
            if winner_name is None:  # Game was quit
                break
                
            # Update wins history
            wins_history = update_wins_history(wins_history, winner_name, agent_list)
            
            # Print episode information
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"Epsilon: {epsilon:.3f}")
            print(f"Winner: {winner_name}")

            # Save models and plot every 50 episodes
            if episode == EPISODES - 1:
                print("\nSaving models...")
                for agent in agent_list:
                    torch.save(agent.model.state_dict(), f"saved_models/poker_agent_{agent.name}_epoch_{episode+1}.pth")
                print("Models saved successfully!")
                
                # Plot and save wins visualization
                plot_wins(wins_history, window_size=50, save_path="viz_pdf/poker_wins.jpg")
                print("Wins plot saved successfully!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        plot_wins(wins_history, window_size=50, save_path="viz_pdf/poker_wins.jpg")
        print("Wins plot saved successfully!")
    finally:
        if rendering:
            pygame.quit()
        print("\nTraining completed")
