# poker_train.py
import numpy as np
import random as rd
import pygame
import torch
import time
from poker_agents import PokerAgent
from poker_game import PokerGame, GamePhase, PlayerAction
import matplotlib
matplotlib.use('Agg')

# Hyperparameters
EPISODES = 10000
GAMMA = 0.9985
ALPHA = 0.003
EPS_DECAY = 1 # 0.9998
STATE_SIZE = 166
RENDERING = True

def set_seed(seed=42):
    rd.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Function to run a single episode
def run_test_games(agent_list, env):
    # Main game loop
    while True:
        # Check if game should end (only one player with chips)
        active_players = [p for p in env.players if p.stack >= env.big_blind]
        if len(active_players) < 2:
            print("Game over - only one player remains with chips")
            env.reset()
            return

        # Get current player and corresponding agent
        current_player = env.players[env.current_player_idx]
        
        # Skip if player is not active
        if not current_player.is_active:
            env._next_player()
            continue
            
        print(f"Current player: {current_player.name}")
        current_agent = agent_list[env.current_player_idx]
        
        # Get current state and valid actions
        state = env.get_state()
        valid_actions = [a for a in PlayerAction if env.action_buttons[a].enabled]
        print('valid_actions', valid_actions)
        
        # Handle AI or human turn
        if not current_player.is_human:
            print(f"AI {current_agent.name} is playing")
            # AI agent's turn
            action_chosen, _ = current_agent.get_action(state, 0.99, valid_actions)
            env.step(action_chosen)
            time.sleep(1)
        else:
            print(f"Human is playing")
            # Human player's turn
            human_has_acted = False
            while not human_has_acted:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        mouse_pos = pygame.mouse.get_pos()
                        # Check button clicks
                        for action, button in env.action_buttons.items():
                            if button.rect.collidepoint(mouse_pos) and button.enabled:
                                action_chosen = action
                                env.step(action_chosen)
                                human_has_acted = True
                                time.sleep(1)
                                break
                        
                        # Check bet slider for raise actions
                        if env.bet_slider.collidepoint(mouse_pos):
                            slider_value = (mouse_pos[0] - env.bet_slider.x) / env.bet_slider.width
                            min_raise = max(env.current_bet * 2, env.big_blind * 2)
                            max_raise = current_player.stack + current_player.current_bet
                            bet_range = max_raise - min_raise
                            env.current_bet_amount = min(min_raise + (bet_range * slider_value), max_raise)
                            env.current_bet_amount = max(env.current_bet_amount, min_raise)
                
                # Update display while waiting for human input
                env._draw()
                pygame.display.flip()
                env.clock.tick(30)
        
        # Update display
        env._draw()
        pygame.display.flip()
        env.clock.tick(30)
        
        # Check if game is over
        if env.current_phase == GamePhase.SHOWDOWN:
            env._draw()
            pygame.display.flip()
            pygame.time.wait(2000)  # Show final state for 2 seconds
            break

    # Start new hand after delay
    pygame.time.wait(1000)
    return

if __name__ == "__main__":
    # Initialize agents
    agent_list = []
    
    # Human player (first position)
    human_agent = PokerAgent(
        state_size=STATE_SIZE,
        action_size=5,
        gamma=GAMMA,
        learning_rate=ALPHA
    )
    human_agent.name = "Human_1"
    human_agent.is_human = True
    agent_list.append(human_agent)

    # Add two AI agents
    for i in [2, 3]:
        agent = PokerAgent(
            state_size=STATE_SIZE,
            action_size=5,
            gamma=GAMMA,
            learning_rate=ALPHA,
            load_model=True,
            load_path=f"saved_models/poker_agent_player_{i}.pth"
        )
        agent.name = f"player_{i}"
        agent.is_human = False
        agent_list.append(agent)

    # Run game loop
    try:
        env = PokerGame()
        # Initialize game and sync player names with agents
        for i, agent in enumerate(agent_list):
            env.players[i].name = agent.name
            env.players[i].is_human = agent.is_human
            
        while True:
            # Check active players before starting new hand
            active_players = [p for p in env.players if p.stack >= env.big_blind]
            
            if len(active_players) < 2:
                print("Game over - resetting game")
                env.reset()
                # Re-sync player states after reset
                for i, agent in enumerate(agent_list):
                    env.players[i].name = agent.name
                    env.players[i].is_human = agent.is_human
            else:
                env.start_new_hand()
                
            run_test_games(agent_list, env)
            
    except KeyboardInterrupt:
        print("\nGame ended by user")
    finally:
        pygame.quit()
