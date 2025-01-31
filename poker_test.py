import pygame
from poker_game import PokerGame
from poker_agents import PokerAgent
import torch

def main():
    # Initialize pygame
    pygame.init()
    
    # Initialize game
    game = PokerGame()
    
    # Create and load AI agents for players 2-6
    agent_list = [None] 
    for i in range(2, 7):
        agent = PokerAgent(
            state_size=46,
            action_size=5,
            gamma=0.9985,
            learning_rate=0.003,
            load_model=True,
            load_path=f"saved_models/poker_agent_player_{i}.pth"
        )
        agent.name = f"player_{i}"
        agent_list.append(agent)
    
    # Set player 1 (index 0) as human, others as AI
    game.players[0].is_human = True
    for i in range(1, 6):
        game.players[i].is_human = False
    
    # Run the game loop with mixed human/AI players
    try:
        game.run_mixed_game(agent_list)
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        pygame.quit()
        print("Game ended")

if __name__ == "__main__":
    main()
