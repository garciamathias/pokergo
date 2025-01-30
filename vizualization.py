import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import seaborn as sns
import os

def plot_wins(wins_history: Dict[str, List[int]], window_size: int = 50, save_path: str = "viz_pdf/poker_wins.jpg"):
    """
    Plot the win rate curves for each player with a moving average.
    
    Args:
        wins_history (Dict[str, List[int]]): Dictionary mapping player names to their win histories (1 for win, 0 for loss)
        window_size (int): Size of the moving average window
        save_path (str): Path where to save the JPEG plot
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        plt.clf()
        plt.close('all')
        
        plt.figure(figsize=(12, 8), dpi=300)
        plt.style.use('bmh')
        
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
        
        for i, (player_name, wins) in enumerate(wins_history.items()):
            if len(wins) < window_size:
                print(f"Warning: Not enough data points for {player_name} to calculate win rate")
                continue
                
            # Calculate win rate (moving average)
            win_rate = np.convolve(wins, np.ones(window_size)/window_size, mode='valid')
            episodes = range(len(win_rate))
            
            plt.plot(episodes, win_rate, 
                    label=f'{player_name} Win Rate (MA{window_size})',
                    color=colors[i % len(colors)],
                    linewidth=1.5)
        
        plt.title('Win Rate per Player Over Time', fontsize=14, pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Win Rate', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plt.savefig(save_path, 
                   format='jpg',
                   bbox_inches='tight',
                   dpi=300,
                   pad_inches=0.1)
        
        plt.close('all')
        
        print(f"Successfully saved win rate plot to {save_path}")
        
    except Exception as e:
        print(f"Error while creating plot: {str(e)}")
        plt.close('all')

def update_wins_history(wins_history: Dict[str, List[int]], 
                       winner_name: str,
                       agent_list: List) -> Dict[str, List[int]]:
    """
    Update the wins history dictionary with new episode winner.
    
    Args:
        wins_history (Dict[str, List[int]]): Current wins history
        winner_name (str): Name of the winning player
        agent_list (List): List of agents
        
    Returns:
        Dict[str, List[int]]: Updated wins history
    """
    try:
        for agent in agent_list:
            if agent.name not in wins_history:
                wins_history[agent.name] = []
            # Add 1 for win, 0 for loss
            wins_history[agent.name].append(1 if agent.name == winner_name else 0)
        
        return wins_history
    except Exception as e:
        print(f"Error while updating wins history: {str(e)}")
        return wins_history
