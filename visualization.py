import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import seaborn as sns
import os

def plot_rewards(rewards_history: Dict[str, List[float]], window_size: int = 50, save_path: str = "viz_rewards.jpg"):
    """
    Plot the mean reward curves for each player with a moving average.
    
    Args:
        rewards_history (Dict[str, List[float]]): Dictionary mapping player names to their reward histories
        window_size (int): Size of the moving average window
        save_path (str): Path where to save the JPEG plot
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Create new figure with high DPI
        plt.figure(figsize=(12, 8), dpi=300)
        
        # Set style - using a built-in style instead of seaborn
        plt.style.use('bmh')
        
        # Set color palette manually
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
        
        # Plot for each player
        for i, (player_name, rewards) in enumerate(rewards_history.items()):
            if len(rewards) < window_size:
                print(f"Warning: Not enough data points for {player_name} to calculate moving average")
                continue
                
            # Calculate moving average
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            episodes = range(len(moving_avg))
            
            # Plot the moving average with cycling colors
            plt.plot(episodes, moving_avg, 
                    label=f'{player_name} (MA{window_size})',
                    color=colors[i % len(colors)],
                    linewidth=1.5)
        
        # Customize plot
        plt.title('Mean Reward per Player Over Time', fontsize=14, pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Mean Reward', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot as JPEG with high quality
        plt.savefig(save_path, 
                   format='jpg',
                   bbox_inches='tight',
                   dpi=300,
                   pad_inches=0.1)
        
        # Close all figures to free memory
        plt.close('all')
        
        print(f"Successfully saved plot to {save_path}")
        
    except Exception as e:
        print(f"Error while creating plot: {str(e)}")
        # Clean up in case of error
        plt.close('all')

def update_rewards_history(rewards_history: Dict[str, List[float]], 
                         episode_rewards: List[float], 
                         agent_list: List) -> Dict[str, List[float]]:
    """
    Update the rewards history dictionary with new episode rewards.
    
    Args:
        rewards_history (Dict[str, List[float]]): Current rewards history
        episode_rewards (List[float]): Rewards from the current episode
        agent_list (List): List of agents
        
    Returns:
        Dict[str, List[float]]: Updated rewards history
    """
    try:
        for i, reward in enumerate(episode_rewards):
            player_name = agent_list[i].name
            if player_name not in rewards_history:
                rewards_history[player_name] = []
            rewards_history[player_name].append(reward)
        
        return rewards_history
    except Exception as e:
        print(f"Error while updating rewards history: {str(e)}")
        return rewards_history

def plot_winning_stats(winning_history: Dict[str, List[int]], save_path: str = "viz_wins.jpg"):
    """
    Plot the winning statistics for each player.
    
    Args:
        winning_history (Dict[str, List[int]]): Dictionary mapping player names to their win histories
        save_path (str): Path where to save the JPEG plot
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        plt.clf()
        plt.close('all')
        
        plt.figure(figsize=(12, 8), dpi=300)
        plt.style.use('bmh')
        
        # Calculate win percentages
        win_percentages = {}
        total_games = sum(len(wins) for wins in winning_history.values())
        
        for player_name, wins in winning_history.items():
            win_count = sum(1 for win in wins if win == 1)
            win_percentages[player_name] = (win_count / total_games) * 100
        
        # Create bar plot
        plt.bar(win_percentages.keys(), win_percentages.values())
        
        plt.title('Win Percentage by Player', fontsize=14, pad=20)
        plt.xlabel('Player', fontsize=12)
        plt.ylabel('Win Percentage (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, format='jpg', bbox_inches='tight', dpi=300, pad_inches=0.1)
        plt.close('all')
        
        print(f"Successfully saved winning stats plot to {save_path}")
        
    except Exception as e:
        print(f"Error while creating winning stats plot: {str(e)}")
        plt.close('all')

def update_winning_history(winning_history: Dict[str, List[int]], 
                         episode_winners: List[int], 
                         agent_list: List) -> Dict[str, List[int]]:
    """
    Update the winning history dictionary with new episode winners.
    
    Args:
        winning_history (Dict[str, List[int]]): Current winning history
        episode_winners (List[int]): Binary list indicating winners (1) and losers (0)
        agent_list (List): List of agents
        
    Returns:
        Dict[str, List[int]]: Updated winning history
    """
    try:
        for i, is_winner in enumerate(episode_winners):
            player_name = agent_list[i].name
            if player_name not in winning_history:
                winning_history[player_name] = []
            winning_history[player_name].append(is_winner)
        
        return winning_history
    except Exception as e:
        print(f"Error while updating winning history: {str(e)}")
        return winning_history