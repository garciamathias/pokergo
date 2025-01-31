import matplotlib
matplotlib.use('Agg')  # Use Agg backend that doesn't require GUI
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import seaborn as sns
import os

class TrainingVisualizer:
    def __init__(self):
        # Create three subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Configure plots
        self.ax1.set_title('Average Reward per Agent')
        self.ax2.set_title('Win Rate per Agent')
        self.ax3.set_title('Action Distribution per Agent')
        
        # Initialize data
        self.window_size = 50
        self.rewards_data = {f"Agent {i+1}": [] for i in range(3)}
        self.wins_data = {f"Agent {i+1}": [] for i in range(3)}
        self.action_data = {f"Agent {i+1}": {
            'FOLD': [], 'CHECK': [], 'CALL': [], 'RAISE': [], 'ALL_IN': []
        } for i in range(3)}
        self.episodes = []
        
        # Colors for agents and actions
        self.colors = ['red', 'green', 'blue']
        self.action_colors = {
            'FOLD': '#FF9999',    # Light red
            'CHECK': '#99FF99',   # Light green
            'CALL': '#9999FF',    # Light blue
            'RAISE': '#FFFF99',   # Yellow
            'ALL_IN': '#FF99FF'   # Pink
        }
        
        # Configure axes
        self.ax1.set_xlabel('Episodes')
        self.ax1.set_ylabel('Average Reward')
        self.ax2.set_xlabel('Episodes')
        self.ax2.set_ylabel('Win Rate (%)')
        self.ax3.set_xlabel('Episodes')
        self.ax3.set_ylabel('Action Distribution (%)')
        
        # Add grids
        self.ax1.grid(True)
        self.ax2.grid(True)
        self.ax3.grid(True)
        
        # Initialize lines for rewards and wins
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
        
        # Add counter for file saves
        self.save_counter = 0
        self.save_interval = 100  # Save every 100 updates

    def update_action_distribution(self, agent_name, action):
        """Update action distribution for an agent"""
        action_name = action.name if hasattr(action, 'name') else action
        for act in self.action_data[agent_name].keys():
            self.action_data[agent_name][act].append(1 if act == action_name else 0)
    
    def plot_action_distribution(self):
        """Plot action distribution for each agent"""
        self.ax3.clear()
        self.ax3.set_title('Action Distribution per Agent')
        self.ax3.grid(True)
        
        bar_width = 0.15  # Reduced from 0.25
        agent_positions = np.arange(len(self.rewards_data))
        
        # Calculate offset to center the group of bars
        total_width = bar_width * len(self.action_data['Agent 1'].keys())
        offset = total_width / 2 - bar_width / 2
        
        for action_idx, action_name in enumerate(self.action_data['Agent 1'].keys()):
            action_percentages = []
            for agent_name in self.rewards_data.keys():
                action_counts = self.action_data[agent_name][action_name]
                if action_counts:
                    percentage = sum(action_counts[-self.window_size:]) / min(len(action_counts), self.window_size) * 100
                else:
                    percentage = 0
                action_percentages.append(percentage)
            
            self.ax3.bar(agent_positions + action_idx * bar_width - offset,
                        action_percentages,
                        bar_width,
                        label=action_name,
                        color=self.action_colors[action_name])
        
        self.ax3.set_xticks(agent_positions)
        self.ax3.set_xticklabels(self.rewards_data.keys())
        self.ax3.legend()
        self.ax3.set_ylim(0, 100)
        
    def update_plots(self, episode, rewards, wins, actions):
        """Update all plots with new data"""
        # Update data
        self.episodes.append(episode)
        
        # Update rewards and wins
        for i, agent_name in enumerate(self.rewards_data.keys()):
            self.rewards_data[agent_name].append(rewards[i])
            self.wins_data[agent_name].append(wins[i])
            if actions and i < len(actions):
                self.update_action_distribution(agent_name, actions[i])
            
            # Calculate moving average of rewards
            if len(self.rewards_data[agent_name]) >= self.window_size:
                moving_avg = [
                    sum(self.rewards_data[agent_name][max(0, j-self.window_size):j])/min(j, self.window_size)
                    for j in range(1, len(self.rewards_data[agent_name])+1)
                ]
                self.reward_lines[agent_name].set_data(self.episodes, moving_avg)
            
            # Calculate cumulative win rate
            if len(self.wins_data[agent_name]) > 0:
                win_rates = [
                    sum(self.wins_data[agent_name][:j+1])/(j+1) * 100
                    for j in range(len(self.wins_data[agent_name]))
                ]
                self.win_lines[agent_name].set_data(self.episodes, win_rates)
        
        # Update action distribution
        self.plot_action_distribution()
        
        # Adjust axis limits
        for ax in [self.ax1, self.ax2]:
            ax.relim()
            ax.autoscale_view()
        
        # Adjust y-axis limits for win rate
        self.ax2.set_ylim([-5, 105])
        
        # Save file less frequently
        self.save_counter += 1
        if self.save_counter >= self.save_interval:
            plt.tight_layout()
            self.fig.savefig('viz_pdf/training_progress.jpg')
            self.save_counter = 0

def plot_rewards(rewards_history: dict, window_size: int = 50, save_path: str = "viz_pdf/poker_rewards.jpg"):
    """Plot evolution of average rewards for each agent"""
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'blue']
    
    for (agent_name, rewards), color in zip(rewards_history.items(), colors):
        if len(rewards) >= window_size:
            moving_avg = [
                sum(rewards[i:i+window_size])/window_size 
                for i in range(len(rewards)-window_size+1)
            ]
            episodes = range(window_size-1, len(rewards))
            plt.plot(episodes, moving_avg, label=agent_name, color=color)
    
    plt.title(f'Average Rewards ({window_size} episode window)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_winning_stats(winning_history: dict, window_size: int = 50, save_path: str = "viz_pdf/poker_wins.jpg"):
    """Plot evolution of win rate for each agent"""
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'blue']
    
    for (agent_name, wins), color in zip(winning_history.items(), colors):
        if len(wins) >= window_size:
            win_rate = [
                sum(wins[i:i+window_size])/window_size * 100 
                for i in range(len(wins)-window_size+1)
            ]
            episodes = range(window_size-1, len(wins))
            plt.plot(episodes, win_rate, label=agent_name, color=color)
    
    plt.title(f'Win Rate ({window_size} episode window)')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
