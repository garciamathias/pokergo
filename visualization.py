# visualization.py
import matplotlib
matplotlib.use('Agg')  # Use Agg backend that doesn't require GUI
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class TrainingVisualizer:
    def __init__(self, save_interval: int = 1000):
        # Create subplots
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Configure plots
        self.ax1.set_title('Average Reward per Agent')
        self.ax2.set_title('Win Rate per Agent')
        self.ax3.set_title('Action Distribution per Agent')
        self.ax4.set_title('Hand Strength Analysis')
        
        # Initialize data
        self.window_size = 50
        self.rewards_data = {f"Agent {i+1}": [] for i in range(3)}
        self.wins_data = {f"Agent {i+1}": [] for i in range(3)}
        self.action_data = {f"Agent {i+1}": {
            'FOLD': [], 'CHECK': [], 'CALL': [], 'RAISE': [], 'ALL_IN': []
        } for i in range(3)}
        self.hand_strength_data = {f"Agent {i+1}": {'strength': [], 'action': []} for i in range(3)}
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
        self.ax4.set_xlabel('Hand Strength')
        self.ax4.set_ylabel('Action Tendency')
        
        # Add grids
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.grid(True)
        
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
        self.save_interval = save_interval // 10

    def update_action_distribution(self, agent_name, action):
        """Update action distribution for an agent"""
        action_name = action.name if hasattr(action, 'name') else action
        for act in self.action_data[agent_name].keys():
            self.action_data[agent_name][act].append(1 if act == action_name else 0)

    def update_hand_strength_data(self, agent_name, strength, action):
        """Update hand strength and action data"""
        self.hand_strength_data[agent_name]['strength'].append(strength)
        self.hand_strength_data[agent_name]['action'].append(action.name)

    def plot_action_distribution(self):
        """Plot action distribution for each agent"""
        self.ax3.clear()
        self.ax3.set_title('Action Distribution per Agent')
        self.ax3.grid(True)
        
        bar_width = 0.15
        agent_positions = np.arange(len(self.rewards_data))
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

    def plot_hand_strength_analysis(self):
        """Plot hand strength correlation with actions"""
        self.ax4.clear()
        action_mapping = {
            'FOLD': 0,
            'CHECK': 1,
            'CALL': 2,
            'RAISE': 3,
            'ALL_IN': 4
        }
        
        # Use the same colors as defined in __init__
        for i, (agent_name, color) in enumerate(zip(self.hand_strength_data.keys(), self.colors)):
            strengths = self.hand_strength_data[agent_name]['strength']
            actions = [action_mapping[a] for a in self.hand_strength_data[agent_name]['action']]
            
            if len(strengths) > 0 and len(actions) > 0:
                # Calculate rolling correlation with error handling
                correlations = []
                for j in range(len(strengths)):
                    start = max(0, j - self.window_size)
                    end = j + 1
                    
                    # Ensure we have enough data points and variance
                    window_strengths = strengths[start:end]
                    window_actions = actions[start:end]
                    
                    if len(window_strengths) > 1 and len(set(window_strengths)) > 1 and len(set(window_actions)) > 1:
                        try:
                            corr = np.corrcoef(window_strengths, window_actions)[0,1]
                            correlations.append(corr if not np.isnan(corr) else 0)
                        except Exception:
                            correlations.append(0)
                    else:
                        correlations.append(0)
                
                # Only plot if we have valid correlations
                if correlations:
                    # Use consistent color from self.colors
                    self.ax4.plot(self.episodes[-len(correlations):], correlations, 
                                label=agent_name, color=self.colors[i])
        
        self.ax4.set_title('Hand Strength-Action Correlation')
        self.ax4.set_xlabel('Episodes')
        self.ax4.set_ylabel('Correlation Coefficient')
        self.ax4.legend()
        self.ax4.grid(True)
        self.ax4.set_ylim(-1, 1)

    def update_plots(self, episode, rewards, wins, actions, hand_strengths):
        """Update all plots with new data"""
        # Update data
        self.episodes.append(episode)
        
        # Update rewards and wins
        for i, agent_name in enumerate(self.rewards_data.keys()):
            self.rewards_data[agent_name].append(rewards[i])
            self.wins_data[agent_name].append(wins[i])
            if actions and i < len(actions):
                self.update_action_distribution(agent_name, actions[i])
                self.update_hand_strength_data(agent_name, hand_strengths[i], actions[i])
            
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
        
        # Update plots
        self.plot_action_distribution()
        self.plot_hand_strength_analysis()
        
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
    """
    Plot the total number of wins for each agent as bars.
    """
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'blue']
    
    agent_names = list(winning_history.keys())
    total_wins = []
    
    # Calculate total wins for each agent
    for agent_name in agent_names:
        wins = winning_history[agent_name]
        total_wins.append(sum(wins))
    
    # Create bar positions
    positions = np.arange(len(agent_names))
    
    # Create bars
    plt.bar(positions, total_wins, color=colors)
    
    # Customize the plot
    plt.title('Total Wins per Agent')
    plt.xlabel('Agents')
    plt.ylabel('Number of Wins')
    
    # Set x-axis labels to agent names
    plt.xticks(positions, agent_names)
    
    # Add value labels on top of each bar
    for i, v in enumerate(total_wins):
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    plt.grid(True, axis='y')
    plt.savefig(save_path)
    plt.close()
