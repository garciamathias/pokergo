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
        
        # Create new figure for metrics
        self.metrics_fig, self.metrics_axs = plt.subplots(3, 2, figsize=(15, 12))
        self.metrics_axs = self.metrics_axs.flatten()
        
        # Configure plots
        self.ax1.set_title('Average Reward per Agent')
        self.ax2.set_title('Win Rate per Agent')
        self.ax3.set_title('Action Distribution per Agent')
        self.ax4.set_title('Hand Strength vs Win Rate Correlation')
        
        # Initialize data
        self.window_size = 50
        self.rewards_data = {f"Agent {i+1}": [] for i in range(3)}
        self.wins_data = {f"Agent {i+1}": [] for i in range(3)}
        self.action_data = {f"Agent {i+1}": {
            'FOLD': [], 'CHECK': [], 'CALL': [], 'RAISE': [], 'ALL_IN': []
        } for i in range(3)}
        self.hand_strength_data = {f"Agent {i+1}": {'strength': [], 'win': []} for i in range(3)}
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
        self.ax4.set_xlabel('Episodes')
        self.ax4.set_ylabel('Win Rate Correlation')
        
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
        
        # Initialize metrics tracking
        self.metrics_history = defaultdict(lambda: defaultdict(list))
        self.episodes_history = []
        
        # Configure metrics axes
        metric_titles = ['Approx KL', 'Entropy Loss', 'Value Loss', 
                        'Advantage STD', 'Learning Rate', 'Total Loss']
        for ax, title in zip(self.metrics_axs, metric_titles):
            ax.set_title(title)
            ax.grid(True)
            ax.set_xlabel('Episodes')
        
        # Add counter for file saves
        self.save_counter = 0
        self.save_interval = save_interval // 10

    def update_action_distribution(self, agent_name, action):
        """Update action distribution for an agent"""
        action_name = action.name if hasattr(action, 'name') else action
        for act in self.action_data[agent_name].keys():
            self.action_data[agent_name][act].append(1 if act == action_name else 0)

    def update_hand_strength_data(self, agent_name, strength, win_status):
        """Update hand strength and win data"""
        self.hand_strength_data[agent_name]['strength'].append(strength)
        self.hand_strength_data[agent_name]['win'].append(1 if win_status else 0)

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
        """Plot hand strength correlation with wins"""
        self.ax4.clear()
        self.ax4.set_title('Hand Strength vs Win Rate Correlation')
        
        for i, (agent_name, color) in enumerate(zip(self.hand_strength_data.keys(), self.colors)):
            strengths = self.hand_strength_data[agent_name]['strength']
            wins = self.hand_strength_data[agent_name]['win']
            
            if len(strengths) > 0 and len(wins) > 0:
                correlations = []
                for j in range(len(strengths)):
                    start = max(0, j - self.window_size)
                    end = j + 1
                    
                    window_strengths = strengths[start:end]
                    window_wins = wins[start:end]
                    
                    # Check if we have enough variance in both arrays
                    if (len(window_strengths) > 1 and 
                        len(set(window_strengths)) > 1 and 
                        len(set(window_wins)) > 1):
                        try:
                            # Suppress numpy warnings during correlation calculation
                            with np.errstate(divide='ignore', invalid='ignore'):
                                corr = np.corrcoef(window_strengths, window_wins)[0,1]
                            # Check if correlation is valid
                            if corr is not None and not np.isnan(corr):
                                correlations.append(corr)
                            else:
                                correlations.append(0)
                        except Exception:
                            correlations.append(0)
                    else:
                        correlations.append(0)
                
                if correlations:
                    self.ax4.plot(self.episodes[-len(correlations):], correlations, 
                                label=agent_name, color=color)
        
        self.ax4.legend()
        self.ax4.grid(True)
        self.ax4.set_ylim(-1, 1)
        self.ax4.axhline(0, color='gray', linestyle='--')

    def update_metrics(self, episode, metrics_list):
        """Update training metrics history"""
        self.episodes_history.append(episode)
        
        for i, metrics in enumerate(metrics_list):
            if metrics is not None:
                agent_name = f"Agent {i+1}"
                for metric_name, value in metrics.items():
                    self.metrics_history[agent_name][metric_name].append(value)

    def plot_metrics(self):
        """Plot training metrics"""
        metric_keys = ['approx_kl', 'entropy_loss', 'value_loss', 
                      'std', 'learning_rate', 'loss']
        
        for ax, metric_key in zip(self.metrics_axs, metric_keys):
            ax.clear()
            ax.set_title(metric_key.replace('_', ' ').title())
            ax.grid(True)
            
            for agent_name, metrics in self.metrics_history.items():
                if metric_key in metrics:
                    values = metrics[metric_key]
                    if len(values) > 0:
                        # Calculate moving average
                        window = min(50, len(values))
                        smoothed_values = np.convolve(values, 
                                                    np.ones(window)/window, 
                                                    mode='valid')
                        episodes = self.episodes_history[-len(smoothed_values):]
                        
                        color = self.colors[int(agent_name[-1])-1]
                        ax.plot(episodes, smoothed_values, 
                               label=agent_name, color=color)
            
            ax.legend()
            ax.set_xlabel('Episodes')
        
        plt.tight_layout()
        self.metrics_fig.savefig('viz_pdf/training_metrics.jpg')

    def update_plots(self, episode, rewards, wins, actions_dict, hand_strengths, metrics_list=None):
        """Update all plots with new data"""
        self.episodes.append(episode)
        
        # Update rewards and wins
        for i, agent_name in enumerate(self.rewards_data.keys()):
            self.rewards_data[agent_name].append(rewards[i])
            self.wins_data[agent_name].append(wins[i])
            
            # Update action distribution if actions are provided
            if actions_dict and agent_name in actions_dict and actions_dict[agent_name]:
                for action in actions_dict[agent_name]:
                    self.update_action_distribution(agent_name, action)
            
            # Update hand strength data with win status
            if hand_strengths and i < len(hand_strengths):
                self.update_hand_strength_data(agent_name, hand_strengths[i], wins[i])
            
            # Calculate moving average of rewards
            if len(self.rewards_data[agent_name]) >= self.window_size:
                moving_avg = []
                for j in range(1, len(self.rewards_data[agent_name])+1):
                    # Get window of data
                    start_idx = max(0, j-self.window_size)
                    window_data = self.rewards_data[agent_name][start_idx:j]
                    
                    # Remove extreme values (0.5% from each end)
                    if len(window_data) > 4:  # Only trim if we have enough data points
                        window_data = np.array(window_data)
                        lower_percentile = np.percentile(window_data, 0.5)
                        upper_percentile = np.percentile(window_data, 99.5)
                        trimmed_data = window_data[(window_data >= lower_percentile) & 
                                                 (window_data <= upper_percentile)]
                        avg = np.mean(trimmed_data) if len(trimmed_data) > 0 else np.mean(window_data)
                    else:
                        avg = np.mean(window_data)
                    
                    moving_avg.append(avg)
                
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
        
        # Update and plot metrics if provided
        if metrics_list is not None:
            self.update_metrics(episode, metrics_list)
        
        # Save plots at intervals
        self.save_counter += 1
        if self.save_counter >= self.save_interval:
            plt.tight_layout()
            self.fig.savefig('viz_pdf/training_progress.jpg')
            self.plot_metrics()
            self.save_counter = 0

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
