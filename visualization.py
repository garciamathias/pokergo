from typing import Dict, List

def plot_rewards(rewards_history: Dict[str, List[float]], window_size: int = 50, save_path: str = "viz_rewards.jpg"):
    # ...
    # Set color palette manually for 3 players
    colors = ['#FF0000', '#00FF00', '#0000FF']  # Rouge, Vert, Bleu pour 3 joueurs
    # ... 