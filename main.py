from poker_train import main_training_loop, set_seed, EPISODES, GAMMA, ALPHA, STATE_SIZE, RENDERING
from poker_agents import PokerAgent

# Set seed for reproducibility
set_seed(42)

agent_list = []


# Create the Q-learning agents for 3 players
for i in range(3):  # Changed from 6 to 3
    agent = PokerAgent(
        state_size=STATE_SIZE,
        action_size=5,  # [check, call, fold, raise, all-in]
        gamma=GAMMA,
        learning_rate=ALPHA,
        load_model=False,
        load_path=f"saved_models/poker_agent_player_{i+1}.pth"
    )
    agent.name = f"player_{i+1}"
    agent_list.append(agent)

"""
# Create the Q-learning agent (and duplicate it 6 times)
agent = PokerAgent(
    state_size=STATE_SIZE,
    action_size=[5], # [check, call, fold, raise, all-in]
    gamma=GAMMA,
    learning_rate=ALPHA,
    load_model=False,
)
agent.name = "all_players"
for _ in range(6):
    agent_list.append(agent)
"""

# Start the training loop
main_training_loop(agent_list, episodes=EPISODES, rendering=RENDERING, render_every=1)