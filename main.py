from poker_train import main_training_loop, set_seed, EPISODES, GAMMA, ALPHA, STATE_SIZE, RENDERING
from poker_agents import PokerAgent

# Set seed for reproducibility
set_seed(42)

agent_list = []

# Create the Q-learning agent
for i in range(6):
    agent = PokerAgent(
        state_size=STATE_SIZE,
        action_sizes=[4], # [check, call, fold, raise]
        gamma=GAMMA,
        learning_rate=ALPHA,
        load_model=False,
    )
    agent.name = f"player_{i+1}"  # Add name for model saving
    agent_list.append(agent)

# Start the training loop
main_training_loop(agent_list, episodes=EPISODES, rendering=RENDERING, render_every=1)