Architecture:

- poker_game.py: PokerGame class that manages the poker game defined in pygame (if the file is executed, the poker game launches in human free play mode)
- poker_agents.py: PokerAgent class that defines the architecture and training methods for agents (each agent is defined by the same class)
- poker_train.py: Orchestrates the training of agents
- main.py: Launches the training
- config.py: Contains the training parameters






# 6-Player Poker: Rules and Gameplay

**6-player poker**, also known as **6-Max**, is a Texas Hold'em variant played with a maximum of **6 players per table**. This format is popular in online games and tournaments as it encourages more aggressive play.

## 1. Game Flow

### 1.1 Setup
- Each player receives **two private cards**.
- The dealer is designated by a **dealer button** that rotates clockwise after each hand.
- Two players place the **blinds**:
  - **Small Blind (SB)**: player to the left of the button.
  - **Big Blind (BB)**: player to the left of the Small Blind.

### 1.2 Betting Rounds
The game proceeds in **four phases** with betting:

1. **Pre-flop**: After cards are dealt, betting begins with the player left of the Big Blind.
2. **Flop**: Three community cards are revealed on the table followed by a betting round.
3. **Turn**: A fourth community card is added, followed by another betting round.
4. **River**: A fifth and final community card is revealed, followed by a final betting round.

### 1.3 Player Options
During each round, a player can:
- **Fold**: Abandon their hand.
- **Call**: Match the current bet.
- **Raise**: Increase the bet.
- **Check**: If no bet has been made, pass without betting.

## 2. Hand Rankings
Winning combinations from strongest to weakest:
1. **Royal Flush** (10, J, Q, K, A of the same suit)
2. **Straight Flush** (five consecutive cards of the same suit)
3. **Four of a Kind** (four cards of the same value)
4. **Full House** (three of a kind + a pair)
5. **Flush** (five cards of the same suit, not consecutive)
6. **Straight** (five consecutive cards of different suits)
7. **Three of a Kind** (three cards of the same value)
8. **Two Pair** (two pairs)
9. **Pair** (two cards of the same value)
10. **High Card** (highest card when no other combination is formed)

## 3. 6-Max Specific Strategies
- **Aggressive Play**: Fewer players means medium-strength hands have more value
- **Position Importance**: Playing in position (acting after others) allows for better decision-making
- **Blind Stealing**: With only two blinds and fewer players, blinds are attacked more frequently

## 4. Hand Completion
- If all but one player folds, the remaining player wins the pot
- If multiple players reach showdown, the best hand wins the pot
- In case of a tie, the pot is split
