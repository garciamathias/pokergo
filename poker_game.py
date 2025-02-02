# poker_game.py
import pygame
import random as rd
from enum import Enum
from typing import List, Dict, Optional, Tuple
import pygame.font
from collections import Counter
import torch
import time

class Card:
    """
    Represents a playing card with a suit and value.
    """
    def __init__(self, suit: str, value: int):
        """
        Initialize a card with a suit and value.
        Args:
            suit (str): The card's suit (♠, ♥, ♦, ♣)
            value (int): The card's value (2-14, where 14 is Ace)
        """
        self.suit = suit
        self.value = value
        
    def __str__(self):
        """
        Convert card to string representation.
        Returns:
            str: String representation of the card (e.g., "A♠")

        Example:
        >>> card = Card('♠', 14)
        >>> print(card)
        'A♠'
        """
        values = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        value_str = values.get(self.value, str(self.value))
        return f"{value_str}{self.suit}"

class HandRank(Enum):
    """
    Enumeration of possible poker hand rankings from lowest to highest.
    """
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9

class PlayerAction(Enum):
    """
    Enumeration of possible player actions during their turn.
    """
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all-in"

class GamePhase(Enum):
    """
    Enumeration of poker game phases from deal to showdown.
    """
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"

class Button:
    """
    Represents a clickable button in the game UI.
    """
    def __init__(self, x: int, y: int, width: int, height: int, text: str, color: Tuple[int, int, int]):
        """
        Initialize a button with position, size, text, and color.
        Args:
            x (int): X-coordinate position
            y (int): Y-coordinate position
            width (int): Button width
            height (int): Button height
            text (str): Button text
            color (Tuple[int, int, int]): RGB color tuple
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.is_hovered = False
        self.enabled = True

    def draw(self, screen, font):
        """
        Draw the button on the screen.
        Args:
            screen: Pygame screen surface
            font: Pygame font object for text rendering
        """
        if not self.enabled:
            # Gray out disabled buttons
            color = (128, 128, 128)
            text_color = (200, 200, 200)
        else:
            color = (min(self.color[0] + 30, 255), min(self.color[1] + 30, 255), min(self.color[2] + 30, 255)) if self.is_hovered else self.color
            text_color = (255, 255, 255)
        
        pygame.draw.rect(screen, color, self.rect)
        text_surface = font.render(self.text, True, text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

class Player:
    """
    Represents a poker player with their cards, stack, and game state.
    """
    def __init__(self, name: str, stack: int, position: int):
        """
        Initialize a player with name, starting stack, and table position.
        Args:
            name (str): Player's name
            stack (int): Starting chip stack
            position (int): Table position (0-2)
        """
        self.name = name
        self.stack = stack
        self.position = position
        self.cards: List[Card] = []
        self.is_active = True
        self.current_bet = 0
        self.is_human = True
        self.has_acted = False
        positions = [
            (550, 650),  # Bottom (Player 1)
            (100, 300),  # Left (Player 2)
            (1000, 300)  # Right (Player 3)
        ]
        self.x, self.y = positions[position]

class PokerGame:
    """
    Main game class that manages the poker game state and logic.
    """
    def __init__(self, num_players: int = 3, small_blind: int = 10, big_blind: int = 20):
        """
        Initialize the poker game with players and blinds.
        Args:
            num_players (int): Number of players (default: 3)
            small_blind (int): Starting small blind amount (default: 10)
            big_blind (int): Starting big blind amount (default: 20)
        """
        pygame.init()
        pygame.font.init()
        self.SCREEN_WIDTH = 1400
        self.SCREEN_HEIGHT = 900
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("3-Max Poker")
        self.font = pygame.font.SysFont('Arial', 24)
        self.num_players = 3
        
        # Structure des blindes
        self.blind_levels = [
            (10, 20),   # Niveau 1
            (15, 30),   # Niveau 2
            (20, 40),   # Niveau 3
            (30, 60),   # Niveau 4
            (40, 80),   # Niveau 5
            (50, 100),  # Niveau 6
            (60, 120),  # Niveau 7
            (80, 160),  # Niveau 8
            (100, 200)  # Niveau 9
        ]
        self.current_blind_level = 0
        self.hands_until_blind_increase = 4
        self.hands_played = 0
        
        self.small_blind = self.blind_levels[0][0]
        self.big_blind = self.blind_levels[0][1]
        self.starting_stack = 500  # Stack de départ fixé à 500 jetons
        
        self.pot = 0
        self.deck: List[Card] = self._create_deck()
        self.community_cards: List[Card] = []
        self.current_phase = GamePhase.PREFLOP
        self.players = self._initialize_players()
        self.button_position = rd.randint(0, self.num_players - 1)
        self.current_player_idx = (self.button_position + 1) % self.num_players
        self.current_bet = self.big_blind
        self.last_raiser = None
        self.clock = pygame.time.Clock()
        self.round_number = 0
        self.number_raise_this_round = 0
        
        # Initialize UI elements
        self.action_buttons = self._create_action_buttons()
        self.bet_slider = pygame.Rect(50, self.SCREEN_HEIGHT - 100, 200, 20)
        self.current_bet_amount = self.big_blind
        
        # Add action history tracking
        self.action_history = []
        
        # Add winner info tracking
        self.winner_info = None
        
        # Add winner display timing
        self.winner_display_start = 0
        self.winner_display_duration = 5000  # 5 seconds in milliseconds

        # Add last AI action time
        self.last_ai_action_time = 0
        
        self.start_new_hand()

    def reset(self):
        """
        Reset the game state for a new hand. Reinitializes the game.
        """
        # Reset game state variables
        self.pot = 0
        self.community_cards = []
        self.current_phase = GamePhase.PREFLOP
        self.deck = self._create_deck()
        self.winner_info = None
        self.winner_display_start = 0
        self.round_number = 0
        self.number_raise_this_round = 0
        self.action_history = []
        
        # Reset player states
        for player in self.players:
            player.cards = []
            player.current_bet = 0
            player.is_active = True
            player.has_acted = False
            player.stack = 500  # Reset to starting stack
        
        # Reset button and blinds
        self.button_position = (self.button_position + 1) % self.num_players
        sb_pos = (self.button_position + 1) % self.num_players
        bb_pos = (self.button_position + 2) % self.num_players
        
        # Post blinds
        self.players[sb_pos].stack -= self.small_blind
        self.players[sb_pos].current_bet = self.small_blind
        self.players[bb_pos].stack -= self.big_blind
        self.players[bb_pos].current_bet = self.big_blind
        
        self.pot = self.small_blind + self.big_blind
        self.current_bet = self.big_blind
        
        # Deal cards
        self.deal_cards()
        
        # Set starting player (UTG)
        self.current_player_idx = (bb_pos + 1) % self.num_players
        
        # Reset betting state
        self.last_raiser = None
        
        return self.get_state()

    def start_new_hand(self):
        """
        Reset game state and start a new hand.
        Posts blinds, deals cards, and sets up the initial betting round.
        """
        # Update blinds before starting new hand
        self.update_blinds()
        
        # Reset game state
        self.pot = 0
        self.community_cards = []
        self.current_phase = GamePhase.PREFLOP
        self.current_bet = self.big_blind
        self.last_raiser = None
        self.round_number = 0
        
        # Reset player states and check for bankrupted players
        active_players_count = 0
        for player in self.players:
            player.cards = []
            player.current_bet = 0
            player.has_acted = False
            if player.stack >= self.big_blind:
                player.is_active = True
                active_players_count += 1
            else:
                player.is_active = False
                print(f"{player.name} is out of the game (insufficient funds: ${player.stack})")
        
        # Check if we have enough players to continue
        if active_players_count < 2:
            print(f"Not enough players with sufficient funds ({active_players_count}). Need at least 2 players.")
            self.reset()
            return
        
        # Move button position
        self.button_position = (self.button_position + 1) % self.num_players
        
        # En heads-up (2 joueurs actifs), le bouton est la petite blind
        active_players = [p for p in self.players if p.is_active]
        if len(active_players) == 2:
            # Le bouton (petite blind) agit en premier preflop, mais reçoit ses cartes en dernier
            sb_pos = self.button_position  # Le bouton est la petite blind
            bb_pos = (self.button_position + 1) % self.num_players
            
            # Post blinds
            self.players[sb_pos].stack -= self.small_blind
            self.players[sb_pos].current_bet = self.small_blind
            self.players[bb_pos].stack -= self.big_blind
            self.players[bb_pos].current_bet = self.big_blind
            
            self.pot = self.small_blind + self.big_blind
            
            # Deal cards - BB reçoit en premier, SB en dernier
            self.deal_cards_heads_up(bb_pos, sb_pos)
            
            # En preflop, BB agit en premier
            self.current_player_idx = bb_pos
        else:
            # Plus de 2 joueurs - logique normale
            sb_pos = (self.button_position + 1) % self.num_players
            bb_pos = (self.button_position + 2) % self.num_players
            
            # Post blinds
            self.players[sb_pos].stack -= self.small_blind
            self.players[sb_pos].current_bet = self.small_blind
            self.players[bb_pos].stack -= self.big_blind
            self.players[bb_pos].current_bet = self.big_blind
            
            self.pot = self.small_blind + self.big_blind
            
            # Deal cards normalement
            self.deal_cards()
            
            # Set starting player (UTG)
            self.current_player_idx = (bb_pos + 1) % self.num_players

    def deal_cards_heads_up(self, bb_pos: int, sb_pos: int):
        """
        Deal cards specifically for heads-up play.
        BB receives cards first, SB (button) receives last.
        """
        self.deck = self._create_deck()
        
        # Clear previous hands
        for player in self.players:
            player.cards = []
        self.community_cards = []
        
        # Deal two cards to each active player in the correct order
        for _ in range(2):
            # BB gets cards first
            self.players[bb_pos].cards.append(self.deck.pop())
            # SB (button) gets cards last
            self.players[sb_pos].cards.append(self.deck.pop())

    def evaluate_hand(self, player: Player) -> Tuple[HandRank, List[int]]:
        """
        Evaluate a player's best possible poker hand.
        Args:
            player (Player): The player whose hand to evaluate
        Returns:
            Tuple[HandRank, List[int]]: The hand rank and list of card values for comparison
        """
        all_cards = player.cards + self.community_cards
        values = [card.value for card in all_cards]
        suits = [card.suit for card in all_cards]
        
        # Check for flush
        suit_counts = Counter(suits)
        flush_suit = next((suit for suit, count in suit_counts.items() if count >= 5), None)
        
        # Check for straight
        unique_values = sorted(set(values))
        straight = False
        straight_high = None
        for i in range(len(unique_values) - 4):
            if unique_values[i+4] - unique_values[i] == 4:
                straight = True
                straight_high = unique_values[i+4]
        
        # Special case for Ace-low straight
        if set([14, 2, 3, 4, 5]).issubset(set(values)):
            straight = True
            straight_high = 5
        
        # Count values
        value_counts = Counter(values)
        
        # Determine hand rank
        if straight and flush_suit:
            flush_cards = [card for card in all_cards if card.suit == flush_suit]
            if straight_high == 14 and all(v in [10, 11, 12, 13, 14] for v in values):
                return (HandRank.ROYAL_FLUSH, [14])
            return (HandRank.STRAIGHT_FLUSH, [straight_high])
        
        if 4 in value_counts.values():
            quads = [v for v, count in value_counts.items() if count == 4][0]
            return (HandRank.FOUR_OF_A_KIND, [quads])
        
        if 3 in value_counts.values() and 2 in value_counts.values():
            trips = [v for v, count in value_counts.items() if count == 3][0]
            pair = [v for v, count in value_counts.items() if count == 2][0]
            return (HandRank.FULL_HOUSE, [trips, pair])
        
        if flush_suit:
            flush_cards = sorted([card.value for card in all_cards if card.suit == flush_suit], reverse=True)
            return (HandRank.FLUSH, flush_cards[:5])
        
        if straight:
            return (HandRank.STRAIGHT, [straight_high])
        
        if 3 in value_counts.values():
            trips = [v for v, count in value_counts.items() if count == 3][0]
            kickers = sorted([v for v in values if v != trips], reverse=True)[:2]
            return (HandRank.THREE_OF_A_KIND, [trips] + kickers)
        
        pairs = [v for v, count in value_counts.items() if count == 2]
        if len(pairs) >= 2:
            pairs.sort(reverse=True)
            kicker = max(v for v in values if v not in pairs[:2])
            return (HandRank.TWO_PAIR, pairs[:2] + [kicker])
        
        if pairs:
            kickers = sorted([v for v in values if v != pairs[0]], reverse=True)[:3]
            return (HandRank.PAIR, pairs + kickers)
        
        return (HandRank.HIGH_CARD, sorted(values, reverse=True)[:5])

    def check_round_completion(self):
        """
        Check if the current betting round is complete.
        Returns:
            bool: True if round is complete, False otherwise
        """
        active_players = [p for p in self.players if p.is_active]
        
        # If only one player remains, round is complete
        if len(active_players) == 1:
            return True
        
        # If only two players remain and both are all-in, proceed to showdown
        if len(active_players) == 2:
            all_in_players = [p for p in active_players if p.stack == 0]
            if len(all_in_players) == 2:
                print("Two players remain and both are all-in - proceeding to showdown")
                # Deal remaining community cards if needed
                while len(self.community_cards) < 5:
                    self.community_cards.append(self.deck.pop())
                self.handle_showdown()
                return True
        
        # Check if all active players have acted and bets are equal
        all_acted = all(p.has_acted for p in active_players)
        bets_equal = len(set(p.current_bet for p in active_players)) == 1
        
        return all_acted and bets_equal

    def advance_phase(self):
        """
        Move the game to the next phase and deal appropriate community cards.
        Resets player states and betting for the new phase.
        """
        print(f"current_phase {self.current_phase}")
        
        # Check if all active players are all-in
        active_players = [p for p in self.players if p.is_active]
        all_in_players = [p for p in active_players if p.stack == 0]
        
        if len(all_in_players) == len(active_players) and len(active_players) > 1:
            print("All players are all-in - proceeding directly to showdown")
            # Deal all remaining community cards
            while len(self.community_cards) < 5:
                self.community_cards.append(self.deck.pop())
            self.handle_showdown()
            return
        
        # Normal phase progression
        if self.current_phase == GamePhase.PREFLOP:
            self.current_phase = GamePhase.FLOP
        elif self.current_phase == GamePhase.FLOP:
            self.current_phase = GamePhase.TURN
        elif self.current_phase == GamePhase.TURN:
            self.current_phase = GamePhase.RIVER
        
        # Increment round number when moving to a new phase
        self.round_number += 1
        self.number_raise_this_round = 0
        
        # Deal community cards for the new phase
        self.deal_community_cards()
        
        # Reset betting for new phase
        self.current_bet = 0
        for player in self.players:
            if player.is_active:
                player.has_acted = False
                player.current_bet = 0
        
        # Set first player after dealer button
        self.current_player_idx = (self.button_position + 1) % self.num_players
        while not self.players[self.current_player_idx].is_active:
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players

    def process_action(self, player: Player, action: PlayerAction, bet_amount: Optional[int] = None):
        """
        Process a player's action during their turn.
        """
        # Don't process actions during showdown
        if self.current_phase == GamePhase.SHOWDOWN:
            return action
            
        # Debug print for action start
        print(f"\n=== Action by {player.name} ===")
        print(f"Action: {action.value}")
        print(f"Current phase: {self.current_phase}")
        print(f"Current pot: ${self.pot}")
        print(f"Current bet: ${self.current_bet}")
        print(f"Player stack before: ${player.stack}")
        print(f"Player current bet: ${player.current_bet}")
        
        # Record the action
        action_text = f"{player.name}: {action.value}"
        if bet_amount is not None and action == PlayerAction.RAISE:
            action_text += f" ${bet_amount}"
        elif action == PlayerAction.RAISE:
            # Calculate minimum and maximum possible raise amounts
            min_raise = max(self.current_bet * 2, self.big_blind * 2)
            bet_amount = min_raise

            action_text += f" ${bet_amount}"

        self.action_history.append(action_text)
        if len(self.action_history) > 10:
            self.action_history.pop(0)
        
        # Process the action
        if action == PlayerAction.FOLD:
            player.is_active = False
            print(f"{player.name} folds")
            
        elif action == PlayerAction.CHECK:
            player.has_acted = True
            print(f"{player.name} checks")
            
        elif action == PlayerAction.CALL:
            call_amount = self.current_bet - player.current_bet
            player.stack -= call_amount
            player.current_bet = self.current_bet
            self.pot += call_amount
            print(f"{player.name} calls ${call_amount}")
            
        elif action == PlayerAction.RAISE and bet_amount is not None:
            total_to_put_in = bet_amount - player.current_bet
            player.stack -= total_to_put_in
            player.current_bet = bet_amount
            self.current_bet = bet_amount
            self.pot += total_to_put_in
            self.last_raiser = player
            print(f"{player.name} raises to ${bet_amount}")
            for p in self.players:
                if p != player and p.is_active:
                    p.has_acted = False
        
        elif action == PlayerAction.ALL_IN:
            all_in_amount = player.stack + player.current_bet
            total_to_put_in = player.stack
            player.stack = 0
            player.current_bet = all_in_amount
            self.pot += total_to_put_in
            
            if all_in_amount > self.current_bet:
                self.current_bet = all_in_amount
                self.last_raiser = player
                for p in self.players:
                    if p != player and p.is_active:
                        p.has_acted = False
            
            print(f"{player.name} fait tapis avec ${all_in_amount}")
        
        player.has_acted = True
        
        # Debug print post-action state
        print(f"Player stack after: ${player.stack}")
        print(f"New pot: ${self.pot}")
        print(f"Active players: {sum(1 for p in self.players if p.is_active)}")
        
        # Check for all-in situations after the action
        active_players = [p for p in self.players if p.is_active]
        all_in_players = [p for p in active_players if p.stack == 0]
        
        # If all active players are all-in or only one player remains active, go to showdown
        if (len(all_in_players) == len(active_players) and len(active_players) > 1) or len(active_players) == 1:
            print("Proceeding to showdown (all players all-in or only one player remains)")
            while len(self.community_cards) < 5:
                self.community_cards.append(self.deck.pop())
            self.handle_showdown()
            return action
        
        # Check if round is complete and handle next phase
        if self.check_round_completion():
            print("Round complete - advancing phase")
            if self.current_phase == GamePhase.RIVER:
                print("River complete - going to showdown")
                self.handle_showdown()
            else:
                self.advance_phase()
                print(f"Advanced to {self.current_phase}")
        else:
            self._next_player()
            print(f"Next player: {self.players[self.current_player_idx].name}")
        
        return action

    def handle_showdown(self):
        """
        Handle the showdown phase where remaining players reveal their hands.
        Evaluates hands, determines winner(s), and awards the pot.
        """
        print("\n=== SHOWDOWN ===")
        self.current_phase = GamePhase.SHOWDOWN
        active_players = [p for p in self.players if p.is_active]
        print(f"Active players in showdown: {[p.name for p in active_players]}")
        
        # Disable all action buttons during showdown
        for button in self.action_buttons.values():
            button.enabled = False
        
        if len(active_players) == 1:
            winner = active_players[0]
            winner.stack += self.pot
            self.winner_info = f"{winner.name} wins ${self.pot} (all others folded)"
            print(f"Winner by fold: {winner.name}")
            print(f"Winning amount: ${self.pot}")
        else:
            # Make sure all community cards are dealt for all-in situations
            while len(self.community_cards) < 5:
                self.community_cards.append(self.deck.pop())
            
            # Evaluate hands and find winner
            player_hands = [(player, self.evaluate_hand(player)) for player in active_players]
            for player, (hand_rank, _) in player_hands:
                print(f"{player.name}'s hand: {[str(card) for card in player.cards]}")
                print(f"{player.name}'s hand rank: {hand_rank.name}")
            
            player_hands.sort(key=lambda x: (x[1][0].value, x[1][1]), reverse=True)
            winner = player_hands[0][0]
            winner.stack += self.pot
            winning_hand = player_hands[0][1][0].name.replace('_', ' ').title()
            self.winner_info = f"{winner.name} wins ${self.pot} with {winning_hand}"
            print(f"Winner at showdown: {winner.name}")
            print(f"Winning hand: {winning_hand}")
            print(f"Winning amount: ${self.pot}")
        
        # Set the winner display start time
        self.winner_display_start = pygame.time.get_ticks()
        self.winner_display_duration = 5000  # 5 seconds in milliseconds

    def _create_action_buttons(self) -> Dict[PlayerAction, Button]:
        """
        Create and initialize the action buttons for player interaction.
        Returns:
            Dict[PlayerAction, Button]: Dictionary mapping actions to button objects
        """
        buttons = {
            PlayerAction.FOLD: Button(300, self.SCREEN_HEIGHT - 100, 100, 40, "Fold", (200, 0, 0)),
            PlayerAction.CHECK: Button(450, self.SCREEN_HEIGHT - 100, 100, 40, "Check", (0, 200, 0)),
            PlayerAction.CALL: Button(600, self.SCREEN_HEIGHT - 100, 100, 40, "Call", (0, 0, 200)),
            PlayerAction.RAISE: Button(750, self.SCREEN_HEIGHT - 100, 100, 40, "Raise", (200, 200, 0)),
            PlayerAction.ALL_IN: Button(900, self.SCREEN_HEIGHT - 100, 100, 40, "All-in", (150, 0, 150))
        }
        return buttons

    def _create_deck(self) -> List[Card]:
        """
        Create and shuffle a new deck of 52 cards.
        Returns:
            List[Card]: A shuffled deck of cards
        """
        suits = ['♠', '♥', '♦', '♣']
        values = range(2, 15)  # 2-14 (Ace is 14)
        deck = [Card(suit, value) for suit in suits for value in values]
        rd.shuffle(deck)
        return deck
    
    def _initialize_players(self) -> List[Player]:
        """
        Create and initialize all players for the game.
        Returns:
            List[Player]: List of initialized player objects
        """
        players = []
        for i in range(self.num_players):
            player = Player(f"Player {i+1}", self.starting_stack, i)
            players.append(player)
        return players
    
    def deal_cards(self):
        """
        Deal two hole cards to each active player.
        Resets and shuffles the deck before dealing.
        """
        self.deck = self._create_deck()  # Reset and shuffle deck
        # Clear previous hands
        for player in self.players:
            player.cards = []
        self.community_cards = []
        
        # Deal two cards to each active player
        for _ in range(2):
            for player in self.players:
                if player.is_active:
                    player.cards.append(self.deck.pop())
    
    def deal_community_cards(self):
        """
        Deal community cards based on the current game phase.
        Deals 3 cards for flop, 1 for turn, and 1 for river.
        """
        if self.current_phase == GamePhase.FLOP:
            for _ in range(3):
                self.community_cards.append(self.deck.pop())
        elif self.current_phase in [GamePhase.TURN, GamePhase.RIVER]:
            self.community_cards.append(self.deck.pop())
    
    def _draw_card(self, card: Card, x: int, y: int):
        """
        Draw a single card on the screen.
        Args:
            card (Card): The card to draw
            x (int): X-coordinate for card position
            y (int): Y-coordinate for card position
        """
        # Draw card background
        card_width, card_height = 50, 70
        pygame.draw.rect(self.screen, (255, 255, 255), (x, y, card_width, card_height))
        pygame.draw.rect(self.screen, (0, 0, 0), (x, y, card_width, card_height), 1)
        
        # Draw card text
        color = (255, 0, 0) if card.suit in ['♥', '♦'] else (0, 0, 0)
        text = self.font.render(str(card), True, color)
        self.screen.blit(text, (x + 5, y + 5))
    
    def _draw_player(self, player: Player):
        """
        Draw a player's information and cards on the screen.
        Args:
            player (Player): The player to draw
        """
        # Draw neon effect for active player
        if player.position == self.current_player_idx:
            # Draw multiple circles with decreasing alpha for glow effect
            for radius in range(40, 20, -5):
                glow_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                alpha = int(255 * (1 - (radius - 20) / 20))  # Fade from center
                pygame.draw.circle(glow_surface, (0, 255, 255, alpha), (radius, radius), radius)  # Cyan glow
                self.screen.blit(glow_surface, (player.x + 25 - radius, player.y + 35 - radius))
        
        # Draw player info with 2 decimal places
        name_color = (0, 255, 255) if player.position == self.current_player_idx else (255, 255, 255)
        name_text = self.font.render(f"{player.name} (${player.stack:.2f})", True, name_color)
        self.screen.blit(name_text, (player.x - 50, player.y - 40))
        
        # Draw player cards
        if player.is_active and player.cards:
            if player.is_human or self.current_phase == GamePhase.SHOWDOWN:
                for i, card in enumerate(player.cards):
                    self._draw_card(card, player.x + i * 60, player.y)
            else:
                # Draw card backs for non-human players
                for i in range(2):
                    pygame.draw.rect(self.screen, (200, 0, 0), (player.x + i * 60, player.y, 50, 70))
        
        # Draw current bet with 2 decimal places
        if player.current_bet > 0:
            bet_text = self.font.render(f"Bet: ${player.current_bet:.2f}", True, (255, 255, 0))
            self.screen.blit(bet_text, (player.x - 30, player.y + 80))
    
    def _draw(self):
        """
        Draw the complete game state on the screen.
        Includes table, players, cards, pot, and UI elements.
        """
        # Clear screen
        self.screen.fill((0, 100, 0))  # Green felt background
        
        # Draw table (smaller for 3 players)
        pygame.draw.ellipse(self.screen, (139, 69, 19), (200, 100, 800, 500))
        pygame.draw.ellipse(self.screen, (165, 42, 42), (220, 120, 760, 460))
        
        # Draw community cards
        for i, card in enumerate(self.community_cards):
            self._draw_card(card, 400 + i * 60, 350)
        
        # Draw pot with 2 decimal places
        pot_text = self.font.render(f"Pot: ${self.pot:.2f}", True, (255, 255, 255))
        self.screen.blit(pot_text, (550, 300))
        
        # Draw players
        for player in self.players:
            self._draw_player(player)
        
        # Draw current player indicator in bottom right
        current_player = self.players[self.current_player_idx]
        current_player_text = self.font.render(f"Current Player: {current_player.name}", True, (255, 255, 255))
        self.screen.blit(current_player_text, (self.SCREEN_WIDTH - 300, self.SCREEN_HEIGHT - 50))
        
        # Draw dealer button (D)
        button_player = self.players[self.button_position]
        button_x = button_player.x + 52
        button_y = button_player.y + 80
        pygame.draw.circle(self.screen, (255, 255, 255), (button_x, button_y), 15)
        dealer_text = self.font.render("D", True, (0, 0, 0))
        dealer_rect = dealer_text.get_rect(center=(button_x, button_y))
        self.screen.blit(dealer_text, dealer_rect)
        
        # Update button states before drawing
        self._update_button_states()
        
        # Draw action buttons for current player's turn
        for button in self.action_buttons.values():
            button.draw(self.screen, self.font)
        
        # Draw bet slider with min and max values
        current_player = self.players[self.current_player_idx]
        min_raise = max(self.current_bet * 2, self.big_blind * 2)
        max_raise = current_player.stack + current_player.current_bet
        
        pygame.draw.rect(self.screen, (200, 200, 200), self.bet_slider)
        bet_text = self.font.render(f"Bet: ${int(self.current_bet_amount)}", True, (255, 255, 255))
        min_text = self.font.render(f"Min: ${min_raise}", True, (255, 255, 255))
        max_text = self.font.render(f"Max: ${max_raise}", True, (255, 255, 255))
        
        self.screen.blit(bet_text, (50, self.SCREEN_HEIGHT - 75))
        self.screen.blit(min_text, (self.bet_slider.x, self.SCREEN_HEIGHT - 125))
        self.screen.blit(max_text, (self.bet_slider.x, self.SCREEN_HEIGHT - 150))
        
        # Draw action history in top right corner
        history_x = self.SCREEN_WIDTH - 200
        history_y = 50
        history_text = self.font.render("Action History:", True, (255, 255, 255))
        self.screen.blit(history_text, (history_x, history_y - 30))
        
        for i, action in enumerate(self.action_history):
            text = self.font.render(action, True, (255, 255, 255))
            self.screen.blit(text, (history_x, history_y + i * 25))
        
        # Draw game info
        game_info_text = self.font.render(f"Game Info: {self.current_phase}", True, (255, 255, 255))
        self.screen.blit(game_info_text, (50, 50))
        
        # Draw winner announcement if there is one and within display duration
        if self.winner_info:
            current_time = pygame.time.get_ticks()
            if current_time - self.winner_display_start < self.winner_display_duration:
                # Create semi-transparent overlay
                overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
                overlay.fill((0, 0, 0))
                overlay.set_alpha(128)
                self.screen.blit(overlay, (0, 0))
                
                # Draw winner text with shadow for better visibility
                winner_font = pygame.font.SysFont('Arial', 48, bold=True)
                shadow_text = winner_font.render(self.winner_info, True, (0, 0, 0))  # Shadow
                winner_text = winner_font.render(self.winner_info, True, (255, 215, 0))  # Gold color
                
                # Position for center of screen
                text_rect = winner_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
                
                # Draw shadow slightly offset
                shadow_rect = text_rect.copy()
                shadow_rect.x += 2
                shadow_rect.y += 2
                self.screen.blit(shadow_text, shadow_rect)
                
                # Draw main text
                self.screen.blit(winner_text, text_rect)
            else:
                # After display duration, start new hand
                self.winner_info = None
                self.button_position = (self.button_position + 1) % self.num_players
                self.start_new_hand()

        # Ajouter l'affichage des blindes actuelles
        blind_text = self.font.render(f"Blindes: {self.small_blind}/{self.big_blind}", True, (255, 255, 255))
        self.screen.blit(blind_text, (50, 25))
        
        hands_left_text = self.font.render(
            f"Mains avant augmentation: {self.hands_until_blind_increase - (self.hands_played % self.hands_until_blind_increase)}", 
            True, 
            (255, 255, 255)
        )
        self.screen.blit(hands_left_text, (50, 75))

    def handle_input(self, event):
        """
        Handle player input events.
        Args:
            event: Pygame event object to process
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            current_player = self.players[self.current_player_idx]
            
            # Check button clicks
            for action, button in self.action_buttons.items():
                if button.rect.collidepoint(mouse_pos) and button.enabled:
                    bet_amount = self.current_bet_amount if action == PlayerAction.RAISE else None
                    # Validate bet amount doesn't exceed player's stack
                    if action == PlayerAction.RAISE:
                        max_bet = current_player.stack + current_player.current_bet
                        min_bet = max(self.current_bet * 2, self.big_blind * 2)
                        bet_amount = min(bet_amount, max_bet)
                        bet_amount = max(bet_amount, min_bet)
                    self.process_action(current_player, action, bet_amount)
            
            # Check bet slider
            if self.bet_slider.collidepoint(mouse_pos):
                # Calculate minimum raise (2x current bet)
                min_raise = max(self.current_bet * 2, self.big_blind * 2)
                # Calculate maximum raise (player's stack + current bet)
                max_raise = current_player.stack + current_player.current_bet
                
                # Calculate bet amount based on slider position
                slider_value = (mouse_pos[0] - self.bet_slider.x) / self.bet_slider.width
                bet_range = max_raise - min_raise
                self.current_bet_amount = min(min_raise + (bet_range * slider_value), max_raise)
                self.current_bet_amount = max(self.current_bet_amount, min_raise)

    def _next_player(self):
        """
        Move to the next active player in the game.
        Skips players who have folded.
        """
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        while not self.players[self.current_player_idx].is_active:
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players

    def _update_button_states(self):
        """
        Update the enabled/disabled state of action buttons based on current game state.
        """
        current_player = self.players[self.current_player_idx]
        
        # First check if all active players are all-in
        active_players = [p for p in self.players if p.is_active]
        all_in_players = [p for p in active_players if p.stack == 0]
        
        if len(all_in_players) == len(active_players) and len(active_players) > 1:
            # Disable all buttons as we should auto-proceed to showdown
            for button in self.action_buttons.values():
                button.enabled = False
            return
        
        # Enable all buttons by default
        for button in self.action_buttons.values():
            button.enabled = True
        
        # Disable check if there's a bet to call
        if current_player.current_bet < self.current_bet:
            self.action_buttons[PlayerAction.CHECK].enabled = False
        
        # Disable call if no bet to call or not enough chips
        if current_player.current_bet == self.current_bet:
            self.action_buttons[PlayerAction.CALL].enabled = False
        elif current_player.stack < (self.current_bet - current_player.current_bet):
            self.action_buttons[PlayerAction.CALL].enabled = False
        
        # Disable raise if not enough chips for minimum raise
        min_raise = max(self.current_bet * 2, self.big_blind * 2)
        if current_player.stack + current_player.current_bet < min_raise:
            self.action_buttons[PlayerAction.RAISE].enabled = False
        
        # Also if the player has already raised 3 times, disable raise
        if self.number_raise_this_round >= 4:
            self.action_buttons[PlayerAction.RAISE].enabled = False

        # If Check is enabled, disable fold
        if self.action_buttons[PlayerAction.CHECK].enabled:
            self.action_buttons[PlayerAction.FOLD].enabled = False
        
        # All-in toujours disponible si le joueur a des jetons
        self.action_buttons[PlayerAction.ALL_IN].enabled = current_player.stack > 0
    
    def get_state(self):
        """
        Get the current state of the game for the RL agent.
        Returns a list containing:
        - Cards info (player's cards and community cards)
        - Current hand rank
        - Bets info (current bets of all players)
        - Position info (relative positions of players)
        - Activity info (which players are still active)
        - Round info (current phase, current bet)
        - Available actions (which actions are valid)
        - Previous actions
        - Money left (stack sizes)
        """
        current_player = self.players[self.current_player_idx]
        state = []

        # 1. Cards information (normalized values 2-14 -> 0-1)
        # Player's cards
        for card in current_player.cards:
            state.append((card.value - 2) / 12)  # Normalize card values
        
        # Community cards
        flop = [-1] * 3
        turn = [-1]
        river = [-1]
        for i, card in enumerate(self.community_cards):
            if i < 3:
                flop[i] = (card.value - 2) / 12
            elif i == 3:
                turn[0] = (card.value - 2) / 12
            else:
                river[0] = (card.value - 2) / 12
        state.extend(flop + turn + river)

        # 2. Current hand rank (if enough cards are visible)
        if len(current_player.cards) + len(self.community_cards) >= 5:
            hand_rank, _ = self.evaluate_hand(current_player)
            state.append(hand_rank.value / len(HandRank))  # Normalize rank value
        else:
            state.append(-1)  # No hand rank available yet

        # 3. Round information
        phase_values = {
            GamePhase.PREFLOP: 0,
            GamePhase.FLOP: 0.25,
            GamePhase.TURN: 0.5,
            GamePhase.RIVER: 0.75,
            GamePhase.SHOWDOWN: 1
        }
        state.append(phase_values[self.current_phase])

        # 4. Round number
        state.append(self.round_number)

        # 5. Current bet normalized by big blind
        state.append(self.current_bet / self.big_blind)

        # 6. Money left (stack sizes normalized by initial stack)
        initial_stack = 200 * self.big_blind
        for player in self.players:
            state.append(player.stack / initial_stack)

        # 7. Bets information (normalized by big blind)
        for player in self.players:
            state.append(player.current_bet / self.big_blind)

        # 8. Activity information (extreme binary: active/folded)
        for player in self.players:
            state.append(1 if player.is_active else -1)

        # 9. Position information (one-hot encoded relative positions)
        relative_positions = [0.1] * self.num_players
        for i in range(self.num_players):
            relative_pos = (i - self.current_player_idx) % self.num_players
            relative_positions[relative_pos] = 1
        state.extend(relative_positions)

        # 10. Available actions (extreme binary: available/unavailable)
        action_availability = []
        for action in PlayerAction:
            if action in self.action_buttons and self.action_buttons[action].enabled:
                action_availability.append(1)
            else:
                action_availability.append(-1)
        state.extend(action_availability)

        # 11. Previous actions (last action of each player, encoded)
        action_encoding = {
            None: 0,
            PlayerAction.FOLD: 1,
            PlayerAction.CHECK: 2,
            PlayerAction.CALL: 3,
            PlayerAction.RAISE: 4,
            PlayerAction.ALL_IN: 5  # Added ALL_IN action encoding
        }
        last_actions = [0] * self.num_players
        for action_text in reversed(self.action_history[-self.num_players:]):
            if ":" in action_text:
                player_name, action = action_text.split(":")
                player_idx = int(player_name.split("_")[-1]) - 1
                action = action.strip()
                for action_type in PlayerAction:
                    if action_type.value in action:
                        last_actions[player_idx] = action_encoding[action_type]
                        
        state.extend(last_actions)

        # 12. Win probability estimation
        active_players = [p for p in self.players if p.is_active]
        num_opponents = len(active_players) - 1
        
        if num_opponents <= 0:
            win_prob = 1.0  # No opponents left
        else:
            # Get hand strength
            hand_strength = self._evaluate_hand_strength(current_player)
            
            # Adjust for number of opponents (2-3)
            win_prob = hand_strength ** num_opponents
            
            # Pre-flop specific adjustment (using Sklansky-Chubukov approximation)
            if self.current_phase == GamePhase.PREFLOP:
                # Reduce confidence in pre-flop estimates
                win_prob *= 0.8
        
        state.append(win_prob)

        return state

    def step(self, action: PlayerAction) -> Tuple[List[float], float]:
        """
        Execute an action and return the new state with the calculated reward.
        The reward considers hand strength, positional play, pot odds, showdown outcomes, and chip changes.
        """
        current_player = self.players[self.current_player_idx]
        initial_stack = current_player.stack
        reward = 0.0

        # Capture game state before processing the action for pot odds calculation
        call_amount_before = self.current_bet - current_player.current_bet
        pot_before = self.pot

        # --- Strategic Action Rewards ---
        # Reward based on action relative to hand strength
        hand_strength = self._evaluate_hand_strength(current_player)
        pot_potential = self.pot / (self.big_blind * 100)
    
        if action == PlayerAction.RAISE:
            reward += 0.2 * hand_strength  # Scale reward with hand strength
            if hand_strength > 0.7:
                reward += 0.5 * pot_potential  # Bonus for aggressive play with strong hands
                
        elif action == PlayerAction.ALL_IN:
            reward += 0.3 * hand_strength
            if hand_strength > 0.8:
                reward += 1.0 * pot_potential
                
        elif action == PlayerAction.CALL:
            reward += 0.1 * min(hand_strength, 0.6)  # Diminished returns for passive play

        elif action == PlayerAction.FOLD:
            if hand_strength < 0.4:
                reward += 0.3  # Reward good folds
            else:
                reward -= 0.5  # Penalize folding decent hands

        # --- Positional Bonus ---
        # Bonus for aggressive actions in late position (button)
        if current_player.position == self.button_position:
            if action in [PlayerAction.RAISE, PlayerAction.ALL_IN]:
                reward += 0.2

        # --- Phase-Specific Adjustments ---
        # Penalize over-aggression in early phases
        if action in [PlayerAction.RAISE, PlayerAction.ALL_IN]:
            if self.current_phase in [GamePhase.PREFLOP, GamePhase.FLOP]:
                reward -= 0.2
            else:
                reward += 0.1  # Encourage aggression in later streets

        # Process the action (updates game state)
        self.process_action(current_player, action)

        # --- Pot Odds Evaluation ---
        if action == PlayerAction.CALL and call_amount_before > 0:
            total_pot_after_call = pot_before + call_amount_before
            pot_odds = call_amount_before / total_pot_after_call if total_pot_after_call > 0 else 0
            if hand_strength > pot_odds:
                reward += 0.3  # Mathematically sound call
            else:
                reward -= 0.3  # Poor call considering odds

        # --- Bankruptcy Penalty ---
        if current_player.stack <= 0:
            reward -= 5.0  # Severe penalty for busting

        # --- Core Chip Change Reward ---
        # Direct reward/penalty based on net chips gained/lost
        stack_change = current_player.stack - initial_stack / self.starting_stack
        reward += stack_change  # Normalized by current big blind

        return self.get_state(), reward

    def _evaluate_hand_strength(self, player) -> float:
        """
        Évalue la force relative d'une main (0 à 1)
        """
        if self.current_phase == GamePhase.PREFLOP:
            return self._evaluate_preflop_strength(player.cards)
        
        # Évaluer la main complète avec les cartes communes
        hand = self.evaluate_hand(player)
        # Normaliser le score entre 0 et 1
        return hand[0].value / 9  # 9 est la valeur max (ROYAL_FLUSH)

    def _evaluate_preflop_strength(self, cards) -> float:
        """
        Évalue la force d'une main preflop
        """
        card1, card2 = cards
        # Paires
        if card1.value == card2.value:
            return 0.5 + (card1.value / 28)  # Plus haute est la paire, plus fort est le score
        
        # Cartes assorties
        suited = card1.suit == card2.suit
        # Connecteurs
        connected = abs(card1.value - card2.value) == 1
        
        # Score de base basé sur les valeurs des cartes
        base_score = (card1.value + card2.value) / 28  # Normaliser par le max possible
        
        # Bonus pour suited et connected
        if suited:
            base_score += 0.1
        if connected:
            base_score += 0.05
        
        return min(base_score, 1.0)  # Garantir que le score est entre 0 et 1

    def manual_run(self):
        """
        Main game loop.
        Handles game flow, player turns, and updates display.
        """
        self.deal_cards()  # Initial deal
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_SPACE:
                        self.start_new_hand()
                    if event.key == pygame.K_r:
                        self.reset_game()
                    if event.key == pygame.K_s:
                        print(self.get_state())
                
                self.handle_input(event)
                
                # Update button hover states
                mouse_pos = pygame.mouse.get_pos()
                for button in self.action_buttons.values():
                    button.is_hovered = button.rect.collidepoint(mouse_pos)
            
            self._draw()
            pygame.display.flip()
        
        pygame.quit()

    def update_blinds(self):
        """
        Met à jour les blindes tous les 4 mains si possible
        """
        self.hands_played += 1
        if self.hands_played % self.hands_until_blind_increase == 0:
            if self.current_blind_level < len(self.blind_levels) - 1:
                self.current_blind_level += 1
                self.small_blind, self.big_blind = self.blind_levels[self.current_blind_level]
                print(f"Blindes augmentées à {self.small_blind}/{self.big_blind}")

if __name__ == "__main__":
    game = PokerGame()
    game.manual_run()

