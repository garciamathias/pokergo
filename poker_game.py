import pygame
import random
from enum import Enum
from typing import List, Dict, Optional, Tuple
import pygame.font
from collections import Counter

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
            position (int): Table position (0-5)
        """
        self.name = name
        self.stack = stack
        self.position = position
        self.cards: List[Card] = []
        self.is_active = True
        self.current_bet = 0
        self.is_human = True  # Make all players human
        self.has_acted = False
        self.is_all_in = False  # Nouveau statut pour le tapis
        positions = [
            (600, 700),  # Bottom (Player 1)
            (200, 600),  # Bottom Left (Player 2)
            (150, 300),  # Middle Left (Player 3)
            (600, 100),  # Top (Player 4)
            (1050, 300), # Middle Right (Player 5)
            (1000, 600)  # Bottom Right (Player 6)
        ]
        self.x, self.y = positions[position]

class PokerGame:
    """
    Main game class that manages the poker game state and logic.
    """
    def __init__(self, num_players: int = 6, small_blind: int = 1, big_blind: int = 2):
        """
        Initialize the poker game with players and blinds.
        Args:
            num_players (int): Number of players (default: 6)
            small_blind (int): Small blind amount (default: 1)
            big_blind (int): Big blind amount (default: 2)
        """
        pygame.init()
        pygame.font.init()
        self.SCREEN_WIDTH = 1400
        self.SCREEN_HEIGHT = 900
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("6-Max Poker")
        self.font = pygame.font.SysFont('Arial', 24)
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.pot = 0
        self.deck: List[Card] = self._create_deck()
        self.community_cards: List[Card] = []
        self.current_phase = GamePhase.PREFLOP
        self.players = self._initialize_players()
        self.button_position = 0
        self.current_player_idx = 0
        self.current_bet = self.big_blind
        self.last_raiser = None
        self.round_ended = False
        
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
        
        self.game_over = False  # Nouveau flag pour indiquer la fin du jeu
        self.game_over_message = None  # Message à afficher quand le jeu est terminé
        
        self.raise_count = 0  # Nombre de relances dans le tour actuel
        self.max_raises = 4  # Maximum de relances par tour (3 + mise initiale)
        
        self.start_new_hand()

    def start_new_hand(self):
        """
        Reset game state and start a new hand.
        """
        print("\n=== Starting New Hand ===")
        
        # Vérifier d'abord si le jeu peut continuer
        players_with_chips = [p for p in self.players if p.stack > 0]
        print(f"Players with chips: {[(p.name, p.stack) for p in players_with_chips]}")
        
        if len(players_with_chips) < 2:
            print("Not enough players with chips to continue!")
            self.game_over = True
            self.game_over_message = "Game Over - Not enough players with chips!"
            return
        
        # Reset game state
        self.community_cards = []
        self.current_phase = GamePhase.PREFLOP
        self.current_bet = self.big_blind
        self.last_raiser = None
        self.round_ended = False
        self.pot = 0
        
        # Reset player states
        for player in self.players:
            player.cards = []
            player.current_bet = 0
            player.has_acted = False
            player.is_all_in = False
            # Réinitialiser is_active basé sur le stack
            player.is_active = player.stack > 0
        
        print(f"Active players after reset: {[p.name for p in self.players if p.is_active]}")
        
        # Déterminer les joueurs actifs
        active_players = [p for p in self.players if p.is_active]
        print(f"Active players with chips: {[p.name for p in active_players]}")
        
        if len(active_players) < 2:
            print("Not enough players with chips to continue!")
            self.game_over = True
            self.game_over_message = "Game Over - Not enough players with chips!"
            return
        
        # Avancer le bouton
        self.button_position = (self.button_position + 1) % self.num_players
        while self.players[self.button_position].stack <= 0:
            self.button_position = (self.button_position + 1) % self.num_players
        
        print(f"Button moved to: Player {self.button_position + 1}")
        
        # Trouver SB et BB
        sb_pos = self.button_position
        bb_pos = self.button_position
        
        # Trouver la petite blind
        while True:
            sb_pos = (sb_pos + 1) % self.num_players
            if self.players[sb_pos].stack > 0:
                break
        
        # Trouver la grosse blind
        while True:
            bb_pos = (bb_pos + 1) % self.num_players
            if bb_pos != sb_pos and self.players[bb_pos].stack > 0:
                break
        
        # Poster les blindes
        sb_player = self.players[sb_pos]
        bb_player = self.players[bb_pos]
        
        # Gérer les all-ins sur les blindes
        sb_amount = min(self.small_blind, sb_player.stack)
        bb_amount = min(self.big_blind, bb_player.stack)
        
        sb_player.stack -= sb_amount
        sb_player.current_bet = sb_amount
        if sb_amount == sb_player.stack:
            sb_player.is_all_in = True
            print(f"Player {sb_pos + 1} is all-in on small blind")
        
        bb_player.stack -= bb_amount
        bb_player.current_bet = bb_amount
        if bb_amount == bb_player.stack:
            bb_player.is_all_in = True
            print(f"Player {bb_pos + 1} is all-in on big blind")
        
        self.pot = sb_amount + bb_amount
        print(f"Posted SB (${sb_amount}) from Player {sb_pos + 1}")
        print(f"Posted BB (${bb_amount}) from Player {bb_pos + 1}")
        print(f"Initial pot: ${self.pot}")
        
        # Trouver UTG (Under the Gun)
        utg_pos = bb_pos
        while True:
            utg_pos = (utg_pos + 1) % self.num_players
            if self.players[utg_pos].stack > 0:
                break
        
        self.current_player_idx = utg_pos
        print(f"First to act (UTG): Player {utg_pos + 1}")
        
        # Deal cards
        self.deal_cards()
        
        # Reset action history and winner info
        self.action_history = []
        self.winner_info = None

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
        active_players = [p for p in self.players if p.is_active and not p.is_all_in]
        
        # Si tous les joueurs sont à tapis ou ont fold sauf un, le tour est terminé
        if len(active_players) <= 1:
            return True
        
        # Vérifier si tous les joueurs actifs (non all-in) ont agi et ont misé le même montant
        all_acted = all(p.has_acted for p in active_players)
        bets_equal = len(set(p.current_bet for p in active_players)) == 1
        
        return all_acted and bets_equal

    def advance_phase(self):
        """
        Move to the next phase.
        """
        print(f"\n=== Advancing Phase ===")
        print(f"Current phase: {self.current_phase}")
        
        if self.current_phase == GamePhase.PREFLOP:
            self.current_phase = GamePhase.FLOP
            print("Dealing Flop")
        elif self.current_phase == GamePhase.FLOP:
            self.current_phase = GamePhase.TURN
            print("Dealing Turn")
        elif self.current_phase == GamePhase.TURN:
            self.current_phase = GamePhase.RIVER
            print("Dealing River")
        
        # Deal community cards
        self.deal_community_cards()
        print("Community cards:", [str(card) for card in self.community_cards])
        
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
        print(f"\n=== Processing Action ===")
        print(f"Player {player.name} action: {action.value}")
        
        if action == PlayerAction.RAISE and bet_amount is not None:
            # Calculer le montant minimum de relance
            previous_raise = self.current_bet - player.current_bet
            min_raise = self.current_bet + previous_raise
            
            if self.current_phase == GamePhase.PREFLOP and self.current_bet == self.big_blind:
                min_raise = self.big_blind * 2
            
            print(f"Minimum raise amount: ${min_raise}")
            print(f"Player stack: ${player.stack}")
            
            # Vérifier si le joueur a assez pour la mise minimale
            total_possible = player.stack + player.current_bet
            
            # Si le joueur n'a pas assez pour la mise minimale, autoriser l'all-in
            if total_possible <= min_raise:
                print(f"Player doesn't have enough for minimum raise, allowing all-in")
                bet_amount = total_possible
                player.stack = 0
                player.current_bet = bet_amount
                player.is_all_in = True
                self.current_bet = max(self.current_bet, bet_amount)
                self.pot += bet_amount - player.current_bet
                self.last_raiser = player
                print(f"{player.name} goes all-in for ${bet_amount}")
                print(f"Pot is now ${self.pot}")
            else:
                # Vérifier si le montant est valide pour une relance normale
                if bet_amount < min_raise:
                    print(f"Invalid raise amount: ${bet_amount} (minimum: ${min_raise})")
                    return False
                
                # Gérer la relance normale ou l'all-in volontaire
                total_to_put_in = bet_amount - player.current_bet
                if total_to_put_in >= player.stack:
                    print(f"{player.name} going all-in")
                    total_to_put_in = player.stack
                    bet_amount = player.current_bet + total_to_put_in
                    player.is_all_in = True
                
                player.stack -= total_to_put_in
                player.current_bet = bet_amount
                self.current_bet = bet_amount
                self.pot += total_to_put_in
                self.last_raiser = player
                
                print(f"{player.name} raises to ${bet_amount}")
                print(f"Pot is now ${self.pot}")
            
            # Réinitialiser les actions des autres joueurs
            for p in self.players:
                if p != player and p.is_active and not p.is_all_in:
                    p.has_acted = False
        else:
            # Record the action with bet amount if applicable
            action_text = f"{player.name}: {action.value}"
            if bet_amount is not None and action == PlayerAction.RAISE:
                action_text += f" ${bet_amount}"
            self.action_history.append(action_text)
            # Keep only the last 10 actions
            if len(self.action_history) > 10:
                self.action_history.pop(0)
            
            if action == PlayerAction.FOLD:
                print(f"{player.name} folds")
                player.is_active = False
            elif action == PlayerAction.CHECK:
                print(f"{player.name} checks")
                player.has_acted = True
            elif action == PlayerAction.CALL:
                call_amount = self.current_bet - player.current_bet
                if call_amount >= player.stack:
                    call_amount = player.stack
                    player.is_all_in = True
                    print(f"{player.name} calls all-in for ${call_amount}")
                else:
                    print(f"{player.name} calls ${call_amount}")
                player.stack -= call_amount
                player.current_bet += call_amount
                self.pot += call_amount
                print(f"Pot is now ${self.pot}")
        
        player.has_acted = True
        
        # Check if round is complete and handle next phase
        if self.check_round_completion():
            if self.current_phase == GamePhase.RIVER:
                self.handle_showdown()
            else:
                self.advance_phase()
        else:
            self._next_player()

    def handle_showdown(self):
        """
        Handle the showdown phase and side pots.
        """
        print("\n=== Showdown ===")
        self.current_phase = GamePhase.SHOWDOWN  # Ajout de cette ligne importante
        active_players = [p for p in self.players if p.is_active]
        print(f"Active players: {[p.name for p in active_players]}")
        
        if len(active_players) == 1:
            winner = active_players[0]
            winner.stack += self.pot
            print(f"{winner.name} wins ${self.pot} uncontested")
            self.winner_info = f"{winner.name} wins ${self.pot}"
            return
        
        # Afficher les mains des joueurs
        print("\nPlayer hands:")
        for player in active_players:
            hand_rank, values = self.evaluate_hand(player)
            print(f"{player.name}: {[str(card) for card in player.cards]} - {hand_rank.name}")
        
        print("\nCommunity cards:", [str(card) for card in self.community_cards])
        
        # Créer les pots divisés
        all_in_players = sorted(
            [p for p in active_players if p.is_all_in],
            key=lambda p: p.current_bet
        )
        
        side_pots = []
        remaining_pot = self.pot
        previous_bet = 0
        
        # Calculer les side pots pour chaque joueur all-in
        for all_in_player in all_in_players:
            current_bet = all_in_player.current_bet
            pot_contribution = current_bet - previous_bet
            eligible_players = [p for p in active_players if p.current_bet >= current_bet]
            pot_size = pot_contribution * len(eligible_players)
            remaining_pot -= pot_size
            side_pots.append((pot_size, eligible_players))
            previous_bet = current_bet
        
        # Ajouter le pot principal s'il reste des jetons
        if remaining_pot > 0:
            non_all_in_players = [p for p in active_players if not p.is_all_in]
            side_pots.append((remaining_pot, non_all_in_players))
        
        # Distribuer chaque pot
        winners_info = []
        for pot_size, eligible_players in side_pots:
            if pot_size > 0:
                player_hands = [(p, self.evaluate_hand(p)) for p in eligible_players]
                player_hands.sort(key=lambda x: (x[1][0].value, x[1][1]), reverse=True)
                best_hand = player_hands[0][1]
                pot_winners = [p for p, hand in player_hands if hand == best_hand]
                split_amount = pot_size / len(pot_winners)
                
                for winner in pot_winners:
                    winner.stack += split_amount
                    winners_info.append(f"{winner.name} wins ${split_amount:.2f}")
        
        self.winner_info = " | ".join(winners_info)
        
        # Après avoir distribué les gains, démarrer une nouvelle main
        print("\n=== Hand Complete ===")
        # Attendre un peu pour montrer le résultat
        self.winner_display_start = pygame.time.get_ticks()
        
        # Vérifier si la partie peut continuer
        players_with_chips = [p for p in self.players if p.stack > 0]
        if len(players_with_chips) < 2:
            print("Game over - not enough players with chips!")
            self.game_over = True
            self.game_over_message = "Game Over - Not enough players with chips!"
            return
        
        # Avancer le bouton pour la prochaine main
        self.button_position = (self.button_position + 1) % self.num_players
        while self.players[self.button_position].stack <= 0:
            self.button_position = (self.button_position + 1) % self.num_players

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
            PlayerAction.RAISE: Button(750, self.SCREEN_HEIGHT - 100, 100, 40, "Raise", (200, 200, 0))
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
        random.shuffle(deck)
        return deck
    
    def _initialize_players(self) -> List[Player]:
        """
        Create and initialize all players for the game.
        Returns:
            List[Player]: List of initialized player objects
        """
        players = []
        starting_stack = 200 * self.big_blind
        for i in range(self.num_players):
            player = Player(f"Player {i+1}", starting_stack, i)
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
        
        # Draw all-in status if applicable
        if player.is_all_in:
            all_in_text = self.font.render("ALL IN", True, (255, 0, 0))  # Rouge pour le texte ALL IN
            self.screen.blit(all_in_text, (player.x - 30, player.y + 110))
    
    def _draw(self):
        """
        Draw the complete game state on the screen.
        """
        # Clear screen
        self.screen.fill((0, 100, 0))  # Green felt background
        
        if self.game_over:
            # Afficher le message de fin de partie
            game_over_font = pygame.font.SysFont('Arial', 48, bold=True)
            text = game_over_font.render(self.game_over_message, True, (255, 0, 0))
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(text, text_rect)
            
            # Afficher un message pour quitter
            quit_font = pygame.font.SysFont('Arial', 24)
            quit_text = quit_font.render("Press ESC to quit", True, (255, 255, 255))
            quit_rect = quit_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 50))
            self.screen.blit(quit_text, quit_rect)
            return
        
        # Draw table
        pygame.draw.ellipse(self.screen, (139, 69, 19), (100, 100, 1000, 600))
        pygame.draw.ellipse(self.screen, (165, 42, 42), (120, 120, 960, 560))
        
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
                # Après le délai d'affichage, démarrer une nouvelle main
                self.winner_info = None
                if not self.game_over:  # Vérifier que le jeu n'est pas terminé
                    print("\nStarting new hand after winner display")
                    self.start_new_hand()

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
                    if action == PlayerAction.RAISE and button.text == "ALL-IN":
                        # Cas du tapis forcé
                        bet_amount = current_player.stack + current_player.current_bet
                    else:
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
        """
        initial_position = self.current_player_idx
        while True:
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players
            current_player = self.players[self.current_player_idx]
            
            # Vérifier si on est revenu au joueur initial
            if self.current_player_idx == initial_position:
                print("Round complete - all players have acted")
                break
            
            # Vérifier si le joueur peut agir
            if current_player.is_active and not current_player.is_all_in and current_player.stack > 0:
                print(f"Next to act: {current_player.name}")
                break

    def _update_button_states(self):
        """
        Update the enabled/disabled state of action buttons based on current game state.
        """
        current_player = self.players[self.current_player_idx]
        
        # Enable all buttons by default
        for button in self.action_buttons.values():
            button.enabled = True
        
        # Disable check if there's a bet to call
        if current_player.current_bet < self.current_bet:
            self.action_buttons[PlayerAction.CHECK].enabled = False
        
        # Gérer le cas où le joueur n'a pas assez pour suivre
        call_amount = self.current_bet - current_player.current_bet
        if call_amount >= current_player.stack:
            # Le joueur peut seulement faire all-in
            self.action_buttons[PlayerAction.CALL].enabled = False
            self.action_buttons[PlayerAction.RAISE].text = "ALL-IN"
            self.action_buttons[PlayerAction.RAISE].enabled = True
        else:
            # Configuration normale des boutons
            if current_player.current_bet == self.current_bet:
                self.action_buttons[PlayerAction.CALL].enabled = False
            
            # Toujours permettre le bouton RAISE pour l'all-in
            self.action_buttons[PlayerAction.RAISE].enabled = True
            if current_player.stack + current_player.current_bet < self.current_bet * 2:
                self.action_buttons[PlayerAction.RAISE].text = "ALL-IN"
            else:
                self.action_buttons[PlayerAction.RAISE].text = "RAISE"

    def run(self):
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
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and self.game_over:
                    running = False
                
                if not self.game_over:
                    self.handle_input(event)
                    
                    # Update button hover states
                    mouse_pos = pygame.mouse.get_pos()
                    for button in self.action_buttons.values():
                        button.is_hovered = button.rect.collidepoint(mouse_pos)
            
            # AI players' turns (seulement si le jeu n'est pas terminé)
            if not self.game_over:
                current_player = self.players[self.current_player_idx]
                if not current_player.is_human and current_player.is_active:
                    action = random.choice([PlayerAction.CALL, PlayerAction.FOLD])
                    self.process_action(current_player, action)
        
            self._draw()
            pygame.display.flip()
        
        pygame.quit()

if __name__ == "__main__":
    game = PokerGame()
    game.run()