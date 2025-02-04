# poker_game.py
import pygame
import random as rd
from enum import Enum
from typing import List, Dict, Optional, Tuple
import pygame.font
from collections import Counter
import numpy as np
import torch
import time

class Card:
    """
    Représente une carte à jouer avec une couleur et une valeur.
    """
    def __init__(self, suit: str, value: int):
        """
        Initialise une carte avec une couleur et une valeur.
        
        Args:
            suit (str): La couleur de la carte (♠, ♥, ♦, ♣)
            value (int): La valeur de la carte (2-14, où 14 est l'As)
        """
        self.suit = suit
        self.value = value
        
    def __str__(self):
        """
        Convertit la carte en représentation textuelle.
        
        Returns:
            str: Représentation textuelle de la carte (ex: "A♠")

        Exemple:
        >>> card = Card('♠', 14)
        >>> print(card)
        'A♠'
        """
        values = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        value_str = values.get(self.value, str(self.value))
        return f"{value_str}{self.suit}"

class HandRank(Enum):
    """
    Énumération des combinaisons possibles au poker, de la plus faible à la plus forte.
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
    Énumération des actions possibles pour un joueur pendant son tour.
    """
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all-in"

class GamePhase(Enum):
    """
    Énumération des phases d'une partie de poker, de la distribution au showdown.
    """
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"

class Button:
    """
    Représente un bouton cliquable dans l'interface utilisateur du jeu.
    """
    def __init__(self, x: int, y: int, width: int, height: int, text: str, color: Tuple[int, int, int]):
        """
        Initialise un bouton avec sa position, sa taille, son texte et sa couleur.
        
        Args:
            x (int): Position X sur l'écran
            y (int): Position Y sur l'écran
            width (int): Largeur du bouton en pixels
            height (int): Hauteur du bouton en pixels
            text (str): Texte affiché sur le bouton
            color (Tuple[int, int, int]): Couleur RGB du bouton
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.is_hovered = False
        self.enabled = True

    def draw(self, screen, font):
        """
        Dessine le bouton sur l'écran avec les effets visuels appropriés.
        
        Args:
            screen: Surface Pygame sur laquelle dessiner le bouton
            font: Police Pygame pour le rendu du texte
        """
        if not self.enabled:
            # Griser les boutons désactivés
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
    Représente un joueur de poker avec ses cartes, son stack et son état de jeu.
    """
    def __init__(self, name: str, stack: int, position: int):
        """
        Initialise un joueur avec son nom, son stack de départ et sa position à la table.
        
        Args:
            name (str): Nom du joueur
            stack (int): Stack de départ en jetons
            position (int): Position à la table (0-2)
        """
        self.name = name
        self.stack = stack
        self.position = position
        self.cards: List[Card] = []
        self.is_active = True  # Indique si le joueur a assez de jetons pour jouer
        self.has_folded = False  # Indique si le joueur s'est couché pendant la main en cours
        self.current_bet = 0
        self.is_human = True
        self.has_acted = False
        positions = [
            (550, 650),  # Bas (Joueur 1)
            (100, 300),  # Gauche (Joueur 2)
            (1000, 300)  # Droite (Joueur 3)
        ]
        self.x, self.y = positions[position]

class PokerGame:
    """
    Classe principale qui gère l'état et la logique du jeu de poker.
    """
    def __init__(self, num_players: int = 3):
        """
        Initialise la partie de poker avec les joueurs et les blindes.
        
        Args:
            num_players (int): Nombre de joueurs (défaut: 3)
            small_blind (int): Montant de la petite blinde (défaut: 10)
            big_blind (int): Montant de la grosse blinde (défaut: 20)
        """
        pygame.init()
        pygame.font.init()
        self.SCREEN_WIDTH = 1400
        self.SCREEN_HEIGHT = 900
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("3-Max Poker")
        self.font = pygame.font.SysFont('Arial', 24)
        self.num_players = num_players
        
        # Structure des blindes avec progression
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
        
        # Initialiser les éléments de l'interface
        self.action_buttons = self._create_action_buttons()
        self.bet_slider = pygame.Rect(50, self.SCREEN_HEIGHT - 100, 200, 20)
        self.current_bet_amount = self.big_blind
        
        # Ajouter le suivi de l'historique des actions
        self.action_history = []
        
        # Ajouter le suivi des informations du gagnant
        self.winner_info = None
        
        # Ajouter le timing d'affichage du gagnant
        self.winner_display_start = 0
        self.winner_display_duration = 2000  # 2 secondes en millisecondes

        # Ajouter le temps de la dernière action IA
        self.last_ai_action_time = 0
        
        self.start_new_hand()
        self._update_button_states()

    def reset(self):
        """
        Réinitialise complètement l'état du jeu pour une nouvelle partie.
        
        Returns:
            List[float]: État initial du jeu après réinitialisation
        """
        # Réinitialiser les variables d'état du jeu
        self.pot = 0
        self.community_cards = []
        self.current_phase = GamePhase.PREFLOP
        self.deck = self._create_deck()
        self.winner_info = None
        self.winner_display_start = 0
        self.round_number = 0
        self.number_raise_this_round = 0
        self.action_history = []
        
        # Réinitialiser l'état des joueurs
        for player in self.players:
            player.cards = []
            player.current_bet = 0
            player.is_active = True
            player.has_folded = False
            player.has_acted = False
            player.stack = 500  # Réinitialiser au stack de départ
        
        # Réinitialiser le bouton et les blindes
        self.button_position = (self.button_position + 1) % self.num_players
        self.sb_pos = (self.sb_pos + 1) % self.num_players
        self.bb_pos = (self.bb_pos + 1) % self.num_players
        
        # Poster les blindes
        self.players[self.sb_pos].stack -= self.small_blind
        self.players[self.sb_pos].current_bet = self.small_blind
        self.players[self.bb_pos].stack -= self.big_blind
        self.players[self.bb_pos].current_bet = self.big_blind
        
        self.pot = self.small_blind + self.big_blind
        self.current_bet = self.big_blind
        
        # Distribuer les cartes
        self.deal_cards()
        
        # Définir le premier joueur (UTG)
        self.current_player_idx = (self.bb_pos + 1) % self.num_players
        
        # Réinitialiser l'état des mises
        self.last_raiser = None

        self._update_button_states()

        return self.get_state()

    def start_new_hand(self):
        """
        Démarre une nouvelle main de poker en réinitialisant l'état approprié.
        Met à jour les blindes et vérifie les conditions de jeu.
        """
        # Mettre à jour les blindes avant de commencer une nouvelle main
        self.update_blinds()
        
        # Réinitialiser les variables d'état du jeu
        self.pot = 0
        self.community_cards = []
        self.current_phase = GamePhase.PREFLOP
        self.current_bet = self.big_blind
        self.last_raiser = None
        self.round_number = 0
        
        # Clear previous action history and add initial round separator
        self.action_history = []
        self.action_history.append("=== NEW HAND ===")
        self.action_history.append(f"--- {GamePhase.PREFLOP.value.upper()} ---")
        
        # Construire une nouvelle liste des joueurs actifs (uniquement ceux avec assez de fonds)
        active_players = []
        for player in self.players:
            player.cards = []
            player.current_bet = 0
            player.has_acted = False
            player.has_folded = False
            if player.stack >= self.big_blind:
                player.is_active = True
                active_players.append(player)
            else:
                player.is_active = False
                print(f"{player.name} est hors jeu (fonds insuffisants: ${player.stack})")
        
        # S'il y a moins de 2 joueurs actifs, réinitialiser le jeu
        if len(active_players) < 2:
            print("Pas assez de joueurs pour continuer.")
            self.reset()
            return
        
        # Compter les joueurs actifs pour la structure des blindes
        num_active_players = len(active_players)
        
        # Déplacer le bouton avant de définir les positions des blindes
        self.button_position = (self.button_position + 1) % self.num_players
        while not self.players[self.button_position].is_active:
            self.button_position = (self.button_position + 1) % self.num_players
        
        if num_active_players == 2:  # Heads-up
            # En heads-up, le bouton est SB et agit en dernier preflop, premier postflop
            self.sb_pos = self.button_position
            self.bb_pos = (self.button_position + 1) % self.num_players
            while not self.players[self.bb_pos].is_active:
                self.bb_pos = (self.bb_pos + 1) % self.num_players
        else:
            # Structure normale à 3 joueurs
            self.sb_pos = (self.button_position + 1) % self.num_players
            while not self.players[self.sb_pos].is_active:
                self.sb_pos = (self.sb_pos + 1) % self.num_players
                
            self.bb_pos = (self.sb_pos + 1) % self.num_players
            while not self.players[self.bb_pos].is_active:
                self.bb_pos = (self.bb_pos + 1) % self.num_players
        
        # Poster les blindes
        self.players[self.sb_pos].stack -= self.small_blind
        self.players[self.sb_pos].current_bet = self.small_blind
        self.players[self.bb_pos].stack -= self.big_blind
        self.players[self.bb_pos].current_bet = self.big_blind
        
        self.pot = self.small_blind + self.big_blind
        self.deal_cards()
        
        # UTG agit en premier preflop (après BB)
        self.current_player_idx = (self.bb_pos + 1) % self.num_players
        while not self.players[self.current_player_idx].is_active:
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players

        self._update_button_states()
        return True

    def _next_active_position(self, current_pos):
        """
        Trouve la prochaine position active à partir de la position donnée.
        
        Args:
            current_pos (int): Position actuelle à la table
            
        Returns:
            int: Prochaine position active dans le sens horaire
        """
        next_pos = (current_pos + 1) % self.num_players
        while not self.players[next_pos].is_active:
            next_pos = (next_pos + 1) % self.num_players
        return next_pos

    def evaluate_final_hand(self, player: Player) -> Tuple[HandRank, List[int]]:
        """
        Évalue la meilleure main possible d'un joueur avec les cartes communes.
        
        Args:
            player (Player): Le joueur dont on évalue la main
            
        Returns:
            Tuple[HandRank, List[int]]: Le rang de la main et les valeurs pour départager
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

    def evaluate_current_hand(self, player) -> Tuple[HandRank, List[int]]:
        """
        Évalue la main actuelle d'un joueur avec les cartes communes disponibles meme si il y en a moins de 5.
        
        Args:
            player (Player): Le joueur dont on évalue la main
        """
        # Si le joueur n'a pas de cartes ou a foldé
        if not player.cards or player.has_folded:
            return (HandRank.HIGH_CARD, [0])
        
        # Obtenir toutes les cartes disponibles
        all_cards = player.cards + self.community_cards
        values = [card.value for card in all_cards]
        suits = [card.suit for card in all_cards]
        
        # Au pré-flop, évaluer uniquement les cartes du joueur
        if self.current_phase == GamePhase.PREFLOP:
            # Paire de départ
            if player.cards[0].value == player.cards[1].value:
                return (HandRank.PAIR, [player.cards[0].value])
            # Cartes hautes
            return (HandRank.HIGH_CARD, sorted([c.value for c in player.cards], reverse=True))
        
        # Compter les occurrences des valeurs et couleurs
        value_counts = Counter(values)
        suit_counts = Counter(suits)
        
        # Vérifier les combinaisons possibles avec les cartes disponibles
        # Paire
        pairs = [v for v, count in value_counts.items() if count >= 2]
        if pairs:
            if len(pairs) >= 2:  # Double paire
                pairs.sort(reverse=True)
                kicker = max(v for v in values if v not in pairs[:2])
                return (HandRank.TWO_PAIR, pairs[:2] + [kicker])
            # Simple paire
            kickers = sorted([v for v in values if v != pairs[0]], reverse=True)[:3]
            return (HandRank.PAIR, pairs + kickers)
        
        # Brelan
        trips = [v for v, count in value_counts.items() if count >= 3]
        if trips:
            kickers = sorted([v for v in values if v != trips[0]], reverse=True)[:2]
            return (HandRank.THREE_OF_A_KIND, [trips[0]] + kickers)
        
        # Couleur potentielle (4 cartes de la même couleur)
        flush_suit = next((suit for suit, count in suit_counts.items() if count >= 4), None)
        if flush_suit:
            flush_cards = sorted([card.value for card in all_cards if card.suit == flush_suit], reverse=True)
            if len(flush_cards) >= 5:
                return (HandRank.FLUSH, flush_cards[:5])
        
        # Quinte potentielle
        unique_values = sorted(set(values))
        for i in range(len(unique_values) - 3):
            if unique_values[i+3] - unique_values[i] == 3:  # 4 cartes consécutives
                return (HandRank.STRAIGHT, [unique_values[i+3]])
        
        # Si aucune combinaison, retourner la plus haute carte
        return (HandRank.HIGH_CARD, sorted(values, reverse=True)[:5])

    def check_round_completion(self):
        """
        Vérifie si le tour d'enchères actuel est terminé.
        Un tour est terminé quand tous les joueurs actifs ont misé le même montant.
        """
        # Ne considérer que les joueurs actifs et n'ayant pas fold
        active_players = [p for p in self.players if p.is_active and not p.has_folded]
        
        # S'il ne reste qu'un joueur, le tour est terminé
        if len(active_players) == 1:
            return True
        
        # Vérifier si tous les joueurs actifs ont agi et ont la même mise
        all_acted = all(p.has_acted for p in active_players)
        bets_equal = len(set(p.current_bet for p in active_players)) == 1
        
        return all_acted and bets_equal

    def advance_phase(self):
        """
        Passe à la phase suivante du jeu (préflop -> flop -> turn -> river).
        Distribue les cartes communes appropriées et réinitialise les mises.
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
        Traite l'action d'un joueur pendant son tour et met à jour l'état du jeu.
        
        Args:
            player (Player): Le joueur qui effectue l'action
            action (PlayerAction): L'action choisie
            bet_amount (Optional[int]): Le montant de la mise si applicable
        """
        # Check if player has sufficient funds for any action
        if player.stack <= 0:
            self._next_player()
            return False
        
        # Don't process actions during showdown
        if self.current_phase == GamePhase.SHOWDOWN:
            return action
            
        # Debug print for action start
        print(f"\n=== Action by {player.name} ===")
        print(f"Player activity: {player.is_active}")
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

        # Add round separator before action if phase is changing
        if self.check_round_completion() and self.current_phase != GamePhase.SHOWDOWN:
            self.action_history.append(f"--- {self.current_phase.value.upper()} ---")
        
        self.action_history.append(action_text)
        if len(self.action_history) > 10:
            self.action_history.pop(0)
        
        # Process the action
        if action == PlayerAction.FOLD:
            player.has_folded = True
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
        active_players = [p for p in self.players if p.is_active and not p.has_folded]
        all_in_players = [p for p in active_players if p.stack == 0]
        
        # Check if only one player remains (others folded or inactive)
        if len(active_players) == 1:
            print("Moving to showdown (only one player remains)")
            self.handle_showdown()
            return action
        
        # Check if all remaining active players are all-in
        if (len(all_in_players) == len(active_players)) and (len(active_players) > 1):
            print("Moving to showdown (all remaining players are all-in)")
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
                # Réinitialiser has_acted pour tous les joueurs actifs et non fold au début d'une nouvelle phase
                for p in self.players:
                    if p.is_active and not p.has_folded:
                        p.has_acted = False
        else:
            self._next_player()
            print(f"Next player: {self.players[self.current_player_idx].name}")
        
        return action

    def handle_showdown(self):
        """
        Gère la phase de showdown où les joueurs restants révèlent leurs mains.
        Évalue les mains, détermine le(s) gagnant(s) et attribue le pot.
        """
        print("\n=== SHOWDOWN ===")
        self.current_phase = GamePhase.SHOWDOWN
        active_players = [p for p in self.players if p.is_active and not p.has_folded]
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
            player_hands = [(player, self.evaluate_final_hand(player)) for player in active_players]
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
        self.winner_display_duration = 2000  # 2 seconds in milliseconds

    def _create_action_buttons(self) -> Dict[PlayerAction, Button]:
        """
        Crée et initialise les boutons d'action pour l'interaction des joueurs.
        
        Returns:
            Dict[PlayerAction, Button]: Dictionnaire associant les actions aux objets boutons
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
        Crée et mélange un nouveau jeu de 52 cartes.
        
        Returns:
            List[Card]: Un jeu de cartes mélangé
        """
        suits = ['♠', '♥', '♦', '♣']
        values = range(2, 15)  # 2-14 (Ace is 14)
        deck = [Card(suit, value) for suit in suits for value in values]
        rd.shuffle(deck)
        return deck
    
    def _initialize_players(self) -> List[Player]:
        """
        Crée et initialise tous les joueurs pour la partie.
        
        Returns:
            List[Player]: Liste des objets joueurs initialisés
        """
        players = []
        for i in range(self.num_players):
            player = Player(f"Player_{i+1}", self.starting_stack, i)
            players.append(player)
        return players
    
    def deal_cards(self):
        """
        Distribue deux cartes à chaque joueur actif.
        Réinitialise et mélange le jeu avant la distribution.
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
        Distribue les cartes communes selon la phase de jeu actuelle.
        Distribue 3 cartes pour le flop, 1 pour le turn et 1 pour la river.
        """
        if self.current_phase == GamePhase.FLOP:
            for _ in range(3):
                self.community_cards.append(self.deck.pop())
        elif self.current_phase in [GamePhase.TURN, GamePhase.RIVER]:
            self.community_cards.append(self.deck.pop())
    
    def _draw_card(self, card: Card, x: int, y: int):
        """
        Dessine une carte sur l'écran avec sa valeur et sa couleur.
        
        Args:
            card (Card): La carte à dessiner
            x (int): Position X sur l'écran
            y (int): Position Y sur l'écran
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
        Dessine les informations d'un joueur sur l'écran (cartes, stack, mises).
        
        Args:
            player (Player): Le joueur à dessiner
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
        if player.is_active and not player.has_folded and player.cards:
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
    
        # Draw dealer button (D) - Updated positioning logic
        if player.position == self.button_position:  # Only draw if this player is the dealer
            button_x = player.x + 52
            button_y = player.y + 80
            pygame.draw.circle(self.screen, (255, 255, 255), (button_x, button_y), 15)
            dealer_text = self.font.render("D", True, (0, 0, 0))
            dealer_rect = dealer_text.get_rect(center=(button_x, button_y))
            self.screen.blit(dealer_text, dealer_rect)
    
    def _draw(self):
        """
        Dessine l'état complet du jeu sur l'écran.
        Inclut la table, les joueurs, les cartes communes, le pot et l'interface.
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
        
        # Draw action history in top right corner with better formatting
        history_x = self.SCREEN_WIDTH - 300
        history_y = 50
        history_text = self.font.render("Action History:", True, (255, 255, 255))
        self.screen.blit(history_text, (history_x, history_y - 30))
        
        for i, action in enumerate(self.action_history):
            # Use different colors for different types of text
            if action.startswith("==="):  # New hand separator
                color = (255, 215, 0)  # Gold
            elif action.startswith("---"):  # Round separator
                color = (0, 255, 255)  # Cyan
            else:  # Normal action
                color = (255, 255, 255)  # White
            
            text = self.font.render(action, True, color)
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
                active_players = [p for p in self.players if p.stack >= self.big_blind]
                if len(active_players) > 1:
                    self.start_new_hand()
                else:
                    self.reset()

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
        Gère les événements d'entrée des joueurs (souris, clavier).
        
        Args:
            event: Objet événement Pygame à traiter
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
        Passe au prochain joueur actif et n'ayant pas fold dans le sens horaire.
        """
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        while not self.players[self.current_player_idx].is_active or self.players[self.current_player_idx].has_folded:
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players

    def _update_button_states(self):
        """
        Met à jour l'état activé/désactivé des boutons d'action.
        Prend en compte la phase de jeu et les règles du poker.
        """
        current_player = self.players[self.current_player_idx]
        
        # Activer tous les boutons par défaut
        for button in self.action_buttons.values():
            button.enabled = True
        
        # Règles spécifiques au preflop
        if self.current_phase == GamePhase.PREFLOP:
            # Le joueur ne peut check que si sa mise actuelle est égale à la mise courante
            if current_player.current_bet < self.current_bet:
                self.action_buttons[PlayerAction.CHECK].enabled = False
        else:
            # Post-flop: on peut check seulement si personne n'a misé
            if current_player.current_bet < self.current_bet:
                self.action_buttons[PlayerAction.CHECK].enabled = False
        
        # Désactiver call si pas de mise à suivre ou pas assez de jetons
        if current_player.current_bet == self.current_bet:
            self.action_buttons[PlayerAction.CALL].enabled = False
        elif current_player.stack < (self.current_bet - current_player.current_bet):
            self.action_buttons[PlayerAction.CALL].enabled = False
        
        # Désactiver raise si pas assez de jetons pour la mise minimale
        min_raise = max(self.current_bet * 2, self.big_blind * 2)
        if current_player.stack + current_player.current_bet < min_raise:
            self.action_buttons[PlayerAction.RAISE].enabled = False
        
        # Désactiver raise si déjà 4 relances dans le tour
        if self.number_raise_this_round >= 4:
            self.action_buttons[PlayerAction.RAISE].enabled = False
        
        # All-in toujours disponible si le joueur a des jetons
        self.action_buttons[PlayerAction.ALL_IN].enabled = current_player.stack > 0

    def get_state(self):
        """
        Obtient l'état actuel du jeu pour l'apprentissage par renforcement.
        
        Returns:
            List[float]: État normalisé du jeu incluant:
            - Cartes (joueur et communes)
            - Rang de la main
            - Mises et positions
            - État des joueurs
            - Phase et actions disponibles
        """
        current_player = self.players[self.current_player_idx]
        state = []

        # Correspondance des couleurs avec des nombres ♠, ♥, ♦, ♣
        suit_map = {
            "♠" : 0,
            "♥" : 1,
            "♦" : 2,
            "♣" : 3
        }

        # 1. Informations sur les cartes (encodage one-hot)
        # Cartes du joueur
        for card in current_player.cards:
            value_range = [0.01] * 13
            value_range[card.value - 2] = 1
            state.extend(value_range)  # Extension pour la valeur
            suit_range = [0.01] * 4
            suit_range[suit_map[card.suit]] = 1
            state.extend(suit_range)  # Extension pour la couleur
        
        # Ajout de remplissage pour les cartes manquantes du joueur
        remaining_player_cards = 2 - len(current_player.cards)
        for _ in range(remaining_player_cards):
            state.extend([0.01] * 13)  # Remplissage des valeurs
            state.extend([0.01] * 4)   # Remplissage des couleurs
        
        # Cartes communes
        for i, card in enumerate(self.community_cards):
            value_range = [0.01] * 13
            value_range[card.value - 2] = 1
            state.extend(value_range)  # Extension
            suit_range = [0.01] * 4
            suit_range[suit_map[card.suit]] = 1
            state.extend(suit_range)  # Extension
        
        # Ajout de remplissage pour les cartes communes manquantes
        remaining_community_cards = 5 - len(self.community_cards)
        for _ in range(remaining_community_cards):
            state.extend([0.01] * 13)  # Remplissage des valeurs
            state.extend([0.01] * 4)   # Remplissage des couleurs
        
        # 2. Rang de la main actuelle (si assez de cartes sont visibles)
        if len(current_player.cards) + len(self.community_cards) >= 5:
            hand_rank, _ = self.evaluate_final_hand(current_player)
            state.append(hand_rank.value / len(HandRank))  # Normalisation de la valeur du rang (taille = 1)
        else:
            hand_rank, _ = self.evaluate_current_hand(current_player)
            state.append(hand_rank.value / len(HandRank))  # Normalisation de la valeur du rang (taille = 1)

        # 3. Informations sur le tour
        phase_values = {
            GamePhase.PREFLOP: 0,
            GamePhase.FLOP: 1,
            GamePhase.TURN: 2,
            GamePhase.RIVER: 3,
            GamePhase.SHOWDOWN: 4
        }

        phase_range = [0.01] * 5
        phase_range[phase_values[self.current_phase]] = 1
        state.extend(phase_range)

        # 4. Numéro du tour
        state.append(self.round_number)  # Normalisation du numéro de tour (taille = 1)

        # 5. Mise actuelle normalisée par la grosse blinde
        state.append(self.current_bet / self.big_blind)  # Normalisation de la mise (taille = 1)

        # 6. Argent restant (tailles des stacks normalisées par le stack initial)
        initial_stack = self.starting_stack
        for player in self.players:
            state.append(player.stack / initial_stack) # (taille = 3)

        # 7. Informations sur les mises (normalisées par la grosse blinde)
        for player in self.players:
            state.append(player.current_bet / initial_stack) # (taille = 3)

        # 8. Informations sur l'activité (binaire extrême : actif/ruiné)
        for player in self.players:
            state.append(1 if player.is_active else -1) # (taille = 3)
        
        # 9. Informations sur l'activité in game (binaire extrême : en jeu/a foldé)
        for player in self.players:
            state.append(1 if player.has_folded else -1) # (taille = 3)

        # 10. Informations sur la position (encodage one-hot des positions relatives)
        relative_positions = [0.1] * self.num_players
        relative_pos = (self.current_player_idx - self.button_position) % self.num_players
        relative_positions[relative_pos] = 1
        state.extend(relative_positions) # (taille = 3)

        # 11. Actions disponibles (binaire extrême : disponible/indisponible)
        action_availability = []
        for action in PlayerAction:
            if action in self.action_buttons and self.action_buttons[action].enabled:
                action_availability.append(1)
            else:
                action_availability.append(-1)
        state.extend(action_availability) # (taille = 3)

        # 12. Actions précédentes (dernière action de chaque joueur, encodée en vecteurs one-hot)
        action_encoding = {
            None: 0,
            PlayerAction.FOLD: 1,
            PlayerAction.CHECK: 2,
            PlayerAction.CALL: 3,
            PlayerAction.RAISE: 4,
            PlayerAction.ALL_IN: 5
        }

        # Initialisation du tableau des dernières actions avec des zéros
        last_actions = [[0.1] * 6 for _ in range(self.num_players)]  # 6 actions possibles (y compris None)

        # Traitement des actions récentes
        for action_text in reversed(self.action_history[-self.num_players:]):
            if ":" in action_text:
                player_name, action = action_text.split(":")
                player_idx = int(player_name.split("_")[-1]) - 1
                action = action.strip()
                
                # Recherche du type d'action correspondant
                for action_type in PlayerAction:
                    if action_type.value in action:
                        # Création de l'encodage one-hot
                        last_actions[player_idx] = [0.1] * 6  # Réinitialisation à zéro
                        last_actions[player_idx][action_encoding[action_type]] = 1
                        break

        # 12. Liste des actions précédentes des 3 joueurs
        flattened_actions = [val for sublist in last_actions for val in sublist]
        state.extend(flattened_actions)  

        # 13. Estimation de la probabilité de victoire
        active_players = [p for p in self.players if p.is_active and not p.has_folded]
        num_opponents = len(active_players) - 1
        
        if num_opponents <= 0:
            win_prob = 1.0  # Plus d'adversaires
        else:
            # Obtention de la force de la main
            hand_strength = self._evaluate_hand_strength(current_player)
            
            # Ajustement pour le nombre d'adversaires (2-3)
            win_prob = hand_strength ** num_opponents
            
            # Ajustement spécifique au pré-flop (utilisant l'approximation Sklansky-Chubukov)
            if self.current_phase == GamePhase.PREFLOP:
                # Réduction de la confiance dans les estimations pré-flop
                win_prob *= 0.8
        
        state.append(win_prob) # (taille = 1)

        # 14. Cotes du pot
        call_amount = self.current_bet - current_player.current_bet
        pot_odds = call_amount / (self.pot + call_amount) if (self.pot + call_amount) > 0 else 0
        state.append(pot_odds) # (taille = 1)

        # 15. Équité
        equity = self._evaluate_equity(current_player)
        state.append(equity) # (taille = 1)

        # 16. Facteur d'agressivité
        state.append(self.number_raise_this_round / 4) # (taille = 1)

        # Avant de retourner, conversion en tableau numpy
        state = np.array(state, dtype=np.float32)
        return state

    def step(self, action: PlayerAction) -> Tuple[List[float], float]:
        """
        Exécute une action et calcule la récompense associée.
        
        Args:
            action (PlayerAction): L'action à exécuter
            
        Returns:
            Tuple[List[float], float]: Nouvel état et récompense
        """
        current_player = self.players[self.current_player_idx]
        reward = 0.0

        # Capturer l'état du jeu avant de traiter l'action pour le calcul des cotes du pot
        call_amount_before = self.current_bet - current_player.current_bet
        pot_before = self.pot

        # --- Récompenses stratégiques des actions ---
        # Récompense basée sur l'action par rapport à la force de la main
        hand_strength = self._evaluate_hand_strength(current_player)
        pot_potential = self.pot / (self.big_blind * 100)
    
        if action == PlayerAction.RAISE:
            reward += 0.2 * hand_strength  # Ajuster la récompense selon la force de la main
            if hand_strength > 0.7:
                reward += 0.5 * pot_potential  # Bonus pour jeu agressif avec main forte
                
        elif action == PlayerAction.ALL_IN:
            reward += 0.3 * hand_strength
            if hand_strength > 0.8:
                reward += 1.0 * pot_potential
                
        elif action == PlayerAction.CALL:
            reward += 0.1 * min(hand_strength, 0.6)  # Rendements décroissants pour jeu passif
        
        elif action == PlayerAction.CHECK: # Pénaliser le check si la main est forte
            if hand_strength > 0.5:
                reward -= 0.5 * pot_potential
            else:
                reward += 0.3

        elif action == PlayerAction.FOLD:
            if hand_strength < 0.2:
                reward += 0.3  # Récompenser les bons folds
            else:
                reward -= 0.5  # Pénaliser les folds avec bonnes mains
            
        # --- Bonus de position ---
        # Bonus pour actions agressives en dernière position (bouton)
        if current_player.position == self.button_position:
            if action in [PlayerAction.RAISE, PlayerAction.ALL_IN]:
                reward += 0.2
        
        # Traiter l'action (met à jour l'état du jeu)
        self.process_action(current_player, action)

        # --- Évaluation des cotes du pot ---
        if action == PlayerAction.CALL and call_amount_before > 0:
            total_pot_after_call = pot_before + call_amount_before
            pot_odds = call_amount_before / total_pot_after_call if total_pot_after_call > 0 else 0
            if hand_strength > pot_odds:
                reward += 0.3  # Call mathématiquement justifié
            else:
                reward -= 0.3  # Mauvais call considérant les cotes

        return self.get_state(), reward

    def _evaluate_preflop_strength(self, cards) -> float:
        """
        Évalue la force d'une main preflop selon des heuristiques.
        
        Args:
            cards (List[Card]): Les deux cartes à évaluer
            
        Returns:
            float: Force de la main entre 0 et 1
        """
        # Vérification de sécurité pour mains vides ou incomplètes
        if not cards or len(cards) < 2:
            return 0.0
        
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

    def _evaluate_hand_strength(self, player) -> float:
        """
        Évalue la force relative d'une main (0 à 1) similaire à _evaluate_preflop_strength 
        mais avec les cartes communes
        
        Args:
            player (Player): Le joueur dont on évalue la main
            
        Returns:
            float: Force de la main entre 0 (très faible) et 1 (très forte)
        """
        # Retourner 0 si le joueur a foldé ou n'a pas de cartes
        if not player.is_active or not player.cards:
            return 0.0
        
        # Au pré-flop, utiliser l'évaluation spécifique pré-flop
        if self.current_phase == GamePhase.PREFLOP:
            return self._evaluate_preflop_strength(player.cards)
        
        # Obtenir toutes les cartes disponibles (main + cartes communes)
        all_cards = player.cards + self.community_cards
        
        # Évaluer la main actuelle
        hand_rank, kickers = self.evaluate_final_hand(player)
        base_score = hand_rank.value / len(HandRank)  # Score de base normalisé
        
        # Bonus/malus selon la phase de jeu et les kickers
        phase_multiplier = {
            GamePhase.FLOP: 0.8,   # Moins certain au flop
            GamePhase.TURN: 0.9,   # Plus certain au turn
            GamePhase.RIVER: 1.0,  # Certitude maximale à la river
        }.get(self.current_phase, 1.0)
        
        # Calculer le score des kickers (normalisé)
        kicker_score = sum(k / 14 for k in kickers) / len(kickers) if kickers else 0
        
        # Vérifier les tirages possibles
        draw_potential = 0.0
        
        # Compter les cartes de chaque couleur
        suits = [card.suit for card in all_cards]
        suit_counts = Counter(suits)
        
        # Compter les cartes consécutives
        values = sorted(set(card.value for card in all_cards))
        
        # Tirage couleur
        flush_draw = any(count == 4 for count in suit_counts.values())
        if flush_draw:
            draw_potential += 0.15
        
        # Tirage quinte
        for i in range(len(values) - 3):
            if values[i+3] - values[i] == 3:  # 4 cartes consécutives
                draw_potential += 0.15
                break
        
        # Tirage quinte flush
        if flush_draw and any(values[i+3] - values[i] == 3 for i in range(len(values) - 3)):
            draw_potential += 0.1
        
        # Le potentiel de tirage diminue à mesure qu'on avance dans les phases
        draw_potential *= {
            GamePhase.FLOP: 1.0,
            GamePhase.TURN: 0.5,
            GamePhase.RIVER: 0.0,
        }.get(self.current_phase, 0.0)
        
        # Calculer le score final
        final_score = (
            base_score * 0.7 +      # Score de base (70% du score)
            kicker_score * 0.2 +    # Score des kickers (20% du score)
            draw_potential          # Potentiel de tirage (jusqu'à 10% supplémentaires)
        ) * phase_multiplier
        
        return min(1.0, max(0.0, final_score))  # Garantir un score entre 0 et 1
    
    def _evaluate_equity(self, player) -> float:
        """
        Calcule l'équité au pot pour la main d'un joueur.
        Prend en compte la position, les cotes et la phase de jeu.
        
        Args:
            player (Player): Le joueur dont on évalue l'équité
            
        Returns:
            float: Équité entre 0 et 1
        """
        # Return 0 equity if player has folded or has no cards
        if not player.is_active or not player.cards:
            return 0.0
        
        # Get base equity from hand strength
        hand_strength = self._evaluate_hand_strength(player)
        
        # Count active players
        active_players = [p for p in self.players if p.is_active]
        num_active = len(active_players)
        if num_active <= 1:
            return 1.0  # Only player left
        
        # Position multiplier (better position = higher equity)
        # Calculate relative position from button (0 = button, 1 = SB, 2 = BB)
        relative_pos = (player.position - self.button_position) % self.num_players
        position_multiplier = 1.0 + (0.1 * (self.num_players - relative_pos) / self.num_players)
        
        # Pot odds consideration
        total_pot = self.pot + sum(p.current_bet for p in self.players)
        call_amount = self.current_bet - player.current_bet
        if call_amount > 0 and total_pot > 0:
            pot_odds = call_amount / (total_pot + call_amount)
            # Adjust equity based on pot odds
            if hand_strength > pot_odds:
                equity_multiplier = 1.2  # Good pot odds
            else:
                equity_multiplier = 0.8  # Poor pot odds
        else:
            equity_multiplier = 1.0
        
        # Phase multiplier (later streets = more accurate equity)
        phase_multipliers = {
            GamePhase.PREFLOP: 0.7,  # Less certain
            GamePhase.FLOP: 0.8,
            GamePhase.TURN: 0.9,
            GamePhase.RIVER: 1.0     # Most certain
        }
        phase_multiplier = phase_multipliers.get(self.current_phase, 1.0)
        
        # Calculate final equity
        equity = (
            hand_strength 
            * position_multiplier 
            * equity_multiplier 
            * phase_multiplier
        )
        
        # Clip to [0, 1]
        return np.clip(equity, 0.0, 1.0)

    def _create_side_pot(self, all_in_amount):
        """
        Creates a side pot when a player is all-in for a smaller amount.
        
        Args:
            all_in_amount (int): The amount the all-in player has bet
        """
        # TODO: Implement side pot logic
        # For now, just continue the hand without creating actual side pots
        pass

    def manual_run(self):
        """
        Lance le jeu en mode manuel avec interface graphique.
        Gère la boucle principale du jeu et les événements.
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
                        self.reset()
                    if event.key == pygame.K_s:
                        state = self.get_state()
                        print('--------------------------------')
                        
                        # Print player cards (2 cards, each with value and suit)
                        print("Player cards:")
                        for i in range(2):  # 2 cards
                            value_range = state[i*17:(i*17)+13]  # 13 possible values
                            suit_range = state[i*17+13:(i*17)+17]  # 4 possible suits
                            value = value_range.index(1) + 2  # Convert back to card value
                            suit = ["♠", "♥", "♦", "♣"][suit_range.index(1)]  # Convert back to suit
                            print(f"Card {i+1}: {value}{suit}")
                        
                        # Print community cards (up to 5 cards)
                        print("\nCommunity cards:")
                        for i in range(5):  # Up to 5 community cards
                            base_idx = 34 + (i*17)  # Starting index for each community card
                            value_range = state[base_idx:base_idx+13]
                            suit_range = state[base_idx+13:base_idx+17]
                            if 1 in value_range:  # Check if card exists
                                value = value_range.index(1) + 2
                                suit = ["♠", "♥", "♦", "♣"][suit_range.index(1)]
                                print(f"Card {i+1}: {value}{suit}")
                        
                        # Print rest of state information
                        print(f"\nHand rank: {state[119] * len(HandRank)}")  # Index after card encodings
                        print(f"Game phase: {state[120:125]}")  # 5 values for game phase
                        print(f"Round number: {state[125]}")
                        print(f"Current bet: {state[126]}")
                        print(f"Stack sizes: {[x * self.starting_stack for x in state[127:130]]}")
                        print(f"Current bets: {state[130:133]}")
                        print(f"Player activity: {state[133:136]}")
                        print(f"Relative positions: {state[136:139]}")
                        print(f"Available actions: {state[139:144]}")
                        print(f"Previous actions Player 1: {state[144:150]}")
                        print(f"Previous actions Player 2: {state[150:156]}")
                        print(f"Previous actions Player 3: {state[156:162]}")
                        print(f"Win probability: {state[162]}")
                        print(f"Pot odds: {state[163]}")
                        print(f"Equity: {state[164]}")
                        print(f"Aggression factor: {state[165]}")
                        print('--------------------------------')
                
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
        Met à jour les blindes selon la structure définie.
        Augmente les blindes tous les N mains si possible.
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
