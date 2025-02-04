#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module de calcul et d'actualisation de la range adverse.
Ce module propose deux fonctionnalités principales :
1. Actualiser (filtrer) la range adverse à partir des actions passées (preflop, flop, turn, river) et du board,
   en retournant à la fois la range mise à jour et un score moyen de force.
2. Évaluer la force de notre main par rapport à la range adverse (équité).
"""

import itertools
from itertools import combinations
from enum import Enum
from typing import List, Tuple, Optional, Dict

# On utilise la classe Card, définie dans poker_game.py
from poker_game import Card


class PlayerAction(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all-in"


class HandRange:
    """
    Représente une range de mains avec leurs fréquences/pondérations.
    
    Cette classe permet de :
    - Stocker des mains avec une fréquence associée (0-1)
    - Gérer l'ajout/suppression de mains
    - Calculer le nombre total de combinaisons
    
    La fréquence représente la probabilité que l'adversaire joue cette main dans sa range.
    Par exemple, une fréquence de 0.5 signifie qu'il joue cette main 50% du temps.
    """
    def __init__(self):
        self.hands: Dict[Tuple[Card, Card], float] = {}  # Main -> Fréquence (0-1)
        
    def add_hand(self, hand: Tuple[Card, Card], frequency: float = 1.0):
        self.hands[hand] = min(1.0, max(0.0, frequency))
        
    def remove_hand(self, hand: Tuple[Card, Card]):
        if hand in self.hands:
            del self.hands[hand]
            
    def get_frequency(self, hand: Tuple[Card, Card]) -> float:
        return self.hands.get(hand, 0.0)
        
    def total_combinations(self) -> float:
        return sum(self.hands.values())


def init_range(excluded_cards: Optional[List[Card]] = None) -> List[Tuple[Card, Card]]:
    """
    Initialise la range de l'adversaire avec toutes les combinaisons de 2 cartes possibles
    en excluant les cartes connues (notre main et le board).

    Args:
        excluded_cards (Optional[List[Card]]): Liste de cartes à exclure

    Returns:
        List[Tuple[Card, Card]]: Liste de mains potentielles (tuple de 2 cartes)
    """
    # Construction d'un deck complet
    deck = []
    suits = ['♠', '♥', '♦', '♣']
    values = list(range(2, 15))  # 2 à 14 (l'As étant représenté par 14)
    for suit in suits:
        for value in values:
            deck.append(Card(suit, value))
    if excluded_cards:
        # On exclut les cartes déjà vues (notre main, board, etc.)
        deck = [card for card in deck if all(str(card) != str(ex) for ex in excluded_cards)]
    # On crée toutes les combinaisons possibles de 2 cartes
    all_hands = list(combinations(deck, 2))
    return all_hands


def eval_preflop_strength(hand: Tuple[Card, Card]) -> float:
    """
    Évalue la force d'une main preflop sur une échelle de 0 à 1.
    Pour une paire, on ajoute un bonus en fonction de la valeur.
    Pour deux cartes différentes, on se base sur la somme des valeurs,
    avec un bonus si elles sont assorties et/ou connectées.

    Args:
        hand (Tuple[Card, Card]): La main (2 cartes)

    Returns:
        float: Force de la main preflop
    """
    card1, card2 = hand
    if card1.value == card2.value:
        score = 0.5 + (card1.value / 28)  # 28 permet de normaliser (valeur max 14)
    else:
        score = (card1.value + card2.value) / 28
        if card1.suit == card2.suit:
            score += 0.1
        if abs(card1.value - card2.value) == 1:
            score += 0.05
    return min(score, 1.0)


def eval_postflop_strength(hand: Tuple[Card, Card], board: List[Card]) -> float:
    """
    Évalue la force d'une main postflop (flop, turn, river) sur une échelle de 0 à 1.
    On part de la force preflop et on apporte un bonus si une ou les deux cartes de la main
    correspondent à une carte du board (ce qui peut indiquer que la main s'est améliorée).

    Args:
        hand (Tuple[Card, Card]): La main (2 cartes)
        board (List[Card]): Les cartes communes retournées

    Returns:
        float: Force de la main postflop
    """
    score = eval_preflop_strength(hand)
    board_values = [card.value for card in board]
    occurrence = 0
    if hand[0].value in board_values:
        occurrence += 1
    if hand[1].value in board_values:
        occurrence += 1
    bonus = 0.2 * occurrence  # Bonus pour 1 ou 2 correspondances sur le board
    return min(score + bonus, 1.0)


def evaluer_force_raise(montant_raise: float, pot_size: float, is_reraise: bool = False, previous_raise: float = 0.0) -> float:
    """
    Évalue la force d'une raise sur une échelle de 0 à 1.
    
    Facteurs pris en compte:
    1. Ratio raise/pot
    2. Si c'est un reraise (3bet, 4bet, etc.)
    3. Taille par rapport à la raise précédente
    
    Args:
        montant_raise: Taille de la raise en BB
        pot_size: Taille du pot avant la raise en BB
        is_reraise: True si c'est un reraise (3bet+)
        previous_raise: Taille de la raise précédente en BB
        
    Returns:
        float: Score de force (0-1)
    """
    # Score de base basé sur le ratio raise/pot
    ratio_pot = montant_raise / pot_size
    score = min(ratio_pot / 2, 1.0)  # Un ratio de 2x pot donne un score max
    
    # Bonus pour les reraises
    if is_reraise:
        ratio_previous = montant_raise / previous_raise
        if ratio_previous >= 3:  # 3x la raise précédente est très fort
            score += 0.3
        elif ratio_previous >= 2:  # 2x est modérément fort
            score += 0.2
        else:
            score += 0.1
            
    # Ajustements selon les seuils typiques
    if montant_raise >= 25:  # 25BB+ est généralement très fort
        score += 0.2
    elif montant_raise >= 15:  # 15BB+ est fort
        score += 0.1
        
    return min(score, 1.0)


def filtrer_range_preflop(range_initial: List[Tuple[Card, Card]], 
                         action: PlayerAction,
                         raise_size: float = 0.0,
                         pot_size: float = 1.5,  # 1.5BB par défaut (SB + BB)
                         is_reraise: bool = False,
                         previous_raise: float = 0.0) -> List[Tuple[Card, Card]]:
    """
    Filtre la range preflop en fonction de l'action et de la force de la raise
    """
    if action in [PlayerAction.RAISE, PlayerAction.ALL_IN]:
        force_raise = evaluer_force_raise(raise_size, pot_size, is_reraise, previous_raise)
        # Plus la raise est forte, plus le seuil est élevé
        threshold = 0.3 + (force_raise * 0.4)  # Seuil entre 0.3 et 0.7
    elif action == PlayerAction.CALL:
        threshold = 0.3
    else:
        threshold = 0.0
        
    return [hand for hand in range_initial if eval_preflop_strength(hand) >= threshold]


def filtrer_range_postflop(range_actuelle: List[Tuple[Card, Card]], board: List[Card], 
                          action: PlayerAction, pot_odds: float = 0.0,
                          stack_bb: float = 50.0) -> List[Tuple[Card, Card]]:
    """
    Filtre la range postflop en tenant compte de la texture du board,
    des pot odds et du stack
    """
    board_texture = analyser_texture_board(board)
    board_texture['board'] = board  # Ajout du board dans le dictionnaire
    range_temp = HandRange()
    
    # Convertir la liste de mains en HandRange
    for hand in range_actuelle:
        range_temp.add_hand(hand)
    
    # Ajustement des seuils selon le stack
    seuil_bluff = 0.3  # valeur par défaut
    if stack_bb < 15:
        seuil_bluff = 0.15  # Moins de bluffs en short stack
    elif stack_bb > 100:
        seuil_bluff = 0.4  # Plus de bluffs en deep stack
    
    filtered_hands = []
    for hand in range_actuelle:
        force = eval_postflop_strength(hand, board)
        continuation_freq = calculer_frequence_continuation(
            hand, board_texture, action, pot_odds, stack_bb
        )
        
        # Ajustement selon le stack
        if stack_bb < 15 and not _est_premium(hand):
            continuation_freq *= 0.7  # Moins agressif en short stack avec des mains faibles
        
        if continuation_freq > seuil_bluff:
            filtered_hands.append(hand)
            
    return filtered_hands


def analyser_texture_board(board: List[Card]) -> dict:
    """
    Analyse la texture d'un board pour déterminer ses caractéristiques principales.
    
    Returns:
        dict: Dictionnaire avec les caractéristiques suivantes:
            - paired: True si le board contient une paire
            - suited: True si 3+ cartes de la même couleur
            - connected: True si cartes connectées (max 1 gap)
            - high_cards: Nombre de cartes >= 10
            - dynamic: Score de dynamisme (0-1)
            - main_suit: Couleur dominante si suited
    """
    if not board:
        return {
            'paired': False,
            'suited': False,
            'connected': False,
            'high_cards': 0,
            'dynamic': 0.0,
            'main_suit': None
        }
        
    # Analyse des couleurs
    suits_count = {}
    for card in board:
        suits_count[card.suit] = suits_count.get(card.suit, 0) + 1
    max_suited = max(suits_count.values())
    main_suit = max(suits_count.items(), key=lambda x: x[1])[0] if max_suited >= 2 else None
    
    # Analyse des valeurs
    values = sorted([card.value for card in board])
    
    # Caractéristiques
    paired = len(set(values)) < len(values)
    suited = max_suited >= 3
    high_cards = sum(1 for v in values if v >= 10)
    
    # Vérification des connexions
    connected = False
    if len(values) >= 2:
        gaps = [values[i+1] - values[i] for i in range(len(values)-1)]
        connected = any(gap <= 2 for gap in gaps)
    
    # Calcul du dynamisme
    dynamic = _evaluer_dynamisme_board(board)
    
    return {
        'paired': paired,
        'suited': suited,
        'connected': connected,
        'high_cards': high_cards,
        'dynamic': dynamic,
        'main_suit': main_suit
    }


def calculer_frequence_continuation(hand: Tuple[Card, Card], 
                                  board_texture: dict,
                                  action: PlayerAction,
                                  pot_odds: float,
                                  stack_bb: float) -> float:
    """
    Calcule la probabilité qu'un joueur continue avec une main donnée.
    
    La fréquence est basée sur :
    1. Type de main (overpair, top pair, tirage...)
    2. Texture du board (statique vs dynamique)
    3. Action précédente (check vs raise)
    4. Pot odds (rentabilité pour suivre)
    5. Taille du stack (plus agressif en deep stack)
    
    Par exemple :
    - Overpair sur board sec : haute fréquence (0.9)
    - Tirage sur board coordonné : fréquence basée sur les pot odds
    - Bluff sur board dynamique : fréquence modérée (0.3)
    
    Args:
        hand: Main du joueur
        board_texture: Caractéristiques du board
        action: Action précédente
        pot_odds: Rapport mise/pot
        stack_bb: Stack en big blinds
        
    Returns:
        float: Fréquence de continuation (0-1)
    """
    base_freq = 0.0
    
    # Ajustement selon le type de main
    if _est_overpair(hand, board_texture):
        base_freq = 0.9
    elif _est_top_pair(hand, board_texture):
        base_freq = 0.8
    elif _est_draw(hand, board_texture):
        base_freq = pot_odds * 1.2  # Légèrement plus que les pot odds
    else:
        base_freq = 0.3  # Bluffs occasionnels
        
    # Ajustement selon l'action
    if action == PlayerAction.CHECK:
        base_freq *= 0.7
    elif action == PlayerAction.RAISE:
        base_freq *= 1.3
        
    # Ajustement selon le stack
    if stack_bb < 15 and not _est_premium(hand):
        base_freq *= 0.7  # Moins agressif en short stack avec des mains faibles
        
    return min(1.0, base_freq)


def actualiser_range(actions: dict, board: List[Card],
                    excluded_cards: Optional[List[Card]] = None,
                    raise_sizes: Optional[dict] = None,
                    pot_sizes: Optional[dict] = None) -> Tuple[List[Tuple[Card, Card]], float]:
    """
    Met à jour la range en prenant en compte les tailles de raise
    
    Args:
        actions: Dictionnaire des actions par street
        board: Cartes communes
        excluded_cards: Cartes à exclure
        raise_sizes: Dictionnaire des tailles de raise par street
        pot_sizes: Dictionnaire des tailles de pot par street
    """
    range_possible = init_range(excluded_cards)
    raise_sizes = raise_sizes or {}
    pot_sizes = pot_sizes or {}

    # Filtrage preflop avec les nouvelles métriques
    if "preflop" in actions:
        is_reraise = False
        previous_raise = 0.0
        current_street_actions = []
        
        # Détecter si c'est un reraise
        for street, action_history in actions.items():
            if street == "preflop":
                if isinstance(action_history, list):
                    current_street_actions = action_history
                else:
                    current_street_actions = [action_history]
                    
        is_reraise = len([a for a in current_street_actions if a == PlayerAction.RAISE]) > 1
        if is_reraise:
            previous_raise = raise_sizes.get("preflop_previous", 0.0)
            
        range_possible = filtrer_range_preflop(
            range_possible, 
            actions["preflop"],
            raise_sizes.get("preflop", 0.0),
            pot_sizes.get("preflop", 1.5),
            is_reraise,
            previous_raise
        )

    # Filtrage sur le flop (si le board comporte au moins 3 cartes)
    if board and len(board) >= 3 and "flop" in actions:
        range_possible = filtrer_range_postflop(range_possible, board, actions["flop"], 0.0, 0.0)

    # Filtrage sur le turn (si le board comporte 4 cartes)
    if board and len(board) >= 4 and "turn" in actions:
        range_possible = filtrer_range_postflop(range_possible, board, actions["turn"], 0.0, 0.0)

    # Filtrage sur la river (si le board comporte 5 cartes)
    if board and len(board) == 5 and "river" in actions:
        range_possible = filtrer_range_postflop(range_possible, board, actions["river"], 0.0, 0.0)

    # Calcul de la force moyenne de la range
    if board and len(board) >= 3:
        forces = [eval_postflop_strength(hand, board) for hand in range_possible]
    else:
        forces = [eval_preflop_strength(hand) for hand in range_possible]
    force_moyenne = sum(forces) / len(forces) if forces else 0.0

    return range_possible, force_moyenne


def evaluer_force_main_contre_range(notre_main: Tuple[Card, Card],
                                    range_adverse: List[Tuple[Card, Card]],
                                    board: List[Card]) -> float:
    """
    Évalue la force de notre main par rapport à la range adverse et au board.
    La méthode consiste à calculer une "équité" approximative :
      - On évalue la force de notre main (preflop ou postflop selon la présence du board).
      - Pour chaque main de la range adverse, on évalue sa force sur la même street.
      - On détermine le pourcentage de mains adverses que notre main bat.

    Args:
        notre_main (Tuple[Card, Card]): Notre main (2 cartes)
        range_adverse (List[Tuple[Card, Card]]): Range adverse filtrée
        board (List[Card]): Cartes communes

    Returns:
        float: Équité estimée de notre main (entre 0 et 1)
    """
    if board:
        notre_force = eval_postflop_strength(notre_main, board)
        adversaire_forces = [eval_postflop_strength(hand, board) for hand in range_adverse]
    else:
        notre_force = eval_preflop_strength(notre_main)
        adversaire_forces = [eval_preflop_strength(hand) for hand in range_adverse]

    # Comptabilise le nombre de mains adverses que notre main bat
    count = sum(1 for force in adversaire_forces if notre_force > force)
    equity = count / len(adversaire_forces) if adversaire_forces else 1.0
    return equity


def ajuster_range_selon_stack(range_hands: HandRange, stack_ratio: float) -> HandRange:
    """
    Ajuste la composition de la range selon la profondeur des stacks.
    
    Ajustements selon le stack :
    
    Short stack (<15 BB):
    - Augmente la fréquence des mains premium (+30%)
    - Réduit les mains spéculatives (-50%)
    - Favorise les mains à haute valeur immédiate
    
    Deep stack (>100 BB):
    - Augmente les mains spéculatives (+20%)
    - Maintient une range plus large
    - Permet plus de jeu post-flop
    
    Stack moyen (15-100 BB):
    - Garde la range équilibrée
    - Pas d'ajustements majeurs
    
    Args:
        range_hands: Range initiale
        stack_ratio: Ratio stack/big blind
        
    Returns:
        HandRange: Range ajustée selon le stack
    """
    range_ajustee = HandRange()
    
    # Ajustements selon la profondeur de stack
    if stack_ratio < 15:  # Short stack
        for hand, freq in range_hands.hands.items():
            if _est_premium(hand):
                # Augmente la fréquence des mains premium en short stack
                range_ajustee.add_hand(hand, min(1.0, freq * 1.3))
            elif _est_speculative(hand):
                # Réduit la fréquence des mains spéculatives
                range_ajustee.add_hand(hand, freq * 0.5)
            else:
                range_ajustee.add_hand(hand, freq)
                
    elif stack_ratio > 100:  # Deep stack
        for hand, freq in range_hands.hands.items():
            if _est_speculative(hand):
                # Augmente la fréquence des mains spéculatives en deep stack
                range_ajustee.add_hand(hand, min(1.0, freq * 1.2))
            else:
                range_ajustee.add_hand(hand, freq)
    
    else:  # Stack moyen
        return range_hands
        
    return range_ajustee


def _est_premium(hand: Tuple[Card, Card]) -> bool:
    """Détermine si une main est premium (AA, KK, QQ, AK)"""
    card1, card2 = hand
    if card1.value == card2.value and card1.value >= 12:  # Paires de Q+
        return True
    if card1.value + card2.value >= 27 and (card1.value == 14 or card2.value == 14):  # AK
        return True
    return False


def _est_speculative(hand: Tuple[Card, Card]) -> bool:
    """Détermine si une main est spéculative (petites paires, suited connectors)"""
    card1, card2 = hand
    # Petites paires
    if card1.value == card2.value and card1.value <= 10:
        return True
    # Suited connectors
    if card1.suit == card2.suit and abs(card1.value - card2.value) == 1:
        return True
    return False


def get_preflop_range(position: str, action: PlayerAction, stack_bb: float) -> HandRange:
    """
    Retourne une range preflop typique selon la position, l'action et le stack
    
    Args:
        position: Position du joueur ('UTG', 'MP', 'CO', 'BTN', 'SB', 'BB')
        action: Action réalisée
        stack_bb: Stack en big blinds
    """
    range_hands = HandRange()
    
    # Définition des ranges de base par position
    position_ranges = {
        'UTG': ['AA', 'KK', 'QQ', 'JJ', 'TT', 'AK', 'AQs'],  # Très serré
        'MP': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', 'AK', 'AQ', 'AJs'],  # Serré
        'CO': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', 'AK', 'AQ', 'AJ', 'AT', 'KQ'],  # Medium
        'BTN': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', 'AK', 'AQ', 'AJ', 'AT', 'A9s+', 'KQ', 'KJ'],  # Large
        'SB': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', 'AK', 'AQ', 'AJ'],  # Polarisé
        'BB': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', 'AK', 'AQ', 'AJ', 'AT', 'KQ']  # Défensif
    }
    
    # Ajustement selon l'action
    if action in [PlayerAction.RAISE, PlayerAction.ALL_IN]:
        # Range plus serrée pour les raises
        range_hands = _convert_range_to_hands(position_ranges[position][:len(position_ranges[position])//2])
    else:
        range_hands = _convert_range_to_hands(position_ranges[position])
        
    # Ajustement selon le stack
    range_hands = ajuster_range_selon_stack(range_hands, stack_bb)
    
    return range_hands


def evaluer_coherence_actions(actions_historique: List[Tuple[str, PlayerAction]], 
                            board: List[Card],
                            range_estimee: HandRange) -> float:
    """
    Évalue si les actions passées sont cohérentes avec la range estimée.
    
    Exemples d'incohérences :
    - Fold avec une range forte (>0.6)
    - Raise avec une range faible (<0.4)
    - All-in avec une range polarisée sur board dangereux
    
    Le score de cohérence diminue de 30% pour chaque action incohérente.
    
    Cette analyse permet de :
    1. Détecter des tells dans le jeu adverse
    2. Affiner l'estimation de la range
    3. Identifier des tendances exploitables
    
    Args:
        actions_historique: Liste des actions par street
        board: Cartes communes
        range_estimee: Range estimée de l'adversaire
        
    Returns:
        float: Score de cohérence (0-1)
    """
    coherence = 1.0
    
    for street, action in actions_historique:
        if not _action_coherente_avec_range(action, range_estimee, board):
            coherence *= 0.7
            
    return coherence

def _action_coherente_avec_range(action: PlayerAction, 
                                range_hands: HandRange,
                                board: List[Card]) -> bool:
    force_moyenne = calculer_force_moyenne(range_hands, board)
    
    if action == PlayerAction.RAISE and force_moyenne < 0.4:
        return False
    if action == PlayerAction.FOLD and force_moyenne > 0.6:
        return False
        
    return True

def _evaluer_dynamisme_board(board: List[Card]) -> float:
    """
    Évalue le dynamisme du board (potentiel de tirage)
    Retourne un score entre 0 (statique) et 1 (très dynamique)
    """
    score = 0.0
    
    # Vérifier les tirages couleur
    suits_count = {}
    for card in board:
        suits_count[card.suit] = suits_count.get(card.suit, 0) + 1
    if max(suits_count.values()) >= 3:
        score += 0.3
        
    # Vérifier les tirages quinte
    values = sorted([card.value for card in board])
    gaps = [values[i+1] - values[i] for i in range(len(values)-1)]
    if any(gap == 1 for gap in gaps):
        score += 0.3
    if len(set(values)) == len(board):  # Pas de paire
        score += 0.2
        
    return min(score, 1.0)

def _est_overpair(hand: Tuple[Card, Card], board_texture: dict) -> bool:
    """Vérifie si la main est une paire supérieure aux cartes du board"""
    if 'board' not in board_texture or not board_texture['board']:
        return False
        
    card1, card2 = hand
    if card1.value != card2.value:
        return False
    board_max = max(card.value for card in board_texture['board'])
    return card1.value > board_max

def _est_top_pair(hand: Tuple[Card, Card], board_texture: dict) -> bool:
    """Vérifie si la main forme la meilleure paire possible avec le board"""
    if 'board' not in board_texture or not board_texture['board']:
        return False
        
    card1, card2 = hand
    board_values = [card.value for card in board_texture['board']]
    board_max = max(board_values)
    return (card1.value == board_max or card2.value == board_max)

def _est_draw(hand: Tuple[Card, Card], board_texture: dict) -> bool:
    """Vérifie si la main a un tirage (couleur ou quinte)"""
    return _est_flush_draw(hand, board_texture) or _est_straight_draw(hand, board_texture)

def _est_flush_draw(hand: Tuple[Card, Card], board_texture: dict) -> bool:
    """Vérifie si la main a un tirage couleur"""
    if not board_texture.get('board'):
        return False
    
    suits_count = {}
    for card in list(hand) + board_texture.get('board', []):
        suits_count[card.suit] = suits_count.get(card.suit, 0) + 1
    return max(suits_count.values()) == 4

def _est_straight_draw(hand: Tuple[Card, Card], board_texture: dict) -> bool:
    """Vérifie si la main a un tirage quinte"""
    if not board_texture.get('board'):
        return False
        
    values = sorted(set([card.value for card in list(hand) + board_texture.get('board', [])]))
    if len(values) < 4:
        return False
        
    # Vérifier les trous dans la séquence
    gaps = [values[i+1] - values[i] for i in range(len(values)-1)]
    return sum(gaps) <= 4 and len(values) >= 4

def _convert_range_to_hands(range_notation: List[str]) -> HandRange:
    """
    Convertit une notation de range (ex: ['AA', 'KK', 'AKs']) en HandRange
    """
    range_hands = HandRange()
    
    for hand_str in range_notation:
        hands = _convert_hand_notation(hand_str)
        for hand in hands:
            range_hands.add_hand(hand)
            
    return range_hands

def _convert_hand_notation(hand_str: str) -> List[Tuple[Card, Card]]:
    """
    Convertit une notation de main (ex: 'AKs') en liste de combinaisons de cartes
    """
    value_map = {
        'T': 10, 
        'J': 11, 
        'Q': 12, 
        'K': 13, 
        'A': 14,
        '2': 2, '3': 3, '4': 4, '5': 5,
        '6': 6, '7': 7, '8': 8, '9': 9
    }
    suits = ['♠', '♥', '♦', '♣']
    
    # Conversion des valeurs des cartes
    card1_val = value_map[hand_str[0]]
    card2_val = value_map[hand_str[1]]
    
    hands = []
    if len(hand_str) == 3 and hand_str[2] == 's':  # Suited
        for suit in suits:
            hands.append((Card(suit, card1_val), Card(suit, card2_val)))
    else:  # Offsuit ou paire
        for suit1 in suits:
            for suit2 in suits:
                if card1_val == card2_val and suit1 >= suit2:
                    continue
                hands.append((Card(suit1, card1_val), Card(suit2, card2_val)))
                
    return hands

def calculer_force_moyenne(range_hands: HandRange, board: List[Card]) -> float:
    """
    Calcule la force moyenne d'une range sur un board donné
    """
    if not range_hands.hands:
        return 0.0
        
    total_force = 0.0
    total_freq = 0.0
    
    for hand, freq in range_hands.hands.items():
        if board:
            force = eval_postflop_strength(hand, board)
        else:
            force = eval_preflop_strength(hand)
        total_force += force * freq
        total_freq += freq
        
    return total_force / total_freq if total_freq > 0 else 0.0

def evaluer_range_vs_range(range1: HandRange, range2: HandRange, board: List[Card]) -> float:
    """
    Évalue l'équité d'une range contre une autre range sur un board donné.
    
    Cette fonction est utile pour :
    - Comparer la force de deux ranges
    - Évaluer l'avantage d'une position
    - Identifier les déséquilibres dans les ranges
    
    Args:
        range1: Première range à évaluer
        range2: Range adverse
        board: Cartes communes
        
    Returns:
        float: Équité de range1 vs range2 (0-1)
    """
    total_equity = 0.0
    total_weight = 0.0
    
    for hand1, freq1 in range1.hands.items():
        hand_equity = 0.0
        for hand2, freq2 in range2.hands.items():
            if _mains_compatibles(hand1, hand2, board):
                if board:
                    force1 = eval_postflop_strength(hand1, board)
                    force2 = eval_postflop_strength(hand2, board)
                else:
                    force1 = eval_preflop_strength(hand1)
                    force2 = eval_preflop_strength(hand2)
                
                hand_equity += (force1 > force2) * freq2
                
        total_equity += hand_equity * freq1
        total_weight += freq1
        
    return total_equity / total_weight if total_weight > 0 else 0.5

def _mains_compatibles(hand1: Tuple[Card, Card], hand2: Tuple[Card, Card], 
                      board: Optional[List[Card]] = None) -> bool:
    """
    Vérifie si deux mains sont compatibles (pas de cartes en commun)
    """
    cards1 = set(hand1)
    cards2 = set(hand2)
    board_cards = set(board) if board else set()
    
    return not (cards1 & cards2 or cards1 & board_cards or cards2 & board_cards)

def identifier_bluff_spots(board: List[Card], range_adverse: HandRange) -> float:
    """
    Identifie les spots favorables pour bluffer en analysant la texture du board
    et la range adverse.
    
    Facteurs considérés :
    1. Texture du board (statique vs dynamique)
    2. Force moyenne de la range adverse
    3. Présence de tirages manqués
    4. Distribution des mains dans la range
    
    Returns:
        float: Score de bluff (0-1), plus le score est élevé, 
              plus le spot est favorable pour bluffer
    """
    board_texture = analyser_texture_board(board)
    force_moyenne = calculer_force_moyenne(range_adverse, board)
    
    bluff_score = 0.0
    
    # Board sec = plus de bluffs car moins de calls
    if not board_texture['paired'] and not board_texture['suited']:
        bluff_score += 0.3
        
    # Range adverse faible = plus de bluffs
    if force_moyenne < 0.4:
        bluff_score += 0.3
        
    # Board avec tirages manqués = bon pour bluffer
    if board_texture['dynamic'] > 0.6:
        bluff_score += 0.2
        
    # Beaucoup de mains faibles dans la range = bon pour bluffer
    weak_hands = sum(1 for hand, freq in range_adverse.hands.items() 
                    if eval_postflop_strength(hand, board) < 0.3)
    if weak_hands / len(range_adverse.hands) > 0.5:
        bluff_score += 0.2
        
    return min(bluff_score, 1.0)

def ajuster_range_multiway(range_hands: HandRange, nb_joueurs: int) -> HandRange:
    """
    Ajuste une range pour le jeu multiway (plusieurs joueurs dans le coup).
    
    Ajustements :
    - Moins de bluffs
    - Plus de mains nutted
    - Moins de mains marginales
    - Plus de mains à potentiel
    
    Args:
        range_hands: Range initiale
        nb_joueurs: Nombre de joueurs actifs
        
    Returns:
        HandRange: Range ajustée pour le multiway
    """
    range_ajustee = HandRange()
    
    for hand, freq in range_hands.hands.items():
        # Réduire la fréquence selon le nombre de joueurs
        new_freq = freq
        
        # Mains premium gardent leur fréquence
        if _est_premium(hand):
            pass
        # Mains spéculatives réduites mais pas éliminées
        elif _est_speculative(hand):
            new_freq *= max(0.2, 1.0 / nb_joueurs)
        # Mains marginales fortement réduites
        else:
            new_freq *= max(0.1, 0.5 / nb_joueurs)
            
        if new_freq > 0:
            range_ajustee.add_hand(hand, new_freq)
            
    return range_ajustee

def analyser_tendances_adversaire(historique_actions: List[Tuple[str, PlayerAction, HandRange]],
                                min_mains: int = 20) -> dict:
    """
    Analyse les tendances de jeu d'un adversaire à partir de son historique.
    
    Tendances analysées :
    - Fréquence de continuation bet
    - Fréquence de fold face aux raises
    - Agressivité par position
    - Cohérence des ranges
    
    Args:
        historique_actions: Liste de tuples (street, action, range)
        min_mains: Nombre minimum de mains pour l'analyse
        
    Returns:
        dict: Statistiques et tendances identifiées
    """
    if len(historique_actions) < min_mains:
        return {"fiabilite": 0.0, "message": "Pas assez de mains"}
        
    stats = {
        "freq_cbet": 0.0,
        "freq_fold_vs_raise": 0.0,
        "coherence_moyenne": 0.0,
        "tendance_bluff": 0.0
    }
    
    # Analyser chaque action
    for street, action, range_estimee in historique_actions:
        # Calculer les statistiques
        if street == "flop":
            stats["freq_cbet"] += (action == PlayerAction.RAISE)
        if action == PlayerAction.FOLD:
            stats["freq_fold_vs_raise"] += 1
            
        # Évaluer la cohérence
        coherence = evaluer_coherence_actions([(street, action)], [], range_estimee)
        stats["coherence_moyenne"] += coherence
        
        # Détecter les tendances de bluff
        if action == PlayerAction.RAISE and calculer_force_moyenne(range_estimee, []) < 0.4:
            stats["tendance_bluff"] += 1
            
    # Normaliser les statistiques
    n = len(historique_actions)
    stats = {k: v/n for k, v in stats.items()}
    stats["fiabilite"] = min(1.0, len(historique_actions) / min_mains)
    
    return stats

def calculer_outs(hand: Tuple[Card, Card], board: List[Card]) -> int:
    """
    Calcule le nombre d'outs (cartes qui améliorent notre main)
    
    Prend en compte :
    - Tirages couleur (9 outs)
    - Tirages quinte (8 outs typiquement)
    - Overcards (6 outs max)
    - Paires (2-3 outs)
    
    Args:
        hand: Notre main (2 cartes)
        board: Cartes du board
        
    Returns:
        int: Nombre total d'outs
    """
    outs = 0
    used_cards = list(hand) + board
    
    # Vérifier les tirages couleur
    if _est_flush_draw(hand, {'board': board}):
        suit_target = hand[0].suit if hand[0].suit == hand[1].suit else None
        if suit_target:
            outs += 9 - len([c for c in used_cards if c.suit == suit_target])
    
    # Vérifier les tirages quinte
    if _est_straight_draw(hand, {'board': board}):
        values = sorted(set([c.value for c in used_cards]))
        gaps = []
        for i in range(len(values)-1):
            gap = values[i+1] - values[i]
            if gap > 1:
                gaps.extend(range(values[i]+1, values[i+1]))
        # Retirer les cartes déjà utilisées
        used_values = [c.value for c in used_cards]
        gaps = [v for v in gaps if v not in used_values]
        outs += len(gaps) * 4  # 4 cartes possibles par valeur
        
    # Vérifier les overcards
    board_max = max([c.value for c in board])
    for card in hand:
        if card.value > board_max:
            remaining = 3  # 3 autres cartes de même valeur
            outs += remaining - len([c for c in board if c.value == card.value])
            
    # Vérifier les paires possibles
    for card in hand:
        if card.value not in [c.value for c in board]:
            remaining = 3  # 3 autres cartes de même valeur
            outs += remaining - len([c for c in used_cards if c.value == card.value])
            
    return outs

if __name__ == "__main__":
    print("=== Tests du module de range ===\n")
    
    # Test 1 : Analyse d'une situation preflop + flop
    print("Test 1 : Situation preflop + flop")
    actions = {
        "preflop": PlayerAction.RAISE,
        "flop": PlayerAction.CALL
    }
    board = [Card('♠', 10), Card('♥', 3), Card('♦', 7)]
    my_cards = [Card('♣', 14), Card('♣', 13)]  # AK clubs
    excluded = my_cards + board

    print(f"Actions adverses: {[f'{street}: {action.value}' for street, action in actions.items()]}")
    print(f"Board: {[str(c) for c in board]}")
    print(f"Notre main: {[str(c) for c in my_cards]}")
    
    range_adverse, range_score = actualiser_range(actions, board, excluded)
    print(f"\nRésultats:")
    print(f"- Nombre de mains dans la range adverse : {len(range_adverse)}")
    print(f"- Score moyen de la range adverse : {range_score:.2f}")
    equity = evaluer_force_main_contre_range(tuple(my_cards), range_adverse, board)
    print(f"- Equity de AKc contre la range adverse : {equity:.2f}\n")

    # Test 2 : Analyse de la texture du board
    print("Test 2 : Analyse de la texture du board")
    print(f"Board analysé: {[str(c) for c in board]}")
    board_texture = analyser_texture_board(board)
    print(f"\nCaractéristiques du board:")
    print(f"- Paired (cartes appariées): {board_texture['paired']}")
    print(f"- Suited (cartes assorties): {board_texture['suited']}")
    print(f"- Connected (cartes connectées): {board_texture['connected']}")
    print(f"- Nombre de cartes hautes (T+): {board_texture['high_cards']}")
    print(f"- Score de dynamisme: {board_texture['dynamic']:.2f} (0=statique, 1=très dynamique)\n")

    # Test 3 : Range preflop selon la position
    print("Test 3 : Range preflop par position")
    print("Action: RAISE, Stack: 50BB")
    positions = ['UTG', 'MP', 'BTN']
    for pos in positions:
        range_pos = get_preflop_range(pos, PlayerAction.RAISE, 50.0)
        print(f"- {pos}: {range_pos.total_combinations():.0f} combinaisons possibles")
        # Afficher quelques mains représentatives de la range
        sample_hands = list(range_pos.hands.items())[:3]
        print(f"  Exemples de mains: {[str(h[0][0])+str(h[0][1]) for h in sample_hands]}")

    # Test 4 : Analyse d'un spot de bluff
    print("\nTest 4 : Analyse d'un spot de bluff")
    board_bluff = [Card('♠', 14), Card('♠', 10), Card('♥', 9)]  # A♠ T♠ 9♥
    print(f"Board: {[str(c) for c in board_bluff]} (board avec tirage couleur)")
    
    range_bluff = HandRange()
    range_bluff.add_hand((Card('♣', 7), Card('♦', 7)))  # 77
    range_bluff.add_hand((Card('♣', 8), Card('♦', 8)))  # 88
    print("Range adverse: paires moyennes (77, 88)")
    
    bluff_score = identifier_bluff_spots(board_bluff, range_bluff)
    print(f"Score de bluff: {bluff_score:.2f} (0=mauvais spot, 1=excellent spot)")
    if bluff_score > 0.6:
        print("→ Bon spot pour bluffer (board dangereux vs range faible)")
    elif bluff_score > 0.3:
        print("→ Spot correct pour bluffer occasionnellement")
    else:
        print("→ Mauvais spot pour bluffer")

    # Test 5 : Ajustement multiway
    print("\nTest 5 : Ajustement multiway")
    range_base = HandRange()
    range_base.add_hand((Card('♠', 14), Card('♠', 13)))  # AKs
    range_base.add_hand((Card('♥', 7), Card('♥', 6)))   # 76s
    range_base.add_hand((Card('♦', 5), Card('♣', 5)))   # 55
    
    print("Range de base:")
    print("- AK suited (main premium)")
    print("- 76 suited (main spéculative)")
    print("- 55 (petite paire)")
    
    range_hw = ajuster_range_multiway(range_base, 2)
    range_mw = ajuster_range_multiway(range_base, 4)
    
    print(f"\nAjustements selon le nombre de joueurs:")
    print(f"- Range originale: {range_base.total_combinations():.2f} combinaisons")
    print(f"- Heads-up (2 joueurs): {range_hw.total_combinations():.2f} combinaisons")
    print(f"- Multiway (4 joueurs): {range_mw.total_combinations():.2f} combinaisons")
    print("→ La range se resserre avec plus de joueurs (moins de mains spéculatives)")

    # Test 6 : Simulation d'une main complète avec 3 joueurs
    print("\nTest 6 : Simulation d'une main complète (3 joueurs)")
    print("=" * 50)
    
    # Setup initial
    hero_cards = (Card('♠', 14), Card('♠', 13))  # Notre main: AK suited
    board = {
        'flop': [Card('♥', 13), Card('♣', 7), Card('♦', 2)],  # K♥ 7♣ 2♦
        'turn': Card('♠', 9),  # 9♠
        'river': Card('♥', 3)  # 3♥
    }
    
    # Actions de chaque joueur par street
    villain1_actions = {
        'preflop': PlayerAction.RAISE,    # UTG raise
        'flop': PlayerAction.CALL,        # Call notre cbet
        'turn': PlayerAction.CHECK,       # Check-call
        'river': PlayerAction.FOLD        # Fold face à notre value bet
    }
    
    villain2_actions = {
        'preflop': PlayerAction.CALL,     # BB call
        'flop': PlayerAction.CALL,        # Call
        'turn': PlayerAction.FOLD         # Fold face au turn
    }
    
    print(f"Notre main: {[str(c) for c in hero_cards]}")
    print("Position: Bouton (BTN)")
    print("\nScénario: UTG raise, nous 3-bet BTN, BB call, UTG call")
    
    # Informations de raise et tailles de pot
    raise_sizes = {
        'preflop': 9.0,        # UTG open 3BB
        'preflop_previous': 3.0,  # Pour calculer la force du 3bet
        'flop': 15.0,          # Cbet 15BB
        'turn': 30.0,          # Turn bet 30BB
        'river': 60.0          # River bet 60BB
    }
    
    pot_sizes = {
        'preflop': 1.5,    # SB + BB
        'flop': 27.0,      # Après le 3bet call
        'turn': 57.0,      # Après le cbet call
        'river': 117.0     # Après le turn bet call
    }
    
    # Simulation street par street
    streets = ['preflop', 'flop', 'turn', 'river']
    current_board = []
    
    for street in streets:
        print(f"\n{street.upper()}:")
        if street in board:
            if isinstance(board[street], list):
                current_board.extend(board[street])
            else:
                current_board.append(board[street])
            print(f"Board: {[str(c) for c in current_board]}")
            print(f"Pot: {['UTG: raise', 'BTN: 3-bet', 'BB: call', 'UTG: call'] if street == 'preflop' else 'Pot: {pot_sizes[street]:.0f} BB'}")
        
        # Analyse Villain 1 (UTG)
        if street in villain1_actions:
            current_v1_actions = {s: villain1_actions[s] for s in villain1_actions if s <= street}
            v1_range, v1_score = actualiser_range(
                current_v1_actions, 
                current_board, 
                list(hero_cards) + current_board,
                raise_sizes,
                pot_sizes
            )
            print(f"\nVillain 1 (UTG):")
            print(f"Action: {villain1_actions[street].value}")
            print(f"Range: {len(v1_range)} combinaisons")
            print(f"Force moyenne: {v1_score:.2f}")
            if len(v1_range) <= 10:
                print(f"Mains probables: {[str(h[0])+str(h[1]) for h in v1_range[:3]]}")
        
        # Analyse Villain 2 (BB)
        if street in villain2_actions:
            current_v2_actions = {s: villain2_actions[s] for s in villain2_actions if s <= street}
            v2_range, v2_score = actualiser_range(
                current_v2_actions, 
                current_board, 
                list(hero_cards) + current_board,
                raise_sizes,
                pot_sizes
            )
            print(f"\nVillain 2 (BB):")
            print(f"Action: {villain2_actions[street].value}")
            print(f"Range: {len(v2_range)} combinaisons")
            print(f"Force moyenne: {v2_score:.2f}")
            if len(v2_range) <= 10:
                print(f"Mains probables: {[str(h[0])+str(h[1]) for h in v2_range[:3]]}")
        
        # Notre équité contre les ranges actives
        if street != 'preflop':
            active_ranges = []
            if street in villain1_actions and villain1_actions[street] != PlayerAction.FOLD:
                active_ranges.append(v1_range)
            if street in villain2_actions and villain2_actions[street] != PlayerAction.FOLD:
                active_ranges.append(v2_range)
            
            if active_ranges:
                print("\nNotre situation:")
                for i, r in enumerate(active_ranges, 1):
                    equity = evaluer_force_main_contre_range(hero_cards, r, current_board)
                    print(f"Équité vs Villain {i}: {equity:.2f}")
                
                # Analyse du board
                board_texture = analyser_texture_board(current_board)
                if board_texture['dynamic'] > 0.5:
                    print("Board dynamique - Beaucoup de tirages possibles")
                else:
                    print("Board statique - Peu de tirages")
                    
                # Recommandation d'action
                print("\nRecommandation:")
                if all(equity > 0.6 for equity in [evaluer_force_main_contre_range(hero_cards, r, current_board) for r in active_ranges]):
                    print("→ Value bet recommandé (forte équité vs toutes les ranges)")
                elif any(equity < 0.4 for equity in [evaluer_force_main_contre_range(hero_cards, r, current_board) for r in active_ranges]):
                    print("→ Check/Fold recommandé (faible équité)")
                else:
                    print("→ Check/Call recommandé (équité moyenne)")

    # Test 7 : Simulation d'une partie Expresso
    print("\nTest 7 : Simulation d'une partie Expresso")
    print("=" * 50)
    
    # Setup initial
    hero_cards = (Card('♠', 14), Card('♠', 10))  # ATs
    board = {
        'flop': [Card('♠', 13), Card('♠', 5), Card('♥', 2)],  # K♠ 5♠ 2♥
        'turn': Card('♣', 10),  # T♣
        'river': Card('♦', 5)   # 5♦
    }
    
    # Tailles des mises par street
    raise_sizes = {
        'preflop': 2.0,         # Min-raise à 2BB
        'flop': 4.0,           # Demi-pot
        'turn': 8.0,          # Environ pot
        'river': 11.0         # All-in restant
    }
    
    pot_sizes = {
        'preflop': 1.5,    # SB + BB
        'flop': 6.0,       # 2BB (raise) + 2BB (call) + 1.5BB (blindes)
        'turn': 14.0,      # 6BB (pot flop) + 8BB (action flop)
        'river': 30.0      # 14BB (pot turn) + 16BB (action turn)
    }
    
    # Stacks initiaux et évolution
    stacks = {
        'hero': 25.0,    # 500 jetons = 25BB
        'v1': 25.0,
        'v2': 25.0
    }
    
    # Historique des mises pour chaque joueur
    mises = {
        'hero': {'preflop': 0.5, 'flop': 0, 'turn': 0, 'river': 0},  # Commence avec SB
        'v1': {'preflop': 0, 'flop': 0, 'turn': 0, 'river': 0},
        'v2': {'preflop': 1.0, 'flop': 0, 'turn': 0, 'river': 0}     # Commence avec BB
    }
    
    print(f"Notre main: {[str(c) for c in hero_cards]} (Tirage couleur + Paire)")
    print("Position: Bouton (BTN)")
    print(f"Stacks: BTN {stacks['hero']}BB, UTG {stacks['v1']}BB, BB {stacks['v2']}BB")
    print("\nScénario: Format Expresso (500 jetons)")
    
    # Simulation street par street
    streets = ['preflop', 'flop', 'turn', 'river']
    current_board = []
    
    for street in streets:
        print(f"\n{street.upper()}:")
        if street in board:
            if isinstance(board[street], list):
                current_board.extend(board[street])
            else:
                current_board.append(board[street])
            print(f"Board: {[str(c) for c in current_board]}")
            print(f"Pot: {pot_sizes[street]:.1f}BB")
            
            # Analyse détaillée du board
            board_texture = analyser_texture_board(current_board)
            print("\nAnalyse détaillée du board:")
            print("1. Texture:")
            if board_texture['suited']:
                nb_suited = max([len([c for c in current_board if c.suit == suit]) for suit in ['♠', '♥', '♦', '♣']])
                print(f"→ {nb_suited} cartes assorties ({[str(c) for c in current_board if c.suit == board_texture['main_suit']]})")
            if board_texture['paired']:
                # Trouver toutes les valeurs qui apparaissent plus d'une fois
                values = [c.value for c in current_board]
                paired_values = [v for v in set(values) if values.count(v) > 1]
                for value in paired_values:
                    # Convertir les valeurs numériques en notation poker
                    value_str = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T'}.get(value, str(value))
                    print(f"→ Paire de {value_str}")
            if board_texture['connected']:
                values = sorted([c.value for c in current_board])
                gaps = [values[i+1] - values[i] for i in range(len(values)-1)]
                print(f"→ Cartes connectées avec gaps: {gaps}")
                
            print("\n2. Calcul du dynamisme:")
            # Contribution des tirages couleur
            suited_score = 0.3 if board_texture['suited'] else 0.0
            print(f"- Tirages couleur: {suited_score:.2f}")
            
            # Contribution des tirages quinte
            straight_score = 0.0
            values = sorted([c.value for c in current_board])
            gaps = [values[i+1] - values[i] for i in range(len(values)-1)]
            if any(gap == 1 for gap in gaps):
                straight_score = 0.3
            print(f"- Tirages quinte: {straight_score:.2f}")
            
            # Contribution des cartes non pairées
            unpaired_score = 0.2 if len(set(values)) == len(current_board) else 0.0
            print(f"- Cartes non pairées: {unpaired_score:.2f}")
            
            print(f"Score total dynamisme: {board_texture['dynamic']:.2f}")
            
            print("\n3. Force de notre main:")
            # Analyse de notre main sur ce board
            hand_strength = eval_postflop_strength(hero_cards, current_board)
            print(f"- Force brute: {hand_strength:.2f}")
            
            # Détails des tirages
            if _est_flush_draw(hero_cards, {'board': current_board}):
                nb_flush = len([c for c in current_board + list(hero_cards) if c.suit == hero_cards[0].suit])
                print(f"- Tirage couleur: {nb_flush} cartes à {hero_cards[0].suit}")
                
            if _est_straight_draw(hero_cards, {'board': current_board}):
                values = sorted(set([c.value for c in current_board + list(hero_cards)]))
                print(f"- Tirage quinte: {values}")
                
            # Analyse des outs
            outs = calculer_outs(hero_cards, current_board)
            if outs > 0:
                print(f"- Nombre d'outs: {outs}")
                print(f"- Équité sur tirage: {(outs * 2):.1f}% au turn, {(outs * 4):.1f}% total")
            
        # Mise à jour des stacks avec détails
        if street in raise_sizes:
            print("\nDétail des mises:")
            for joueur in ['hero', 'v1', 'v2']:
                if mises[joueur][street] > 0:
                    print(f"- {joueur.upper()}: -{mises[joueur][street]}BB")
                    print(f"  Stack avant: {stacks[joueur] + mises[joueur][street]:.1f}BB")
                    print(f"  Stack après: {stacks[joueur]:.1f}BB")
                    
            print(f"\nConstruction du pot:")
            pot_street = sum([mises[j][street] for j in ['hero', 'v1', 'v2']])
            print(f"- Mises street actuelle: {pot_street:.1f}BB")
            pot_total = sum([sum(v.values()) for v in mises.values()])
            print(f"- Pot total: {pot_total:.1f}BB")
            print(f"- Pot odds: {(raise_sizes[street] / pot_total):.2f}")
        
        # Mise à jour des stacks
        if street in raise_sizes:
            # Préflop: ajouter les blindes
            if street == 'preflop':
                stacks['hero'] -= mises['hero']['preflop']  # SB
                stacks['v2'] -= mises['v2']['preflop']      # BB
            
            # Ajouter les mises de la street actuelle
            if street in villain1_actions and villain1_actions[street] != PlayerAction.FOLD:
                mises['v1'][street] = raise_sizes[street]
                stacks['v1'] -= raise_sizes[street]
            
            if street in villain2_actions and villain2_actions[street] != PlayerAction.FOLD:
                mises['v2'][street] = raise_sizes[street]
                stacks['v2'] -= raise_sizes[street]
            
            # Notre action (supposée être un call/raise à chaque street active)
            if any(v != PlayerAction.FOLD for v in [villain1_actions.get(street), villain2_actions.get(street)]):
                mises['hero'][street] = raise_sizes[street]
                stacks['hero'] -= raise_sizes[street]
        
        print(f"\nStacks restants: BTN {max(0, stacks['hero']):.1f}BB, "
              f"UTG {max(0, stacks['v1']):.1f}BB, "
              f"BB {max(0, stacks['v2']):.1f}BB")
        print(f"Pot actuel: {sum([sum(v.values()) for v in mises.values()]):.1f}BB")
        
        # Analyse Villain 1 (UTG)
        if street in villain1_actions:
            current_v1_actions = {s: villain1_actions[s] for s in villain1_actions if s <= street}
            v1_range, v1_score = actualiser_range(
                current_v1_actions, 
                current_board, 
                list(hero_cards) + current_board,
                raise_sizes,
                pot_sizes
            )
            print(f"\nVillain 1 (UTG):")
            print(f"Action: {villain1_actions[street].value} ({raise_sizes[street]}BB)")
            print(f"Range: {len(v1_range)} combinaisons")
            print(f"Force moyenne: {v1_score:.2f}")
            if len(v1_range) <= 10:
                print(f"Mains probables: {[str(h[0])+str(h[1]) for h in v1_range[:3]]}")
        
        # Analyse Villain 2 (BB)
        if street in villain2_actions:
            current_v2_actions = {s: villain2_actions[s] for s in villain2_actions if s <= street}
            v2_range, v2_score = actualiser_range(
                current_v2_actions, 
                current_board, 
                list(hero_cards) + current_board,
                raise_sizes,
                pot_sizes
            )
            print(f"\nVillain 2 (BB):")
            print(f"Action: {villain2_actions[street].value}")
            print(f"Range: {len(v2_range)} combinaisons")
            print(f"Force moyenne: {v2_score:.2f}")
            if len(v2_range) <= 10:
                print(f"Mains probables: {[str(h[0])+str(h[1]) for h in v2_range[:3]]}")
        
        # Notre situation et recommandations
        if street != 'preflop':
            active_ranges = []
            if street in villain1_actions and villain1_actions[street] != PlayerAction.FOLD:
                active_ranges.append(v1_range)
            if street in villain2_actions and villain2_actions[street] != PlayerAction.FOLD:
                active_ranges.append(v2_range)
            
            if active_ranges:
                print("\nNotre situation:")
                for i, r in enumerate(active_ranges, 1):
                    equity = evaluer_force_main_contre_range(hero_cards, r, current_board)
                    print(f"Équité vs Villain {i}: {equity:.2f}")
                
                # Recommandation spécifique Expresso
                print("\nRecommandation Expresso:")
                if stacks['hero'] < 5:  # Très short stack
                    print("→ Push/Fold mode (moins de 5BB)")
                elif any(equity > 0.7 for equity in [evaluer_force_main_contre_range(hero_cards, r, current_board) for r in active_ranges]):
                    print("→ All-in recommandé (très forte équité)")
                elif all(equity > 0.55 for equity in [evaluer_force_main_contre_range(hero_cards, r, current_board) for r in active_ranges]):
                    print(f"→ Value bet {min(pot_sizes[street], stacks['hero'])}BB (bonne équité)")
                else:
                    print("→ Check/Fold (équité insuffisante en format court stack)") 