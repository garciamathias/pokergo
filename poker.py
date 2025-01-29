import pygame
import random
from collections import defaultdict

# Initialisation Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Poker Texas Hold'em")
font = pygame.font.SysFont('Arial', 20)

# Couleurs
GREEN = (0, 128, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GOLD = (255, 215, 0)

class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
        
    def __repr__(self):
        return f"{self.rank}{self.suit}"

class Deck:
    def __init__(self):
        self.cards = []
        suits = ['♠', '♣', '♥', '♦']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        for suit in suits:
            for rank in ranks:
                self.cards.append(Card(suit, rank))
                
    def shuffle(self):
        random.shuffle(self.cards)
        
    def deal(self):
        return self.cards.pop()

class PokerGame:
    def __init__(self):
        self.deck = Deck()
        self.deck.shuffle()
        self.player_hand = []
        self.computer_hand = []
        self.community_cards = []
        self.pot = 0
        self.game_phase = 0  # 0: Pre-flop, 1: Flop, 2: Turn, 3: River
        self.buttons = []
        
        # Distribution des cartes initiales
        for _ in range(2):
            self.player_hand.append(self.deck.deal())
            self.computer_hand.append(self.deck.deal())
            
    def draw_card(self, card, x, y, hidden=False):
        if hidden:
            pygame.draw.rect(screen, WHITE, (x, y, 70, 100))
        else:
            pygame.draw.rect(screen, WHITE, (x, y, 70, 100))
            text = font.render(f"{card.rank}{card.suit}", True, BLACK)
            screen.blit(text, (x+5, y+5))
            
    def draw_interface(self):
        screen.fill(GREEN)
        
        # Cartes du joueur
        for i, card in enumerate(self.player_hand):
            self.draw_card(card, 300 + i*80, 450)
            
        # Cartes de l'ordinateur
        for i in range(len(self.computer_hand)):
            self.draw_card(None, 300 + i*80, 50, hidden=True)
            
        # Cartes communes
        for i, card in enumerate(self.community_cards):
            self.draw_card(card, 200 + i*80, 250)
            
        # Affichage du pot
        pot_text = font.render(f"Pot: ${self.pot}", True, WHITE)
        screen.blit(pot_text, (20, 20))
        
        # Boutons
        self.buttons = []
        if self.game_phase < 4:
            actions = ['Check', 'Bet 50', 'Fold']
            for i, action in enumerate(actions):
                rect = pygame.Rect(20 + i*120, 500, 100, 50)
                pygame.draw.rect(screen, GOLD, rect)
                text = font.render(action, True, BLACK)
                screen.blit(text, (rect.x + 10, rect.y + 15))
                self.buttons.append((rect, action))
                
    def evaluate_hand(self, hand):
        ranks = sorted([c.rank for c in hand], key=lambda x: '23456789TJQKA'.index(x))
        suits = [c.suit for c in hand]
        
        def is_flush():
            return len(set(suits)) == 1
        
        def is_straight():
            index = ['2','3','4','5','6','7','8','9','10','J','Q','K','A'].index
            indices = sorted([index(r) for r in ranks])
            return len(set(indices)) == 5 and indices[-1] - indices[0] == 4
        
        count = defaultdict(int)
        for r in ranks:
            count[r] += 1
        count = sorted(count.values(), reverse=True)
        
        if is_straight() and is_flush():
            return (8, "Quinte Flush")
        elif count[0] == 4:
            return (7, "Carré")
        elif count[0] == 3 and count[1] == 2:
            return (6, "Full House")
        elif is_flush():
            return (5, "Flush")
        elif is_straight():
            return (4, "Quinte")
        elif count[0] == 3:
            return (3, "Brelan")
        elif count[0] == 2 and count[1] == 2:
            return (2, "Double Paire")
        elif count[0] == 2:
            return (1, "Paire")
        else:
            return (0, "Carte Haute")

    def handle_bet(self, amount):
        self.pot += amount
        
    def next_phase(self):
        if self.game_phase == 0:
            for _ in range(3):
                self.community_cards.append(self.deck.deal())
        elif self.game_phase in [1, 2]:
            self.community_cards.append(self.deck.deal())
        self.game_phase += 1
        
    def check_winner(self):
        player_full = self.player_hand + self.community_cards
        computer_full = self.computer_hand + self.community_cards
        player_score = self.evaluate_hand(player_full)
        computer_score = self.evaluate_hand(computer_full)
        
        if player_score[0] > computer_score[0]:
            return "Joueur gagne avec " + player_score[1]
        elif computer_score[0] > player_score[0]:
            return "Ordinateur gagne avec " + computer_score[1]
        else:
            return "Égalité"

def main():
    game = PokerGame()
    running = True
    
    while running:
        game.draw_interface()
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                for rect, action in game.buttons:
                    if rect.collidepoint(pos):
                        if action == 'Bet 50':
                            game.handle_bet(50)
                            game.next_phase()
                        elif action == 'Check':
                            game.next_phase()
                        elif action == 'Fold':
                            print("Vous avez abandonné!")
                            running = False
                            
        if game.game_phase >= 4:
            result = game.check_winner()
            print(result)
            running = False
            
    pygame.quit()

if __name__ == "__main__":
    main()