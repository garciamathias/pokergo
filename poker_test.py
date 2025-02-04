# poker_train.py
import numpy as np
import random as rd
import pygame
import torch
import time
from poker_agents import PokerAgent
from poker_game import PokerGame, GamePhase, PlayerAction
import matplotlib
matplotlib.use('Agg')

# Hyperparameters
EPISODES = 10000
GAMMA = 0.9985
ALPHA = 0.003
EPS_DECAY = 0.9998
STATE_SIZE = 169
RENDERING = True

def set_seed(seed=42):
    rd.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def load_trained_agents():
    """
    Charge les agents entraînés à partir des modèles sauvegardés.
    
    Returns:
        list: Liste des agents chargés avec leurs modèles entraînés
    """
    agent_list = []    
    # Créer et charger les agents
    for i in range(3):
        agent = PokerAgent(
            state_size=STATE_SIZE,
            action_size=5,
            gamma=0.9985,
            learning_rate=0.001,
            load_model=True,  # Charger le modèle sauvegardé
            load_path=f"saved_models/poker_agent_player_{i+1}.pth"
        )
        agent.name = f"player_{i+1}"
        agent.is_human = False  # Les agents sont des IA
        agent_list.append(agent)
    
    # Définir le premier joueur comme humain pour les tests
    agent_list[0].is_human = True
    return agent_list

def run_test_games(agent_list, env):
    """
    Exécute des parties de test avec les agents chargés.
    
    Args:
        agent_list (list): Liste des agents à tester
        env (PokerGame): Environnement du jeu
    """
    running = True
    
    while running:
        # Gérer les événements Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    env.start_new_hand()
                if event.key == pygame.K_r:
                    env.reset()
            
            # Gérer les entrées du joueur humain 
            current_player = env.players[env.current_player_idx]
            current_agent = agent_list[env.current_player_idx]
            
            if current_player.is_human:
                env.handle_input(event)
            else:
                print(f"Current player: {current_player.name}")
                state = env.get_state()
                env._update_button_states()
                valid_actions = [a for a in PlayerAction if env.action_buttons[a].enabled]
                print('valid_actions', valid_actions)
                
                print(f"AI {current_player.name} is playing")
                # Ne pas décompresser le retour de get_action
                action_chosen = current_agent.get_action(state, 0.01, valid_actions)
                
                # Ajouter un délai pour voir l'action de l'IA
                time.sleep(1)
                
                # Exécuter l'action choisie
                env.process_action(current_player, action_chosen)
        
        # Mettre à jour l'affichage
        env._draw()
        pygame.display.flip()
        env.clock.tick(30)  # 30 FPS
    
    pygame.quit()

if __name__ == "__main__":
    # Initialiser l'environnement
    env = PokerGame()
    
    # Charger les agents entraînés
    agent_list = load_trained_agents()
    
    # Synchroniser les noms des joueurs
    for i, agent in enumerate(agent_list):
        env.players[i].name = agent.name
        env.players[i].is_human = agent.is_human
    
    # Lancer les parties de test
    run_test_games(agent_list, env)
