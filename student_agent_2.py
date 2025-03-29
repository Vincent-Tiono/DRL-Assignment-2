
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
import torch
import pickle
from game_env import Game2048Env
from value_approx import NTupleApproximator
from collections import defaultdict


import copy
import random
import math
import numpy as np

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class UCTNode:
    def __init__(self, env, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class UCTMCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        best_value = float("-inf")
        best_child = None

        for child in node.children.values():
            q = child.total_reward / child.visits if child.visits > 0 else 0.0
            uct = q + self.c * math.sqrt(
                math.log(node.visits) / child.visits
            )
            if uct > best_value:
                best_value = uct
                best_child = child
            # print (q, self.c * math.sqrt(
            #     math.log(node.visits) / child.visits
            # ))
        return best_child


    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        initial_score = self.approximator.value(sim_env.board)
        final_score = initial_score
        decay_factor = 0.95
        cnt = 1
        for _ in range(1):
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break
            # action = np.random.choice(legal_moves)
            for action in legal_moves:
                sim_env_copy = copy.deepcopy(sim_env)
                sim_env_copy.board = sim_env.board.copy()
                sim_env_copy.score = sim_env.score
                
                board, reward, moved, done, _ = sim_env_copy.act(action)
                final_score += self.approximator.value(board) * decay_factor
                
            decay_factor *= self.gamma
            cnt += 1
            if done:
                break
        # print (self.approximator.value(sim_env.board), initial_score)
        # print (self.approximator.value(sim_env.board))
        # return self.approximator.value(sim_env.board)
        # print (final_score, initial_score)
        return final_score


    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        while node is not None:
            node.visits += 1
            node.total_reward += (reward - node.total_reward) / node.visits
            node = node.parent


    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded() and node.children and not sim_env.is_game_over():
            node = self.select_child(node)
            _, _, done, _ = sim_env.step(node.action)
            if done:
                break


        # TODO: Expansion: If the node is not terminal, expand an untried action.
        if (not node.fully_expanded()) and (not sim_env.is_game_over()):
            action = node.untried_actions.pop()
            sim_env.step(action)

            child = UCTNode(sim_env, state=sim_env.board.copy(), score=sim_env.score, parent=node, action=action)
            node.children[action] = child
            node = child

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution
   

def load_approximator(filepath='2048_model.pkl'):
    """Load the trained N-Tuple Approximator from a file"""
    with open('2048_model.pkl', 'rb') as f:
        state = pickle.load(f)

    # Reconstruct the NTupleApproximator object
    approximator = NTupleApproximator(
        board_size=state['board_size'],
        patterns=state['patterns']
    )

    # Restore weights (convert dictionaries back to defaultdicts)
    approximator.weights = [defaultdict(float, weight) for weight in state['weights']]
    
    return approximator

approximator = load_approximator('2048_model.pkl')
previous_score = 0
def get_action(state, score):
    # print (score)
    global previous_score
    env = Game2048Env()
    env.board = state.copy()
    env.score = score    


    td_mcts = UCTMCTS(env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=10, gamma=0.99)
        
    # Create the root node from the current state
    root = UCTNode(env, state, env.score)

    # Run multiple simulations to build the MCTS tree
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    # Select the best action (based on highest visit count)
    best_act, _ = td_mcts.best_action_distribution(root)

    return best_act  
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    
    best_value = -1e9
    best_action = None
    for a in legal_moves:
        sim_env = copy.deepcopy(env)
        sim_env.board = state.copy()
        sim_env.score = previous_score

        next_state, next_score, _, _, _ = sim_env.act(a)
        r = next_score - previous_score  # immediate reward
        v_next = approximator.value(next_state)
        if r + v_next > best_value:
            best_value = r + v_next
            best_action = a
    # print (r, previous_score)
    action = best_action
    state, reward, done, _ = env.step(action)  # Apply the selected action

    previous_score = reward
    return action
    # You can submit this random agent to evaluate the performance of a purely random strategy.
