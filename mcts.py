import copy
import random
import math
import numpy as np
from value_approx import NTupleApproximator  # Import only what you need


# UCT Node for MCTS
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
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
		# A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


class UCTMCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10):
        self.env = env
        self.iterations = iterations
        self.c = exploration_constant  # Balances exploration and exploitation
        self.rollout_depth = rollout_depth
        self.approximator = approximator

    def create_env_from_state(self, state, score):
        """
        Creates a deep copy of the environment with a given board state and score.
        """
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        """
        Select a child node using UCT formula enhanced with approximator values
        """
        selected_child = None
        best_score = -float('inf')
        
        for action, child in node.children.items():
            if child.visits == 0:
                return child
            
            # Standard UCT formula
            uct_value = child.total_reward / child.visits + self.c * math.sqrt(math.log(node.visits) / child.visits)
            
            approx_value = self.approximator.value(child.state) / self.approximator.average_value()
            uct_value += 0.1 * approx_value  # Small weight to avoid dominating UCT
        
            if selected_child is None or uct_value > best_score:
                best_score = uct_value
                selected_child = child
                
        return selected_child


    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout from the current state up to the specified depth.
        initial_score = sim_env.score
        cumulative_reward = 0
        discount_factor = 0.99  # You can adjust this value

        # Play some random moves first (for exploration)
        for i in range(min(depth, self.rollout_depth)):
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break  # Game over
            action = random.choice(legal_moves)
            _, new_score, done, info = sim_env.step(action)
            
            # Calculate discounted reward
            reward = new_score - initial_score
            cumulative_reward += (discount_factor ** i) * reward
            
            if done:
                break

        state_value = self.approximator.value(sim_env.board)
        v_norm = self.approximator.average_value()
        
        # Combine cumulative reward with estimated future value
        return (cumulative_reward + state_value) / v_norm


    def backpropagate(self, node, reward):
        # TODO: Propagate the reward up the tree, updating visit counts and total rewards.
        while node:
          node.visits += 1
          node.total_reward += (reward - node.total_reward) / node.visits
          node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching a non-fully expanded node.
        while node.fully_expanded():
          node = self.select_child(node)
          _, _, done, _ = sim_env.step(node.action)

        # TODO: Expansion: if the node has untried actions, expand one.
        if len(node.untried_actions) > 0:
          action = random.choice(node.untried_actions)
          node.untried_actions.remove(action)
          next_state, next_score, done, _ = sim_env.step(action)
          node.children[action] = UCTNode(self.env, state=next_state, score=next_score, parent=node, action=action)
          node = node.children[action]

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagation: Update the tree with the rollout reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        '''
        Computes the visit count distribution for each action at the root node.
        '''
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