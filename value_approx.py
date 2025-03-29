import copy
import random
import math
import numpy as np
from collections import defaultdict
from game_env import Game2048Env  # Changed import


class NTupleApproximator:
    def __init__(self, board_size, patterns):
        self.board_size = board_size
        self.weights = [defaultdict(float) for _ in patterns]
        self.patterns = patterns

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        Ensures all tiles are converted to a consistent index representation.
        """
        if tile == 0:
            return 0
        return int(math.log2(tile))

    def get_feature(self, board, coords):
        """
        Extract feature tuple from board based on given coordinates.
        """
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)

    def value(self, board):
        """
        Estimate the board value by summing values from all patterns.
        """
        return sum(
            self.weights[i][self.get_feature(board, pattern)] 
            for i, pattern in enumerate(self.patterns)
        )

    def update(self, board, delta, alpha):
        """
        Update weights for all patterns based on the TD error.
        """
        for i, pattern in enumerate(self.patterns):
            feature = self.get_feature(board, pattern)
            self.weights[i][feature] += alpha * delta
            
    def average_value(self):
        """
        Compute the average weight value over all patterns.
        If no weights are present, return 1.0 to avoid division by zero.
        """
        total = 0.0
        count = 0
        for weight_dict in self.weights:
            for value in weight_dict.values():
                total += value
                count += 1
        return total / count if count > 0 else 1.0

def choose_best_action(env, approximator, legal_moves, gamma):
    """
    Choose the best action based on estimated future value.
    """
    action_values = []
    for action in legal_moves:
        # Create a deep copy to simulate the move
        test_env = copy.deepcopy(env)
        prev_score = test_env.score
        
        # Simulate the move
        next_state, new_score, _, info = test_env.step(action)
        
        # Calculate incremental reward
        incremental_reward = new_score - prev_score
        
        # Use the state before random tile for value estimation
        next_state_before_random = info["state_before_random"]
        
        # Combine immediate reward with estimated future value
        action_value = incremental_reward + gamma * approximator.value(next_state_before_random)
        action_values.append((action, action_value))
    
    # Return the action with the highest estimated value
    return max(action_values, key=lambda x: x[1])[0]

def td_learning(env, approximator, num_episodes=50000, alpha=0.001, gamma=0.99, epsilon=0.1):
    """
    Train the 2048 agent using TD-Learning with after-state learning.
    """
    final_scores = []
    success_rates = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        previous_score = 0
        max_tile = 0
        trajectory = []

        while not done:
            # Find legal moves
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            
            if not legal_moves:
                break

            # Choose action using epsilon-greedy strategy
            action = choose_best_action(env, approximator, legal_moves, gamma)

            # Take the action
            next_state, new_score, done, info = env.step(action)
            
            # Calculate incremental reward
            incremental_reward = new_score - previous_score
            previous_score = new_score
            
            # Track maximum tile
            max_tile = max(max_tile, np.max(next_state))

            # Store transition: 
            # (current_state, action, reward, state_before_random, next_state)
            trajectory.append((
                state, 
                action, 
                incremental_reward, 
                info["state_before_random"], 
                next_state
            ))

            # Update state for next iteration
            state = next_state

        # Perform backward TD learning
        next_value = 0
        for s, a, r, s_after, s_next in reversed(trajectory):
            # Compute TD error
            delta = r + gamma * next_value - approximator.value(s_after)
            
            # Update weights
            approximator.update(s_after, delta, alpha)
            
            # Update next value for next iteration
            next_value = approximator.value(s_after)

        # Record episode statistics
        final_scores.append(previous_score)
        success_rates.append(1 if max_tile >= 2048 else 0)

        # Print progress periodically
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.mean(success_rates[-100:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Score: {avg_score:.2f} | "
                  f"Success Rate: {success_rate:.2f}")

    return final_scores, approximator

# Define patterns (same as before)
patterns = [
    # horizontal 4-tuples
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(1, 0), (1, 1), (1, 2), (1, 3)],
    [(2, 0), (2, 1), (2, 2), (2, 3)],
    [(3, 0), (3, 1), (3, 2), (3, 3)],
    
    # vertical 4-tuples
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 1), (1, 1), (2, 1), (3, 1)],
    [(0, 2), (1, 2), (2, 2), (3, 2)],
    [(0, 3), (1, 3), (2, 3), (3, 3)],
    
    # all 4-tile squares
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 1), (0, 2), (1, 1), (1, 2)],
    [(0, 2), (0, 3), (1, 2), (1, 3)],
    [(1, 0), (1, 1), (2, 0), (2, 1)],
    [(1, 1), (1, 2), (2, 1), (2, 2)],
    [(1, 2), (1, 3), (2, 2), (2, 3)],
    [(2, 0), (2, 1), (3, 0), (3, 1)],
    [(2, 1), (2, 2), (3, 1), (3, 2)],
    [(2, 2), (2, 3), (3, 2), (3, 3)],
]

# if __name__ == "__main__":
#     # Create approximator and environment
#     approximator = NTupleApproximator(board_size=4, patterns=patterns)
#     env = Game2048Env()

#     # Run TD-Learning training
#     final_scores = td_learning(
#         env, 
#         approximator, 
#         num_episodes=1000,  # Can increase for better learning
#         alpha=0.01,  # Smaller learning rate for stability
#         gamma=0.99, 
#         epsilon=0.1
#     )

#     # Save the model
#     import torch

#     # Define a dictionary to hold the model's state
#     state = {
#         'board_size': approximator.board_size,
#         'patterns': approximator.patterns,
#         'weights': [dict(weight) for weight in approximator.weights]
#     }

#     # Save the state to a file
#     torch.save(state, '2048.pt')

if __name__ == "__main__":
    # Create approximator and environment
    approximator = NTupleApproximator(board_size=4, patterns=patterns)
    env = Game2048Env()

    # Run TD-Learning training
    final_scores, approximator = td_learning(
        env, 
        approximator, 
        num_episodes=300,  # Adjust as needed
        alpha=0.01, 
        gamma=0.99, 
        epsilon=0.1
    )

    # Save the trained model
    import pickle

    state = {
        'board_size': approximator.board_size,
        'patterns': approximator.patterns,
        'weights': [dict(weight) for weight in approximator.weights]
    }

    with open('2048_model.pkl', 'wb') as f:
        pickle.dump(state, f)

    print("Model saved to 2048_model.pkl")
