# train_model.py
import torch
from student_agent import Game2048Env
from value_approx import NTupleApproximator, td_learning, patterns

def train_and_save_model():
    # Create approximator and environment
    approximator = NTupleApproximator(board_size=4, patterns=patterns)
    env = Game2048Env()
    
    # Run TD-Learning training
    final_scores, trained_approximator = td_learning(
        env, 
        approximator, 
        num_episodes=100,
        alpha=0.01,
        gamma=0.99, 
        epsilon=0.1
    )
    
    # Save the model
    state = {
        'board_size': trained_approximator.board_size,
        'patterns': trained_approximator.patterns,
        'weights': [dict(weight) for weight in trained_approximator.weights]
    }
    
    # Save the state to a file
    torch.save(state, '2048.pt')
    print("Model saved successfully as 2048.pt")
    
    return trained_approximator

if __name__ == "__main__":
    train_and_save_model()