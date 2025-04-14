from student_agent import Game2048Env
import os
from TD_utils import NTupleApproximator, patterns, td_learning

approximator = NTupleApproximator(board_size=4, patterns=patterns, weights_path='/content/weights.pkl')

env = Game2048Env()

# Run TD-Learning training
# Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
# However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.
# Create log directory if not exists
os.makedirs("logs", exist_ok=True)  

final_scores = td_learning(env, approximator, num_episodes=100000, alpha=0.1, gamma=1, epsilon=0.1)
# training takes 28 hours

