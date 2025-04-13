import copy
import random
import math
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import os
from student_agent import Game2048Env

# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------

def rot(pattern):
  # rotates a pattern 90 degree clockwise
  # pattern = [(0, 0), (0, 1), (0, 2), (0, 3)] for example
  rotated_pattern = []
  for coord in pattern:
    x, y = coord
    rotated_pattern.append((y, 3-x))

  return rotated_pattern

def flip(pattern):
  # flips a pattern wrt y the axis in the middle of the board
  flipped_pattern = []
  for coord in pattern:
    x, y = coord
    flipped_pattern.append((x, 3-y))

  return flipped_pattern


class NTupleApproximator:
    def __init__(self, board_size, patterns, weights_path=None):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        for pattern in self.patterns:

            syms = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)

            # for syms_ in syms:
            #     self.symmetry_patterns.append(syms_)

        if os.path.exists(weights_path):
          print(f"Loading weights from {weights_path}...")
          self.load_weights(weights_path)
        else:
          print("No saved weights found. Starting fresh.")

    def save_weights(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.weights, f)

    def load_weights(self, filename):
      with open(filename, 'rb') as f:
          self.weights = pickle.load(f)



    def visualize(self, pat):
      # Create a 2D array filled with zeros
      array = np.zeros((4, 4))

      # Set the coordinates to 1
      for x, y in pat:
          array[x, y] = 1

      # Visualize
      plt.imshow(array, cmap='gray_r')
      plt.title("Visualized Coordinates")
      plt.grid(True)
      plt.show()


    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        # pat1 = [(0, 0), (0, 1), (0, 2), (0, 3)]
        sym1 = pattern
        sym2 = rot(sym1)
        sym3 = rot(sym2)
        sym4 = rot(sym3)
        sym5 = flip(sym1)
        sym6 = rot(sym5)
        sym7 = rot(sym6)
        sym8 = rot(sym7)

        # visualize the 8 tuples of the same pattern
        # for i, sym in enumerate([sym1, sym2, sym3, sym4, sym5, sym6, sym7, sym8]):
        #   self.visualize(sym)

        return [sym1, sym2, sym3, sym4, sym5, sym6, sym7, sym8]

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
      # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
      return tuple(self.tile_to_index(board[x, y]) for x, y in coords)


    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.

        value = 0
        for i, pat_group in enumerate(self.symmetry_patterns):
          for sym in pat_group:
            feat = self.get_feature(board, sym)
            v = self.weights[i][feat]
            value += v

        return value


    def update(self, board, delta, alpha):
        alpha = alpha / (len(self.patterns) * 8)

        # TODO: Update weights based on the TD error.
        # update current state value
        # update weights associated with each pattern group
        # alpha should be devided by the number of patterns?
        # part of the entries are updated for each LUT
        for i, pat_group in enumerate(self.symmetry_patterns):
          # the ith lookup table, self.weights[i], is used
          for sym in pat_group:
            feat = self.get_feature(board, sym)
            self.weights[i][feat] += alpha * delta


def create_env_from_state(env, state, score):
  # Create a deep copy of the environment with the given state and score.
  new_env = copy.deepcopy(env)
  new_env.board = state.copy()
  new_env.score = score
  return new_env

def TD_action_selection(env, approximator, gamma):
  # for simplicity, assume state transition is deterministic, though it's not
  # Q(s, a) = rt + gamma * V(next_state)
  legal_moves = [action for action in [0, 1, 2, 3] if env.is_move_legal(action)]
  best_Q = -float('inf')
  best_action = None
  prev_score = env.score
  for action in legal_moves:
    # try executing this action
    sim_env = create_env_from_state(env, env.board, env.score)
    next_state, new_score, done, _ = sim_env.step(action)
    value = approximator.value(next_state)
    Q = (new_score - prev_score) + gamma * value
    if Q > best_Q:
      best_Q = Q
      best_action = action

  # if random.random() < epsilon:
  #   # Random move (exploration)
  #   best_action = random.choice(legal_moves)

  return best_action


def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """
    final_scores = []
    success_flags = []
    best_avg = 0

    for episode in range(num_episodes):
        state = env.reset().copy()
        trajectory = []  # Store trajectory data if needed
        previous_score = 0
        done = False
        max_tile = np.max(state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            # TODO: action selection based on the value
            # Note: TD learning works fine on 2048 without explicit exploration, but you can still try some exploration methods.

            # epsilon greedy exploration seems to hinder training
            action = TD_action_selection(env, approximator, gamma, epsilon)  # from current state from env

            next_state, new_score, done, _ = env.step(action) # change state of the env


            next_state = next_state.copy()

            incremental_reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            # TODO: Store trajectory or just update depending on the implementation
            if done:
              incremental_reward = 0

            trajectory.append((state, action, incremental_reward, next_state))

            state = next_state

        # TODO: If you are storing the trajectory, consider updating it now depending on your implementation.

        for t in reversed(range(len(trajectory))):

            state, action, incremental_reward, next_state = trajectory[t]
            current_val = approximator.value(state)

            if t == len(trajectory) - 1:
              next_val = 0
            else:
              next_val = approximator.value(next_state)

            delta = incremental_reward + gamma * next_val - current_val # one step TD target: rt + γV(st+1, θ)
            approximator.update(state, delta, alpha)


        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")
            if avg_score > best_avg:
                print('new best, saving weights')
                best_avg = avg_score
                approximator.save_weights("weights.pkl")

    return final_scores


# Matsuzaki’s 8×6-tuple network.
pat1 = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
pat2 = [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (1, 3)]
pat3 = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1)]
pat4 = [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (2, 2)]
pat5 = [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2), (2, 2)]
pat6 = [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (2, 3)]
pat7 = [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2), (1, 3)]
pat8 = [(0, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
patterns = [pat1, pat2, pat3, pat4, pat5, pat6, pat7, pat8]

# pat1 = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
# pat2 = [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)]
# pat3 = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
# pat4 = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
# patterns = [pat1, pat2, pat3, pat4]

approximator = NTupleApproximator(board_size=4, patterns=patterns, weights_path='/content/weights.pkl')

env = Game2048Env()

# Run TD-Learning training
# Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
# However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.

final_scores = td_learning(env, approximator, num_episodes=100000, alpha=0.1, gamma=0.99, epsilon=0.1)

# training takes 28 hours

