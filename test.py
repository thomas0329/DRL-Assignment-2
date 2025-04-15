import copy
import random
import math
import numpy as np
from student_agent import Game2048Env
from TD_utils import NTupleApproximator, patterns
# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None):
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
class TD_MCTS:
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
        best_uct = -float('inf')

        for action, child in node.children.items():
          Q = child.total_reward / child.visits
          uct = Q + self.c * math.sqrt(np.log(node.visits) / child.visits)
          if uct > best_uct:
            best_uct = uct
            best_child = child

        return best_child

    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        # check this
        cum_incremental_rwds = 0
        discount = [self.gamma ** i for i in range(depth)]
        prev_score = sim_env.score
        p = 0

        for _ in range(depth):
          legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
          if len(legal_moves) == 0: # done before even taking an act
            return 0

          action = random.choice(legal_moves)
          _, score, done, _ = sim_env.step(action)
          if done:
            # incremental_score = 0
            return cum_incremental_rwds

          incremental_score = score - prev_score

          cum_incremental_rwds += incremental_score * (self.gamma ** p)
          p += 1
          prev_score = score


        terminal_val = self.approximator.value(sim_env.board)

        return cum_incremental_rwds + (self.gamma ** depth) * terminal_val
        # the value of the expanded node


    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        # reward: rollout_reward

        # from the expanded node
        while node is not None:
          node.visits += 1
          node.total_reward += reward
          node = node.parent


    def run_simulation(self, root):
        # print('simulation starts')
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        
        while node.fully_expanded():
          node = self.select_child(node)
          board, score, done, _ = sim_env.step(node.action)
        

        # TODO: Expansion: If the node is not terminal, expand an untried action.
        new_act = random.choice(node.untried_actions)
        board, score, done, _ = sim_env.step(new_act)
        
        node.untried_actions.remove(new_act)

        # point the current node to the expanded node
        node.children[new_act] = TD_MCTS_Node(board, score, parent=node, action=new_act)
        node = node.children[new_act]

        # Rollout: Simulate a random game from the expanded node.
        # print('rollout starts')
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # print('rollout ends')
        # Backpropagate the obtained reward.
        # from the expanded node!
        self.backpropagate(node, rollout_reward)
        # print('simulation ends')

    def best_action_distribution(self, root):
        # root seems to be the current node
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
        # most visited child

        # the target_distribution used by the policy approximator
        # the visit distr for each action


env = Game2048Env()

approximator = NTupleApproximator(board_size=4, patterns=patterns, weights_path='weights_.pkl')

td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=10, gamma=0.99)

state = env.reset()
# env.render()

done = False
while not done:
    # Create the root node from the current state
    root = TD_MCTS_Node(state, env.score)

    # Run multiple simulations to build the MCTS tree
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    # Select the best action (based on highest visit count)
    best_act, _ = td_mcts.best_action_distribution(root)
    print("TD-MCTS selected action:", best_act)

    # Execute the selected action and update the state
    state, reward, done, _ = env.step(best_act)
    print(reward)
    # env.render(action=best_act)

print("Game over, final score:", env.score)