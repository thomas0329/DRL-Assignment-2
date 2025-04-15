import copy
import random
import math
import numpy as np
from TD_utils import create_env_from_state
# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.


class TD_MCTS_Node:
    # an additional parameter env is passed
    # env.copy()
    def __init__(self, env, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        # create a snapshot of the env
        # self.env = create_env_from_state(env, state, score)
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

# class DecisionNode(TD_MCTS_Node):
#     def __init__(self, env, state, score, parent=None, action=None):
#         super().__init__(state, score, parent, action)
#         self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]
        
# class ChanceNode(TD_MCTS_Node):
    # to be figured out
        
  

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
        ucts = []
        assigned = 0
        for action, child in node.children.items():
          Q = child.total_reward / child.visits
          uct = Q + self.c * math.sqrt(np.log(node.visits) / child.visits)
          ucts.append(uct)
          if uct > best_uct:
            best_uct = uct
            best_child = child
            assigned = 1
        if assigned == 0:
          #  print('ucts', ucts)  # ucts []
          print('node.fully expanded', node.fully_expanded())
          print('node.children.items()', node.children.items())
          print('node', hex(id(node)))

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


    def backpropagate(self, node, rollout_reward, selection_rwds):
      # to be verified
      # selection_rwds includes the reward from the expansion step
      reward = rollout_reward

      # from the expanded node
      while node is not None:
        node.visits += 1
        node.total_reward += reward
        # claculate next reward in advance
        if len(selection_rwds) != 0:
          reward = reward * self.gamma + selection_rwds[-1]
          selection_rwds.pop()

        node = node.parent
        
    def selection(self, node, sim_env):
        prev_score = node.score
        selection_rwds = []
        done = False
        while node.fully_expanded():
          
          node = self.select_child(node)
          board, score, done, _ = sim_env.step(node.action)
          
          selection_rwds.append(score - prev_score)
          prev_score = score

        return node, selection_rwds, done
    
    def expansion(self, node, selection_rwds, sim_env, done):
        if done:
          return node
        prev_score = node.score
        new_act = random.choice(node.untried_actions)
        board, score, done, _ = sim_env.step(new_act)
        
        incremental_score = score - prev_score
        selection_rwds.append(incremental_score)

        node.untried_actions.remove(new_act)
        
        node.children[new_act] = TD_MCTS_Node(self.env, board, score, parent=node, action=new_act)

        return node.children[new_act] # expanded node


    def run_simulation(self, root):
        
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)
        
        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        node, selection_rwds, done = self.selection(node, sim_env)
        
        # TODO: Expansion: If the node is not terminal, expand an untried action.
        node = self.expansion(node, selection_rwds, sim_env, done)
        
        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        
        # Backpropagate the obtained reward from the expanded node
        self.backpropagate(node, rollout_reward, selection_rwds)

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


  