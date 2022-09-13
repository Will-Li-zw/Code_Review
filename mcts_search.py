import math
import random
import time
import gc
from agents.MCTS.node import Node

class MCTS:
    def __init__(self, my_pos, adv_pos, root_board):
        # create the root
        self.root_node = Node(my_pos, adv_pos, False, -1)
        self.cur_node = self.root_node
        self.cur_board = root_board
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))  # Moves (Up, Right, Down, Left)
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}  # Opposite Directions
        
    """stimulate the game to develop the tree"""
    def search(self, search_time):
        start_time = time.time()
        self.cur_node.get_next_state(self.cur_board)
        self.cur_node = self.root_node

        """stimulate game and update reward until time expired"""
        while time.time() - start_time <= search_time:
            score = self.game_play()    # start the stimulation of a game
            self.backpropagate(score)   # update the result of this stimulation
            self.cur_node = self.root_node  # reset the pointer

        """select the best move for children"""
        best_node = self.root_node.children[0]
        max_visit = 0
        for node in self.root_node.children:    # find the most visited node
            if node.visits > max_visit and node.reward > 0:
                max_visit = node.visits
                best_node = node
        self.root_node = best_node  # update the tree and board according to the best move
        self.cur_node = self.root_node
        self.update_cur_board()
        self.cur_node.get_next_state(self.cur_board)
        return best_node.my_pos, best_node.dir_barrier

    def game_play(self):
        game_result = self.cur_node.get_game_result(self.cur_board)
        """select next node to explore"""
        self.cur_node = self.select_best_move()
        self.update_cur_board()

        """random play the game until it end"""
        while not game_result[0]:
            self.cur_node = self.cur_node.get_one_child(self.cur_board)
            self.update_cur_board()
            game_result = self.cur_node.get_game_result(self.cur_board)

        "return the result of game according to the game play"
        if game_result[1] > game_result[2]:
            return 1
        elif game_result[1] == game_result[2]:
            return 0.5
        else:
            return 0

    """function to select the best node according to UCT"""
    def select_best_move(self):
        best_score = float('-inf')
        best_moves = []

        for cn in self.cur_node.children:
            score_flag = 1 if cn.turn else -1
            # use UCT formula to calculate the score
            if cn.visits == 0:
                child_node_score = math.inf
            else:
                child_node_score = score_flag * cn.reward / cn.visits + math.sqrt(
                    2 * math.log(self.cur_node.visits / cn.visits))

            # update the best score and the list of best moves
            if child_node_score > best_score:
                best_score = child_node_score
                best_moves = [cn]
            elif child_node_score == best_score and cn.reward >= 0:
                best_moves.append(cn)

        return random.choice(best_moves)

    """ update the visit and result of current tree """
    def backpropagate(self, score: float):
        while self.cur_node != self.root_node:
            # update the node's reward and visits
            self.cur_node.reward += score
            self.cur_node.visits += 1
            self.reset_cur_board()
            # goes to the node's parent
            self.cur_node = self.cur_node.parent
        self.cur_node.reward += score   # update the root
        self.cur_node.visits += 1
