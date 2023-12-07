#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import queue
import random
import copy

import chess
import numpy as np
import sys
import time
import ast

from itertools import permutations


class Aichess():
    """
    A class to represent the game of chess.

    ...

     Attributes:
    -----------
    chess : Chess
        Represents the current state of the chess game.
    listNextStates : list
        A list to store the next possible states in the game.
    listVisitedStates : list
        A list to store visited states during search.
    pathToTarget : list
        A list to store the path to the target state.
    currentStateW : list
        Represents the current state of the white pieces.
    depthMax : int
        The maximum depth to search in the game tree.
    checkMate : bool
        A boolean indicating whether the game is in a checkmate state.
    dictVisitedStates : dict
        A dictionary to keep track of visited states and their depths.
    dictPath : dict
        A dictionary to reconstruct the path during search.

    Methods:
    --------
    getCurrentState() -> list
        Returns the current state for the whites.

    getListNextStatesW(myState) -> list
        Retrieves a list of possible next states for the white pieces.

    isSameState(a, b) -> bool
        Checks whether two states are the same.

    isVisited(mystate) -> bool
        Checks if a state has been visited.

    isCheckMate(mystate) -> bool
        Checks if a state represents a checkmate.

    DepthFirstSearch(currentState, depth) -> bool
        Depth-first search algorithm.

    worthExploring(state, depth) -> bool
        Checks if a state is worth exploring during search using the optimised DFS algorithm.

    DepthFirstSearchOptimized(currentState, depth) -> bool
        Optimized depth-first search algorithm.

    reconstructPath(state, depth) -> None
        Reconstructs the path to the target state. Updates pathToTarget attribute.

    canviarEstat(start, to) -> None
        Moves a piece from one state to another.

    movePieces(start, depthStart, to, depthTo) -> None
        Moves all pieces between states.

    BreadthFirstSearch(currentState, depth) -> None
        Breadth-first search algorithm.

    h(state) -> int
        Calculates a heuristic value for a state using Manhattan distance.

    AStarSearch(currentState) 
        A* search algorithm -> To be implemented by you

    translate(s) -> tuple
        Translates traditional chess coordinates to list indices.

    """

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)
        self.q_table = dict()
        self.listNextStates = []
        self.listVisitedStates = []
        self.pathToTarget = []
        self.currentStateW = self.chess.boardSim.currentStateW;
        self.depthMax = 8
        self.checkMate = False

        # Prepare a dictionary to control the visited state and at which
        # depth they were found
        self.dictVisitedStates = {}
        # Dictionary to reconstruct the BFS path
        self.dictPath = {}
        self.TA = TA

    def getCurrentState(self):

        return self.chess.board.currentStateW.copy()

    def getListNextStatesW(self, myState):

        self.chess.board.getListNextStatesW(myState)
        self.listNextStates = self.chess.board.listNextStates.copy()

        return self.listNextStates

    def isSameState(self, a, b):

        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):

            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):

            if b[k] not in a:
                isSameState2 = False

        isSameState = isSameState1 and isSameState2
        return isSameState

    def isVisited(self, mystate):

        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedStates)):

                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True

            return isVisited
        else:
            return False

    def isCheckMate(self, mystate):

        # list of possible check mate states
        listCheckMateStates = [[[0, 0, 2], [2, 4, 6]], [[0, 1, 2], [2, 4, 6]], [[0, 2, 2], [2, 4, 6]],
                               [[0, 6, 2], [2, 4, 6]], [[0, 7, 2], [2, 4, 6]]]

        # Check all state permuations and if they coincide with a list of CheckMates
        for permState in list(permutations(mystate)):
            if list(permState) in listCheckMateStates:
                return True

        return False

    def DepthFirstSearch(self, currentState, depth):

        # We visited the node, therefore we add it to the list
        # In DF, when we add a node to the list of visited, and when we have
        # visited all noes, we eliminate it from the list of visited ones
        self.listVisitedStates.append(currentState)

        # is it checkmate?
        if self.isCheckMate(currentState):
            self.pathToTarget.append(currentState)
            return True

        if depth + 1 <= self.depthMax:
            for son in self.getListNextStatesW(currentState):
                if not self.isVisited(son):
                    # in the state son, the first piece is the one just moved
                    # We check the position of currentState
                    # matched by the piece moved
                    if son[0][2] == currentState[0][2]:
                        fitxaMoguda = 0
                    else:
                        fitxaMoguda = 1

                    # we move the piece to the new position
                    self.chess.moveSim(currentState[fitxaMoguda], son[0])
                    # We call again the method with the son, 
                    # increasing depth
                    if self.DepthFirstSearch(son, depth + 1):
                        # If the method returns True, this means that there has
                        # been a checkmate
                        # We ad the state to the list pathToTarget
                        self.pathToTarget.insert(0, currentState)
                        return True
                    # we reset the board to the previous state
                    self.chess.moveSim(son[0], currentState[fitxaMoguda])

        # We eliminate the node from the list of visited nodes
        # since we explored all successors
        self.listVisitedStates.remove(currentState)

    def worthExploring(self, state, depth):

        # First of all, we check that the depth is bigger than depthMax
        if depth > self.depthMax: return False
        visited = False
        # check if the state has been visited
        for perm in list(permutations(state)):
            permStr = str(perm)
            if permStr in list(self.dictVisitedStates.keys()):
                visited = True
                # If there state has been visited at a epth bigger than 
                # the current one, we are interestted in visiting it again
                if depth < self.dictVisitedStates[perm]:
                    # We update the depth associated to the state
                    self.dictVisitedStates[permStr] = depth
                    return True
        # Whenever not visited, we add it to the dictionary 
        # at the current depth
        if not visited:
            permStr = str(state)
            self.dictVisitedStates[permStr] = depth
            return True

    def DepthFirstSearchOptimized(self, currentState, depth):
        # is it checkmate?
        if self.isCheckMate(currentState):
            self.pathToTarget.append(currentState)
            return True

        for son in self.getListNextStatesW(currentState):
            if self.worthExploring(son, depth + 1):

                # in state 'son', the first piece is the one just moved
                # we check which position of currentstate matche
                # the piece just moved
                if son[0][2] == currentState[0][2]:
                    fitxaMoguda = 0
                else:
                    fitxaMoguda = 1

                # we move the piece to the novel position
                self.chess.moveSim(currentState[fitxaMoguda], son[0])
                # we call the method with the son again, increasing depth
                if self.DepthFirstSearchOptimized(son, depth + 1):
                    # If the method returns true, this means there was a checkmate
                    # we add the state to the list pathToTarget
                    self.pathToTarget.insert(0, currentState)
                    return True
                # we return the board to its previous state
                self.chess.moveSim(son[0], currentState[fitxaMoguda])

    def reconstructPath(self, state, depth):
        # When we found the solution, we obtain the path followed to get to this        
        for i in range(depth):
            self.pathToTarget.insert(0, state)
            # Per cada node, mirem quin és el seu pare
            state = self.dictPath[str(state)]

        self.pathToTarget.insert(0, state)

    def canviarEstat(self, start, to):
        # We check which piece has been moved from one state to the next
        if start[0] == to[0]:
            fitxaMogudaStart = 1
            fitxaMogudaTo = 1
        elif start[0] == to[1]:
            fitxaMogudaStart = 1
            fitxaMogudaTo = 0
        elif start[1] == to[0]:
            fitxaMogudaStart = 0
            fitxaMogudaTo = 1
        else:
            fitxaMogudaStart = 0
            fitxaMogudaTo = 0
        # move the piece changed
        self.chess.moveSim(start[fitxaMogudaStart], to[fitxaMogudaTo])

    def movePieces(self, start, depthStart, to, depthTo):

        # To move from one state to the next for BFS we will need to find
        # the state in common, and then move until the node 'to'
        moveList = []
        # We want that the depths are equal to find a common ancestor
        nodeTo = to
        nodeStart = start
        # if the depth of the node To is larger than that of start, 
        # we pick the ancesters of the node until being at the same
        # depth
        while (depthTo > depthStart):
            moveList.insert(0, to)
            nodeTo = self.dictPath[str(nodeTo)][0]
            depthTo -= 1
        # Analogous to the previous case, but we trace back the ancestors
        # until the node 'start'
        while (depthStart > depthTo):
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # We move the piece the the parerent state of nodeStart
            self.canviarEstat(nodeStart, ancestreStart)
            nodeStart = ancestreStart
            depthStart -= 1

        moveList.insert(0, nodeTo)
        # We seek for common node
        while nodeStart != nodeTo:
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # Move the piece the the parerent state of nodeStart
            self.canviarEstat(nodeStart, ancestreStart)
            # pick the parent of nodeTo
            nodeTo = self.dictPath[str(nodeTo)][0]
            # store in the list
            moveList.insert(0, nodeTo)
            nodeStart = ancestreStart
        # Move the pieces from the node in common
        # until the node 'to'
        for i in range(len(moveList)):
            if i < len(moveList) - 1:
                self.canviarEstat(moveList[i], moveList[i + 1])

    def BreadthFirstSearch(self, currentState, depth):
        """
        Check mate from currentStateW
        """
        BFSQueue = queue.Queue()
        # The node root has no parent, thus we add None, and -1, which would be the depth of the 'parent node'
        self.dictPath[str(currentState)] = (None, -1)
        depthCurrentState = 0
        BFSQueue.put(currentState)
        self.listVisitedStates.append(currentState)
        # iterate until there is no more candidate nodes
        while BFSQueue.qsize() > 0:
            # Find the optimal configuration
            node = BFSQueue.get()
            depthNode = self.dictPath[str(node)][1] + 1
            if depthNode > self.depthMax:
                break
            # If it not the root node, we move the pieces from the previous to the current state
            if depthNode > 0:
                self.movePieces(currentState, depthCurrentState, node, depthNode)

            if self.isCheckMate(node):
                # Si és checkmate, construïm el camí que hem trobat més òptim
                self.reconstructPath(node, depthNode)
                break

            for son in self.getListNextStatesW(node):
                if not self.isVisited(son):
                    self.listVisitedStates.append(son)
                    BFSQueue.put(son)
                    self.dictPath[str(son)] = (node, depthNode)
            currentState = node
            depthCurrentState = depthNode

    def h(self, state):

        if state[0][2] == 2:
            posicioRei = state[1]
            posicioTorre = state[0]
        else:
            posicioRei = state[0]
            posicioTorre = state[1]
        # With the king we wish to reach configuration (2,4), calculate Manhattan distance
        fila = abs(posicioRei[0] - 2)
        columna = abs(posicioRei[1] - 4)
        # Pick the minimum for the row and column, this is when the king has to move in diagonal
        # We calculate the difference between row an colum, to calculate the remaining movements
        # which it shoudl go going straight        
        hRei = min(fila, columna) + abs(fila - columna)
        # with the tower we have 3 different cases
        if posicioTorre[0] == 0 and (posicioTorre[1] < 3 or posicioTorre[1] > 5):
            hTorre = 0
        elif posicioTorre[0] != 0 and posicioTorre[1] >= 3 and posicioTorre[1] <= 5:
            hTorre = 2
        else:
            hTorre = 1
        # In our case, the heuristics is the real cost of movements
        return hRei + hTorre

    def AStarSearch(self, currentState):
        node_inicial = currentState  # Guardem el node inicial
        frontera = [(currentState, self.h(currentState))]  # Posem el primer node a la forntera

        # Inicialitzem el dict on guardem la profunditat de cada node
        dict_depth = {str(currentState): 0}
        # Inicialitzem el dict dels estats visitats com a keys on guardem el valor de la funcio succcesor al value
        self.dictVisitedStates = {str(currentState): self.h(currentState) + dict_depth[str(currentState)]}
        # Inicialitzem el dict on guardem el pare de cada node per aixi poder trobar el camí a la solució
        self.dictPath = {str(currentState): currentState}

        # Mentre la forntera no estigui buida s'itera el while
        while frontera:
            # Ordenem la frontera segons la minima heuristica
            frontera = sorted(frontera, key=lambda x: x[1])
            # Guardem el següent estat i valor de la primera tupla de frontera
            next_state, current_cost = frontera.pop(0)
            # Movem la les peces necessaries per canviar de l'estat actual al següent
            self.canviarEstat(currentState, next_state)
            # Definim com a estat actual el estat següent
            currentState = next_state

            # Comprovem si l'estat actual es Check Mate
            if self.isCheckMate(currentState):
                # Reconstruim el camí fins l'inicial
                self.reconstructPath(currentState, dict_depth[str(currentState)])

            # Iterem sobre tots els possibles nous estats
            for next_state in self.getListNextStatesW(currentState):
                # Si no hem visitat l'estat nou o el seu valor successor es menor del guardat, entrem al if
                if str(next_state) not in self.dictVisitedStates.keys() or (
                        self.h(next_state) + dict_depth[str(next_state)]) < self.dictVisitedStates[str(next_state)]:
                    # Definim com a pare del node següent al node actual
                    self.dictPath[str(next_state)] = currentState
                    # Definim la profunditat del node següent sumant 1 a la profunditat de l'actual
                    dict_depth[str(next_state)] = dict_depth[str(currentState)] + 1
                    # Fiquem el següent estat als visitats on tambe guardem el seu valor successor
                    self.dictVisitedStates[str(next_state)] = self.h(next_state) + dict_depth[str(next_state)]
                    # Fiquem el node a la forntera guardant l'estat i l'heuristica
                    frontera.append((next_state, (self.h(next_state))))

    def get_moving_states(self, state, next_state):
        print(type(next_state))
        if state[0] != next_state[0]:
            a = state[0]
            b = next_state[0]
        else:
            a = state[1]
            b = next_state[1]
        return a, b

    def sort_state(self, state):
        if state[0][2] == 6:
            aux = state[0]
            state[0] = state[1]
            state[1] = aux
        return state

    def update_q_part1(self, state, next_state, alpha, gamma):
        if self.isCheckMate(next_state):
            reward = 100
        else:
            reward = -1
        next_movements = [(self.sort_state(elemento)) for elemento in self.getListNextStatesW(next_state)]
        if str(next_state) not in self.q_table:
            next_moves_str = [str(elemento) for elemento in next_movements]
            self.q_table[str(next_state)] = {key: random.uniform(-1, 1) for key in next_moves_str}
        q_state = self.q_table[str(next_state)]
        if len(q_state.values()) != 0:
            max_next_q = max(q_state.values())
        else:
            max_next_q = float('-inf')
        q_value = self.q_table[str(state)][str(next_state)] + alpha * (reward + gamma * max_next_q - self.q_table[str(state)][str(next_state)])
        self.q_table[str(state)][str(next_state)] = q_value
        return reward

    def update_q_part2(self, state, next_state, alpha, gamma):
        reward = -self.h(next_state)
        if self.isCheckMate(next_state):
            reward = 100
        next_movements = [(self.sort_state(elemento)) for elemento in self.getListNextStatesW(next_state)]
        if str(next_state) not in self.q_table:
            next_moves_str = [str(elemento) for elemento in next_movements]
            self.q_table[str(next_state)] = {key: random.uniform(-1, 1) for key in next_moves_str}
        q_state = self.q_table[str(next_state)]
        if len(q_state.values()) != 0:
            max_next_q = max(q_state.values())
        else:
            max_next_q = float('-inf')
        q_value = self.q_table[str(state)][str(next_state)] + alpha * (reward + gamma * max_next_q - self.q_table[str(state)][str(next_state)])
        self.q_table[str(state)][str(next_state)] = q_value
        return reward

    def q_learning_chess(self, factor_epsilon, iterations, alpha, gamma, part=1):
        self.chess.board.print_board()
        self.q_table = dict()
        self.cell_values = [[-1 for _ in range(8)] for _ in range(8)]
        score = 0
        count = 0
        epsilon = 1
        state = copy.deepcopy(self.getCurrentState())
        next_movements = [(self.sort_state(elemento)) for elemento in self.getListNextStatesW(state)]
        next_moves_str = [str(elemento) for elemento in next_movements]
        self.q_table[str(state)] = {key: random.uniform(-1, 1) for key in next_moves_str}

        while count < iterations:
            q_state = self.q_table[str(state)]
            random_value = random.random()
            next_movements = [(self.sort_state(elemento)) for elemento in self.getListNextStatesW(state)]
            random_move = False
            next_state = None
            while next_state is None or (next_state[0][0] == 0 and next_state[0][1] == 4) or (next_state[1][0] == 0 and next_state[1][1] == 4):
                if random_value < 1 - epsilon:
                    next_state = max(q_state, key=q_state.get)
                    next_state = ast.literal_eval(next_state)
                    if next_state[0][0] == 0 and next_state[0][1] == 4 or next_state[1][0] == 0 and next_state[1][1] == 4:
                        del q_state[str(next_state)]
                        next_state = max(q_state, key=q_state.get)
                        next_state = ast.literal_eval(next_state)
                else:
                    next_state = random.choice(next_movements)
                    random_move = True
            a, b = self.get_moving_states(state, next_state)
            print(next_state)
            self.chess.move(a, b)
            self.chess.board.print_board()
            print(random_move)
            if part == 1:
                score += self.update_q_part1(state, next_state, alpha, gamma)
            else:
                score += self.update_q_part2(state, next_state, alpha, gamma)
            state = next_state
            if self.isCheckMate(state):
                if epsilon > 0:
                    epsilon -= factor_epsilon
                count += 1
                print("ha arribat al objectiu!")
                print("score:", score, "| iteració", count)
                self.chess.board.print_board()
                self.chess = chess.Chess(self.TA, True)
                state = copy.deepcopy(self.getCurrentState())
                score = 0
                if count > 100 and count % 20 == 0:
                    time.sleep(1)

    def drunken_sailor(self, factor_epsilon, iterations, alpha, gamma):
        self.chess.board.print_board()
        self.q_table = dict()
        self.cell_values = [[-1 for _ in range(8)] for _ in range(8)]
        score = 0
        count = 0
        epsilon = 1
        state = copy.deepcopy(self.getCurrentState())
        next_movements = [(self.sort_state(elemento)) for elemento in self.getListNextStatesW(state)]
        next_moves_str = [str(elemento) for elemento in next_movements]
        self.q_table[str(state)] = {key: random.uniform(-1, 1) for key in next_moves_str}

        while count < iterations:
            q_state = self.q_table[str(state)]
            random_value = random.random()
            next_movements = [(self.sort_state(elemento)) for elemento in self.getListNextStatesW(state)]
            random_move = False
            next_state = None
            while next_state is None or (next_state[0][0] == 0 and next_state[0][1] == 4) or (next_state[1][0] == 0 and next_state[1][1] == 4):
                if random_value < 1 - epsilon:
                    if random.random() > 0.95:
                        next_state = random.choice(next_movements)
                        random_move = True
                    else:
                        next_state = max(q_state, key=q_state.get)
                        next_state = ast.literal_eval(next_state)
                        if next_state[0][0] == 0 and next_state[0][1] == 4 or next_state[1][0] == 0 and next_state[1][1] == 4:
                            del q_state[str(next_state)]
                            next_state = max(q_state, key=q_state.get)
                            next_state = ast.literal_eval(next_state)
                else:
                    next_state = random.choice(next_movements)
                    random_move = True
            a, b = self.get_moving_states(state, next_state)
            print(next_state)
            self.chess.move(a, b)
            self.chess.board.print_board()
            print(random_move)
            score += self.update_q_part2(state, next_state, alpha, gamma)
            state = next_state
            if self.isCheckMate(state):
                if epsilon > 0:
                    epsilon -= factor_epsilon
                count += 1
                print("ha arribat al objectiu!")
                print("score:", score, "| iteració", count)
                self.chess.board.print_board()
                self.chess = chess.Chess(self.TA, True)
                state = copy.deepcopy(self.getCurrentState())
                score = 0
                if count > 100 and count % 20 == 0:
                    time.sleep(1)
def translate(s):
    """
    Translates traditional board coordinates of chess into list indices
    """
    try:
        row = int(s[0])
        col = s[1]
        if row < 1 or row > 8:
            print(s[0] + "is not in the range from 1 - 8")
            return None
        if col < 'a' or col > 'h':
            print(s[1] + "is not in the range from a - h")
            return None
        dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        return (8 - row, dict[col])
    except:
        print(s + "is not in the format '[number][letter]'")
        return None



if __name__ == "__main__":

    if len(sys.argv) < 1:
        sys.exit(1)
    # intiialize board
    TA = np.zeros((8, 8))
    # load initial state
    # white pieces
    TA[7][0] = 2
    TA[7][4] = 6
    TA[0][4] = 12

    # initialise bord
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    currentState = aichess.chess.board.currentStateW.copy()
    print("printing board")
    aichess.chess.boardSim.print_board()

    aichess.q_learning_chess(part=1, factor_epsilon=0.003,iterations=1000, alpha=0.2,gamma=0.8)
    aichess.q_learning_chess(part=2, factor_epsilon=0.005, iterations=1000, alpha=0.9, gamma=0.1)
    aichess.drunken_sailor(factor_epsilon=0.005, iterations=1000, alpha=0.9, gamma=0.1)
