
import time

import numpy as np
import random

from tablero import Tablero

all_movements = [0, 1, 2 ,3]
def get_Q(tablero, movement, position, alpha, gamma):
    movements = tablero.get_possible_next_movements()
    max_next_q = float('-inf')
    next_pos = tablero.posicion_actual
    reward = tablero.get_actual_reward()
    q_table = tablero.Q_matrix
    value = q_table[position][movement] + alpha * (reward + gamma * np.max(q_table[next_pos]) - q_table[position][movement])
    q_table[position][movement] = value
    tablero.Q_matrix = q_table

def Q_learning(tablero, factor_epsilon, iterations, alpha, gamma):
    # Establim el valor de les cel·les a -1 menys la objectiu a 100
    cell_values = tablero.cell_values
    tablero.init_final_states([(0, 3)])
    # Imprimim l'estat inicial del tauler
    tablero.imprimir_tablero()
    solution = False
    score = 0
    count = 0
    epsilon = 1
    while count < iterations:
        next_movements, a = tablero.get_possible_next_movements()
        position = tablero.posicion_actual
        random_move = False
        movement = None
        random_value = random.random()
        if random_value < 1 - epsilon:
            movement = np.argmax(tablero.Q_matrix[position])
        else:
            movement = random.choice(all_movements)
            random_move = True
        tablero.move(movement)
        tablero.print_Q()
        tablero.imprimir_tablero()
        print(random_move)
        # print(tablero.Q_matrix)

        get_Q(tablero, movement, position, alpha, gamma)
        position = tablero.posicion_actual
        score += cell_values[position[0]][position[1]]
        # time.sleep(0.2)

        if tablero.is_final_state():
            if epsilon > 0:
                epsilon -= factor_epsilon
            count += 1
            print("ha arribat al objectiu!")
            print("score:", score, count)
            time.sleep(0.3)
            tablero.tablero[tablero.posicion_actual[0]][
                tablero.posicion_actual[1]] = ' '  # Marcar la posición actual con 'X'
            tablero.posicion_actual = (2, 0)  # Empieza en la esquina inferior izquierda (posición 2, 0)
            tablero.tablero[tablero.posicion_actual[0]][
                tablero.posicion_actual[1]] = 'X'  # Marcar la posición actual con 'X'
            tablero.imprimir_tablero()
            score = 0
            time.sleep(0.5)

def drunken_sailor(tablero, factor_epsilon, iterations, alpha, gamma):
    # Establim el valor de les cel·les a -1 menys la objectiu a 100
    cell_values = tablero.cell_values
    tablero.init_final_states([(0, 3)])
    # Imprimim l'estat inicial del tauler
    tablero.imprimir_tablero()
    score = 0
    count = 0
    epsilon = 1
    while count < iterations:
        position = tablero.posicion_actual
        random_move = False
        random_value = random.random()
        if random_value < 1 - epsilon:
            if random.random() < 0.99:
                movement = np.argmax(tablero.Q_matrix[position])
            else:
                movement = random.choice(all_movements)
        else:
            movement = random.choice(all_movements)
            random_move = True
        tablero.move(movement)
        tablero.print_Q()
        tablero.imprimir_tablero()
        print(random_move)
        # print(tablero.Q_matrix)

        get_Q(tablero, movement, position, alpha, gamma)
        position = tablero.posicion_actual
        score += cell_values[position[0]][position[1]]
        # time.sleep(0.2)

        if tablero.is_final_state():
            if epsilon > 0:
                epsilon -= factor_epsilon
            count += 1
            print("ha arribat al objectiu!")
            print("score:", score, count)
            time.sleep(0.3)
            tablero.tablero[tablero.posicion_actual[0]][
                tablero.posicion_actual[1]] = ' '  # Marcar la posición actual con 'X'
            tablero.posicion_actual = (2, 0)  # Empieza en la esquina inferior izquierda (posición 2, 0)
            tablero.tablero[tablero.posicion_actual[0]][
                tablero.posicion_actual[1]] = 'X'  # Marcar la posición actual con 'X'
            tablero.imprimir_tablero()
            score = 0
            time.sleep(0.5)

def ex1_a():
    # Definim tauler per ex1_a
    Q_learning(Tablero((2, 0), 1),factor_epsilon=0.11,iterations=16, alpha=0.3,gamma=0.8)

def ex1_b():
    # Definim tauler per ex1_b
    Q_learning(Tablero((2, 0), 2),factor_epsilon=0.15,iterations=16, alpha=0.7,gamma=0.3)

def ex1_c():
    drunken_sailor(Tablero((2, 0), 2),factor_epsilon=0.15,iterations=16, alpha=0.7,gamma=0.3)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ex1_b()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
