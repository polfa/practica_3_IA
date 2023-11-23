import random
import time

from tablero import Tablero

def get_Q(tablero):
    a, movements = tablero.get_possible_next_movements()
    max_next_q = float('-inf')
    for move in movements:
        if tablero.Q_matrix[move[0]][move[1]] > max_next_q:
            max_next_q = tablero.Q_matrix[move[0]][move[1]]
    pos = tablero.posicion_actual
    alfa = 0.5
    tablero.Q_matrix[pos[0]][pos[1]] = tablero.Q_matrix[pos[0]][pos[1]] + alfa * (tablero.get_actual_reward() + tablero.discount_factor * max_next_q - tablero.Q_matrix[pos[0]][pos[1]])


def ex1_a():
    # Definim tauler
    tablero = Tablero((2, 0))
    # Establim el valor de les cel·les a -1 menys la objectiu a 100
    cell_values = tablero.init_cell_values_to(-1, goal_value=100)
    tablero.init_final_states([(0, 3)])
    # Imprimim l'estat inicial del tauler
    tablero.imprimir_tablero()
    solution = False
    score = 0
    while not solution:
        next_movements, a = tablero.get_possible_next_movements()
        movement = random.choice(next_movements)
        tablero.move(movement)
        position = tablero.posicion_actual
        score += cell_values[position[0]][position[1]]
        print("score: ", score)
        get_Q(tablero)
        print(tablero.Q_matrix)
        tablero.imprimir_tablero()
        time.sleep(1)

        if tablero.is_final_state():
            print("ha arribat al objectiu!")
            time.sleep(1)
            tablero.tablero[tablero.posicion_actual[0]][
                tablero.posicion_actual[1]] = ' '  # Marcar la posición actual con 'X'
            tablero.posicion_actual = (2,0)  # Empieza en la esquina inferior izquierda (posición 2, 0)
            tablero.tablero[tablero.posicion_actual[0]][tablero.posicion_actual[1]] = 'X'  # Marcar la posición actual con 'X'
            tablero.imprimir_tablero()







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ex1_a()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
