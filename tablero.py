from enum import Enum

import numpy as np


class Direccion(Enum):
    ARRIBA = 0
    ABAJO = 1
    IZQUIERDA = 2
    DERECHA = 3

class Tablero:
    def __init__(self, pos, num_reward):
        self.filas = 3
        self.columnas = 4
        self.tablero = [[' ' for _ in range(self.columnas)] for _ in range(self.filas)]
        self.tablero[1][1] = '-'
        self.posicion_actual = pos  # Empieza en la esquina inferior izquierda (posición 2, 0)
        self.tablero[self.posicion_actual[0]][self.posicion_actual[1]] = 'X'  # Marcar la posición actual con 'X'
        self.cell_values = [[]]
        self.final_states = []
        self.discount_factor = 0.90
        self.Q_matrix = None
        self.mean = 1
        if num_reward == 1:
            self.init_matrix_reward_1()
        if num_reward == 2:
            self.init_matrix_reward_2()

    def init_matrix_reward_1(self):
        num = -1
        goal_value = 100
        self.cell_values = [[num for _ in range(4)] for _ in range(3)]
        self.cell_values[0][3] = goal_value
        self.Q_matrix = np.random.uniform(low=-1, high=1, size=[3,4,4])

    def init_matrix_reward_2(self):
        num = -1
        goal_value = 100
        self.cell_values = [[-3, -2, -1, 100],[-4, 0, -2, -1],[-5, -4, -3, -2]]
        self.Q_matrix = np.random.uniform(low=-1, high=1, size=[3,4,4])

    def print_Q(self):
        for i in range(self.Q_matrix.shape[0]):
            print(f"Matriz {i + 1}:")
            print(self.Q_matrix[i])
            print()

    def init_final_states(self, state_list):
        self.final_states = state_list

    def is_final_state(self):
        if self.posicion_actual in self.final_states:
            return True
        return False

    def get_possible_next_movements(self):
        next_movements = []
        positions = []
        pos = self.posicion_actual
        if self.check_movement(Direccion.ARRIBA):
            next_movements.append(Direccion.ARRIBA.value)
            positions.append((pos[0] - 1, pos[1]))
        if self.check_movement(Direccion.ABAJO):
            next_movements.append(Direccion.ABAJO.value)
            positions.append((pos[0] + 1, pos[1]))
        if self.check_movement(Direccion.DERECHA):
            next_movements.append(Direccion.DERECHA.value)
            positions.append((pos[0], pos[1] + 1))
        if self.check_movement(Direccion.IZQUIERDA):
            next_movements.append(Direccion.IZQUIERDA.value)
            positions.append((pos[0], pos[1] - 1))
        return next_movements, positions
    def imprimir_tablero(self):
        print('------------------')
        for fila in self.tablero:
            print('|', end=' ')
            for casilla in fila:
                print(casilla, end=' | ')
            print()
        print('------------------')

    def move(self, direccion):

        if direccion == Direccion.ARRIBA.value:
            if self.posicion_actual[0] > 0 and (self.posicion_actual[0]-1, self.posicion_actual[1]) != (1, 1):
                self.tablero[self.posicion_actual[0]][self.posicion_actual[1]] = ' '  # Limpiar la posición actual
                self.posicion_actual = (self.posicion_actual[0] - 1, self.posicion_actual[1])  # Mover hacia arriba
                self.tablero[self.posicion_actual[0]][self.posicion_actual[1]] = 'X'  # Marcar nueva posición con 'X'
                return True
            else:
                print("¡No puedes moverte arriba")
                return False
        elif direccion == Direccion.ABAJO.value:
            if self.posicion_actual[0] < self.filas - 1 and (self.posicion_actual[0] + 1, self.posicion_actual[1]) != (1, 1):
                self.tablero[self.posicion_actual[0]][self.posicion_actual[1]] = ' '  # Limpiar la posición actual
                self.posicion_actual = (self.posicion_actual[0] + 1, self.posicion_actual[1])  # Mover hacia abajo
                self.tablero[self.posicion_actual[0]][self.posicion_actual[1]] = 'X'  # Marcar nueva posición con 'X'
                return True
            else:
                print("¡No puedes moverte abajo!")
                return False
        elif direccion == Direccion.IZQUIERDA.value:
            if self.posicion_actual[1] > 0 and (self.posicion_actual[0], self.posicion_actual[1] - 1) != (1, 1):
                self.tablero[self.posicion_actual[0]][self.posicion_actual[1]] = ' '  # Limpiar la posición actual
                self.posicion_actual = (self.posicion_actual[0], self.posicion_actual[1] - 1)  # Mover hacia la izquierda
                self.tablero[self.posicion_actual[0]][self.posicion_actual[1]] = 'X'  # Marcar nueva posición con 'X'
                return True
            else:
                print("¡No puedes moverte a la izquierda!")
                return False
        elif direccion == Direccion.DERECHA.value:
            if self.posicion_actual[1] < self.columnas - 1 and (self.posicion_actual[0], self.posicion_actual[1] + 1) != (1, 1):
                self.tablero[self.posicion_actual[0]][self.posicion_actual[1]] = ' '  # Limpiar la posición actual
                self.posicion_actual = (self.posicion_actual[0], self.posicion_actual[1] + 1)  # Mover hacia la derecha
                self.tablero[self.posicion_actual[0]][self.posicion_actual[1]] = 'X'  # Marcar nueva posición con 'X'
                return True
            else:
                print("¡No puedes moverte a la derecha!")
                return False

    def check_movement(self, direccion):
        if direccion == Direccion.ARRIBA:
            if self.posicion_actual[0] > 0 and (self.posicion_actual[0]-1, self.posicion_actual[1]) != (1, 1):
                return True
            else:
                return False
        elif direccion == Direccion.ABAJO:
            if self.posicion_actual[0] < self.filas - 1 and (self.posicion_actual[0] + 1, self.posicion_actual[1]) != (1, 1):
                return True
            else:
                return False
        elif direccion == Direccion.IZQUIERDA:
            if self.posicion_actual[1] > 0 and (self.posicion_actual[0], self.posicion_actual[1] - 1) != (1, 1):
                return True
            else:
                return False
        elif direccion == Direccion.DERECHA:
            if self.posicion_actual[1] < self.columnas - 1 and (self.posicion_actual[0], self.posicion_actual[1] + 1) != (1, 1):
                return True
            else:
                return False

    def get_actual_reward(self):
        return self.cell_values[self.posicion_actual[0]][self.posicion_actual[1]]






