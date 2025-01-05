###
import numpy as np
from torch.nn.functional import one_hot
from . eda.TSP import TSP_Instance, TSP_Environment, TSP_State
from . eda.solveTSP_v2 import solve
env = TSP_Environment


def distance(punto1, punto2):
    return np.sqrt((punto1[0] - punto2[0])**2 + (punto1[1] - punto2[1])**2)

# función para transformar un estado tsp en una secuencia de vectores
# para el modelo basado en capas de atención
def state2vecSeq(self):
    # creamos dos diccionarios para mantenre un mapeo de los
    # movimientos con los índices de la secuencia del modelo de aprendizaje

    city_locations = self.inst_info.city_locations

    idx2move = dict()
    move2idx = dict()
    origin = city_locations[self.visited[-1]]
    destination = city_locations[self.visited[0]]

    origin_dist = 0.0
    dest_dist = distance(origin, destination)

    seq = [list(origin) + [1,0] + [origin_dist, dest_dist], # Última ciudad visitada (origen)
           list(destination) + [0, 1] + [dest_dist, 0.0]]  # Ciudad final

    idx2move[0] = None
    idx2move[1] = ("constructive", self.visited[0])
    move2idx[self.visited[0]] = 1

    idx = 2
    for i in self.not_visited:
        point = list(city_locations[i])
        origin_dist = distance( point, origin)
        dest_dist = distance( point, destination)
        city_vector = point + [0, 0] + [origin_dist, dest_dist] # Otras ciudades

        seq.append(city_vector)
        idx2move[idx] = ("constructive", i)
        move2idx[i] = idx
        idx += 1

    return seq, idx2move, move2idx
