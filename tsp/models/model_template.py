from .. eda.TSP import TSP_Instance, TSP_State, TSP_Environment
env = TSP_Environment
from .. eda.solveTSP_v2 import solve
import numpy as np
from torch.nn.functional import one_hot
## Todos los modelos serán entrenados con el mismo dataset
# X: [20000, 11, 6], Y: [20000, 11]
# donde X: (nb_sample, max_cities + 1, param_count), Y: (nb_sample, max_cities+1)
import torch
from tqdm import tqdm
from .. import state2vecSeq
class Model:
    # El modelo se genera en el constructor y se guarda en self.model
    def __init__(self, 
         input_dim = 6,
         num_heads = 10,
         head_dim = 64,
         city_count = 50,
                 
         batch_size = 512,
         train_split = 0.5,
         nb_samples = 20000,
         epochs = 10):

        self.city_count = city_count # Número de ciudades a evaluar
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## Parámetros modelo
        self.input_dim = input_dim  # Dimensión de la entrada
        self.num_heads = num_heads  # Número de cabezas en la atención multi-cabeza
        self.head_dim = head_dim  # Dimensión de cada cabeza
        ## Parámetros entrenamiento
        self.batch_size = batch_size
        self.train_split = train_split
        self.nb_samples = nb_samples
        self.epochs = 10

        
        self.model = None
    
    
    def load_model(self):
        raise NotImplementedError("La función 'load_model' debe ser declarada")
    def unload_model(self):
        raise NotImplementedError("La función 'unload_model' debe ser implementada");

    def train(self, x, y):
        raise NotImplementedError("La función 'train' debe ser declarada")
      

    def generate_data(self, use_progress_bar=False):
        X = []  # Lista para almacenar las secuencias de entrada
        Y = []  # Lista para almacenar las etiquetas objetivo (las siguientes ciudades a visitar)
        seq_len = self.city_count + 1  # Longitud de la secuencia, ajustada para incluir una ciudad extra
        
        # If the flag is set, initialize the progress bar
        pbar = tqdm(total=self.nb_samples, desc="Generating data", unit="sample", position=0, leave=True) if use_progress_bar else None
        
        # Bucle para generar datos hasta alcanzar el número deseado de muestras
        while True:
            # 1. Generamos instancia aleatoria
            n_cities = self.city_count
            dim = 2  # Dimensión para las coordenadas de la ciudad (2D: x, y)
            city_points = np.random.rand(n_cities, dim)  # Generar puntos aleatorios para las ciudades
            inst_info = TSP_Instance(city_points)
    
            # 2. Resolvemos TSP usando algoritmo tradicional
            solution = solve(city_points)  # Resolver el TSP y obtener un estado final
    
            # 3. Iteramos sobre los movimientos de la solución final para generar varias muestras:
            # estado (X) -> movimiento (Y)
            current_state = TSP_State(inst_info)
            env.state_transition(current_state, ("constructive", solution.visited[0]))
            samples_per_sol = self.city_count - 1  # Número máximo de muestras por solución
            
            for move in [("constructive", city) for city in solution.visited[1:]]:
                seq, _, move2idx = state2vecSeq(current_state)  # Convertir el estado actual a secuencia vectorizada
    
                X.append(torch.tensor(seq))  # Añadir la secuencia a X
                Y.append(one_hot(torch.tensor(move2idx[move[1]]), num_classes=seq_len))
                #Y.append(to_categorical(move2idx[move[1]], num_classes=seq_len))  # Añadir el movimiento como categoría a Y
    
                env.state_transition(current_state, move)  # Hacer la transición al siguiente estado
    
                # Actualizar el progreso de la barra si se está usando
                if use_progress_bar:
                    pbar.update(1)
    
                # Condiciones de parada basadas en el número de ciudades visitadas/no visitadas o muestras generadas
                if len(current_state.visited) > samples_per_sol or len(X) >= self.nb_samples:
                    break
    
            # Romper el bucle externo si se ha alcanzado el número deseado de muestras
            if len(X) >= self.nb_samples:
                break
        
        # Close the progress bar if it was used
        if use_progress_bar:
            pbar.close()
    
        X_padded = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        
        return X_padded, torch.stack(Y)


Model()