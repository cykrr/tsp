from . import TrainableModel

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(TrainableModel):
    def load_model(self):
        self.model = BaseModel.CustomModel(self.input_dim, self.num_heads, self.head_dim)
        
    class CustomModel(nn.Module):
        def __init__(self, input_dim, num_heads, head_dim, dropout_rate=0.2):
            super(BaseModel.CustomModel, self).__init__()
            #self.seq_length = seq_length  # Asumiendo una longitud fija de secuencia para simplificar
            self.num_heads = num_heads
            self.head_dim = head_dim

            # Proyección de entrada
            self.input_projection = nn.Linear(input_dim, num_heads * head_dim)

            # Capa de atención multi-cabeza
            self.multihead_attention = nn.MultiheadAttention(embed_dim=num_heads * head_dim,
                                                            num_heads=num_heads,
                                                            dropout=dropout_rate)

            # Capas lineales individuales para cada posición de la secuencia
            # Esto es un cambio respecto al código original para aplicar una capa lineal por posición de salida
            self.positionwise_linear = nn.Linear(num_heads * head_dim, 1)

            # Capa de salida final, después de un flatten, para aplicar Softmax
            # Nota: Softmax se aplica después del flatten, por lo tanto no se define aquí como una capa pero sí en el forward

        def generate_attention_mask(self, x, padding_value=0):
            # Identificar posiciones de padding en x
            mask = (x.sum(dim=-1) == padding_value)  # Asumiendo que el padding se puede identificar sumando los valores de la característica y comparando con 0
            mask = mask.to(dtype=torch.bool)  # Convierte a bool para usar como máscara
            # PyTorch espera una máscara con True y False donde True indica donde aplicar la máscara
            return mask


        def forward(self, x, seq_lengths=10, return_probabilities=False):
            # x: [batch_size, seq_length, input_dim]
            x = x.float()

            max_len = x.shape[1]

            # Generar máscara de atención basada en las longitudes de las secuencias
            attn_mask = None

            # Aplicar proyección de entrada
            x_proj = self.input_projection(x)
            x_proj = x_proj.permute(1, 0, 2)  # Reordenar para multihead_attention: [seq_length, batch_size, num_heads*head_dim]


            # Aplicar atención multi-cabeza
            attn_output, _ = self.multihead_attention(x_proj, x_proj, x_proj, attn_mask=attn_mask)
            attn_output = attn_output.permute(1, 0, 2)  # Reordenar de vuelta: [batch_size, seq_length, num_heads*head_dim]

            # Aplicar la capa lineal posición por posición
            # Usamos una capa lineal que se aplica a cada vector de salida de la atención de forma independiente
            positionwise_output = self.positionwise_linear(attn_output)

            # Flatten
            flat_output = positionwise_output.view(positionwise_output.size(0), -1)  # [batch_size, seq_length]

            # Softmax
            if return_probabilities:
                output = F.softmax(flat_output, dim=-1)
                return output
            else: #return logits
                return flat_output


    
    def __init__(self):
        super().__init__()
    def load_model(self):
        self.model = BaseModel.CustomModel(input_dim=self.input_dim, num_heads=self.num_heads, head_dim=self.head_dim)
        self.model.to(self.device)

