import torch
import torch.nn as nn
import torch.nn.functional as F

from . import TrainableModel
class TransformerBlockModel(TrainableModel):

  class CustomModel(nn.Module):
    class TransformerBlock(nn.Module):
        def __init__(self, input_dim, num_heads, head_dim, ff_dim, dropout_rate=0.2):
            super(TransformerBlockModel.CustomModel.TransformerBlock, self).__init__()
            self.attention = nn.MultiheadAttention(embed_dim=num_heads * head_dim, num_heads=num_heads, dropout=dropout_rate)
            self.norm1 = nn.LayerNorm(input_dim)  # LayerNorm based on input_dim
            self.ff = nn.Sequential(
                nn.Linear(input_dim, ff_dim),
                nn.ReLU(),
            )
            self.norm2 = nn.LayerNorm(input_dim)  # LayerNorm based on input_dim
            self.dropout = nn.Dropout(dropout_rate)
    
        def forward(self, x):
            # Attention block
            attn_output, _ = self.attention(x, x, x)  # x: (seq_len, batch_size, input_dim)
            x = self.norm1(x + self.dropout(attn_output))  # Residual + Norm
            
            # Feed-forward block
            ff_output = self.ff(x)
            x = self.norm2(x + self.dropout(ff_output))  # Residual + Norm
        
            return x

    def __init__(self, input_dim, num_heads, head_dim, dropout_rate=0.2):
        super(TransformerBlockModel.CustomModel, self).__init__()
        #self.seq_length = seq_length  # Asumiendo una longitud fija de secuencia para simplificar
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Proyección de entrada
        self.input_projection = nn.Linear(input_dim, num_heads * head_dim)
        
        self.attention_blocks = nn.ModuleList([
          TransformerBlockModel.CustomModel.TransformerBlock(input_dim=num_heads * head_dim,
                                        num_heads=num_heads,
                                        head_dim=head_dim,
                                        ff_dim=num_heads * head_dim,
                                        dropout_rate=dropout_rate)
        ])
    
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
      attn_output = x_proj
      x_proj = x_proj.permute(1, 0, 2)  # Reordenar para multihead_attention: [seq_length, batch_size, num_heads*head_dim]
    
    
      # Aplicar atención multi-cabeza
      attn_output = self.attention_blocks[0](x_proj)
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

  def load_model(self):
    self.model = TransformerBlockModel.CustomModel(input_dim=self.input_dim, num_heads=self.num_heads, head_dim=self.head_dim)
    self.model = self.model.to(self.device)
