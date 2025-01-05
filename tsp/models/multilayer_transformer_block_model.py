from . import TrainableModel, TransformerBlockModel

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultilayerTransformerBlockModel(TrainableModel):
  class CustomModel(TransformerBlockModel.CustomModel):
      def __init__(self, input_dim, num_heads, head_dim, dropout_rate=0.2):
          super().__init__(input_dim, num_heads, head_dim, dropout_rate)
          self.attention_blocks = nn.ModuleList([TransformerBlockModel.CustomModel.TransformerBlock(input_dim=num_heads * head_dim,
                                            num_heads=num_heads,
                                            head_dim=head_dim,
                                            ff_dim=num_heads * head_dim,
                                            dropout_rate=dropout_rate)]*6)
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
          for i in range(len(self.attention_blocks)):
              attn_output = self.attention_blocks[i](x_proj)
          
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
    self.model = self.CustomModel(input_dim=self.input_dim, num_heads=self.num_heads, head_dim=self.head_dim)
    self.model = self.model.to(self.device)