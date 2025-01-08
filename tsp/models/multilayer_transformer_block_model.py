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
                                            dropout_rate=dropout_rate) for _ in range(6)])
      def forward(self, x, seq_lengths=10, return_probabilities=False):
           x = x.float()

           # Projection
           x_proj = self.input_projection(x)
           # print(x_proj.shape)
           x_proj = x_proj.permute(1, 0, 2)  # [seq_length, batch_size, num_heads * head_dim]

           # Multi-head attention blocks
           for block in self.attention_blocks:
               x_proj = block(x_proj)
        
           x_proj = x_proj.permute(1, 0, 2)  # [batch_size, seq_length, num_heads * head_dim]
           # print(x_proj.shape)

           # Position-wise linear layer
           logits = self.positionwise_linear(x_proj)
           # print(logits.shape)

           # Flatten
           flat_output = logits.view(logits.size(0), -1)  # [batch_size, seq_length]

           if return_probabilities:
               return F.softmax(flat_output, dim=-1)
           return flat_output  # Return logits directly for CrossEntropyLoss

  def __init__(self):
    super().__init__()
  def load_model(self):
    self.model = self.CustomModel(input_dim=self.input_dim, num_heads=self.num_heads, head_dim=self.head_dim)
    self.model = self.model.to(self.device)
