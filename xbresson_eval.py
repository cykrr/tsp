import numpy as np
import torch
from tsp.models.xbresson_model import TSP_net, compute_tour_length
from eda.agents import GreedyAgent, SingleAgentSolver

device = torch.device("cuda")


trained_model_data = torch.load("./checkpoint/checkpoint_25-01-05--16-58-30-n50-gpu0.pkl", 
                                map_location=device, weights_only=True)
trained_model_state_dict = trained_model_data['model_train']
model = TSP_net(2, 128, 512, 6, 2, 8, 1000)
model.load_state_dict(trained_model_state_dict)

# instances = [
#     TSP_Instance(
#         np.random.rand(50, 2)
#     ) for _ in tqdm(
#             range(50),
#             desc='Instances',
#             unit="instance",
#             position=0
#         )
# ]
instances = torch.tensor(np.random.rand(50, 50, 2)).float()

solutions = model(instances)[0]
costs = compute_tour_length(instances, solutions)
print(costs)

mean_cost = torch.mean(costs)
print(mean_cost)
