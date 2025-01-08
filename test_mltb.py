from tsp.models import MultilayerTransformerBlockModel
import numpy as np
import torch
from eda.TSP import TSP_Instance, TSP_State,TSP_Environment
env=TSP_Environment
import pandas as pd
import gc
from eda.agents import GreedyAgent, SingleAgentSolver
from tsp.model_eval_actions import ModelEvalActions
from tqdm import tqdm

m = MultilayerTransformerBlockModel()
m.load_model()
m.model.load_state_dict(torch.load(r"C:\Users\krr\Downloads\TSP\checkpoint\MultilayerTransformerBlockModel-25-01-08--01-32-44.bin", weights_only=True))
m
dfb = pd.DataFrame(columns=["Model Name", "cities", "iter", "cost"])

model=m
nb_eval=100
instances = [
            TSP_Instance(np.random.rand(model.city_count, 2)) for _ in tqdm(
                range(nb_eval), desc="Instances", unit="instance", position=0, leave=True
            )
        ]
greedy = SingleAgentSolver(env, GreedyAgent(ModelEvalActions(m.model)))
solutions = []

for instance in tqdm(instances, desc="Solving Instances", unit="instance", position=0, leave=True):
    solution, *_ = greedy.solve(TSP_State(instance, visited=[0]))
    dfb = pd.concat([dfb, pd.DataFrame([{
        "ModelName" : type(model).__name__,
        "cities": model.city_count,
        "cost": solution.cost,                
}]) ])

dfb.to_csv("last.csv")
