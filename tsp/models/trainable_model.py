## Definición base de modelo para usar el mismo entrenamiento en los distintos modelos
from .model_template import Model
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import pandas as pd
import gc
from .. eda.TSP import TSP_Instance, TSP_State
from .. eda.agents import GreedyAgent, SingleAgentSolver
from tsp.model_eval_actions import ModelEvalActions
from .. eda.TSP import TSP_Environment
import numpy as np
env = TSP_Environment
class TrainableModel(Model):
    def __init__(self):
        super().__init__()
        # Parámetros del modelo
    def unload_model(self):
        del self.model
        torch.cuda.empty_cache()
        gc.collect()

    def ugly_train(self, xt, yt, xv, yv, num_iter=-1, use_progress_bar=False):
        # self.load_model()
        # Asumiendo que X_padded y Y_stacked ya están definidos y son tensores de PyTorch
        trd = TensorDataset(xt, yt)
        ted = TensorDataset(xv, yv)
    
        # # Dividir el dataset en entrenamiento y prueba
        # train_size = int(self.train_split * len(dataset))
        # test_size = len(dataset) - train_size
        train_dataset, test_dataset = trd, ted
    
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
    
        # Definir el modelo, la función de pérdida y el optimizador
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-4)
    
        # Initialize the DataFrame to store training results
        df = pd.DataFrame(columns=["Model Name", "cities", "iter", "Epoch",
                                   "Training Loss", "Training Accuracy",
                                   "Validation Loss", "Validation Accuracy"])
        dfb = pd.DataFrame(columns=["Model Name", "cities", "iter", "Epoch", "cost"])
    
        # Initialize the progress bar for epochs if required
        epoch_range = range(self.epochs)
        if use_progress_bar:
            epoch_range = tqdm(epoch_range, desc="Training Epochs", unit="epoch", position = 0, leave = True)
        
        print("Entrenando modelo...")
        for epoch in epoch_range:
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()  # Limpia los gradientes
                outputs = self.model(X_batch)  # Obtenemos logits
                loss = loss_function(outputs, y_batch.argmax(dim=1))  # Calcular la pérdida
                loss.backward()  # Backward pass
                optimizer.step()  # Actualizar parámetros
                train_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch.argmax(dim=1)).sum().item()
    
            train_loss /= len(train_loader.dataset)
            train_accuracy = 100 * correct / total
    
            # Validación
            self.model.eval()
            validation_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = self.model(X_batch)
                    loss = loss_function(outputs, y_batch.argmax(dim=1))
                    validation_loss += loss.item() * X_batch.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch.argmax(dim=1)).sum().item()
    
            validation_loss /= len(test_loader.dataset)
            validation_accuracy = 100 * correct / total
    
            # Log results to DataFrame
            df = pd.concat([df, pd.DataFrame([{
                "Model Name": type(self).__name__,
                "cities": self.city_count,
                "iter": num_iter,
                "Epoch": epoch + 1,
                "Training Loss": train_loss,
                "Training Accuracy": train_accuracy,
                "Validation Loss": validation_loss,
                "Validation Accuracy": validation_accuracy
            }])], ignore_index=True)
    
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
            print(f'Epoch {epoch+1}, Val Loss: {validation_loss:.4f}, Val Accuracy: {validation_accuracy:.2f}%')
            nb_eval=50
            instances = [
                TSP_Instance(np.random.rand(self.city_count, 2)) for _ in tqdm(
                    range(nb_eval), desc="Instances", unit="instance", position=0, leave=True
                )
            ]
            greedy = SingleAgentSolver(env, GreedyAgent(ModelEvalActions(self.model)))
            solutions = []

            for instance in tqdm(instances, desc="Solving Instances", unit="instance", position=0, leave=True):
                solution, *_ = greedy.solve(TSP_State(instance, visited=[0]))
                dfb = pd.concat([dfb, pd.DataFrame([{
                    "ModelName" : type(self).__name__,
                    "cities": self.city_count,
                    "Epoch": epoch,
                    "cost": solution.cost,                
            }]) ])
    
        # If tqdm was used, close the progress bar
        if use_progress_bar:
            epoch_range.close()
    
        # self.unload_model()
        return df, dfb;


    def train(self, xt, yt, xv, yv, num_iter=-1, use_progress_bar=False):
        # self.load_model()
        # Asumiendo que X_padded y Y_stacked ya están definidos y son tensores de PyTorch
        trd = TensorDataset(xt, yt)
        ted = TensorDataset(xv, yv)
    
        # # Dividir el dataset en entrenamiento y prueba
        # train_size = int(self.train_split * len(dataset))
        # test_size = len(dataset) - train_size
        train_dataset, test_dataset = trd, ted
    
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
    
        # Definir el modelo, la función de pérdida y el optimizador
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
    
        # Initialize the DataFrame to store training results
        df = pd.DataFrame(columns=["Model Name", "cities", "iter", "Epoch",
                                   "Training Loss", "Training Accuracy",
                                   "Validation Loss", "Validation Accuracy"])
    
        # Initialize the progress bar for epochs if required
        epoch_range = range(self.epochs)
        if use_progress_bar:
            epoch_range = tqdm(epoch_range, desc="Training Epochs", unit="epoch", position = 0, leave = True)
        
        print("Entrenando modelo...")
        for epoch in epoch_range:
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()  # Limpia los gradientes
                outputs = self.model(X_batch)  # Obtenemos logits
                loss = loss_function(outputs, y_batch.argmax(dim=1))  # Calcular la pérdida
                loss.backward()  # Backward pass
                optimizer.step()  # Actualizar parámetros
                train_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch.argmax(dim=1)).sum().item()
    
            train_loss /= len(train_loader.dataset)
            train_accuracy = 100 * correct / total
    
            # Validación
            self.model.eval()
            validation_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = self.model(X_batch)
                    loss = loss_function(outputs, y_batch.argmax(dim=1))
                    validation_loss += loss.item() * X_batch.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch.argmax(dim=1)).sum().item()
    
            validation_loss /= len(test_loader.dataset)
            validation_accuracy = 100 * correct / total
    
            # Log results to DataFrame
            df = pd.concat([df, pd.DataFrame([{
                "Model Name": type(self).__name__,
                "cities": self.city_count,
                "iter": num_iter,
                "Epoch": epoch + 1,
                "Training Loss": train_loss,
                "Training Accuracy": train_accuracy,
                "Validation Loss": validation_loss,
                "Validation Accuracy": validation_accuracy
            }])], ignore_index=True)
            if not use_progress_bar: 
                print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
                print(f'Epoch {epoch+1}, Val Loss: {validation_loss:.4f}, Val Accuracy: {validation_accuracy:.2f}%')
    
        # If tqdm was used, close the progress bar
        if use_progress_bar:
            epoch_range.close()
    
        # self.unload_model()
        return df;
