import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import NPPModel
import os

def mape_loss(pred, target, eps=0.05):

    loss = torch.abs(pred - target) / (target + eps)
    return torch.mean(loss)

def train_npp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = NPPModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = mape_loss

    stages = [501, 101, 21, 1]
    epochs_per_stage = 20 

    for window in stages:
        path = f'baseline/smoothed_data_w{window}.pt'
        if not os.path.exists(path):
            continue
            
        print(f"\n--- Starting Stage: Window {window} ---")
        data = torch.load(path)
        
        dataset = TensorDataset(
            data['g_start'].view(-1, 1), 
            data['delta_g'].view(-1, 1), 
            data['t_pulse'].view(-1, 1)
        )
        loader = DataLoader(dataset, batch_size=1024, shuffle=True)

        for epoch in range(epochs_per_stage):
            model.train()
            total_loss = 0
            for gs, dg, tp in loader:
                gs, dg, tp = gs.to(device), dg.to(device), tp.to(device)
                
                optimizer.zero_grad()
                pred = model(gs, dg)
                loss = criterion(pred, tp)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs_per_stage} | MSE: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), 'baseline/npp_final.pth')
    print("\nTraining Complete. Model saved to baseline/npp_final.pth")

if __name__ == "__main__":
    train_npp()