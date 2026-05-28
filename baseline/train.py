import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from model import NPPModel

class VCMDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'g_start': torch.tensor([item['g_start']], dtype=torch.float32),
            'g_target': torch.tensor([item['g_target']], dtype=torch.float32),
            'direction': torch.tensor([item['direction']], dtype=torch.float32),
            'g_history': item['g_history'].clone().detach().float()
        }

def collate_fn(batch):
    g_starts = torch.stack([item['g_start'] for item in batch])
    g_targets = torch.stack([item['g_target'] for item in batch])
    directions = torch.stack([item['direction'] for item in batch])
    
    histories = [item['g_history'].view(-1) for item in batch]
    max_len = max(h.shape[0] for h in histories)
    
    padded_histories = torch.zeros(len(batch), max_len)
    for i, h in enumerate(histories):
        padded_histories[i, :h.shape[0]] = h
        if h.shape[0] < max_len:
            padded_histories[i, h.shape[0]:] = h[-1]
            
    return g_starts, g_targets, directions, padded_histories

def mapping_loss(out, g_target, g_history, direction, dt=1e-8, kernel_size=1, t_scale=1e-6):
    t_pulse_raw = out * t_scale
    t_pulse_pred = torch.abs(t_pulse_raw)
    
    idx = t_pulse_pred / dt
    idx = torch.clamp(idx, 0, g_history.shape[1] - 1.001)
    
    if kernel_size > 1:
        pad = kernel_size // 2
        h_unsqueezed = g_history.unsqueeze(1)
        kernel = torch.ones(1, 1, kernel_size, device=g_history.device) / kernel_size
        h_padded = F.pad(h_unsqueezed, (pad, pad), mode='replicate')
        g_hist_smooth = F.conv1d(h_padded, kernel).squeeze(1)
    else:
        g_hist_smooth = g_history
        
    idx_floor = torch.floor(idx).long()
    idx_ceil = idx_floor + 1
    
    batch_indices = torch.arange(g_history.shape[0], device=g_history.device).unsqueeze(1)
    
    g_floor = g_hist_smooth[batch_indices, idx_floor]
    g_ceil = g_hist_smooth[batch_indices, idx_ceil]
    
    weight = idx - idx_floor
    g_pred = g_floor + weight * (g_ceil - g_floor)
    
    loss_mse = F.mse_loss(g_pred, g_target)
    
    loss_sign = torch.mean(torch.relu(-out * direction)) * 0.1
    
    return loss_mse + loss_sign

def train_npp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(VCMDataset('data/train.pt'), batch_size=1024, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(VCMDataset('data/val.pt'), batch_size=1024, shuffle=False, collate_fn=collate_fn)
    
    model = NPPModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    schedule = [(20, 1001), (20, 101), (20, 11), (20, 1)]
    dt = 1e-8
        
    current_epoch = 0
    for epochs, kernel_size in schedule:
        print(f"\n--- Phase: kernel_size={kernel_size} ---")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for g_starts, g_targets, directions, g_histories in train_loader:
                g_starts = g_starts.to(device)
                g_targets = g_targets.to(device)
                directions = directions.to(device)
                g_histories = g_histories.to(device)
                
                optimizer.zero_grad()
                out = model(g_starts, g_targets)
                
                out.register_hook(lambda grad: torch.clamp(grad, min=-0.1, max=0.1))
                
                loss = mapping_loss(out, g_targets, g_histories, directions, dt, kernel_size)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for g_starts, g_targets, directions, g_histories in val_loader:
                    g_starts = g_starts.to(device)
                    g_targets = g_targets.to(device)
                    directions = directions.to(device)
                    g_histories = g_histories.to(device)
                    
                    out = model(g_starts, g_targets)
                    loss = mapping_loss(out, g_targets, g_histories, directions, dt, kernel_size)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            current_epoch += 1
            if current_epoch % 5 == 0:
                print(f"Epoch {current_epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
                
    torch.save(model.state_dict(), 'npp_final.pth')
    print("\nModel saved: npp_final.pth")

if __name__ == "__main__":
    train_npp()