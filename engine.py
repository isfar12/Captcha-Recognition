from tqdm import tqdm
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_fn(model, data_loader,optimizer, device):
    model.train()
    total_loss = 0
    for data in tqdm(data_loader,total=len(data_loader)):
        for key, value in data.items():
            data[key] = value.to(device)
        _,loss = model(**data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(data_loader)
    
def eval_fn(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for key, value in data.items():
                data[key] = value.to(device)
            outputs, loss = model(**data)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(data['targets'].cpu().numpy())
            total_loss += loss.item()
    return predictions, true_labels, total_loss / len(data_loader)      