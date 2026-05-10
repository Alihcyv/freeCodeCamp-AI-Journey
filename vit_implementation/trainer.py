import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

from config import Config
from model import VisionTransformer
from data import get_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

torch.manual_seed(Config.SEED)
torch.cuda.manual_seed(Config.SEED)
random.seed(Config.SEED)

train_loader, test_loader = get_dataloaders(Config.BATCH_SIZE)

model = VisionTransformer(
    patch_size=Config.PATCH_SIZE, 
    image_size=Config.IMAGE_SIZE, 
    channels=Config.CHANNELS,
    num_classes=Config.NUM_CLASSES, 
    embed_dim=Config.EMBED_DIM, 
    depth=Config.DEPTH,
    num_heads=Config.NUM_HEADS, 
    mlp_dim=Config.MLP_DIM, 
    drop_rate=Config.DROP_RATE
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (y_pred.argmax(dim=1) == y).type(torch.float).sum().item()
    return total_loss / len(loader.dataset), 100 * correct / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item() * x.size(0)
            correct += (y_pred.argmax(dim=1) == y).type(torch.float).sum().item()
    return total_loss / len(loader.dataset), 100 * correct / len(loader.dataset)

train_accuracies, test_accuracies = [], []
for epoch in range(Config.EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"Epoch: {epoch+1}/{Config.EPOCHS} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    print("-" * 50)

plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Test Accuracy')
plt.show()
