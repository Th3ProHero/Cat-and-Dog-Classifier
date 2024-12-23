from tkinter import Tk, filedialog
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# Transformaciones para preprocesar las imágenes
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Cambiar tamaño a 128x128
    transforms.ToTensor(),         # Convertir imágenes a tensores
    transforms.Normalize((0.5,), (0.5,))  # Normalizar imágenes
])

# Cargar los datasets
train_dataset = datasets.ImageFolder(r"C:\Classificator\dataset\train", transform=transform)
val_dataset = datasets.ImageFolder(r"C:\Classificator\dataset\val", transform=transform)

# Crear DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Clases
classes = train_dataset.classes  # ['gatos', 'perros']
class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 2 clases: perros y gatos
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CatDogClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Adelante
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Atrás
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")
# Guardar
torch.save(model.state_dict(), "cat_dog_classifier.pth")

# Cargar
model.load_state_dict(torch.load("cat_dog_classifier.pth"))
model.eval()
