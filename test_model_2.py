from tkinter import Tk, filedialog
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Asegúrate de tener la misma clase del modelo
class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # 2 clases: perros y gatos
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Función para abrir el explorador de archivos y seleccionar una imagen
def select_image():
    root = Tk()
    root.withdraw()  # Oculta la ventana principal de Tkinter
    file_path = filedialog.askopenfilename(title="Selecciona una imagen", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    return file_path

# Cargar el modelo y el archivo .pth
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CatDogClassifier().to(device)
model.load_state_dict(torch.load("cat_dog_classifier3.pth"))
model.eval()

# Transformaciones necesarias para la entrada
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Abrir el explorador de archivos para seleccionar una imagen
img_path = select_image()
if img_path:
    # Cargar la imagen seleccionada
    img = Image.open(img_path)

    # Convertir la imagen a RGB (en caso de que tenga un canal alfa)
    img = img.convert("RGB")

    # Preprocesar la imagen
    img = transform(img).unsqueeze(0).to(device)  # Añadir dimensión de batch

    # Realizar predicción
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    # Mostrar la predicción
    class_names = ['gatos', 'perros']  # Asegúrate de que las clases estén en el mismo orden
    print(f"Predicción: {class_names[predicted.item()]}")
else:
    print("No se seleccionó ninguna imagen.")
