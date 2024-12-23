import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


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
            nn.Linear(128, 2)  # 2 clases: gatos y perros
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Cargar modelo guardado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CatDogClassifier().to(device)
model.load_state_dict(torch.load("cat_dog_classifier.pth", map_location=device))
model.eval()

# Transformación para preprocesar imágenes
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define las clases
classes = ['gatos', 'perros']

# Función para predecir
def predict_image(image_path, model, transform, classes):
    image = Image.open(image_path)
    
    # Convertir la imagen a RGB si tiene un canal alfa (RGBA) o es monocromática (L)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]  # Convertir a probabilidades
        confidence, predicted = torch.max(probabilities, 0)

    return classes[predicted.item()], confidence.item() * 100

# Función para procesar todas las imágenes de una carpeta
def process_images_in_folder(folder_path, model, transform, classes):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No se encontraron imágenes en la carpeta especificada.")
        return

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        label, confidence = predict_image(image_path, model, transform, classes)
        
        # Mostrar la imagen con clasificación y porcentaje
        image = Image.open(image_path)
        plt.figure()
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"{label} ({confidence:.2f}%)")
        plt.show()

# Ruta de la carpeta `testing`
testing_folder = os.path.join(os.getcwd(), "testing")

# Ejecutar la función
process_images_in_folder(testing_folder, model, transform, classes)
