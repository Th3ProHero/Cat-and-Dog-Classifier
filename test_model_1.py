import torch
import torch.nn as nn
from torchvision import transforms
from tkinter import Tk, filedialog
from PIL import Image

# Definir el modelo
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
model.load_state_dict(torch.load("cat_dog_classifier.pth", map_location=device, weights_only=True))
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

# Función para seleccionar archivo
def select_and_predict(model, transform, classes):
    root = Tk()
    root.withdraw()  # Oculta la ventana principal de Tkinter
    file_path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png")]
    )

    if file_path:
        label, confidence = predict_image(file_path, model, transform, classes)
        print(f"La imagen seleccionada es clasificada como: {label} ({confidence:.2f}% de confianza)")
    else:
        print("No se seleccionó ninguna imagen.")

# Ejecutar la función
select_and_predict(model, transform, classes)
