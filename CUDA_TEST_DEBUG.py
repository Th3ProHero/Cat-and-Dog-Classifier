import torch
print(torch.cuda.is_available())  # Debe imprimir True si CUDA está disponible
# Cargar los pesos del modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("cat_dog_classifier.pth", map_location="cpu")
print(checkpoint.keys())  # Verifica las claves del estado guardado
for key, value in checkpoint.items():
    print(key, value.shape)

state_dict = torch.load("cat_dog_classifier.pth", map_location=device)
print(state_dict.keys())  # Ver las claves en el archivo
