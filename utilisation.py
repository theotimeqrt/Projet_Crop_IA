import torch
from torchvision import transforms
from torchvision.io import read_image
from Updated_ViT_GPU import ViTLoad

# Vérifiez si l'accélération GPU est disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation de l'appareil : {device}")

# Charger le modèle entraîné
model_path = './best_model.pth'
model = ViTLoad(model_path).to(device)
model.eval()  # Mettre en mode évaluation

# Les étapes de prétraitement de l’image sont les mêmes que celles utilisées pendant l'entraînement'.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Assurez-vous que l'image d'entrée comporte 3 canaux (ou développez-la si elle est en niveaux de gris)
def ensure_three_channels(img_tensor):
    if img_tensor.shape[0] == 1:  # S'il s'agit d'une image en niveaux de gris
        img_tensor = img_tensor.expand(3, -1, -1)  # Étendez-le à 3 canaux
    return img_tensor

# Fonction de comparaison d'images
def compare(image_1_path, image_2_path):
    # Chargement et prétraitement de la première image
    image_1 = read_image(image_1_path).float() / 255.0
    image_1 = ensure_three_channels(image_1)
    image_1 = transform(image_1).unsqueeze(0).to(device)

    # Chargement et prétraitement de la deuxième image
    image_2 = read_image(image_2_path).float() / 255.0
    image_2 = ensure_three_channels(image_2)
    image_2 = transform(image_2).unsqueeze(0).to(device)

    # Utilisez forward_two pour extraire les caractéristiques de deux images à la fois
    with torch.no_grad():
        feature_1, feature_2 = model.forward_two(image_1, image_2)

    # Calculer la similarité cosinus et la distance euclidienne
    cosine_similarity = torch.nn.functional.cosine_similarity(feature_1, feature_2).item()
    euclidean_distance = torch.norm(feature_1 - feature_2, p=2).item()

    # Similitude et distance d'impression pour le débogage
    #print(f"Cosine similarity: {cosine_similarity}, Euclidean distance: {euclidean_distance}")

    # Définition du seuil
    similarity_threshold = 0.9995  # Seuil de distance cosinus
    distance_threshold = 1.0  # Seuil de distance euclidienne

    # Combinaison de la similarité cosinus et du jugement de distance euclidienne
    if cosine_similarity > similarity_threshold and euclidean_distance < distance_threshold:
    #if cosine_similarity > similarity_threshold:
        return 1  # similaire
    else:
        return 0  # dissimilaire


# mais
if __name__ == "__main__":
    image_1_path = '../Data_set/images_sorties/check/image12447.jpg'
    image_2_path = '../Data_set/images_sorties/check/image12538.jpg'

    result = compare(image_1_path, image_2_path)
    print(f"比较结果 : {'相似' if result == 1 else '不相似'}")
