import torch
import torchvision.transforms as T
from torchvision.models import vit_b_16
from PIL import Image

# Charger le modèle ViT pré-entraîné
model = vit_b_16(pretrained=True)
model.eval()

# Définir les transformations
transform = T.Compose([
    T.Resize((224, 224)),  # Redimensionner
    T.ToTensor(),  # Convertir en tenseur
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisation
])

def ensure_three_channels(img):
    # Si l'image est une image en niveaux de gris (canal unique), développez-la sur trois canaux
    if img.mode == 'L':  # « L » indique une image en niveaux de gris
        img = img.convert('RGB')
    return img

# Comparer deux images
def compare(image1_path, image2_path):
    # Chargement des images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Assurez-vous que l'image est à trois canaux
    image1 = ensure_three_channels(image1)
    image2 = ensure_three_channels(image2)

    # Conversion d'images
    input1 = transform(image1).unsqueeze(0)  # Ajout de la dimension de "batch"
    input2 = transform(image2).unsqueeze(0)

    # Extraire des fonctionnalités via le modèle
    with torch.no_grad():
        output1 = model(input1)
        output2 = model(input2)

    # Calculer la similarité cosinus
    similarity = torch.nn.functional.cosine_similarity(output1, output2).item()

    # Renvoie 1 (similaire) ou 0 (pas similaire)
    return 1 if similarity > 0.3 else 0

# Si le script est exécuté directement, effectuer une comparaison de test
if __name__ == "__main__":
    image1_path = "R.jpg"
    image2_path = "A_c.jpg"

    result = compare(image1_path, image2_path)
    print(f"Résultat de la comparaison : {'Similaire' if result == 1 else 'Non similaire'}")
