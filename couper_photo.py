import os
from random import randint
from pathlib import Path
import torchvision.transforms as T
from torchvision.io import read_image, write_jpeg
import shutil
from PIL import Image
import pandas as pd

# Chemins des dossiers
folder_path = './Data_set/images_originales'  # Dossier contenant les images originales
output_folder = './Data_set/images_sorties'  # Dossier pour enregistrer les images rognées
output_folder_entrainement = './Data_set/images_sorties/train'  # Dossier pour enregistrer les images rognées pour entraînement
output_folder_validation = './Data_set/images_sorties/check'  # Dossier pour enregistrer les images rognées pour validation
output_folder_test = './Data_set/images_sorties/test'  # Dossier pour enregistrer les images rognées pour test
output_log_path_entrainement = './Data_set/text_infos/logs_output_train.csv'
output_log_path_validation = './Data_set/text_infos/logs_output_check.csv'
output_log_path_test = './Data_set/text_infos/logs_output_test.csv'

# Vider le dossier contenant les images coupées pour entraînement si nécessaire
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

# Créer les dossiers nécessaires s'ils n'existent pas
Path(output_folder_entrainement).mkdir(parents=True, exist_ok=True)
Path(output_folder_validation).mkdir(parents=True, exist_ok=True)
Path(output_folder_test).mkdir(parents=True, exist_ok=True)

# Liste pour enregistrer les données de sortie pour CSV
log_data_entrainement = []
log_data_validation = []
log_data_test = []

# Obtenir tous les fichiers .jpg dans le dossier
image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])

# Si aucune image n'est trouvée, arrête le programme
if len(image_files) == 0:
    print("Aucune image trouvée dans le dossier.")
else:
    print(f"Nombre d'images à traiter : {len(image_files)}")
    # Obtenir le nom de la dernière image (sans extension) pour la première image
    last_image_name, _ = os.path.splitext(image_files[-1])
    new_filename_avant = f"{last_image_name}_cropped.jpg"  # Utilisé pour la première image

    # Parcours de chaque image
    for idx, filename in enumerate(image_files):
        # Extraire le nom de l'image sans l'extension
        original_name, ext = os.path.splitext(filename)

        # Chemin complet de l'image
        img_path = Path(folder_path) / filename

        # Charger l'image
        img = read_image(str(img_path))

        img_PIL = Image.open(img_path)
        image_tensor = T.functional.to_tensor(img_PIL)

        # Obtenir les dimensions de l'image (hauteur et largeur)
        _, height, width = image_tensor.shape

        # Définir une taille de rognage aléatoire
        random_size_w = int(0.1 * width + randint(0, int(0.8 * width)))
        random_size_h = int(0.1 * height + randint(0, int(0.8 * height)))
        crop_size = (random_size_h, random_size_w)  # Taille de rognage
        transform = T.RandomCrop(size=crop_size)

        # Appliquer la transformation de rognage
        cropped_img = transform(img)

        # Créer un nom de fichier pour l'image rognée（Enregistrer le suffixe）
        new_filename = f"{original_name}_cropped.jpg"

        # Définir les limites pour les ensembles d'entraînement, de validation et de test
        train_limit = int(0.7 * len(image_files))
        val_limit = int(0.9 * len(image_files))

        if idx < train_limit:
            new_img_path = Path(output_folder_entrainement) / new_filename

            # Enregistrer l'image rognée
            write_jpeg(cropped_img, str(new_img_path))

            # Ajouter une ligne au journal pour l'image rognée
            log_data_entrainement.append([original_name, os.path.splitext(new_filename)[0], 1])

            # Ajouter une deuxième ligne pour l'image précédente
            log_data_entrainement.append([original_name, os.path.splitext(new_filename_avant)[0], 0])

            # Mettre à jour le nom de la dernière image rognée
            new_filename_avant = new_filename

        elif idx < val_limit:
            new_img_path = Path(output_folder_validation) / new_filename

            # Enregistrer l'image rognée
            write_jpeg(cropped_img, str(new_img_path))

            # Ajouter une ligne au journal pour l'image rognée
            log_data_validation.append([original_name, os.path.splitext(new_filename)[0], 1])

            # Ajouter une deuxième ligne pour l'image précédente
            log_data_validation.append([original_name, os.path.splitext(new_filename_avant)[0], 0])

            # Mettre à jour le nom de la dernière image rognée
            new_filename_avant = new_filename

        else:
            new_img_path = Path(output_folder_test) / new_filename

            # Enregistrer l'image rognée
            write_jpeg(cropped_img, str(new_img_path))

            # Ajouter une ligne au journal pour l'image rognée
            log_data_test.append([original_name, os.path.splitext(new_filename)[0], 1])

            # Ajouter une deuxième ligne pour l'image précédente
            log_data_test.append([original_name, os.path.splitext(new_filename_avant)[0], 0])

            # Mettre à jour le nom de la dernière image rognée
            new_filename_avant = new_filename

# Enregistrer les données dans des fichiers CSV
log_df_entrainement = pd.DataFrame(log_data_entrainement, columns=['Nom original', 'Nom rogné', 'Utilisé l\'image précédente'])
log_df_entrainement.to_csv(output_log_path_entrainement, index=False)

log_df_validation = pd.DataFrame(log_data_validation, columns=['Nom original', 'Nom rogné', 'Utilisé l\'image précédente'])
log_df_validation.to_csv(output_log_path_validation, index=False)

log_df_test = pd.DataFrame(log_data_test, columns=['Nom original', 'Nom rogné', 'Utilisé l\'image précédente'])
log_df_test.to_csv(output_log_path_test, index=False)

# Copier toutes les images originales dans les dossiers de sortie
for idx, file in enumerate(image_files):
    src_file = os.path.join(folder_path, file)

    if idx < train_limit:
        dest_file = os.path.join(output_folder_entrainement, file)
    elif idx < val_limit:
        dest_file = os.path.join(output_folder_validation, file)
    else:
        dest_file = os.path.join(output_folder_test, file)

    shutil.copy2(src_file, dest_file)

    print(f"Image {file} copiée dans le dossier de sortie.")
