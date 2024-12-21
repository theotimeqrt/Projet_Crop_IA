import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
import os
import pandas as pd
from torchvision.io import read_image
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import csv
import time
from torchvision.transforms.functional import to_pil_image

# Vérifier si CUDA est disponible pour accélérer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation de l'appareil : {device}")

# Définir le chemin et les fichiers de la base de données
annotations_file_train = r"./Data_set/text_infos/logs_output_train.csv"
annotations_file_check = r"./Data_set/text_infos/logs_output_check.csv"
img_dir_train = r"./Data_set/images_sorties/train/"
img_dir_valide = r"./Data_set/images_sorties/check/"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),  # Ajouter une rotation aléatoire
    transforms.RandomCrop(size=(224, 224), padding=4),  # Recadrage aléatoire
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Traduction aléatoire
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
])


# ______________ Classe du Dataset __________________
class Data_set(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None, skiprows=1)
        self.img_dir = img_dir
        self.transform = transform
        # Préchargez toutes les images en mémoire pour l'exécution plus rapide
        self.images = {
            os.path.splitext(img)[0]: to_pil_image(read_image(os.path.join(img_dir, img)))
            for img in os.listdir(img_dir) if img.endswith('.jpg')
        }

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name_1 = self.img_labels.iloc[idx, 0]
        img_name_2 = self.img_labels.iloc[idx, 1]
        label = int(self.img_labels.iloc[idx, 2])

        image_1 = self.images[img_name_1]
        image_2 = self.images[img_name_2]

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return (image_1, image_2), label

    
# ______________ Classe du modèle ViT __________________

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = vit_b_16(weights=weights)
        self.vit.heads = nn.Identity()

    def forward_one(self, x):
        return self.vit(x)

    def forward_two(self, x1, x2):
        x_one = self.forward_one(x1)
        x_two = self.forward_one(x2)
        return x_one, x_two

    def forward(self, x):
        return self.forward_one(x)
    

# ______________ Classe du modèle ViT avec chargement des poids __________________
class ViTLoad(ViT):
    def __init__(self, path):
        super(ViTLoad, self).__init__()
        # Charger les poids depuis un fichier .pth
        self.load_state_dict(torch.load(path))
        
# ______________ Perte de contraste avec Cosine Similarity __________________
class ContrastiveLossCosine(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLossCosine, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cosine_distance = 1 - F.cosine_similarity(output1, output2, dim=1)
        pos = (1 - label) * torch.pow(cosine_distance, 2)
        neg = label * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2)
        loss_contrastive = torch.mean(pos + neg)
        return loss_contrastive

# __________________ Fonction pour sauvegarder le modèle ____________________

#def save_model(model, optimizer, epoch, file_path):
#    torch.save({
#        'epoch': epoch,
#        'model_state_dict': model.state_dict(),
#        'optimizer_state_dict': optimizer.state_dict()
#    }, file_path)
#    print(f"Modèle sauvegardé dans : {file_path}")

# Enregistrer uniquement les poids
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)  # Enregistrer uniquement les poids du modèle
    print(f"Modèle sauvegardé dans : {file_path}")


# Définir des stratégies de planification du taux d'apprentissage d'échauffement et de décroissance
def lr_lambda(epoch):
    if epoch < 5:  # Les 5 premières époques sont utilisées pour l'échauffement du taux d'apprentissage
        return epoch / 5
    else:  # Décroissance exponentielle ultérieure
        return 0.95 ** (epoch - 5)
    
# ______________ Classe de la boucle d'entraînement __________________
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, loss_fn, device, epochs=5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        self.writer = SummaryWriter(f'runs/contrastive_trainer_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.best_vloss = float('inf')
        self.epoch_number = 0
        self.scaler = torch.cuda.amp.GradScaler()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2, factor=0.1)


        self.csv_file = 'train_validation_losses.csv'
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Validation_Loss'])

    # Modifier la fonction train_one_epoch pour prendre en charge l'entraînement de précision mixte
    def train_one_epoch(self, epoch_index):
        running_loss = 0.0
        self.model.train()
    
        for i, ((image_1, image_2), labels) in enumerate(self.train_loader):
            image_1, image_2, labels = image_1.to(self.device), image_2.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
    
            # Mode de précision mixte
            with torch.cuda.amp.autocast():  # Allumer AMP
                output_1, output_2 = self.model.forward_two(image_1, image_2)
                loss = self.loss_fn(output_1, output_2, labels)
    
            self.scaler.scale(loss).backward()  # Rétropropagation à l'aide de GradScaler
            self.scaler.step(self.optimizer)  # Mise à jour de l'optimiseur
            self.scaler.update()  # Mettre à jour le scaler
    
            running_loss += loss.item()
            if i % 1000 == 999:
                avg_loss = running_loss / 1000
                print(f'Batch {i + 1}, perte moyenne : {avg_loss:.4f}')
                self.writer.add_scalar('Perte/Entraînement', avg_loss, epoch_index * len(self.train_loader) + i + 1)
                running_loss = 0.0
    
        return running_loss / len(self.train_loader)


    def validate(self):
        running_vloss = 0.0
        self.model.eval()

        with torch.no_grad():
            for vinputs, vlabels in self.val_loader:
                image_1, image_2 = vinputs
                image_1, image_2, vlabels = image_1.to(self.device), image_2.to(self.device), vlabels.to(self.device)
                output_1, output_2 = self.model(image_1), self.model(image_2)
                vloss = self.loss_fn(output_1, output_2, vlabels)
                running_vloss += vloss.item()

        return running_vloss / len(self.val_loader)

    def test(self):
        print("\nÉvaluation sur le jeu de test :")
        running_test_loss = 0.0
        self.model.eval()

        with torch.no_grad():
            for tinputs, tlabels in self.test_loader:
                image_1, image_2 = tinputs
                image_1, image_2, tlabels = image_1.to(self.device), image_2.to(self.device), tlabels.to(self.device)
                output_1, output_2 = self.model(image_1), self.model(image_2)
                tloss = self.loss_fn(output_1, output_2, tlabels)
                running_test_loss += tloss.item()

        avg_test_loss = running_test_loss / len(self.test_loader)
        print(f'Perte moyenne sur le jeu de test : {avg_test_loss:.4f}')

    def train(self):
        for epoch in range(self.epochs):
            torch.cuda.empty_cache()

            print(f"Entraînement de l'EPOCH {epoch + 1}/{self.epochs} :")
            start_time = time.time()  # Epoch début

            train_loss = self.train_one_epoch(self.epoch_number)
            print(f'Train Loss : {train_loss:.4f}')

            avg_vloss = self.validate()
            print(f'Perte de validation : {avg_vloss:.4f}')

            self.writer.add_scalars('Perte/Entraînement et Validation', {
                'Entraînement': train_loss,
                'Validation': avg_vloss
            }, self.epoch_number + 1)
            self.writer.flush()

            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.epoch_number + 1, train_loss, avg_vloss])

            # Mettre à jour le planificateur de taux d'apprentissage
            self.scheduler.step(avg_vloss)
            
            # Sauvegardez uniquement les meilleurs modèles
            if avg_vloss < self.best_vloss:
                self.best_vloss = avg_vloss
                save_model(self.model, 'best_model.pth')

            self.epoch_number += 1

            # Enregistrez l'heure de fin de l'époque et calculez le temps nécessaire
            end_time = time.time()
            epoch_duration = end_time - start_time
            print(f'Temps écoulé pour l\'EPOCH {epoch + 1}/{self.epochs} : {epoch_duration:.2f} secondes.')

            # Pause de 5 minutes (300 secondes)
            print(f'EPOCH {epoch + 1}/{self.epochs} terminé. Pause de 5 minutes avant le prochain EPOCH...')
            time.sleep(300)  # 300 secondes équivalent à 5 minutes

        self.writer.close()
        self.test()

#################### MAIN ######################
if __name__ == "__main__":
    training_set = Data_set(annotations_file_train, img_dir_train, transform=transform)
    validation_set = Data_set(annotations_file_check, img_dir_valide, transform=transform)
    test_set = Data_set(annotations_file_check, img_dir_valide, transform=transform)

    training_loader = DataLoader(training_set, batch_size=16, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    model = ViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = ContrastiveLossCosine(margin=0.5)

    trainer = Trainer(model, training_loader, validation_loader, test_loader, optimizer, loss_fn, device, epochs=20)
    trainer.train()
