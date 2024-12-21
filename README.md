# Projet_Crop_IA
Fine-tuning de ViT, projet IA de 4ème année d'ingénieur en électronique et informatique

Le projet est composé de plusieurs fichiers python.
- Updated_ViT_GPU.py est le programme principal de fine-tuning du ViT pour l'améliorer et lui permettre de reconnaître les photos rognées.
- test_py.py est l'utilisation du ViT original, pré entrainé mais non fine-tuné par nous.
- utilisation.py permet d'utiliser le nouveau modèle sauvegardé.
- comparaison.py permet de comparer l'ancien et le nouveau ViT pour tirer des conclusion sur les performances. Les détails de cette comparaison sont visibles dans le comparaison_results.log.
- couper_photo.py permet de créer notre base de donnée et le csv qui correspond.

La base de donnée utilisée est un mélange de Caltech 101 et PASCAL VOC 2012.

