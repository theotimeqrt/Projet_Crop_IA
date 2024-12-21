import os
import random
import importlib.util

# Paths for Python files to call
utilisation_script = 'utilisation.py'
test_script = 'test_ViT.py'

# Verify if external scripts are loadable
def load_script(script_path):
    spec = importlib.util.spec_from_file_location("module.name", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load external scripts
utilisation = load_script(utilisation_script)
test_ViT = load_script(test_script)

# Determine similarity based on file names
def determine_similarity(image1, image2):
    # Extract numbers from file names
    def extract_number(filename):
        if "_cropped" in filename:
            return filename.split('image')[1].split('_cropped')[0]
        else:
            return filename.split('image')[1].split('.')[0]

    num1 = extract_number(image1)
    num2 = extract_number(image2)

    # If numbers are the same, images are similar
    return 1 if num1 == num2 else 0


# Function to display progress bar
def display_progress(current, total):
    if current == total:
        # S'il s'agit de la dernière tâche, imprimez la progression et terminez
        print(f"\rProgress: {current}/{total} ({(current / total) * 100:.2f}%)", flush=True)
        print()  # Ajouter un saut de ligne
    else:
        # Mettre à jour la barre de progression normalement
        print(f"\rProgress: {current}/{total} ({(current / total) * 100:.2f}%)", end="", flush=True)


# Main function for random comparisons
def main():
    image_folder = "../Data_set/images_sorties/test"
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    # Assurez-vous qu'il y a suffisamment de photos
    if len(images) < 2:
        print("Not enough images for comparison.")
        return

    total_iterations = 500
    correct_utilisation = 0
    correct_test_ViT = 0

    log_file = "comparison_results.log"
    with open(log_file, "w") as log:
        log.write("Comparison Results Log\n")
        log.write("======================\n\n")

        for iteration in range(total_iterations):
            # Afficher la progression
            display_progress(iteration + 1, total_iterations)

            # Sélectionnez au hasard deux images
            image1, image2 = random.sample(images, 2)
            image1_path = os.path.join(image_folder, image1)
            image2_path = os.path.join(image_folder, image2)

            true_similarity = determine_similarity(image1, image2)

            result_utilisation = utilisation.compare(image1_path, image2_path)
            result_test_ViT = test_ViT.compare(image1_path, image2_path)

            correct_utilisation += 1 if result_utilisation == true_similarity else 0
            correct_test_ViT += 1 if result_test_ViT == true_similarity else 0

            log.write(f"Iteration {iteration + 1}:\n")
            log.write(f"  Image 1: {image1_path}\n")
            log.write(f"  Image 2: {image2_path}\n")
            log.write(f"  True Similarity: {true_similarity}\n")
            log.write(f"  utilisation.py Result: {result_utilisation}\n")
            log.write(f"  test_ViT.py Result: {result_test_ViT}\n")
            log.write("\n")

    # Calculer et afficher la précision
    accuracy_utilisation = (correct_utilisation / total_iterations) * 100
    accuracy_test_ViT = (correct_test_ViT / total_iterations) * 100

    print(f"Accuracy of utilisation.py: {accuracy_utilisation:.2f}%")
    print(f"Accuracy of test_ViT.py: {accuracy_test_ViT:.2f}%")

if __name__ == "__main__":
    main()
