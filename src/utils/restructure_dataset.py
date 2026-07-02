import os
import shutil

dataset_path = "../../dataset"

for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    if not os.path.isdir(class_path):
        continue
    for sub_folder in os.listdir(class_path):
        sub_path = os.path.join(class_path, sub_folder)
        if not os.path.isdir(sub_path):
            continue
        images = os.listdir(sub_path)
        for image in images:
            src = os.path.join(sub_path, image)
            dst = os.path.join(class_path, image)
            shutil.move(src, dst)
        os.rmdir(sub_path)
        print(f"  {class_folder}/{sub_folder} — {len(images)} images déplacées")
    print(f"Classe {class_folder} terminée.")

print("Done.")