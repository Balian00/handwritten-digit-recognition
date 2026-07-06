import os
import shutil


def restructure_dataset(dataset_path: str) -> None:
    """Moves images up one level in the dataset structure.

    For each class, moves the images from the subfolders directly
    into the class folder, then removes the empty subfolders.
    On name collision, renames the source file with a "_dup" suffix.

    Args:
        dataset_path: Path to the dataset root folder.
    """
    if not os.path.isdir(dataset_path):
        print(f"Error: folder '{dataset_path}' not found.")
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

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
                if os.path.exists(dst):
                    print(f"  Warning: '{dst}' already exists, renaming the source.")
                    base, ext = os.path.splitext(image)
                    dst = os.path.join(class_path, f"{base}_dup{ext}")
                try:
                    shutil.move(src, dst)
                except OSError as e:
                    print(f"Error: unable to move '{src}' -> '{dst}': {e}")
                    raise
            try:
                os.rmdir(sub_path)
            except OSError as e:
                print(f"Error: unable to remove '{sub_path}': {e}")
                raise
            print(f"  {class_folder}/{sub_folder} — {len(images)} images moved")
        print(f"Class {class_folder} done.")

    print("Done.")


if __name__ == "__main__":
    restructure_dataset("../../dataset")
