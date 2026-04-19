import os
import shutil
import random

# Paths
source_dir = "IQ-OTH_NCCD"   # original dataset folder
output_dir = "dataset"

classes = ['normal', 'benign', 'malignant']

# Create folders
for cls in classes:
    os.makedirs(os.path.join(output_dir, 'train', cls), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', cls), exist_ok=True)

# Split ratio
split_ratio = 0.8

for cls in classes:
    cls_path = os.path.join(source_dir, cls)
    images = os.listdir(cls_path)
    
    random.shuffle(images)
    split_index = int(len(images) * split_ratio)
    
    train_images = images[:split_index]
    test_images = images[split_index:]
    
    # Copy files
    for img in train_images:
        shutil.copy(os.path.join(cls_path, img),
                    os.path.join(output_dir, 'train', cls, img))
    
    for img in test_images:
        shutil.copy(os.path.join(cls_path, img),
                    os.path.join(output_dir, 'test', cls, img))

print("Dataset split completed!")