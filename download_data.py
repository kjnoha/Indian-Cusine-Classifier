import os
import shutil
from bing_image_downloader import downloader

dataset_dir = 'filtered_dataset'
classes = ['pizza food', 'hamburger food', 'sushi food', 'pasta food', 'tacos food', 'sandwich food']

print("Downloading non-Indian food images...")
temp_dir = 'temp_downloads'
for c in classes:
    downloader.download(c, limit=40, output_dir=temp_dir, adult_filter_off=True, force_replace=False, timeout=10)

train_dir = os.path.join(dataset_dir, 'train', 'non_indian')
val_dir = os.path.join(dataset_dir, 'validation', 'non_indian')
test_dir = os.path.join(dataset_dir, 'test', 'non_indian')

for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

all_images = []
for c in classes:
    class_dir = os.path.join(temp_dir, c)
    if os.path.exists(class_dir):
        for f in os.listdir(class_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                all_images.append(os.path.join(class_dir, f))

import random
random.seed(42)
random.shuffle(all_images)
total = len(all_images)

train_split = int(total * 0.75)
val_split = int(total * 0.90)

train_imgs = all_images[:train_split]
val_imgs = all_images[train_split:val_split]
test_imgs = all_images[val_split:]

def copy_files(imgs, target_dir):
    for i, img in enumerate(imgs):
        ext = os.path.splitext(img)[1]
        dest = os.path.join(target_dir, f"non_indian_{i}{ext}")
        shutil.copy(img, dest)

print("Copying to dataset folders...")
copy_files(train_imgs, train_dir)
copy_files(val_imgs, val_dir)
copy_files(test_imgs, test_dir)

print(f"Done! Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
