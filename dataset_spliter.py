import os
import random

import shutil

def split_dataset(labels_dir, images_dir, splited_dataset_dir, train_ratio=0.8, val_ratio=0.1, img_extension=".png"):
    ''' 
    splited_dataset_dir, labels_dir, images_dir are all absolute path
    '''
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"error with labels path and images path")
        return
    # create folder
    if not os.path.exists(splited_dataset_dir):
        os.makedirs(splited_dataset_dir)

    for folder in ['train', 'val', 'test']:
        os.makedirs(os.path.join(splited_dataset_dir, "images", folder), exist_ok=True)
        os.makedirs(os.path.join(splited_dataset_dir, "labels", folder), exist_ok=True)

    all_labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    random.shuffle(all_labels)

    # define break points
    total_labels = len(all_labels)
    train_end = int(total_labels * train_ratio)
    val_end = train_end + int(total_labels * val_ratio)

    # splitting data
    for i, label_file in enumerate(all_labels):
        if i < train_end:
            subset = 'train'
        elif i < val_end:
            subset = 'val'
        else:
            subset = 'test'

        # create source folder path and target folder path
        label_src = os.path.join(labels_dir, label_file)
        label_dst = os.path.join(splited_dataset_dir, "labels", f"{subset}", label_file)

        image_file = label_file.replace('.txt', img_extension)
        image_src = os.path.join(images_dir, image_file)
        image_dst = os.path.join(splited_dataset_dir, "images", f"{subset}", image_file)

        # copy files
        try:
            shutil.copy(label_src, label_dst)
        except:
            print(f"error when copy {label_src}")
        try:
            shutil.copy(image_src, image_dst)
        except:
            print(f"error when copy {image_src}")
    return
# split_dataset(
#     labels_dir='.\\dataset\\labels', 
#     images_dir='.\\dataset\\images',
#     splited_dataset_dir=".\\splited_dataset",
# )
# split_dataset(
#     labels_dir='.\\dataset300\\labels', 
#     images_dir='.\\dataset300\\images',
#     splited_dataset_dir=".\\splited_dataset300",
# )