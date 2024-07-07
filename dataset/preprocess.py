"""
Preprocess the dataset: divide the data set into training set and test set according to a certain proportion
"""
import os
import shutil
import random
import glob

def split_dataset(jpeg_dir, xml_dir, train_ratio=0.8):
    # Define paths for train and test folders
    train_dir = './dataset/train'
    test_dir = './dataset/test'
    
    # Remove existing train and test folders if they exist
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Create new train and test folders
    os.makedirs(train_dir)
    os.makedirs(test_dir)
    
    # Get list of all JPEG and XML files
    jpeg_files = glob.glob(os.path.join(jpeg_dir, '*.jpg'))
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    
    # Sort files to ensure consistency between JPEG and XML lists
    jpeg_files.sort()
    xml_files.sort()
    
    # Ensure the number of JPEG and XML files match
    assert len(jpeg_files) == len(xml_files), "Number of JPEG and XML files do not match"
    
    # Calculate number of files for training and testing
    num_train = int(len(jpeg_files) * train_ratio)
    num_test = len(jpeg_files) - num_train
    
    # Create lists of indices for train and test sets
    indices = list(range(len(jpeg_files)))
    random.shuffle(indices)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]
    
    # Move files to train folder
    for idx in train_indices:
        jpeg_file = jpeg_files[idx]
        xml_file = xml_files[idx]
        shutil.move(jpeg_file, os.path.join(train_dir, os.path.basename(jpeg_file)))
        shutil.move(xml_file, os.path.join(train_dir, os.path.basename(xml_file)))
    
    # Move files to test folder
    for idx in test_indices:
        jpeg_file = jpeg_files[idx]
        xml_file = xml_files[idx]
        shutil.move(jpeg_file, os.path.join(test_dir, os.path.basename(jpeg_file)))
        shutil.move(xml_file, os.path.join(test_dir, os.path.basename(xml_file)))
    
    print(f"Dataset split completed: {num_train} samples in train set, {num_test} samples in test set")

    if os.path.exists(jpeg_dir):
        shutil.rmtree(jpeg_dir)
    if os.path.exists(xml_dir):
        shutil.rmtree(xml_dir)

# Example usage:
jpeg_dir = './dataset/JPEGImages'
xml_dir = './dataset/Annotations'
split_dataset(jpeg_dir, xml_dir, train_ratio=0.8)
