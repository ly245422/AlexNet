import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from lxml import etree
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class CardboardDataset(Dataset):
    def __init__(self, args, transform=None):
        self.img_dir = args.data_path
        self.xml_dir = args.data_path
        self.images = [file for file in os.listdir(args.data_path) if file.endswith('.jpg')]
        self.labels = self._parse_xml_files()
        
        # Define torchvision transforms for image preprocessing
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize image to 224x224
                transforms.ToTensor()  # Convert PIL Image to PyTorch tensor
            ])

    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = self._read_image(img_path)
        label = self.labels[index]
        return image, label.unsqueeze(0)  # Add an extra dimension

    def _parse_xml_files(self):
        labels = []
        for img_name in self.images:
            img_id = os.path.splitext(img_name)[0]  # Extract image ID without extension
            xml_path = os.path.join(self.xml_dir, img_id + '.xml')  # Corresponding XML file
            label = self._parse_single_xml(xml_path)
            labels.append(label)
        return labels

    def _parse_single_xml(self, xml_path):
        tree = etree.parse(xml_path)
        root = tree.getroot()
        
        label = torch.zeros(224, 224, dtype=torch.float)  # Initialize label as a 224x224 tensor
        
        # Extract original image size from XML
        size = root.find('size')
        original_width = int(size.find('width').text)
        original_height = int(size.find('height').text)
        
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Scale bounding box coordinates to match resized image size (224x224)
            width, height = 224, 224  # Resized image dimensions
            scale_x = width / original_width
            scale_y = height / original_height
            
            xmin_scaled = int(xmin * scale_x)
            ymin_scaled = int(ymin * scale_y)
            xmax_scaled = int(xmax * scale_x)
            ymax_scaled = int(ymax * scale_y)
            
            # Convert scaled coordinates to 224x224 indices
            xmin_idx = xmin_scaled
            xmax_idx = xmax_scaled
            ymin_idx = ymin_scaled
            ymax_idx = ymax_scaled
            
            # Ensure bounding box stays within bounds of the image
            xmin_idx = max(0, xmin_idx)
            xmax_idx = min(224, xmax_idx)
            ymin_idx = max(0, ymin_idx)
            ymax_idx = min(224, ymax_idx)
            
            # Set the corresponding region in label tensor to 1
            label[ymin_idx:ymax_idx, xmin_idx:xmax_idx] = 1
        
        return label

    def _read_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image


# Example usage:
def visualize_samples(dataset, num_samples=2):
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        image, label = dataset[i]
        image = transforms.ToPILImage()(image)  # Convert to PIL Image
        
        # Plot image
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image)
        plt.axis('off')
        
        # Plot bounding boxes
        for y in range(label.size(1)):
            for x in range(label.size(2)):
                if label[0, y, x] == 1:  # Accessing the extra dimension
                    xmin = x
                    ymin = y
                    xmax = x + 1
                    ymax = y + 1
                    
                    # Draw rectangle
                    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=2, edgecolor='r', facecolor='none')
                    plt.gca().add_patch(rect)
    
    plt.show()


if __name__=='__main__':
    class Args:
        def __init__(self):
            self.data_path = './dataset/train'

    args = Args()
    transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize image to 224x224
                transforms.ToTensor()  # Convert PIL Image to PyTorch tensor
            ])
    dataset = CardboardDataset(args, transform=transform)

    # Visualize some samples from the dataset
    # visualize_samples(dataset, num_samples=1)
