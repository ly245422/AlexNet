import torch
import torchvision.transforms as transforms
import torchvision.models as models

class ResNet50(torch.nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # Load pretrained ResNet50

        # Modify the last fully connected layer for your specific task
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_ftrs, num_classes)  # Output layer for classification/regression

        self.sigmoid = torch.nn.Sigmoid()

        # Encoder layers
        self.encoder = torch.nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )

        # Decoder layers (upsampling)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid()  # Output as probabilities
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)

        # Resize to (1, 1, 224, 224) using interpolation
        resize = transforms.Resize((224, 224))
        x = resize(x)

        # Convert output to binary values (0 or 1)
        x = torch.where(x > 0.5, torch.tensor(1.0).to(x.device), torch.tensor(0.0).to(x.device))

        return x

if __name__ == '__main__':
    x = torch.rand((1, 3, 224, 224))
    # print(x.shape)
    model = ResNet50(num_classes=1)
    x = model(x)
    print(x.shape)
    print(x)
