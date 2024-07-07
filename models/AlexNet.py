import os
import torch
import torchvision.transforms as transforms

class AlexNet(torch.nn.Module):
    def __init__(self, pretrained_weights_path=None):
        super(AlexNet, self).__init__()
        # 加载预训练权重
        if not pretrained_weights_path is None:
            if not os.path.isfile(pretrained_weights_path):
                raise FileNotFoundError(f"Pretrained weights file '{pretrained_weights_path}' not found.")
            pretrained_dict = torch.load(pretrained_weights_path)
            self.load_state_dict(pretrained_dict, False)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )  # Output size: torch.Size([96, 27, 27])
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )  # Output size: torch.Size([256, 13, 13])
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU()
        )  # Output size: torch.Size([384, 13, 13])
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU()
        )  # Output size: torch.Size([384, 13, 13])
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )  # Output size: torch.Size([256, 6, 6])

        self.flatten = torch.nn.Flatten()  # Output size: torch.Size([1, 9216])

        # Decoder layers (transpose convolutions)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(384, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(384, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid()  # Output as probabilities
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = x.view(-1, 256, 6, 6)  # Reshape to match decoder input size
        x = self.decoder(x)
        
        resize = transforms.Resize((224, 224))  # 或者 transforms.Resize((224, 224), interpolation=2)  # 对应 cv2.INTER_LINEAR

        x = torch.stack([resize(torch.Tensor(img)) for img in x])
        
        
        # Convert output to binary values (0 or 1)
        x = torch.where(x > 0.5, torch.tensor(1.0).to(x.device), torch.tensor(0.0).to(x.device))
        
        return x

if __name__ == '__main__':
    x = torch.rand((1, 3, 224, 224))
    print(x.shape)
    model = AlexNet()
    x = model(x)
    print(x.shape)
    print(x)