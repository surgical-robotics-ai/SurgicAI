import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from torchvision.models import resnet50
from torch.utils.tensorboard import SummaryWriter

# Assuming imgnet is used, which typically uses a ResNet-50 backbone
def load_imgnet_pretrained_model():
    model = resnet50(pretrained=False)  # Normally you would load the imgnet pretrained weights here
    return model

imgnet_model = load_imgnet_pretrained_model()
imgnet_model.eval()  # Set the model to evaluation mode

# Extended model with MLP for behavior cloning
class BehaviorCloningModel(nn.Module):
    def __init__(self, feature_extractor):
        super(BehaviorCloningModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.regressor = nn.Sequential(
            nn.BatchNorm1d(2048 + 7),  # Adjust based on the feature size of imgnet's backbone
            nn.Linear(2048 + 7, 512),
            nn.ReLU(),
            nn.Linear(512, 7)
        )

    def forward(self, x, proprioceptive_data):
        with torch.no_grad():
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)  # Flatten the features
        combined_input = torch.cat((features, proprioceptive_data), dim=1)
        return self.regressor(combined_input)

# Instantiate the model
device = "cuda" if torch.cuda.is_available() else "cpu"
bc_model = BehaviorCloningModel(imgnet_model).to(device)

# Custom Dataset class to handle images and proprioceptive data
class ImageProprioDataset(Dataset):
    def __init__(self, image_paths, actions, proprio_data):
        self.image_paths = image_paths
        self.actions = actions
        self.proprio_data = proprio_data
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        action = torch.from_numpy(self.actions[idx])
        proprioceptive = torch.from_numpy(self.proprio_data[idx])
        return img.to(device), action.to(device), proprioceptive.to(device)

# Example data
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
actions = [np.random.rand(7) for _ in range(2)]
proprioceptive_data = [np.random.rand(7) for _ in range(2)]

# DataLoader
dataset = ImageProprioDataset(image_paths, actions, proprioceptive_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Loss, Optimizer, and TensorBoard
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(bc_model.parameters(), lr=0.001)
writer = SummaryWriter('runs/imgnet_behavior_cloning_experiment')

# Training Loop
def train(model, dataloader, criterion, optimizer, writer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for i, (images, targets, proprio_data) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(images, proprio_data)
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                writer.add_scalar('Training Loss', loss.item(), epoch * len(dataloader) + i)
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}')

# Run training
train(bc_model, dataloader, criterion, optimizer, writer, num_epochs=10)
writer.close()
