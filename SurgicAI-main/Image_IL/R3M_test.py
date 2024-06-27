import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Load the pretrained R3M model
model = torch.hub.load('facebookresearch/r3m', 'r3m_large')
model.eval()  # Freeze the pretrained model

# Extended model with MLP for behavior cloning
class BehaviorCloningModel(nn.Module):
    def __init__(self, pretrained_model):
        super(BehaviorCloningModel, self).__init__()
        self.r3m = pretrained_model
        self.regressor = nn.Sequential(
            nn.BatchNorm1d(2048 + 7),  # Batch normalization on concatenated input
            nn.Linear(2048 + 7, 512),
            nn.ReLU(),
            nn.Linear(512, 7)
        )

    def forward(self, x, proprioceptive_data):
        with torch.no_grad():
            visual_features = self.r3m(x)
        combined_input = torch.cat((visual_features, proprioceptive_data), dim=1)
        return self.regressor(combined_input)

# Instantiate the model
bc_model = BehaviorCloningModel(model)

# Custom Dataset class to handle images and proprioceptive data
class ImageProprioDataset(Dataset):
    def __init__(self, image_paths, actions, proprio_data):
        self.image_paths = image_paths
        self.actions = actions
        self.proprio_data = proprio_data
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
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
        return img, action, proprioceptive

# Example data (replace with your actual paths, actions, and proprioceptive data)
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
actions = [np.random.rand(7) for _ in range(2)]
proprioceptive_data = [np.random.rand(7) for _ in range(2)]

# DataLoader
dataset = ImageProprioDataset(image_paths, actions, proprioceptive_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(bc_model.parameters(), lr=0.001)
writer = SummaryWriter('r3m/ImageBehaviorCloning')

# Training Loop
def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for  i, (images, targets, proprio_data) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(images, proprio_data)
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            if i % 10 == 0:  # Log every 10 mini-batches
                writer.add_scalar('Training Loss', loss.item(), epoch * len(dataloader) + i)
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Run training
train(bc_model, dataloader, criterion, optimizer, num_epochs=10)
writer.close()
