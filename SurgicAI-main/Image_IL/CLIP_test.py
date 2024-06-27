import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import clip
from torch.utils.tensorboard import SummaryWriter

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()  # Freeze the CLIP model

# Extended model with MLP for behavior cloning
class BehaviorCloningModel(nn.Module):
    def __init__(self, clip_model):
        super(BehaviorCloningModel, self).__init__()
        self.clip = clip_model
        self.regressor = nn.Sequential(
            nn.BatchNorm1d(512 + 7),  # Adjust according to the CLIP model's output feature size
            nn.Linear(512 + 7, 512),
            nn.ReLU(),
            nn.Linear(512, 7)
        )

    def forward(self, x, proprioceptive_data):
        with torch.no_grad():
            image_features = self.clip.encode_image(x).float()
        image_features = image_features.view(image_features.size(0), -1)
        combined_input = torch.cat((image_features, proprioceptive_data), dim=1)
        return self.regressor(combined_input)

# Instantiate the model
bc_model = BehaviorCloningModel(clip_model).to(device)

# Custom Dataset class to handle images and proprioceptive data
class ImageProprioDataset(Dataset):
    def __init__(self, image_paths, actions, proprio_data):
        self.image_paths = image_paths
        self.actions = actions
        self.proprio_data = proprio_data
        self.transform = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        action = torch.from_numpy(self.actions[idx])
        proprioceptive = torch.from_numpy(self.proprio_data[idx])
        return img, action, proprioceptive

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
writer = SummaryWriter('clip/ImageBehaviorCloning')

# Training Loop
def train(model, dataloader, criterion, optimizer, writer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for i, (images, targets, proprio_data) in enumerate(dataloader):
            images, targets, proprio_data = images.to(device), targets.to(device), proprio_data.to(device)
            optimizer.zero_grad()
            outputs = model(images, proprio_data)
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            if i % 10 == 0:  # Log every 10 mini-batches
                writer.add_scalar('Training Loss', loss.item(), epoch * len(dataloader) + i)
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}')

# Run training
train(bc_model, dataloader, criterion, optimizer, writer, num_epochs=10)
writer.close()
