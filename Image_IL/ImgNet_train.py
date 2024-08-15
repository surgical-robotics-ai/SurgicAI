import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from torchvision.models import resnet50
import wandb
import argparse
import os
import re
import pickle
import gc

# 命令行参数解析
parser = argparse.ArgumentParser(description='Behavior Cloning Training')
parser.add_argument('--task_name', type=str, required=True, help='Name of the task')
parser.add_argument('--view_name', type=str, required=True, help='Name of the view')
args = parser.parse_args()

task_name = args.task_name
view_name = args.view_name

gc.collect()
torch.cuda.empty_cache()

data_dir = f'/home/jwu220/Trajectory_cloud/Five_task_v2/{task_name}'
model_save_dir = f'/home/jwu220/Trajectory_cloud/IL_model_v2/{task_name}/Imgnet_{view_name}_view/Model'

os.makedirs(model_save_dir, exist_ok=True)

def load_Imgnet_pretrained_model():
    model = resnet50(pretrained=True)
    model.fc = nn.Identity() 
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Imgnet_model = load_Imgnet_pretrained_model()
Imgnet_model.eval()
Imgnet_model.to(device)

class BehaviorCloningModel(nn.Module):
    def __init__(self, feature_extractor):
        super(BehaviorCloningModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.regressor = nn.Sequential(
            nn.BatchNorm1d(2048 + 7),
            nn.Linear(2048 + 7, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
            nn.Tanh()
        ).to(device)

    def forward(self, x, proprioceptive_data):
        with torch.no_grad():
            features = self.feature_extractor(x)
        combined_input = torch.cat((features, proprioceptive_data), dim=1)
        return self.regressor(combined_input)

bc_model = BehaviorCloningModel(Imgnet_model)

class PickleDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        for f in os.listdir(data_dir):
            if re.match(r'episode_\d+\.pkl$', f):
                file_path = os.path.join(data_dir, f)
                with open(file_path, 'rb') as file:
                    trajectory = pickle.load(file)
                    self.data.extend(trajectory)
        
        print(f"Total number of data points: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        step = self.data[idx]
        
        image_data = step['images'][view_name]
        img = Image.fromarray(image_data.astype('uint8'), 'RGB') if isinstance(image_data, np.ndarray) else Image.open(image_data).convert('RGB')
        img = self.transform(img)
        
        proprioceptive = torch.tensor(step['obs']['observation'][0:7], dtype=torch.float32)
        action = torch.tensor(step['action'], dtype=torch.float32)
        
        return img, action, proprioceptive

dataset = PickleDataset(data_dir)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(bc_model.parameters(), lr=0.0001)

wandb.init(project="behavior_cloning_v2", name=f"Imgnet_{task_name}_{view_name}_view")
wandb.config.update(args)

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs, checkpoint_interval):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for i, (images, actions, proprio_data) in enumerate(train_loader):
            images, actions, proprio_data = images.to(device), actions.to(device), proprio_data.to(device)
            optimizer.zero_grad()
            outputs = model(images, proprio_data)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        wandb.log({"Train Loss": avg_train_loss}, step=epoch)
        
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for images, actions, proprio_data in test_loader:
                images, actions, proprio_data = images.to(device), actions.to(device), proprio_data.to(device)
                outputs = model(images, proprio_data)
                loss = criterion(outputs, actions)
                total_test_loss += loss.item()
            avg_test_loss = total_test_loss / len(test_loader)
            wandb.log({"Test Loss": avg_test_loss}, step=epoch)
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == num_epochs:
            torch.save(model.state_dict(), os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth'))
            print(f'Model saved at epoch {epoch+1}')

    torch.save(model.state_dict(), os.path.join(model_save_dir, 'model_final.pth'))
    print('Final model saved')

num_epochs = 40
checkpoint_interval = 20
train_and_evaluate(bc_model, train_loader, test_loader, criterion, optimizer, num_epochs, checkpoint_interval)

wandb.finish()