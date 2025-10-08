import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import random
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from voc_dataset import VOCDataset   
from simple_cnn import SimpleCNN   
from train_q2 import ResNet      

device = torch.device("mps" if torch.cuda.is_available() else "cpu")
num_classes = len(VOCDataset.CLASS_NAMES)
model_path = "checkpoint-model-epoch41_.pth"  

model = ResNet(num_classes=num_classes)
model = torch.load(model_path, map_location=device)
model = model.to(device)
model.eval()

print(f"Loaded trained model from {model_path}")

test_dataset = VOCDataset(split="test", size=224, data_dir="./data/VOCdevkit/VOC2007/")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

N_SAMPLES = 1000
indices = random.sample(range(len(test_dataset)), N_SAMPLES)
features_list, labels_list = [], []

feature_extractor = nn.Sequential(*list(model.resnet.children())[:-1]) 

for idx in tqdm(indices, desc="Extracting features"):
    img, label, _ = test_dataset[idx]
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feature_extractor(img).squeeze().cpu().numpy()
    features_list.append(feat)
    labels_list.append(label.numpy())

features = np.stack(features_list) 
labels = np.stack(labels_list)     

print(f"Feature shape: {features.shape}, Label shape: {labels.shape}")

print("Computing t-SNE embedding...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
features_2d = tsne.fit_transform(features)

num_classes = len(VOCDataset.CLASS_NAMES)
colors = plt.cm.tab20(np.linspace(0, 1, num_classes))  
img_colors = []

for lbl in labels:
    active = np.where(lbl > 0)[0]
    if len(active) == 0:
        img_colors.append(np.array([0.5, 0.5, 0.5, 1.0])) 
    else:
        mean_col = np.mean(colors[active], axis=0)
        img_colors.append(mean_col)
img_colors = np.array(img_colors)

plt.figure(figsize=(12, 8))
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=img_colors, s=15)
plt.title("t-SNE Visualization of ResNet Feature Space on Pascal VOC", fontsize=14)
plt.axis("off")

for i, cls_name in enumerate(VOCDataset.CLASS_NAMES):
    plt.scatter([], [], color=colors[i], label=cls_name)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
plt.tight_layout()
plt.show()
