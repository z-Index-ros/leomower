import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import os
from datetime import datetime


print("Starting at ", datetime.now())
print("Start working in ", os.getcwd())
datasetPath = "dataset"
print("Using dataset located at ", datasetPath)

print("Create dataset instance")
dataset = datasets.ImageFolder(
    datasetPath,
    transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.Resize((640, 480)),
#        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)

print("Split dataset into train and test sets")
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 50, 50])

print("Create data loaders to load data in batches")
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)


print("Define the neural network")
print("CUDA available ", torch.cuda.is_available())
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 2)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Train the neural network")
NUM_EPOCHS = 30
BEST_MODEL_PATH = os.path.join(datasetPath, "best_model_resnet18.pth")
best_accuracy = 0.0

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(NUM_EPOCHS):
    
    for images, labels in iter(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
    
    test_error_count = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
    
    test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
    print('%d: %f - %s' % (epoch, test_accuracy, datetime.now()))
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_accuracy = test_accuracy

print("----------------------")
print("train finished")
print(BEST_MODEL_PATH)
print(datetime.now())
