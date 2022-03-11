import torch
from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image
import torch.nn.functional as F
import os
import numpy as np
from torch.autograd import Variable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)

model.load_state_dict(torch.load('src\\leomower\\scripts\\best_model_resnet18_free_blocked.pth', map_location=device))
model.eval()
#print(model)


imageFilename = "blocked\\4f155a3c-8839-11ec-8cbc-e45f017bf0eb.jpg"
#imageFilename = "free\\677c5602-8839-11ec-8cbc-e45f017bf0eb.jpg"

image = PIL.Image.open(os.path.join("dataset", imageFilename)).convert('RGB')

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

def predict_image(image):
    image_tensor = preprocess(image).float()
    image_tensor = image_tensor.unsqueeze(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    #output = model(image_tensor)
    index = output.data.cpu().numpy().argmax()
    return output  
    #return index  

y = predict_image(image)
y = F.softmax(y, dim=1)
y = y.flatten()
prob_blocked = float(y[0])
print("prob_blocked = ", prob_blocked)
