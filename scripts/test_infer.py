import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image
import torch.nn.functional as F
import os
import numpy as np

# https://vitalflux.com/pytorch-load-predict-pretrained-resnet-model/


model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)

model.load_state_dict(torch.load('src\\leomower\\scripts\\best_model_resnet18_free_blocked.pth', map_location=torch.device('cpu')))

#print(model)

""" device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = model.eval().half()

if torch.cuda.is_available():
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()
else:
    mean = torch.Tensor([0.485, 0.456, 0.406]).half()
    std = torch.Tensor([0.229, 0.224, 0.225]).half()

normalize = torchvision.transforms.Normalize(mean, std) """

imageFilename = "blocked\\9ab0380e-8839-11ec-8cbc-e45f017bf0eb.jpg"
#imageFilename = "free\\8dbd95e2-8839-11ec-8cbc-e45f017bf0eb.jpg"

image = PIL.Image.open(os.path.join("dataset", imageFilename)).convert('RGB')
#image = PIL.Image.fromarray(image)
#image = transforms.functional.to_tensor(image).to(device).half()
#image.sub_(mean[:, None, None]).div_(std[:, None, None])
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

img_preprocessed = preprocess(image)
batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)

y = model(batch_img_tensor)

""" x = image[None, ...]
y = model(x)
 """
y = F.softmax(y, dim=1)
#print(y)
prob_blocked = float(y.flatten()[0])

print("prob_blocked = ", prob_blocked)
