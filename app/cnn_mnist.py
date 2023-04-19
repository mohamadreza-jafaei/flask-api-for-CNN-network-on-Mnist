import torch
from PIL import Image
import io
from pickle import load
import torch.nn as nn
import numpy as np


device = torch.device('cpu')


class CNNNetwork(nn.Module):

    def __init__(self, input_dimention , num_classes):
        super(CNNNetwork , self).__init__()
        self.conv1 = nn.Conv2d(input_dimention , 6 , 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(16*4*4 , 128)
        self.lin2 = nn.Linear(128 , 64)
        self.lin3 = nn.Linear(64, num_classes)

    def forward(self , x):
        # conv 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        #conv 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # x = x.view(-1, 16*4*4)
        x = torch.flatten(x,1)

        # lin 1
        x = self.lin1(x)
        x = self.relu(x)

        # lin 2
        x = self.lin2(x)
        x = self.relu(x)

        # lin 3
        x = self.lin3(x)
        return x

model = CNNNetwork(1,10)
model.load_state_dict(torch.load("./model_state.pth" , map_location=device))
model.eval()
sc = load(open('scaler.pkl', 'rb'))
size = 28

def predict(image_tensor):
    image = image_tensor.view(1,1,28,28)
    predicted = model(image)
    class_predicted = torch.argmax(predicted , 1)
    return class_predicted.item()

def prepare_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    image = image.resize((28,28), Image.Resampling.LANCZOS)

    image = np.asarray(image)
    image = image.reshape(1,784)

    image = sc.transform(image)

    image = torch.from_numpy(image.astype(np.float32))
    return image


