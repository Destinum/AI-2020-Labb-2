# pip3 install torch torchvision
# pip3 install pytmx
# pip3 install pygame

import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
import random
#import Astar
import pytmx
#from pytmx.util_pygame import load_pygame
#import pygame as pg

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

net = Net()

#Create train and test data
train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
#make them sets
trainset = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

#decide loss function and optmizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

#Teach the NN
for datasets in range(200): #10000
    print("Epoch #", datasets)
    #train = CustomDataset(200)
    trainset = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
    for epoch in range(10): # 3 full passes over the data
        for data in trainset:  # `data` is a batch of data
            X, y = data  # X is the batch of features, y is the batch of targets.
            net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
            output = net(X.view(-1,784))  # pass in the reshaped batch (recall they are 28x28 atm)
            loss = F.nll_loss(output, y)  # calc and grab the loss value
            loss.backward()  # apply this loss backwards thru the network's parameters
            optimizer.step()  # attempt to optimize weights to account for loss/gradients


# Test the NN
net.eval() # needed?
correct = 0
total = 0
with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X.view(-1,784))
        #print(output)
        for idx, i in enumerate(output):
            print(torch.argmax(i), y[idx])
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round((correct/total)*100, 3))

## Save and load a model parameters:
##torch.save(net.state_dict(), PATH)
##
##net = Net()   #TheModelClass(*args, **kwargs)
##net.load_state_dict(torch.load(PATH))
##net.eval()













































"""
from matplotlib import pyplot as plt
import numpy as np

data = [(3, 1.5, 1),
        (2, 1, 0),
        (4, 1.5, 1),
        (3, 1, 0),
        (3.5, 0.5, 1),
        (2, 0.5, 0),
        (5.5, 1, 1),
        (1, 1, 0)]

mystery_flower = (4.5, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1 - sigmoid(x))

#""
T = np.linspace(-5, 5, 10)
#print(T)
Y = sigmoid(T)
#print(Y)
plt.plot(T, Y)
#""

#training loop

learning_rate = 0.2
costs = []

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

print("w1:", w1)
print("w2:", w2)
print("b:", b)

for i in range (50000):
    ri = np.random.randint(len(data))
    point = data[ri]
    #print(point)

    z = point[0] * w1 + point[1] * w2 + b
    pred = sigmoid(z)

    target = point[2]
    cost = np.square(pred - target)

    costs.append(cost)

    # Manual derivative?
    dcost_pred = 2 * (pred - target)
    dpred_dz = sigmoid_p(z)
    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1

    dcost_dz = dcost_pred * dpred_dz
    dcost_dw1 = dcost_dz * dz_dw1
    dcost_dw2 = dcost_dz * dz_dw2
    dcost_db = dcost_dz * dz_db

    w1 -= learning_rate * dcost_dw1
    w2 -= learning_rate * dcost_dw2
    b -= learning_rate * dcost_db

print("")
print("w1:", w1)
print("w2:", w2)
print("b:", b)
print("")

for point in data:
    print(point)
    z = point[0] * w1 + point[1] * w2 + b
    pred = sigmoid(z)
    print("pred: {}".format(pred))


z = mystery_flower[0] * w1 + mystery_flower[1] * w2 + b
pred = sigmoid(z)
print("Mystery Flower:", pred)
"""












































































"""
import numpy

def NN(m1, m2, w1, w2, b):
    z = m1 * w1 + m2 * w2 + b
    return sigmoid(z)

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

Dataset = [("Red", 3, 1.5),
        ("Blue", 2, 1),
        ("Red", 4, 1.5),
        ("Blue", 3, 1),
        ("Red", 3.5, 0.5),
        ("Blue", 2, 0.5),
        ("Red", 5.5, 1),
        ("Blue", 1, 1),
        ("???", 4.5, 1)]

w1 = numpy.random.randn()
w2 = numpy.random.randn()
b = numpy.random.randn()

#""
print("w1:", w1)
print("w2:", w2)
print("b:", b)

for Data in Dataset:
    print(NN(Data[1], Data[2], w1, w2, b))
#""


def cost(b):
    return (b - 4) ** 2

def num_slope(b):
    h = 0.0001
    return (cost(b + h) - cost(b)) / h

#print(num_slope(3))
#print(num_slope(5))

def slope(b):
    return 2 * (b - 4)

b = -10

for i in range(100):
    b -= 0.1 * slope(b)
    print(b)

"""














































































































"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REBUILD_DATA = False    #set to true to one once, then back to false unless you want to change something in your training data

#Fisk = torch.cuda.is_available()

class DogsVSCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    pass


        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats:", self.catcount)
        print("Dogs:", self.dogcount)

if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()


training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))

#""
print(training_data[1])

plt.imshow(training_data[1][0], cmap="gray")
plt.show()
#""

#""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#""

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        print(x[0].shape)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x, dim=1)
        #return x

net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X / 255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
print("val_size:", val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print(len(train_X))
print(len(test_X))

BATCH_SIZE = 100
EPOCHS = 1

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        #print(i, i + BATCH_SIZE)
        batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
        batch_y = train_y[i:i+BATCH_SIZE]

        net.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

print(loss)

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1,1,50,50))[0]

        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct+= 1
        total += 1

print("Accuracy", round(correct / total, 3))
"""

























































































"""
import torch
import torchvision
from torchvision import transforms, datasets
#import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


train = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(). __init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)


net = Net()
print(net)

X = torch.rand((28,28))
X = X.view(-1, 28*28)

output = net(X)

optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        #data is a batch of featuresets and labels
        X, y = data
        #print(X[0])
        #print(y[0])
        #break
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0

with torch.no_grad():
    for data in trainset:
        X, y =  data
        output = net(X.view(-1, 28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy:", round(correct/total, 3))
print("")

for Index in range(10):
    plt.imshow(X[Index].view(28,28))
    print(torch.argmax(net(X[Index].view(-1, 784))[0]))
    plt.show()


"""































































"""
for data in trainset:
    print(data)
    break

x, y = data[0][0], data[1][0]

print(y)

plt.imshow(data[0][0].view(28, 28))
#plt.show()

print(data[0][0].shape)

total = 0
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
    Xs, ys = data
    for y in ys:
        counter_dict[int(y)] += 1
        total += 1

print(counter_dict)

for i in counter_dict:
    print(f"{i}: {counter_dict[i]/total*100}")
"""