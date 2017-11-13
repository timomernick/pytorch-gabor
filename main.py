import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
import math
import sys
import numpy as np
import scipy.misc
import time

image_size = 128

def load_input_image():
    input_filename = sys.argv[1]
    image = scipy.misc.imread(input_filename, flatten=True)
    image = scipy.misc.imresize(image, [image_size, image_size])
    image = image.astype(np.float32) * (1.0 / 255.0)
    return image

input_image = load_input_image()
print("input_image", input_image.shape)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.lambda_ = nn.Parameter(torch.rand(1))
        self.theta = nn.Parameter(torch.randn(1) * 1.0)
        self.phi = nn.Parameter(torch.randn(1) * 0.02)
        self.sigma = nn.Parameter(torch.randn(1) * 1.0)
        self.gamma = nn.Parameter(torch.randn(1) * 0.0)
        self.amplitude = nn.Parameter(torch.randn(1))
        self.position = nn.Parameter(torch.randn(2) * 0.5)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        theta = self.sigmoid(self.theta) * math.pi * 2.0
        gamma = 1.0 + (self.gamma * 0.5)
        sigma = 0.1 + (self.sigmoid(self.sigma) * 0.4)
        lambda_ = 0.001 + (self.sigmoid(self.lambda_) * 0.999)
        phi = self.phi
        position = self.position

        x = Variable(torch.arange(-1.0, 1.0, (1.0 / float(image_size)) * 2.0))
        #print("x", x.size())

        y = Variable(torch.arange(-1.0, 1.0, (1.0 / float(image_size)) * 2.0))
        #print("y", y.size())

        x = x - position[1]
        y = y - position[0]

        x = x.view(1, -1).repeat(image_size, 1)
        y = y.view(-1, 1).repeat(1, image_size)

        x1 = (x * torch.cos(theta)) + (y * torch.sin(theta))
        #print("x1", x1.size())

        y1 = (-x * torch.sin(theta)) + (y * torch.cos(theta))
        #print("y1", y1.size())

        sigma_y = sigma/gamma
        g0 = torch.exp(-0.5 * ((x1**2 / sigma**2) + (y1**2 / sigma_y**2)))
        #print("g0", g0.size())

        g1 = torch.cos((2.0 * math.pi * (x1 / lambda_)) + phi)
        #print("g1", g1.size())

        g_real = (g0 * g1) * self.sigmoid(self.amplitude)
        #print("g", g.size())

        return g_real

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        num_generators = 128
        self.generators = []
        for i in range(num_generators):
            generator = Generator()
            self.add_module("gen_" + str(i), generator)
            self.generators.append(generator)

        self.loss = nn.MSELoss()

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        outputs = torch.stack([generator(input) for generator in self.generators], 0)
        output = torch.sum(outputs, 0)
        return self.sigmoid(output)

model = Model()

input = Variable(torch.FloatTensor(input_image))

learning_rate = 0.05
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_iterations = 5000
for iteration in range(num_iterations):
    model.zero_grad()

    output = model(input)

    vutils.save_image(output.data, "output.png")
    vutils.save_image(output.data, "output_" + str(iteration).zfill(4) + ".png")

    loss = model.loss(output, input)
    loss.backward()

    optimizer.step()

    print(iteration, loss.data[0])
