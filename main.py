import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
# import torchvision.utils as tvutils
# import torchvision.models as models
from network import net

BATCH_SIZE = 64


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2, drop_last=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2, drop_last=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 0:    # print every 2000 mini-batches
            print('[%d, %5d]' % (epoch + 1, i + 1))
            print(' Train loss: %.3f' %
                  (running_loss / 2000))
            running_loss = 0.0

            correct = 0
            total = 0

            class_correct = list(0. for i in range(10))
            class_total = list(0. for i in range(10))

            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    c = (predicted == labels).squeeze()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    for j in range(4):
                        label = labels[j]
                        class_correct[label] += c[j].item()
                        class_total[label] += 1

            print('Accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct / total))
            for k in range(10):
                print('Accuracy of %5s : %2d %%' % (
                    classes[k], 100 * class_correct[k] / class_total[k]))


print('Finished Training')
print()