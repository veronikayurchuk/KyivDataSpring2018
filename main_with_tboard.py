import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as tvutils
import torchvision.models as models
from network import net
from tensorboardX import SummaryWriter

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

NUM_CLASSES = len(classes)
BATCH_SIZE = 64
PRINT_EACH = 400


# Setting Tensorboard
# In order to launch tensorboard write:
# tensorboard --logdir "./logs/" --port 8090
# where "./logs/" path with logs
# And connect to http://localhost:8090/ via your favourite browser

save_path = "./logs/test_1"
writer = SummaryWriter(save_path)


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

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

### Dataset Projector
from torchvision import datasets
dataset = datasets.MNIST('mnist', train=False, download=True)
images = dataset.test_data[:300].float()
label = dataset.test_labels[:300]
features = images.view(300, 784)
writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))


running_loss = 0.0
for epoch in range(10):  # loop over the dataset multiple times
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

        running_loss += loss.item()
        if i % PRINT_EACH == 0:
            ### Example of the network
            dummy_input = torch.autograd.Variable(torch.rand(1, 3, 224, 224))
            resnet18 = models.resnet18(False)
            writer.add_graph(resnet18, (dummy_input, ))

            print('[%d, %5d]' % (epoch + 1, i + 1))
            print(' Train loss: %.3f' %
                  (running_loss / PRINT_EACH))

            writer.add_scalar('Train loss', (running_loss / PRINT_EACH), epoch*len(trainloader) + i)

            ### Adding histograms
            for name, param in net.named_parameters():
                writer.add_histogram(name, param.clone().data.numpy(), epoch*len(trainloader) + i)

            ### Images Visualization
            imgs = tvutils.make_grid(inputs, normalize=True, scale_each=True)
            writer.add_image("Image_{}".format(i), imgs)

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
                    for j in range(BATCH_SIZE):
                        label = labels[j]
                        class_correct[label] += c[j].item()
                        class_total[label] += 1

            # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
            writer.add_scalar('Test Accuracy', (100 * correct / total), epoch * len(trainloader) + i)
            for k in range(10):
                # print('Accuracy of %5s : %2d %%' % (classes[k], 100 * class_correct[k] / class_total[k]))
                writer.add_scalar('Accuracy of %5s' % classes[k],
                                  (100 * class_correct[k] / class_total[k]),
                                  epoch * len(trainloader) + i)

print('Finished Training')
print()