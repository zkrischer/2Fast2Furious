import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np
import sklearn.metrics as metrics

import matplotlib.pyplot as plt

from steerDS import SteerDataSet
from model import Net
from loss import EMDLoss

#######################################################################################################################################
####     This tutorial is adapted from the PyTorch "Train a Classifier" tutorial                                                   ####
####     Please review here if you get stuck: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html                   ####
#######################################################################################################################################
torch.manual_seed(0)

#Helper function for visualising images in our dataset
def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    rgbimg = npimg[:,:,::-1]
    plt.imshow(rgbimg)
    plt.show()

#######################################################################################################################################
####     SETTING UP THE DATASET                                                                                                    ####
#######################################################################################################################################


script_path = os.path.dirname(os.path.realpath(__file__))

###################
## Train dataset ##
###################
folder_list = [
    # os.path.join(script_path, '..', 'data', 'train_starter'),
    os.path.join(script_path, '..', 'data', 'train_CW'),
]
# folder_list = [os.path.join(script_path, '..', 'data', 'train_tight_turns_CW'),
#                os.path.join(script_path, '..', 'data', 'train_tight_turns_CCW')]
            #    os.path.join(script_path, '..', 'data', 'train_starter'),
            #    os.path.join(script_path, '..', 'data', 'skipper')]
               #os.path.join(script_path, '..', 'data', 'slamtroopers' ,'05021550_left'),
               #os.path.join(script_path, '..', 'data', 'slamtroopers' ,'05021555_left'),
               #os.path.join(script_path, '..', 'data', 'slamtroopers' ,'05021558_right')]
train_ds = SteerDataSet(folder_list, '.jpg')
print("The train dataset contains %d images " % len(train_ds))

#data loader nicely batches images for the training process and shuffles (if desired)
trainloader = DataLoader(train_ds,batch_size=8,shuffle=True, drop_last=True)
all_y = []
for S in trainloader:
    im, y = S    
    all_y += y.tolist()

print(f'Input to network shape: {im.shape}')

#visualise the distribution of GT labels
all_lbls, all_counts = np.unique(all_y, return_counts = True)
plt.bar(all_lbls, all_counts, width = (all_lbls[1]-all_lbls[0])/2)
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.xticks(all_lbls)
plt.title('Training Dataset')
plt.show()

# visualise some images and print labels -- check these seem reasonable
example_ims, example_lbls = next(iter(trainloader))
print(' '.join(f'{example_lbls[j]}' for j in range(len(example_lbls))))
imshow(torchvision.utils.make_grid(example_ims))


########################
## Validation dataset ##
########################
val_folder_list = [#os.path.join(script_path, '..', 'data', 'slamtroopers' ,'0502160_right'),
                   #os.path.join(script_path, '..', 'data', 'slamtroopers' ,'05021604_left'),
                   #os.path.join(script_path, '..', 'data', 'slamtroopers' ,'04021605_left'),
                   os.path.join(script_path, '..', 'data', 'val_starter')]
val_ds = SteerDataSet(val_folder_list, '.jpg')
print("The train dataset contains %d images " % len(val_ds))

#data loader nicely batches images for the training process and shuffles (if desired)
valloader = DataLoader(val_ds,batch_size=1)
all_y = []
for S in valloader:
    im, y = S    
    all_y += y.tolist()

print(f'Input to network shape: {im.shape}')

#visualise the distribution of GT labels
all_lbls, all_counts = np.unique(all_y, return_counts = True)
plt.bar(all_lbls, all_counts, width = (all_lbls[1]-all_lbls[0])/2)
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.xticks(all_lbls)
plt.title('Validation Dataset')
plt.show()

#######################################################################################################################################
####     INITIALISE OUR NETWORK                                                                                                    ####
#######################################################################################################################################
net = Net()
# net.load_state_dict(torch.load("steer_net.pth",weights_only=True))


#######################################################################################################################################
####     INITIALISE OUR LOSS FUNCTION AND OPTIMISER                                                                                ####
#######################################################################################################################################

#for classification tasks
criterion1 = nn.CrossEntropyLoss()
criterion2 = EMDLoss()
#You could use also ADAM
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#######################################################################################################################################
####     TRAINING LOOP                                                                                                             ####
#######################################################################################################################################
losses = {'train': [], 'val': []}
accs = {'train': [], 'val': []}
best_acc = 0
for epoch in range(10):  # loop over the dataset multiple times

    epoch_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        probability = nn.functional.softmax(outputs, dim=1)
        loss1 = criterion1(outputs, labels.long())
        loss2 = criterion2(probability, labels.long())
        # print(f'{loss1=}')
        # print(f'{loss2=}')
        # print()
        # loss = loss1 + 0.2 * loss2
        loss = loss1
        loss.backward()
        optimizer.step()

        # print statistics
        epoch_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Epoch {epoch + 1} loss: {epoch_loss / len(trainloader)}')
    losses['train'] += [epoch_loss / len(trainloader)]
    accs['train'] += [100.*correct/total]
 
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in val_ds.class_labels}
    total_pred = {classname: 0 for classname in val_ds.class_labels}

    # again no gradients needed
    val_loss = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            loss1 = criterion1(outputs, labels.long())
            loss2 = criterion2(probability, labels.long())
            # print(f'{loss1=}')
            # print(f'{loss2=}')
            # print()
            loss = loss1
            # loss = loss1 + 1.5 * loss2

            val_loss += loss.item()
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[val_ds.class_labels[label.item()]] += 1
                total_pred[val_ds.class_labels[label.item()]] += 1

    # print accuracy for each class
    class_accs = []
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        class_accs += [accuracy]

    accs['val'] += [np.mean(class_accs)]
    losses['val'] += [val_loss/len(valloader)]

    if np.mean(class_accs) > best_acc:
        torch.save(net.state_dict(), 'steer_net.pth')
        best_acc = np.mean(class_accs)

print('Finished Training')

plt.plot(losses['train'], label = 'Training')
plt.plot(losses['val'], label = 'Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(accs['train'], label = 'Training')
plt.plot(accs['val'], label = 'Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


#######################################################################################################################################
####     PERFORMANCE EVALUATION                                                                                                    ####
#######################################################################################################################################
net.load_state_dict(torch.load('steer_net.pth'))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in valloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in val_ds.class_labels}
total_pred = {classname: 0 for classname in val_ds.class_labels}

# again no gradients needed
actual = []
predicted = []
with torch.no_grad():
    for data in valloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)

        actual += labels.tolist()
        predicted += predictions.tolist()

        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[val_ds.class_labels[label.item()]] += 1
            total_pred[val_ds.class_labels[label.item()]] += 1

cm = metrics.confusion_matrix(actual, predicted, normalize = 'true')
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=val_ds.class_labels)
disp.plot()
plt.show()

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f}%')
