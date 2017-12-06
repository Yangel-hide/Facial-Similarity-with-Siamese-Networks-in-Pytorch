import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.optim.lr_scheduler as lrscheduler
import torch.nn.functional as F


#Set of helper functions
def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


#A simple class to manage configuration
class Config():
    #training_dir = "./att_faces/"
    #testing_dir = "./data/faces/testing/"
    training_dir = "./copydata/train/"
    testing_dir = "./copydata/test/"
    train_batch_size = 64
    train_number_epochs = 500

#This dataset generates a pair of images. 0 for geniune pair and 1 for imposter pair
class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        image_index = random.randrange(0, len(self.imageFolderDataset.imgs)-2)
        img0_tuple = self.imageFolderDataset.imgs[image_index]
        #img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            #img1_tuple = self.imageFolderDataset.imgs[image_index+1]
            while True:
                inc = random.randrange(-3, 3)
                img1_tuple = self.imageFolderDataset.imgs[image_index + inc]
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")
        #
        # if self.should_invert:
        #     img0 = PIL.ImageOps.invert(img0)
        #     img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

#Using Image Folder Dataset
folder_dataset = dset.ImageFolder(root=Config.training_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.RandomSizedCrop(299),
                                                                      transforms.RandomHorizontalFlip(),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

#The top row and the bottom row of any column is one pair.
#The 0s and 1s correspond to the column of the image. 0 indiciates dissimilar, and 1 indicates similar.
vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=8)
dataiter = iter(vis_dataloader)


example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())

#Neural Net Definition
#We will use a standard convolutional neural network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.model = models.Inception3()
        self.model.load_state_dict(torch.load('models/inception_v3_google.pth'))
        #self.model_inceptionv3 = models.inception_v3(pretrained=True)
        #num_ftrs = self.model_inceptionv3.fc.in_features
        #self.model_inceptionv3.fc = nn.Linear(num_ftrs, 1024)


    def forward_once(self, x):
        #output = self.model_inceptionv3(x)
        output = self.model(x)
        # output = self.cnn1(x)
        # output = output.view(output.size()[0], -1)
        # output = self.fc1(output)
        return output[0]

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


#Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

#Training Time!
train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=Config.train_batch_size)

net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
lr_scheduler = lrscheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.4)

counter = []
loss_history = []
iteration_number= 0

for epoch in range(0,Config.train_number_epochs):
    lr_scheduler.step()
    for i, data in enumerate(train_dataloader,0):
        img0, img1 , label = data
        img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
        output1,output2 = net(img0,img1)
        optimizer.zero_grad()
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.data[0]))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.data[0])
show_plot(counter,loss_history)

#Some simple testing
#The last 3 subjects were held out from the training, and will be used to test.
#The Distance between each image pair denotes the degree of similarity the model found between the two images.
# Less means it found more similar, while higher values indicate it found them to be dissimilar.
folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Scale((100, 100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                        , should_invert=False)

test_dataloader = DataLoader(siamese_dataset, num_workers=6, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)
x0, _, _ = next(dataiter)

for i in range(10):
    _, x1, label2 = next(dataiter)
    concatenated = torch.cat((x0, x1), 0)

    output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated),
           'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]))