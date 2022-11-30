
import torch
from torchvision import transforms, datasets, models

import json

from torch import nn

from torch import optim


def data_mani(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    val_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform=val_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=64)
    
    return trainloader,valloader,train_data
      
    
def train(trainloader, valloader, lr, hu, epochs, gpu, arch):

    device = torch.device("cuda" if gpu else "cpu")
    if arch == 'densenet161':
        model = models.densenet161(pretrained=True)       
    elif arch == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(2208, int(hu)),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(int(hu), 102),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), float(lr))
    model.to(device);

    steps = 0
    running_loss = 0
    print_every = 5
    for i in range(int(epochs)):
        for inputs, labels in trainloader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)

            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)

                        test_loss += batch_loss.item()

                        ps = torch.exp(log_ps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                print(f"Epoch {i+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(valloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valloader)*100:.3f}")
               
                running_loss = 0
                accuracy = 0
                model.train()

    return model

def save(pth, model, train_data):
    checkpoint = {'classifier' : model.classifier,
                  'class_to_idx' : train_data.class_to_idx,
                  'state_dict': model.state_dict(),
                  }

    torch.save(checkpoint, pth)
    print('model saved')




