#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import logging
import os
import sys

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    running_loss=0
    running_corrects=0
    
    with torch.no_grad():  # Add no_grad for efficiency during testing
        for inputs, labels in test_loader:
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")
    # print(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")
    
#------------------------------------------------------------------------------------------


def train(model, train_loader, validation_loader, epochs, criterion, optimizer, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    best_loss = 1e6
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                # Move data to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples += len(inputs)
                if running_samples % 2000 == 0:
                    accuracy = running_corrects / running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0 * accuracy,
                        )
                    )

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1
            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase,epoch_loss,
                                                                                 epoch_acc,best_loss))
            
        if loss_counter==3:
            print("Finish training because epoch loss increased")            
            break
            
    return model

    
#------------------------------------------------------------------------------------------
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    num_classes = 5
    # load the pretrained model
    model = models.resnet50(pretrained=True)
    
    # # freeze the different parameters of the model to use for feature extraction
    # for param in model.parameters():
    #     param.requires_grad = False
        
    # # find the number of inputs to the final layer of the network
    # num_inputs = model.fc.in_features
    # model.fc = nn.Sequential(
    #                nn.Linear(num_inputs, 128),
    #                nn.ReLU(inplace=True),
    #                nn.Linear(128, num_classes))
        # Freeze early layers but allow fine-tuning of later layers
    # This is better than freezing all parameters
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    
    # Improved classifier head
    num_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_inputs, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    
    return model

#---------------------------------------------------------------------------------------------
def create_data_loaders(data , batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    # creating the dataloaders
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path = os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_data = torchvision.datasets.ImageFolder(
        root=train_data_path,
        transform=train_transform
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    test_data = torchvision.datasets.ImageFolder(
        root=test_data_path,
        transform=test_transform
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    validation_data = torchvision.datasets.ImageFolder(
        root=validation_data_path,
        transform=test_transform,
    )
    validation_data_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_data_loader, test_data_loader, validation_data_loader
    
#----------------------------------------------------------------------------------------------
def main(args):
    # Setup device - use GPU if available, otherwise CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
    
    # logging the used hyperparameters
    logger.info(f'Hyperparameters are LR: {args.lr}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.data}')
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    # Move model to device
    model = model.to(device)
    
    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters() if not isinstance(model, nn.DataParallel) 
                          else model.module.fc.parameters(), args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    # loading the data
    train_loader, test_loader, validation_loader = create_data_loaders(args.data,
                                                                       args.batch_size)

    logger.info("Training the model")
    
    model = train(model, train_loader, validation_loader, args.epochs, loss_criterion, optimizer, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Testing the model")
    
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving the model")
    
    path = os.path.join(args.model_dir, 'model.pth')
    # Handle DataParallel case when saving
    model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(model_state_dict, path)
    
#-------------------------------------------------------------------------------------------------------
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    # Training settings
    # epoch
    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )
    # batch_size
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )
    # lr
    parser.add_argument(
        "--lr", 
        type=float,
        default=0.001
    )

    # Container environment
    parser.add_argument('--data', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir',
                        type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir',
                        type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])

    args=parser.parse_args()
    
    main(args)