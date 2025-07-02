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
import time
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from smdebug import modes
from smdebug.pytorch import get_hook
import smdebug.pytorch as smd

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))



#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, criterion, hook, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(modes.EVAL)
    running_loss=0.0
    running_corrects=0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            running_loss += float(loss.item() * inputs.size(0))
            running_corrects += float(torch.sum(preds == labels.data))

    total_loss = float(running_loss) / float(len(test_loader.dataset))
    total_acc = float(running_corrects) / float(len(test_loader.dataset))

    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")
    
    return all_preds, all_labels

def print_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    '''
    Print and optionally save confusion matrix with detailed metrics
    '''
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print confusion matrix as text
    logger.info("Confusion Matrix:")
    logger.info(f"\n{cm}")
    
    # Print classification report
    logger.info("Classification Report:")
    if class_names:
        report = classification_report(y_true, y_pred, target_names=class_names)
    else:
        report = classification_report(y_true, y_pred)
    logger.info(f"\n{report}")
    
    # Create and save visual confusion matrix if matplotlib is available
    try:
        plt.figure(figsize=(10, 8))
        
        if class_names:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}/confusion_matrix.png")
        
        plt.close()  # Close to free memory
        
    except Exception as e:
        logger.warning(f"Could not create visual confusion matrix: {e}")
    
    return cm


def train(model, train_loader, validation_loader, epochs, criterion, optimizer, scheduler, hook, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    best_loss = 1e6
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0
    # hook.set_mode(modes.TRAIN)

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase == 'train':
                model.train()
                hook.set_mode(modes.TRAIN)
            else:
                model.eval()
                hook.set_mode(modes.EVAL)
            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Add gradient clipping to prevent gradient explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
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
                # Update learning rate scheduler
                scheduler.step(epoch_loss)
                
                # Monitor per-class performance during validation
                from collections import defaultdict
                class_correct = defaultdict(int)
                class_total = defaultdict(int)
                
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validation_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        
                        for label, pred in zip(labels, preds):
                            class_total[label.item()] += 1
                            if label == pred:
                                class_correct[label.item()] += 1
                
                # Log per-class accuracy
                for class_id in range(5):
                    if class_total[class_id] > 0:
                        acc = class_correct[class_id] / class_total[class_id]
                        logger.info(f'Class {class_id+1} accuracy: {acc:.4f}')
                    else:
                        logger.info(f'Class {class_id+1} accuracy: No samples')
                
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    loss_counter = 0  # Reset counter when we improve
                else:
                    loss_counter += 1
                    
            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}, LR: {:.6f}'.format(
                phase, epoch_loss, epoch_acc, best_loss, optimizer.param_groups[0]['lr']))
                
        # More relaxed early stopping
        if loss_counter == 5:  # Changed from 3 to 5
            print("Finish training because epoch loss increased for 5 consecutive epochs")            
            break
    return model
    
def init_weights(m):
    '''
    Initialize weights properly to prevent gradient explosion
    '''
    
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    num_classes = 5
    # load the pretrained model
    model = models.resnet50(pretrained=True)
    
    # Only freeze early layers, allow later layers to adapt
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last residual block for better feature learning
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # find the number of inputs to the final layer of the network
    num_inputs = model.fc.in_features
    
    # Improved classifier with batch normalization and dropout
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
    
    # Initialize the new layers properly
    model.fc.apply(init_weights)
    
    return model

    
def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
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
    

def main(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device: {device}')
    
    # Log GPU information if available
    if device.type == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    logger.info(f'Hyperparameters are LR: {args.lr}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.data}')
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    model = model.to(device)  # Move model to device
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    
    # Get all trainable parameters (includes unfrozen layer4 + fc)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    # Use different learning rates for different parts
    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr': args.lr},
        {'params': model.layer4.parameters(), 'lr': args.lr * 0.1}  # Lower LR for pretrained layers
    ])
    
    # Add learning rate scheduler
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info("create the SMDebug hook and register to the model.")
    
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(loss_criterion)   
    
    train_loader, test_loader, validation_loader = create_data_loaders(args.data,
                                                                       args.batch_size)

    logger.info("Training the model")
    
    model = train(model, train_loader, validation_loader, args.epochs, loss_criterion, optimizer, scheduler, hook, device)
  
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Testing the model")
    
    # Get predictions and true labels for confusion matrix
    test_preds, test_labels = test(model, test_loader, loss_criterion, hook, device)

    class_names = ["1", "2", "3", "4", "5"]
    # Print confusion matrix
    logger.info("Generating confusion matrix...")
    confusion_matrix_result = print_confusion_matrix(
        test_labels, 
        test_preds, 
        class_names=class_names,
        save_path=args.output_dir
    )
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving the model")
    
    # Move model to CPU before saving to ensure compatibility
    model = model.cpu()
    path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    # epoch
    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )
    parser.add_argument('--lr',
                        type=float,
                        default=0.001)  
    parser.add_argument('--batch-size',
                        type=int,
                        default=32)

    parser.add_argument('--data', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir',
                        type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir',
                        type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()
    print(args)

    main(args)