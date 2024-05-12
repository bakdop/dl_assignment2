from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch
from torch.utils.data import DataLoader,Subset

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from utils import AverageMeter, accuracy
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def train(model, data_loader, optimizer, criterion, device, config):
    # TODO set model to train mode
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    model.to(device)
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        # Add more code here ...
        optimizer.zero_grad()
        batch_output = model.forward(batch_inputs)
        
        # tensor = torch.eye(10).to(device)
        # batch_targets = tensor[batch_targets]
        loss = criterion(batch_output,batch_targets)
        loss.backward()        
        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.max_norm)
        optimizer.step()
        with torch.no_grad():
            total = batch_output.size(0)
            prediction = torch.argmax(batch_output,1)
            correct = (prediction==batch_targets).sum().item()
            accuracy = correct/total
            accuracies.update(accuracy)
            losses.update(loss.item())
            # Add more code here ...
            if step % 100 == 0:
                print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    
    return losses.avg, accuracies.avg


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, config):
    # TODO set model to evaluation mode
    with torch.no_grad():
        losses = AverageMeter("Loss")
        accuracies = AverageMeter("Accuracy")
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            # Add more code here ...
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)     
            batch_output = model.forward(batch_inputs)
        
            loss = criterion(batch_output,batch_targets)
            losses.update(loss)
            total = batch_inputs.size(0)
            prediction = torch.argmax(batch_output,1)
            correct = (prediction==batch_targets).sum().item()
            accuracy = correct/total
            accuracies.update(accuracy)
            if step % 100 == 0:
                print(f'[{step}/{len(data_loader)}]', losses, accuracies)
        print(f"losses.avg: {losses.avg}")
    
    return losses.avg.item(), accuracies.avg


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_length = int(config.input_length)
    input_dim = int(config.input_dim)
    num_classes = int(config.num_classes)
    num_hidden = int(config.num_hidden)
    batch_size = int(config.batch_size)
    learning_rate = float(config.learning_rate)
    max_epoch = int(config.max_epoch)
    max_norm = float(config.max_norm)
    data_size =int(config.data_size)
    portion_train = float(config.portion_train)
    # Initialize the model that we are going to use
    print(f"data_size: {data_size}")
    model = VanillaRNN(input_length,input_dim,num_hidden,num_classes)  # fixme
    model.to(device)

    # Initialize the dataset and data loader
    dataset =  PalindromeDataset(input_length=input_length,total_len=data_size) # fixme

    # Split dataset into train and validation sets
    train_indice, test_indice = train_test_split(range(len(dataset)),test_size=1-portion_train)
    
    train_dataset = Subset(dataset,train_indice)
    val_dataset = Subset(dataset,test_indice)  # fixme
    # Create data loaders for training and validation
    train_dloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)  # fixme
    val_dloader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False)  # fixme
    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()  # fixme
    optimizer =  optim.RMSprop(model.parameters(),learning_rate) # fixme
    scheduler = ...  # fixme


    #visualization
    record = []
    for epoch in range(max_epoch):
        # Train the model for one epoch
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f"epoch: {epoch}")
    
        train_loss, train_acc = train(
            model, train_dloader, optimizer, criterion, device, config)
        
        # Evaluate the trained model on the validation set
        val_loss, val_acc = evaluate(
            model, val_dloader, criterion, device, config)
        record.append([train_loss, train_acc, val_loss, val_acc])

  
    torch.save(record,'train_result.pt')

    print(f'final train loss: {train_loss}')
    print(f'final train accuracy: {train_acc}')
    print(f'final validation loss: {val_loss}')
    print(f'final validation accuracy: {val_acc}')
    print('Done training.')


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=19,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='Learning rate')
    parser.add_argument('--max_epoch', type=int,
                        default=100, help='Number of epochs to run for')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--data_size', type=int,
                        default=10000, help='Size of the total dataset')
    parser.add_argument('--portion_train', type=float, default=0.8,
                        help='Portion of the total dataset used for training')

    config = parser.parse_args()
    # Train the model
    main(config)
