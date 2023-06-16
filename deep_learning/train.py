from dataset import get_dataloaders
import csv
import torch.optim as optim
import torch
from torchvision import models
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_epoch(model: nn.Module, optimizer, criterion, scheduler, train_loader):
    model.train()
    for data in train_loader:
        inputs = data['image'].to(device)
        labels = data['label'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    scheduler.step()

def test_accuracy(model: nn.Module, test_loader):
    with torch.no_grad():
        correct = 0
        for data in test_loader:
            inputs = data['image'].to(device)
            labels = data['labels'].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

    return correct / len(test_loader)


def load_network(number_of_classes):
    model_ft = models.resnet34(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, number_of_classes)
    model_ft = model_ft.to(device)
    return model_ft

def run(number_of_epochs=100, test_percentage=0.2, batch_size=4):
    train_loader, test_loader, number_of_classes = get_dataloaders(test_percentage=test_percentage, batch_size=batch_size)
    model = load_network(number_of_classes=number_of_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    with open('log.txt', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'test_accuracy'])
        writer.writeheader()
        for epoch in tqdm(range(number_of_epochs)):
            train_epoch(model=model, criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, train_loader=train_loader)
            accuracy = test_accuracy(model=model, test_loader=test_loader)
            writer.writerow({'epoch': epoch, 'test_accuracy': accuracy})


if __name__ == '__main__':
    run()
