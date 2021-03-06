import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os

# parameters
batch_size_train=64
batch_size_test=64
learning_rate=0.01
training_epochs=100
n_classes=10

model_save_dir="model/"
os.makedirs(model_save_dir,exist_ok=True)
model_save_path = model_save_dir+'mnist_resnet18.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    # build a folder to save the summaries
    writer = SummaryWriter('logs/MNIST_TEST')

    # define input data and labels
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                     torchvision.transforms.Resize((7,7),2),
                                     torchvision.transforms.Grayscale(3),
                                     torchvision.transforms.ToTensor(),
                                     torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                 ])),
        batch_size=batch_size_train, shuffle=True)

    model = torchvision.models.resnet18(pretrained=False, num_classes=n_classes)
    model= model.to('cuda') # convert the model to GPU

    # define MSE loss function
    criterion=nn.CrossEntropyLoss()
    # define optimizer
    optimizer = optim.SGD(model.parameters(),lr=learning_rate, momentum = 0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    print("start training ----")
    for n_iter in range(training_epochs):
        # training middle results
        loss_list = []

        for x, y_true in train_loader:
            # zeroes the gradient buffers of all parameters
            x=x.to(device)
            y_true=y_true.to(device)
            optimizer.zero_grad()
            # calculate the prediction results
            y_pred = model(x)
            # calculate MSE loss
            loss = criterion(y_pred, y_true)
            loss_list.append(loss)
            loss.backward()
            # Perform the training parameters update
            optimizer.step()

        # update the learning rate
        scheduler.step()
        # calculate the mean training loss
        mean_loss=sum(loss_list)/len(loss_list)

        if(n_iter%5==0):
            print("iter: {}, loss: {},learning_rate: {}".format(n_iter,mean_loss, scheduler._get_closed_form_lr()))
            writer.add_scalar('Loss/train', mean_loss, n_iter)
            writer.add_scalar('Learning_rate/train',learning_rate,n_iter)
        if(n_iter%10==9):
            print('------testing--------')
            accuracy=test(model)
            print('-----iter:{}, accuracy:{}------'.format(n_iter,100 * accuracy))
            writer.add_scalar('Accuracy/train', accuracy, n_iter)
            torch.save(model.state_dict(), model_save_path)
            print("model saved at {}".format(model_save_path))
    print("training finished.")


def test(model=None):
    # load testing dataset
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize((7, 7), 2),
                                       torchvision.transforms.Grayscale(3),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)
    # define the model and load pre-trained weights
    if model is None:
        model = torchvision.models.resnet18(pretrained=False, num_classes=n_classes)
        model.load_state_dict(torch.load(model_save_path))
    # convert to GPU
    model.to("cuda")

    # count the correct predicted and total number of images
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y_true in test_loader:
            x=x.to(device)
            y_true=y_true.to(device)
            outputs = model(x)
            _, y_pred = torch.max(outputs.data, 1)
            total += y_true.size(0)
            correct += (y_pred == y_true).sum().item()

    accuracy=correct / total
    return accuracy


if __name__ == "__main__":
    train()
