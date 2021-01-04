import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# network structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.FC1 = nn.Linear(1000, 100)
        self.FC2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.sigmoid(self.FC1(x))
        x = self.FC2(x)
        return x

def calculate_accuracy(y_pred, y_true):
    train_acc = torch.sum(torch.argmax(y_pred,1) == torch.argmax(y_true,1))*0.1/batch_size
    return train_acc

# parameters
batch_size=64
learning_rate=0.01
training_epochs=500
input_data_dimension=1000
output_data_dimension=10

# define the training log file
# build a folder to save the summaries
writer = SummaryWriter('logs/MNIST_TEST')

# define input data and labels
train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('data/', train=True, download=True,
transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

model = Net()
model= model.to('cuda') #convert the model to GPU
# define MSE loss function
criterion=nn.MSELoss() 
# define optimizer
optimizer = optim.SGD(model.parameters(),lr=learning_rate, momentum = 0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

print("start training ----")
for i in range(training_epochs):
    # zeroes the gradient buffers of all parameters
    optimizer.zero_grad()
    # calculate the prediction results
    y_pred = model(x)
    # calculate MSE loss
    loss = criterion(y_pred, y)
    loss.backward()
    # Perform the training parameters update
    optimizer.step()
    scheduler.step()
    if(i%10==0):
        # calculate the accuracy
        training_accu=calculate_accuracy(y_pred,y)
        print("iter: {}, loss: {},accuracy:{},learning_rate: {}".format(i,loss,training_accu, scheduler._get_closed_form_lr()))
        writer.add_scalar('Loss/train', loss, i)
        writer.add_scalar('Accuracy/train', training_accu, i)
        writer.add_scalar('Learning_rate/train',learning_rate,i)

print("training finished.")
