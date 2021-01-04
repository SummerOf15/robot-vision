import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

# parameters
batch_size=64
learning_rate=0.01
training_epochs=500
input_data_dimension=1000
output_data_dimension=10

# define input data and labels
dtype = torch.float
device = torch.device("cuda")
x = torch.randn(batch_size, input_data_dimension,  device=device,dtype=dtype)
y = torch.randn(batch_size, output_data_dimension, device=device, dtype=dtype)

model = Net()
model= model.to('cuda') #convert the model to GPU
# define MSE loss function
criterion=nn.MSELoss() 
# define optimizer
optimizer = optim.SGD(model.parameters(),lr=learning_rate, weight_decay= 1e-3, momentum = 0.9)


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
    if(i%10==0):
        print("iter: {}, loss: {}".format(i,loss))

print("training finished.")
