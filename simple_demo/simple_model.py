import torch
from torch.autograd import Variable
import time
start_time=time.time()
# train data
x_data = Variable(torch.randn(1)*10)
y_data = Variable(x_data*2)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        # One in and one out
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    # our model
model = Model().cuda()

criterion = torch.nn.MSELoss(size_average=False)
# Defined loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Defined optimizer
#  Training: forward, loss, backward, step
#  Training loop
for epoch in range(1000):
    #  Forward pass
    y_pred = model(x_data.cuda())
    # Compute loss
    loss = criterion(y_pred.cuda(), y_data.cuda())
    print(epoch, loss.data[0])
    # Zero gradients
    optimizer.zero_grad()
    # perform backward pass
    loss.backward()
    # update weights
    optimizer.step()


# After training
hour_var = Variable(torch.Tensor([[7.0]]))
print("predict (after training)", 7, model.forward(hour_var.cuda()).data[0][0])
end_time=time.time()
print("耗时：　", end_time-start_time)
