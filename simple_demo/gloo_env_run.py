#!/usr/bin/env python
import os
import torch
import torch as th
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time

def allreduce(send, recv):
    """ Implementation of a ring-reduce. """
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = th.zeros(send.size())
    recv_buff = th.zeros(send.size())
    accum = th.zeros(send.size())
    accum[:] = send[:]
    # th.cuda.synchronize()

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size

    for i in range(size - 1):
        if i % 2 == 0:
            # Send send_buff
            send_req = dist.isend(send_buff, right)
            dist.recv(recv_buff, left)
            accum[:] += recv[:]
        else:
            # Send recv_buff
            send_req = dist.isend(recv_buff, right)
            dist.recv(send_buff, left)
            accum[:] += send[:]
        send_req.wait()
    # th.cuda.synchronize()
    recv[:] = accum[:]


def run(rank, size):
    """ Distributed function to be implemented later. """
    model = Model()
    model = torch.nn.parallel.DistributedDataParallel(model.cuda())
    criterion = torch.nn.MSELoss(size_average=False).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    cudnn.benchmark = True
    x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
    y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
    # train_sampler = torch.utils.data.distributed.DistributedSampler(x_data)

    for epoch in range(100):
        # train_sampler.set_epoch(epoch)

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

    hour_var = Variable(torch.Tensor([[7.0]]))
    print("predict (after training)", 7, model.forward(hour_var).data[0][0])




def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '192.168.0.12'
    os.environ['MASTER_PORT'] = '29555'
    dist.init_process_group(backend, rank=rank, world_size=size)
    torch.cuda.manual_seed(1)
    fn(rank, size)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        # One in and one out
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


def main():

    size = 4
    processes=[]
    for i in range(size):
        p = Process(target=init_processes, args=(i, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    start_time = time.time()
    main()

    end_time = time.time()
    print("耗时：", end_time-start_time)
