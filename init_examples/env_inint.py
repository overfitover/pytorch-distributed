#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import time

def run(rank, size):
    pass


def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '192.168.0.12'
    os.environ['MASTER_PORT'] = '29555'
    dist.init_process_group(backend, rank=rank, world_size=size)
    torch.cuda.manual_seed(1)
    fn(rank, size)
    print("MM")
    print(dist.get_rank())
    print(dist.get_world_size())
    print(dist.is_available())


def main():

    size = 2
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
