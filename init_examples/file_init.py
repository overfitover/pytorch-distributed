import torch
import torch.distributed as dist
import argparse
from time import sleep
from random import randint
from torch.multiprocessing import Process


def initialize(rank, world_size):
    dist.init_process_group(backend='gloo', init_method='file:///home/yxk/share', rank=rank, world_size=world_size)
    print("MM")

def main():

    size = 2
    processes = []
    for i in range(size):
        p = Process(target=initialize, args=(i, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()