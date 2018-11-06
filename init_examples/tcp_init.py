import torch
import torch.distributed as dist
import argparse
from time import sleep
from random import randint
from torch.multiprocessing import Process


def initialize(rank, world_size, ip, port):
    dist.init_process_group(backend='tcp', init_method='tcp://{}:{}'.format(ip, port), rank=rank, world_size=world_size)
    print("MM")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='192.168.0.12')
    parser.add_argument('--port', type=str, default='20000')
    parser.add_argument('--rank', '-r', type=int)
    parser.add_argument('--world-size', '-s', type=int)
    args = parser.parse_args()
    print(args)
    # initialize(args.rank, args.world_size, args.ip, args.port)

    size = 2
    processes = []
    for i in range(size):
        p = Process(target=initialize, args=(i, size, args.ip, args.port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()