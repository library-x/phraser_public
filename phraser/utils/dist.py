import os

import torch


def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in ( 'LOCAL_RANK',):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ( 'RANK',):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ( 'WORLD_SIZE',):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size



def setup_distributed_device(args):
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    if is_using_distributed():
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

        # os.environ['MASTER_ADDR'] = '127.0.0.1'
        # os.environ['MASTER_PORT'] = '5000'

        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        args.distributed = True

        print(f"Distributed training: local_rank={args.local_rank}, "
              f"rank={args.rank}, world_size={args.world_size}, "
              f"pid={os.getpid()}")

    args.is_master = args.local_rank == 0

    if torch.cuda.is_available():
        if args.distributed:
            device = 'cuda:%d' % args.local_rank
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    args.device = device

    return device