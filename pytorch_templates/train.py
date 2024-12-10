import torch
from utils.datautils import prepare_dataloader
from utils.trainingutils import ddp_setup, Trainer, load_train_objs
from utils.input_parser import parse_input_args

from torch.distributed import destroy_process_group
import torch.multiprocessing as mp

import sys


def main(args, rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    args = parse_input_args(sys.argv[1:])
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(args, world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)