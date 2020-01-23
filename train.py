import argparse
import os
import random

import json5
import numpy as np
import torch
from torch.utils.data import DataLoader
from trainer.trainer import Trainer
from torch.nn.utils.rnn import pad_sequence

from util.others import initialize_config


def main(config, resume):
    torch.manual_seed(config["seed"])  # For both GPU and CPU
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    train_dataset = initialize_config(config["train_dataset"])
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        shuffle=config["train_dataloader"]["shuffle"],
        pin_memory=config["train_dataloader"]["pin_memory"],
        collate_fn=train_dataset.pad_batch
    )

    validation_dataset = initialize_config(config["validation_dataset"])
    valid_data_loader = DataLoader(
        dataset=validation_dataset,
        num_workers=1,
        batch_size=1
    )

    generator = initialize_config(config["generator_model"])
    discriminator = initialize_config(config["discriminator_model"])

    generator_optimizer = torch.optim.Adam(
        params=generator.parameters(),
        lr=config["optimizer"]["G_lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )
    discriminator_optimizer = torch.optim.Adam(
        params=discriminator.parameters(),
        lr=config["optimizer"]["D_lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )

    loss_function = initialize_config(config["loss_function"])

    trainer = Trainer(
        config=config,
        resume=resume,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        loss_function=loss_function,
        train_dl=train_data_loader,
        validation_dl=valid_data_loader,
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MetricGAN (PyTorch version)')
    parser.add_argument("-C", "--configuration", required=True, type=str,
                        help="Specify the configuration file for training (*.json).")
    parser.add_argument("-R", "--resume", action="store_true",
                        help="Whether to resume training from a recent breakpoint.")
    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))
    configuration["experiment_name"] = os.path.splitext(os.path.basename(args.configuration))[0]
    configuration["config_path"] = args.configuration

    main(configuration, resume=args.resume)
