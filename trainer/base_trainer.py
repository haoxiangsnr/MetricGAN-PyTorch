import time
from pathlib import Path

import json5
import numpy as np
import torch
import torch.nn as nn

from util import visualization
from util.others import prepare_empty_dir, ExecutionTime


class BaseTrainer:
    def __init__(self, config, resume: bool, G, D, optim_G, optim_D, loss_function):
        self.n_gpus = torch.cuda.device_count()
        self.device = self._prepare_device(self.n_gpus, cudnn_deterministic=config["cudnn_deterministic"])

        self.optimizer_G = optim_G
        self.optimizer_D = optim_D

        self.loss_function = loss_function
        self.loss_adversarial = nn.BCEWithLogitsLoss()

        self.generator = G.to(self.device)
        self.discriminator = D.to(self.device)

        if self.n_gpus > 1:
            self.generator = torch.nn.DataParallel(self.generator, device_ids=list(range(self.n_gpus)))
            self.discriminator = torch.nn.DataParallel(self.discriminator, device_ids=list(range(self.n_gpus)))

        # The configuration items of trainer
        self.epochs = config["trainer"]["epochs"]
        self.save_checkpoint_interval = config["trainer"]["save_checkpoint_interval"]
        self.validation_config = config["trainer"]["validation"]
        self.validation_interval = self.validation_config["interval"]
        self.find_max = self.validation_config["find_max"]
        self.validation_custom_config = self.validation_config["custom"]

        # The following configuration items are not in the config file，we will update it if resume is True in later
        self.start_epoch = 1
        self.best_score = -np.inf if self.find_max else np.inf
        self.root_dir = Path(config["root_dir"]).expanduser().absolute() / config["experiment_name"]
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.logs_dir = self.root_dir / "logs"
        prepare_empty_dir([self.checkpoints_dir, self.logs_dir], resume)

        # Visualization
        self.writer = visualization.writer(self.logs_dir.as_posix())
        self.writer.add_text(
            tag="Configuration",
            text_string=f"<pre>  \n{json5.dumps(config, indent=4, sort_keys=False)}  \n</pre>",
            global_step=1
        )

        with open((self.root_dir / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}.json").as_posix(), "w") as handle:
            json5.dump(config, handle, indent=2, sort_keys=False)

        if resume: self._resume_checkpoint()

        print("Configurations are as follows: ")
        print(json5.dumps(config, indent=2, sort_keys=False))

        self._print_networks([self.generator, self.discriminator])

    def _resume_checkpoint(self):
        """
        Resume experiment from latest checkpoint.

        Notes:
            To be careful at Loading model. if model is an instance of DataParallel, we need to set model.module.*
        """
        latest_model_path = self.checkpoints_dir.expanduser().absolute() / "latest_model.tar"
        assert latest_model_path.exists(), f"{latest_model_path} does not exist, can not load latest checkpoint."

        checkpoint = torch.load(latest_model_path.as_posix(), map_location=self.device)

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_D"])

        if isinstance(self.generator, torch.nn.DataParallel):
            self.generator.module.load_state_dict(checkpoint["generator"])
            self.discriminator.module.load_state_dict(checkpoint["discriminator"])
        else:
            self.generator.load_state_dict(checkpoint["generator"])
            self.discriminator.load_state_dict(checkpoint["discriminator"])

        print(f"Model checkpoint loaded. Training will begin in {self.start_epoch} epoch.")

    def _save_checkpoint(self, epoch, is_best=False):
        """
        Save checkpoint to <root_dir>/checkpoints directory, which contains:
            - current epoch
            - best score in the history
            - the parameters of the optimizers
            - the parameters of the models

        Args:
            is_best(bool):
                if current checkpoint got the best score,
                we will save checkpoint in <root_dir>/checkpoints/best_model.tar.
        """
        print(f"\t Saving {epoch} epoch model checkpoint...")

        # Construct checkpoint tar package
        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict()
        }

        # Parallel
        if isinstance(self.generator, torch.nn.DataParallel):
            state_dict["generator"] = self.generator.module.cpu().state_dict()
            state_dict["discriminator"] = self.discriminator.module.cpu().state_dict()
        else:
            state_dict["generator"] = self.generator.cpu().state_dict()
            state_dict["discriminator"] = self.discriminator.cpu().state_dict()

        """
        Notes:
            - latest_model.tar:
                Contains all checkpoint information, including optimizer parameters, model parameters, etc. 
                New checkpoint will overwrite old one.
            - generator_<epoch>.pth: 
                The parameters of the generator. Follow-up we can specify epoch to inference.
            - best_model.tar:
                The information like latest_model.tar, but only saved when <is_best> is True.
        """
        torch.save(state_dict, (self.checkpoints_dir / "latest_model.tar").as_posix())
        torch.save(state_dict["generator"], (self.checkpoints_dir / f"generator_{str(epoch).zfill(4)}.pth").as_posix())
        if is_best:
            print(f"\t Found best score in {epoch} epoch, saving...")
            torch.save(state_dict, (self.checkpoints_dir / "best_model.tar").as_posix())

        # model.cpu() or model.to("cpu") will migrate the model to CPU, at which point we need re-migrate it back.
        # No matter tensor.cuda() or tensor.to("cuda"), if tensor in CPU, the tensor will not be migrated to GPU.
        self.generator.to(self.device)
        self.discriminator.to(self.device)

    @staticmethod
    def _print_networks(nets: list):
        print(f"This project contains {len(nets)} networks, the number of the parameters: ")
        params_of_all_networks = 0
        for i, net in enumerate(nets, start=1):
            params_of_network = 0
            for param in net.parameters():
                params_of_network += param.numel()

            print(f"\tNetwork {i}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network

        print(f"The amount of parameters in the project is {params_of_all_networks / 1e6} million.")

    def _set_models_to_train_mode(self):
        self.generator.train()
        self.discriminator.train()

    def _set_models_to_eval_mode(self):
        self.generator.eval()
        self.discriminator.eval()

    @staticmethod
    def _prepare_device(n_gpus: int, cudnn_deterministic=False):
        """
        Choose to use CPU or GPU depend on "n_gpus".

        Args:
            n_gpus(int): the number of GPUs used in the experiment.
                if n_gpu == 0, use CPU;
                if n_gpu >= 1, use GPU.
            cudnn_deterministic (bool): repeatability
                cudnn.benchmark will find algorithms to optimize training. if we need to consider the repeatability
                of experiment, set use_cudnn_deterministic to True
        """
        if n_gpus == 0:
            print("Using CPU in the experiment.")
            device = torch.device("cpu")
        else:
            if cudnn_deterministic:
                print("Using deterministic mode of CuDNN in the experiment.")
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            device = torch.device("cuda:0")

        return device

    def _is_best(self, score, find_max=True):
        """
        Check if the current model is the best model
        """
        if find_max and score >= self.best_score:
            self.best_score = score
            return True
        elif not find_max and score <= self.best_score:
            self.best_score = score
            return True
        else:
            return False

    @staticmethod
    def _transform_pesq_range(pesq_score):
        """
        transform [-0.5 ~ 4.5] to [0 ~ 1]
        """
        return (pesq_score + 0.5) / 5

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"============== {epoch} epoch ==============")
            print("[0 seconds] Begin training...")
            timer = ExecutionTime()

            self._set_models_to_train_mode()
            self._train_epoch(epoch)

            if self.save_checkpoint_interval != 0 and (epoch % self.save_checkpoint_interval == 0):
                self._save_checkpoint(epoch)

            if self.validation_interval != 0 and epoch % self.validation_interval == 0:
                print(f"[{timer.duration()} seconds] Training is over, Validation is in progress...")

                self._set_models_to_eval_mode()
                score = self._validation_epoch(epoch)

                if self._is_best(score, find_max=self.find_max):
                    self._save_checkpoint(epoch, is_best=True)

            print(f"[{timer.duration()} seconds] End this epoch.")

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _validation_epoch(self, epoch):
        raise NotImplementedError
