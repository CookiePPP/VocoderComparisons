#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Train Parallel WaveGAN."""

import argparse
import logging
import os
import sys

from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch
import yaml

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob

import parallel_wavegan
import parallel_wavegan.models
import parallel_wavegan.optimizers

from parallel_wavegan.layers import PQMF
from parallel_wavegan.losses import MultiResolutionSTFTLoss
from parallel_wavegan.utils import read_hdf5
from parallel_wavegan.dataset import from_path

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


class Trainer(object):
    """Customized trainer module for Parallel WaveGAN training."""
    
    def __init__(self,
                 steps,
                 epochs,
                 data_loader,
                 sampler,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 config,
                 device=torch.device("cpu"),
                 ):
        """Initialize trainer.
        
        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.
        
        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["checkpoint_path"], max_queue=10000)
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(initial=self.steps,
                         total=self.config["train_max_steps"],
                         desc="[train]")
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        
        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.config["distributed"]:
            state_dict["model"] = {
                "generator": self.model["generator"].module.state_dict(),
                "discriminator": self.model["discriminator"].module.state_dict(),
            }
        else:
            state_dict["model"] = {
                "generator": self.model["generator"].state_dict(),
                "discriminator": self.model["discriminator"].state_dict(),
            }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)
        
        cp_list = sorted(glob(os.path.join(self.config["checkpoint_path"], "checkpoint-????????steps.pkl")))
        # delete old checkpoints
        if len(cp_list) > self.config["n_models_to_keep"]:
            for cp in cp_list[:-self.config["n_models_to_keep"]]:
                open(cp, 'w').close()
                os.unlink(cp)

    def load_checkpoint(self, checkpoint_path=None, load_only_params=False):
        """Load checkpoint.
        
        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.
        
        """
        cp_list = sorted(glob(f'{self.config["checkpoint_path"]}/checkpoint-????????steps.pkl'))
        if checkpoint_path is None and len(cp_list):
            checkpoint_path = cp_list[-1]
        
        if checkpoint_path is None:
            print("No checkpoints found. Skipping"); return
        else:
            print(f"Loading model from '{checkpoint_path}'")
        
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model["generator"].module.load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].module.load_state_dict(state_dict["model"]["discriminator"])
        else:
            self.model["generator"].load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].load_state_dict(state_dict["model"]["discriminator"])
        if not load_only_params:
            if "steps" in state_dict.keys():
                self.steps = state_dict["steps"]
            if "epochs" in state_dict.keys():
                self.epochs = state_dict["epochs"]
            if "optimizer" in state_dict.keys():
                self.optimizer["generator"].load_state_dict(state_dict["optimizer"]["generator"])
                self.optimizer["discriminator"].load_state_dict(state_dict["optimizer"]["discriminator"])
            if "scheduler" in state_dict.keys():
                self.scheduler["generator"].load_state_dict(state_dict["scheduler"]["generator"])
                self.scheduler["discriminator"].load_state_dict(state_dict["scheduler"]["discriminator"])

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        batch = {k: (v.to(self.device) if type(v) == torch.Tensor else v) for k, v in batch.items()}# move Tensors to training device/accelerator
        y = batch["audio"]
        
        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](batch["spect"], x=batch["noise"] if "noise" in batch else None)
        
        # reconstruct the signal from multi-band signal
        if self.config["generator_params"]["out_channels"] > 1:
            y_mb_ = y_
            y_ = self.criterion["pqmf"].synthesis(y_mb_)
        
        # multi-resolution sfft loss
        sc_loss, mag_loss = self.criterion["stft"](y_.squeeze(1), y.squeeze(1))
        self.total_train_loss["train/spectral_convergence_loss"] += sc_loss.item()
        self.total_train_loss["train/log_stft_magnitude_loss"] += mag_loss.item()
        gen_loss = sc_loss + mag_loss

        # subband multi-resolution stft loss
        if self.config.get("use_subband_stft_loss", False):
            gen_loss *= 0.5  # for balancing with subband stft loss
            y_mb = self.criterion["pqmf"].analysis(y)
            y_mb = y_mb.view(-1, y_mb.size(2))  # (B, C, T) -> (B x C, T)
            y_mb_ = y_mb_.view(-1, y_mb_.size(2))  # (B, C, T) -> (B x C, T)
            sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
            self.total_train_loss[
                "train/sub_spectral_convergence_loss"] += sub_sc_loss.item()
            self.total_train_loss[
                "train/sub_log_stft_magnitude_loss"] += sub_mag_loss.item()
            gen_loss += 0.5 * (sub_sc_loss + sub_mag_loss)

        # adversarial loss
        if self.steps > self.config["discriminator_train_start_steps"]:
            p_ = self.model["discriminator"](y_)
            if not isinstance(p_, list):
                # for standard discriminator
                adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
                self.total_train_loss["train/adversarial_loss"] += adv_loss.item()
            else:
                # for multi-scale discriminator
                adv_loss = 0.0
                for i in range(len(p_)):
                    adv_loss += self.criterion["mse"](
                        p_[i][-1], p_[i][-1].new_ones(p_[i][-1].size()))
                adv_loss /= (i + 1)
                self.total_train_loss["train/adversarial_loss"] += adv_loss.item()

                # feature matching loss
                if self.config["use_feat_match_loss"]:
                    # no need to track gradients
                    with torch.no_grad():
                        p = self.model["discriminator"](y)
                    fm_loss = 0.0
                    for i in range(len(p_)):
                        for j in range(len(p_[i]) - 1):
                            fm_loss += self.criterion["l1"](p_[i][j], p[i][j].detach())
                    fm_loss /= (i + 1) * (j + 1)
                    self.total_train_loss["train/feature_matching_loss"] += fm_loss.item()
                    adv_loss += self.config["lambda_feat_match"] * fm_loss

            # add adversarial loss to generator loss
            gen_loss += self.config["lambda_adv"] * adv_loss

        self.total_train_loss["train/generator_loss"] += gen_loss.item()

        # update generator
        self.optimizer["generator"].zero_grad()
        gen_loss.backward()
        if self.config["generator_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.config["generator_grad_norm"])
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()

        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"]:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                y_ = self.model["generator"](batch["spect"], x=batch["noise"] if "noise" in batch else None)
            if self.config["generator_params"]["out_channels"] > 1:
                y_ = self.criterion["pqmf"].synthesis(y_)
            
            # discriminator loss
            p = self.model["discriminator"](batch["audio"])
            p_ = self.model["discriminator"](y_.detach())
            if not isinstance(p, list):
                # for standard discriminator
                real_loss = self.criterion["mse"](p, p.new_ones(p.size()))
                fake_loss = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
                dis_loss = real_loss + fake_loss
            else:
                # for multi-scale discriminator
                real_loss = 0.0
                fake_loss = 0.0
                for i in range(len(p)):
                    real_loss += self.criterion["mse"](
                        p[i][-1], p[i][-1].new_ones(p[i][-1].size()))
                    fake_loss += self.criterion["mse"](
                        p_[i][-1], p_[i][-1].new_zeros(p_[i][-1].size()))
                real_loss /= (i + 1)
                fake_loss /= (i + 1)
                dis_loss = real_loss + fake_loss
            
            self.total_train_loss["train/real_loss"] += real_loss.item()
            self.total_train_loss["train/fake_loss"] += fake_loss.item()
            self.total_train_loss["train/discriminator_loss"] += dis_loss.item()
            
            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config["discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config["discriminator_grad_norm"])
            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
                     f"({self.train_steps_per_epoch} steps per epoch).")

        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        batch = {k: (v.to(self.device) if type(v) == torch.Tensor else v) for k, v in batch.items()}# move Tensors to training device/accelerator
        y = batch["audio"]
        
        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](batch["spect"], x=batch["noise"] if "noise" in batch else None)
        if self.config["generator_params"]["out_channels"] > 1:
            y_mb_ = y_
            y_ = self.criterion["pqmf"].synthesis(y_mb_)
        
        # multi-resolution stft loss
        sc_loss, mag_loss = self.criterion["stft"](y_.squeeze(1), y.squeeze(1))
        aux_loss = sc_loss + mag_loss

        # subband multi-resolution stft loss
        if self.config.get("use_subband_stft_loss", False):
            aux_loss *= 0.5  # for balancing with subband stft loss
            y_mb = self.criterion["pqmf"].analysis(y)
            y_mb = y_mb.view(-1, y_mb.size(2))  # (B, C, T) -> (B x C, T)
            y_mb_ = y_mb_.view(-1, y_mb_.size(2))  # (B, C, T) -> (B x C, T)
            sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
            self.total_eval_loss[
                "eval/sub_spectral_convergence_loss"] += sub_sc_loss.item()
            self.total_eval_loss[
                "eval/sub_log_stft_magnitude_loss"] += sub_mag_loss.item()
            aux_loss += 0.5 * (sub_sc_loss + sub_mag_loss)

        # adversarial loss
        p_ = self.model["discriminator"](y_)
        if not isinstance(p_, list):
            # for standard discriminator
            adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
            gen_loss = aux_loss + self.config["lambda_adv"] * adv_loss
        else:
            # for multi-scale discriminator
            adv_loss = 0.0
            for i in range(len(p_)):
                adv_loss += self.criterion["mse"](
                    p_[i][-1], p_[i][-1].new_ones(p_[i][-1].size()))
            adv_loss /= (i + 1)
            gen_loss = aux_loss + self.config["lambda_adv"] * adv_loss

            # feature matching loss
            if self.config["use_feat_match_loss"]:
                p = self.model["discriminator"](y)
                fm_loss = 0.0
                for i in range(len(p_)):
                    for j in range(len(p_[i]) - 1):
                        fm_loss += self.criterion["l1"](p_[i][j], p[i][j])
                fm_loss /= (i + 1) * (j + 1)
                self.total_eval_loss["eval/feature_matching_loss"] += fm_loss.item()
                gen_loss += self.config["lambda_adv"] * self.config["lambda_feat_match"] * fm_loss

        #######################
        #    Discriminator    #
        #######################
        p = self.model["discriminator"](y)
        p_ = self.model["discriminator"](y_)

        # discriminator loss
        if not isinstance(p_, list):
            # for standard discriminator
            real_loss = self.criterion["mse"](p, p.new_ones(p.size()))
            fake_loss = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
            dis_loss = real_loss + fake_loss
        else:
            # for multi-scale discriminator
            real_loss = 0.0
            fake_loss = 0.0
            for i in range(len(p)):
                real_loss += self.criterion["mse"](
                    p[i][-1], p[i][-1].new_ones(p[i][-1].size()))
                fake_loss += self.criterion["mse"](
                    p_[i][-1], p_[i][-1].new_zeros(p_[i][-1].size()))
            real_loss /= (i + 1)
            fake_loss /= (i + 1)
            dis_loss = real_loss + fake_loss

        # add to total eval loss
        self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()
        self.total_eval_loss["eval/spectral_convergence_loss"] += sc_loss.item()
        self.total_eval_loss["eval/log_stft_magnitude_loss"] += mag_loss.item()
        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()
        self.total_eval_loss["eval/real_loss"] += real_loss.item()
        self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
        self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.data_loader["dev"], desc="[eval]"), 1):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if False and eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)

        logging.info(f"(Steps: {self.steps}) Finished evaluation "
                     f"({eval_steps_per_epoch} steps per epoch).")

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.")

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt
        
        # parse batch
        batch = {k: (v.to(self.device) if type(v) == torch.Tensor else v) for k, v in batch.items()}# move Tensors to training device/accelerator
        y_batch = batch["audio"]
        y_batch_ = self.model["generator"](batch["spect"], x=batch["noise"] if "noise" in batch else None)
        if self.config["generator_params"]["out_channels"] > 1:
            y_batch_ = self.criterion["pqmf"].synthesis(y_batch_)

        # check directory
        dirname = os.path.join(self.config["checkpoint_path"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 1):
            # convert to ndarray
            y, y_ = y.view(-1).cpu().numpy(), y_.view(-1).cpu().numpy()

            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavfile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(figname.replace(".png", "_ref.wav"), y,
                     self.config["sampling_rate"], "PCM_16")
            sf.write(figname.replace(".png", "_gen.wav"), y_,
                     self.config["sampling_rate"], "PCM_16")

            if idx >= self.config["num_save_intermediate_results"]:
                break

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(os.path.join(self.config["checkpoint_path"], f"checkpoint-{self.steps:08}steps.pkl"))
            logging.info(f"Successfully saved checkpoint @ {self.steps:08} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.")
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train Parallel WaveGAN (See detail in parallel_wavegan/bin/train.py).")
    parser.add_argument('--input_training_file', default='filelists/train.txt')
    parser.add_argument('--input_validation_file', default='filelists/val.txt')
    parser.add_argument('--checkpoint_path', default='parallel_wavegan/outdir')
    parser.add_argument('--config', default='parallel_wavegan/config/multi_band_melgan.v2.yaml')
    parser.add_argument('--checkpoint_interval', default=0, type=int)
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--n_models_to_keep', default=1, type=int)
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--rank", "--local_rank", default=0, type=int,
                        help="rank for distributed training. no need to explictly specify.")
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
    
    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")
    
    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")
    
    # check directory existence
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    
    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = parallel_wavegan.__version__  # add version info
    if config["checkpoint_interval"] > 0:
        config["save_interval_steps"] = config["checkpoint_interval"]
    if config["max_steps"] and config["max_steps"] > 0:
        config["train_max_steps"] = config["max_steps"]
    with open(os.path.join(args.checkpoint_path, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    
    data_loader = {}
    data_loader["train"], sampler = from_path(config["input_training_file"  ], config)
    data_loader["dev"  ]          = from_path(config["input_validation_file"], config)[0]
    
    # define models and optimizers
    generator_class = getattr(
        parallel_wavegan.models, config.get("generator_type", "ParallelWaveGANGenerator"),
    )
    discriminator_class = getattr(
        parallel_wavegan.models, config.get("discriminator_type", "ParallelWaveGANDiscriminator"),
    )
    model = {
        "generator":     generator_class(    **config["generator_params"]    ).to(device),
        "discriminator": discriminator_class(**config["discriminator_params"]).to(device),
    }
    criterion = {# reconstruction loss
        "stft": MultiResolutionSTFTLoss(**config["stft_loss_params"]).to(device),
        "mse":  torch.nn.MSELoss().to(device),
    }
    if config.get("use_feat_match_loss", False):  # keep compatibility
        criterion["l1"] = torch.nn.L1Loss().to(device)
    if config["generator_params"]["out_channels"] > 1:
        criterion["pqmf"] = PQMF(
            subbands=config["generator_params"]["out_channels"],
            # keep compatibility
            **config.get("pqmf_params", {})
        ).to(device)
    if config.get("use_subband_stft_loss", False):  # keep compatibility
        assert config["generator_params"]["out_channels"] > 1
        criterion["sub_stft"] = MultiResolutionSTFTLoss(
            **config["subband_stft_loss_params"]).to(device)
    generator_optimizer_class = getattr(
        parallel_wavegan.optimizers, config.get("generator_optimizer_type", "RAdam"),
    )
    discriminator_optimizer_class = getattr(
        parallel_wavegan.optimizers, config.get("discriminator_optimizer_type", "RAdam"),
    )
    optimizer = {
        "generator": generator_optimizer_class(
            model["generator"].parameters(), **config["generator_optimizer_params"],
        ),
        "discriminator": discriminator_optimizer_class(
            model["discriminator"].parameters(), **config["discriminator_optimizer_params"],
        ),
    }
    generator_scheduler_class = getattr(
        torch.optim.lr_scheduler, config.get("generator_scheduler_type", "StepLR"),
    )
    discriminator_scheduler_class = getattr(
        torch.optim.lr_scheduler, config.get("discriminator_scheduler_type", "StepLR"),
    )
    scheduler = {
        "generator": generator_scheduler_class(
            optimizer=optimizer["generator"], **config["generator_scheduler_params"],
        ),
        "discriminator": discriminator_scheduler_class(
            optimizer=optimizer["discriminator"], **config["discriminator_scheduler_params"],
        ),
    }
    if args.distributed:
        # wrap model for distributed training
        try:
            from apex.parallel import DistributedDataParallel
        except ImportError:
            raise ImportError("apex is not installed. please check https://github.com/NVIDIA/apex.")
        model["generator"] = DistributedDataParallel(model["generator"])
        model["discriminator"] = DistributedDataParallel(model["discriminator"])
    logging.info(model["generator"])
    logging.info(model["discriminator"])
    
    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )
    
    # load pretrained parameters from checkpoint
    trainer.load_checkpoint()
    
    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        logging.info(f"Saving checkpoint in 10...")
        for i in range(9, -1, -1):
            print(f"{i}...")
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps:08}steps.pkl"))
        logging.info(f"Successfully saved checkpoint @ {trainer.steps:08}steps.")


if __name__ == "__main__":
    main()
