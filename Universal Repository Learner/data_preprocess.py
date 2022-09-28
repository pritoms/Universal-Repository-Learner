import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
from utils import AverageMeter, data_loader, save_checkpoint, load_checkpoint, save_image, generate_reconstruction, load_config
from model import ConvVAE
import os
import json

class DataPreprocess(object):
    def __init__(self, config, train_loader, eval_loader, test_loader):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.config = config

    def train(self):
        self.model = ConvVAE(self.config, self.config['cuda'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])

        if self.config['cuda']:
            self.model.cuda()
            self.model = nn.DataParallel(self.model)

        best_val_loss = float('inf')

        for epoch in range(self.config['epochs']):
            print('Training...')
            train_loss = self.train_epoch(epoch)

            print('Evaluating...')
            val_loss, val_recon_loss, val_kl_loss, val_accuracy = self.validate()

            is_best = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict() if self.config['cuda'] else self.model.state_dict(),
                'best_val_loss': best_val_loss,
                'optimizer': self.optimizer.state_dict(),
                'config': self.config
            }, is_best, self.config['save_path'])

            print('Epoch: {}/{}'.format(epoch + 1, self.config['epochs']))
            print('Train Loss: {:.4f}, Val Loss: {:.4f}'.format(train_loss, val_loss))
            print('Val Recon Loss: {:.4f}, Val KL Loss: {:.4f}, Val Accuracy: {:.4f}'.format(val_recon_loss, val_kl_loss, val_accuracy))

    def train_epoch(self, epoch):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        recon_losses = AverageMeter()
        kl_losses = AverageMeter()

        end = time.time()
        for i, (input, target) in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            if self.config['cuda']:
                input = input.cuda()

            self.optimizer.zero_grad()
            reconstruction, mu, logvar = self.model(input)
            recon_loss, kl_loss = self.model.loss_function(reconstruction, input, mu, logvar)
            loss = recon_loss + kl_loss
            loss.backward()
            self.optimizer.step()

            recon_losses.update(recon_loss.item(), input.size(0))
            kl_losses.update(kl_loss.item(), input.size(0))
            losses.update(loss.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.config['log_interval'] == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                      'KL Loss {kl_loss.val:.4f} ({kl_loss.avg:.4f})'.format(
                       epoch, i, len(self.train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, recon_loss=recon_losses, kl_loss=kl_losses))

        return losses.avg

    def validate(self):
        self.model.eval()
        losses = AverageMeter()
        recon_losses = AverageMeter()
        kl_losses = AverageMeter()

        with torch.no_grad():
            for i, (input, target) in enumerate(self.eval_loader):
                if self.config['cuda']:
                    input = input.cuda()

                reconstruction, mu, logvar = self.model(input)
                recon_loss, kl_loss = self.model.loss_function(reconstruction, input, mu, logvar)
                loss = recon_loss + kl_loss

                recon_losses.update(recon_loss.item(), input.size(0))
                kl_losses.update(kl_loss.item(), input.size(0))
                losses.update(loss.item(), input.size(0))

        return losses.avg, recon_losses.avg, kl_losses.avg, 0

    def test(self):
        # load best model
        checkpoint = load_checkpoint(self.config['save_path'])
        self.model.load_state_dict(checkpoint['state_dict'])

        self.model.eval()
        losses = AverageMeter()
        recon_losses = AverageMeter()
        kl_losses = AverageMeter()
        accuracies = AverageMeter()

        with torch.no_grad():
            for i, (input, target) in enumerate(self.test_loader):
                if self.config['cuda']:
                    input = input.cuda()

                reconstruction, mu, logvar = self.model(input)
                recon_loss, kl_loss = self.model.loss_function(reconstruction, input, mu, logvar)
                loss = recon_loss + kl_loss

                recon_losses.update(recon_loss.item(), input.size(0))
                kl_losses.update(kl_loss.item(), input.size(0))
                losses.update(loss.item(), input.size(0))

        return losses.avg, recon_losses.avg, kl_losses.avg, 0
