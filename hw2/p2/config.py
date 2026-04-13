# ============================================================================
# File: config.py
# Date: 2026-03-27
# Author: TA
# Description: Experiment configurations.
# ============================================================================

################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
exp_name = "default"  # name of experiment

# Model Options
model_type = "mynet"  # 'mynet' or 'resnet18'

# Learning Options
epochs = 100  # train how many epochs
batch_size = 256  # batch size for dataloader
use_adam = True  # Adam or SGD optimizer
lr = 1e-3  # learning rate
milestones = [16, 32, 45]  # reduce learning rate at 'milestones' epochs
