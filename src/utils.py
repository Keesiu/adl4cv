#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import torchvision
from math import e
from math import log
import time, sys

def show_batch(loader,
               total_means=[0.485, 0.456, 0.406],
               total_stds=[0.229, 0.224, 0.225]):
    
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        fig, ax = plt.subplots(figsize=(10,6))
        inp = inp.numpy().transpose((1, 2, 0))
        inp = inp * total_stds + total_means
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            ax.set_xlabel(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
    
    # Get a batch of training data
    inputs, classes = next(iter(loader))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    class_names = loader.dataset.classes
    imshow(out, title=[class_names[x][0:3] for x in classes])


def plot_training_stats(sol, plot_size=(10,12)):

    fig, (ax1, ax2) = plt.subplots(figsize=plot_size, nrows=2)
    ax1.set_title("Accuracy")

    ax1.set_ylabel('Accuracy')

    ax1.grid()
    ax2.grid()

    x1 = np.linspace(0, sol.num_epochs, len(sol.train_acc_history))
    x2 = np.linspace(0, sol.num_epochs, len(sol.val_acc_history))
    x3 = np.linspace(0, sol.num_epochs, len(sol.train_acc_history_avg))
    x4 = np.linspace(0, sol.num_epochs, len(sol.val_acc_history_avg))

    ax1.plot(x1,sol.train_acc_history, '#D4E6F1')
    ax1.plot(x2,sol.val_acc_history, '#F6DDCC')
    ax1.plot(x3,(sol.train_acc_history_avg), '#2980B9' , label="train_acc")
    ax1.plot(x4,(sol.val_acc_history_avg), '#D35400' , label="val_acc")

    ax1.legend(loc = 4)


    x1 = np.linspace(0, sol.num_epochs, len(sol.train_loss_history))
    x2 = np.linspace(0, sol.num_epochs, len(sol.val_loss_history))
    x3 = np.linspace(0, sol.num_epochs, len(sol.train_loss_history_avg))
    x4 = np.linspace(0, sol.num_epochs, len(sol.val_loss_history_avg))

    ax2.plot(x1,sol.train_loss_history, '#D4E6F1')
    ax2.plot(x2,sol.val_loss_history, '#F6DDCC')
    ax2.plot(x3,(sol.train_loss_history_avg), '#2980B9' , label="train_loss")
    ax2.plot(x4,(sol.val_loss_history_avg), '#D35400' , label="val_loss")
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc = 4)

    plt.show()

#def plot_active_learning_history(perc, )
    #TODO

def get_entropy_of_labels(labels, base=None):
  """ Computes entropy of label distribution. """

  n_labels = len(labels)

  if n_labels <= 1:
    return 0

  value,counts = np.unique(labels, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)

  if n_classes <= 1:
    return 0

  ent = 0.

  # Compute entropy
  base = e if base is None else base
  for i in probs:
    ent -= i * log(i, base)

  return ent


# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 35 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100,2), status)
    sys.stdout.write(text)
    sys.stdout.flush()


def update_train_progress(progress, train_loss, train_acc):
    barLength = 35 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% train_loss: {2} train_acc: {3} {4}".format( "#"*block + "-"*(barLength-block), round(progress*100,2), round(train_loss,3), round(train_acc*100,2), status)
    sys.stdout.write(text)
    sys.stdout.flush()



