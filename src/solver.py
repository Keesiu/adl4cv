import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable
from IPython.core.debugger import set_trace
from .utils import update_progress, update_train_progress
import matplotlib.pyplot as plt
torch.nn.Module.dump_patches = True



class solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.num_epochs = 0
        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

        self.train_loss_history_avg = []
        self.train_acc_history_avg = []
        self.val_acc_history_avg = []
        self.val_loss_history_avg = []
        

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0, show_intermediate_steps = False):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu")
        model.to(device)
        #set_trace()
        print('starting training process.')
        ########################################################################
        # DONE:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        k = 0
        l = 0

        show_steps = show_intermediate_steps
        
        self.num_epochs = 0
        for epoch in range(num_epochs):

            ep_loss = 0
            ep_correct = 0
            ep_total = 0

            ep_val_loss = 0
            ep_correctval = 0
            ep_totalval = 0

            self.num_epochs += 1

            model = model.train()
            torch.enable_grad()
            
            k = 0
            print('')
            print('training epoch [%d]' % (epoch+1))
            for data_idx, data in enumerate(train_loader):
                k += 1
                
                inputs, labels = data
                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

                optim.zero_grad()
                
                outputs = model(inputs)
                
                criterion = self.loss_func.to(device)

                loss = criterion(outputs, labels)
                loss.backward()
                optim.step()
               
                self.train_loss_history.append(loss.data)
                ep_loss += loss.data
                
                
                _, predicted = torch.max(outputs.data, 1)
                
                ep_total += 1
                ep_correct += np.sum(((predicted == labels).cpu().numpy()))/(len(predicted))
                
                self.train_acc_history.append(np.sum(((predicted == labels).cpu().numpy()))/(len(predicted)))

                if k % show_steps == 0 or k == len(train_loader):
                    update_train_progress(k/len(train_loader), loss.data.item(), (np.sum(((predicted == labels).cpu().numpy()))/(len(predicted))))


            epoch_acc = ep_correct * 100 / ep_total 
            ep_loss = ep_loss / ep_total

            self.train_acc_history_avg.append(epoch_acc/100)
            self.train_loss_history_avg.append(ep_loss)
            print('loss_train_avg: %.3f, train_acc_avg: %.3f' %(ep_loss, epoch_acc))
            print('starting validation process')
                    
            model = model.eval()
            torch.no_grad()
            
            for t, data in enumerate(val_loader):
                
                l += 1
                # get the inputs
                inputsval, labelsval = data

                # wrap them in Variable
                inputsval, labelsval = Variable(inputsval).to(device), Variable(labelsval).to(device)

                # forward 
                outputsval = model(inputsval)
                criterionval = self.loss_func.to(device)
                lossval = criterionval(outputsval, labelsval)

                # print statistics

                self.val_loss_history.append(lossval.data)
                ep_val_loss += lossval.data
                _, predicted = torch.max(outputsval.data, 1)
                ep_totalval += 1
                
                ep_correctval += (np.sum((predicted == labelsval).cpu().numpy())/(len(predicted)))

                self.val_acc_history.append(np.sum((predicted == labelsval).cpu().numpy())/(len(predicted)))

                update_progress((t+1)/len(val_loader))

                
            ep_val_loss = ep_val_loss / ep_totalval
            ep_val_acc = ep_correctval * 100 / ep_totalval

            self.val_acc_history_avg.append(ep_val_acc/100)
            self.val_loss_history_avg.append(ep_val_loss)
            
            print('loss_val_avg: %.3f, val_acc_avg: %.3f' %(ep_val_loss, ep_val_acc))
                   
            
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
    


    def plot_training_stats(self, plot_size=(10,12), path=False):

        fig, (ax1, ax2) = plt.subplots(figsize=plot_size, nrows=2)
        ax1.set_title("Accuracy")

        ax1.set_ylabel('Accuracy')

        ax1.grid()
        ax2.grid()

        x1 = np.linspace(0, self.num_epochs, len(self.train_acc_history))
        x2 = np.linspace(0, self.num_epochs, len(self.val_acc_history))
        x3 = np.linspace(0, self.num_epochs, len(self.train_acc_history_avg))
        x4 = np.linspace(0, self.num_epochs, len(self.val_acc_history_avg))

        ax1.plot(x1,self.train_acc_history, '#D4E6F1')
        ax1.plot(x2,self.val_acc_history, '#F6DDCC')
        ax1.plot(x3,(self.train_acc_history_avg), '#2980B9' , label="train_acc")
        ax1.plot(x4,(self.val_acc_history_avg), '#D35400' , label="val_acc")

        ax1.legend(loc = 4)


        x1 = np.linspace(0, self.num_epochs, len(self.train_loss_history))
        x2 = np.linspace(0, self.num_epochs, len(self.val_loss_history))
        x3 = np.linspace(0, self.num_epochs, len(self.train_loss_history_avg))
        x4 = np.linspace(0, self.num_epochs, len(self.val_loss_history_avg))

        ax2.plot(x1,self.train_loss_history, '#D4E6F1')
        ax2.plot(x2,self.val_loss_history, '#F6DDCC')
        ax2.plot(x3,(self.train_loss_history_avg), '#2980B9' , label="train_loss")
        ax2.plot(x4,(self.val_loss_history_avg), '#D35400' , label="val_loss")
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(loc = 4)

        if path:
            plt.savefig(path)

        plt.show()

        
