from . import get_dataloader
from .utils import update_progress
from .neural_network import custom_resnet18
from .solver import solver
from .get_samples_to_add import get_samples_to_add
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.autograd import Variable
import pickle
import datetime
import os


class active_learner():

    def __init__ (self,
                  n_mcd_estimations = 20,
                  n_initial_samples = 1000,
                  low_certainty_threshold = 0.05,
                  high_certainty_threshold = 0,
                  low_certainty_method = 'MCD',
                  high_certainty_method = None,
                  mcd_uncertainty_method='entropy',
                  softmax_uncertainty_method='entropy',
                  al_until_full_train_set = True,
                  n_epochs = [20],
                  RS = 42,
                  BS = 32):

        self.RS = RS
        self.BS = BS
        self.training_finished = False

        np.random.seed(self.RS)
        torch.manual_seed(self.RS)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader.main(
            DATA_DIR = os.path.abspath("./data/256_ObjectCategories"),
            TEST_RATIO = 0.1,
            VAL_RATIO = 0.1,
            RS = self.RS, # Random seed
            BS = self.BS, # Batch size
            TOTAL_MEANS = [0.485, 0.456, 0.406], # Imagenet means per RGB-channel
            # use [0.5520134568214417, 0.533597469329834, 0.5050241947174072] to make dataset specific
            TOTAL_STDS = [0.229, 0.224, 0.225], # Imagenet stds per RGB-channel
            # use [0.03332509845495224, 0.03334072232246399, 0.0340290367603302] to make dataset specific
            RANDOM_AFFINE = 5,
            RESIZE = (224, 224),
            RANDOM_CROP = (160, 160)
        )

        #Defines which percentile of uncertain samples will be used during active learning cycles
        self.low_certainty_threshold = low_certainty_threshold
        self.high_certainty_threshold = high_certainty_threshold

        #Defines how many samples are used for the initial training
        self.n_initial_samples = n_initial_samples

        #Sets monte carlo dropout to False
        self.MCD = False



        #Saves the number of epochs for each AL cycle length of this list defines the number of AL cycles
        self.n_epochs = n_epochs
        self.al_until_full_train_set = al_until_full_train_set

        #Chooses kind of low_certainty measurement we want to use (MCD, softmax)
        self.low_certainty_method = low_certainty_method
        self.high_certainty_method = high_certainty_method
        
        if self.low_certainty_method == 'MCD':
            self.MCD = True

        self.n_mcd_estimations = n_mcd_estimations
        self.mcd_uncertainty_method = mcd_uncertainty_method
        self.softmax_uncertainty_method = softmax_uncertainty_method



        #Create a log that can be saved and loaded which includes which indices have been trained on, and the current AL cycle
        self.log = {}
        initial_targets = [self.train_loader.dataset.targets[i] for i in self.train_loader.sampler.indices]
        initial_idx, _ = train_test_split(
            self.train_loader.sampler.indices,
            train_size=self.n_initial_samples, random_state=self.RS, shuffle=True, stratify=initial_targets)
        self.log['current_train_indices'] = initial_idx
        self.log['current_val_indices'] = self.val_loader.sampler.indices
        self.log['current_al_cycle'] = 0
        self.log['test_scores'] = []
        self.log['current_train_set_percentage'] = []
        self.log['best_epoch'] = []
        self.log['best_val_acc'] = []

        self.current_pseudo_labeled_indices = np.asarray([], dtype = 'int')

        #Save the full train/val/test indices in variable used during AL cycles
        self.full_train_indices = self.train_loader.sampler.indices
        self.full_val_indices = self.val_loader.sampler.indices
        self.test_indices = self.test_loader.sampler.indices


    def _do_training(self, n_epochs, optim_args = {'lr': 0.0001}):

        #set the trainloader.samples to the current selected ones
        self.train_loader.sampler.indices = self.log['current_train_indices']

        #set the validation loader to the right samples. This is crucial since we misused the val_loader instance for the AL_Validation
        self.val_loader.sampler.indices = self.log['current_val_indices']

        #Create a new model and solver
        self.model = custom_resnet18(num_classes=257, p=0.5, MCD = self.MCD)
        self.solver = solver(optim_args = optim_args)

        #use the solver to train the model
        self.solver.train(self.model, self.train_loader, self.val_loader, num_epochs= n_epochs, show_intermediate_steps = 10)

        #Safe the Epoch with best Validation accuracy and acc
        self.log['best_epoch'].append(np.asarray(self.solver.val_loss_history_avg).argmax())
        self.log['best_val_acc'].append(np.asarray(self.solver.val_loss_history_avg).max())

        # Delete Pseudo_labeled_samples
        self.log['current_train_indices'] = np.setdiff1d(self.log['current_train_indices'], self.current_pseudo_labeled_indices)
        self.log['current_train_set_percentage'].append(self.log['current_train_indices'].shape[0]/self.full_train_indices.shape[0])
        self._test_model(self.model, self.val_loader)

    def _add_samples(self):

        #Find all indices of trainset that where not used for previous training (Active_learning_validation_samples)
        current_al_val_indices = np.setdiff1d(self.full_train_indices, self.log['current_train_indices'])

        #Use validation_loader instance on these samples (no augmentation)
        self.val_loader.sampler.indices = current_al_val_indices

        lc_samples_to_add, hc_samples_to_add = get_samples_to_add(self.model,
                                                                  self.val_loader,
                                                                  self.low_certainty_threshold,
                                                                  self.high_certainty_threshold,
                                                                  self.low_certainty_method,
                                                                  self.high_certainty_method,
                                                                  self.n_mcd_estimations,
                                                                  self.mcd_uncertainty_method,
                                                                  self.softmax_uncertainty_method,
                                                                  self.RS)
        self.current_pseudo_labeled_indices = hc_samples_to_add

        #append the found indices to the current training indices
        self.log['current_train_indices'] = np.append(self.log['current_train_indices'], lc_samples_to_add)

        print("")
        print("{} least confident samples have been added to the trainingsset".format(lc_samples_to_add.shape[0]))
        #append the found indices to the current training indices
        
        self.log['current_train_indices'] = np.append(self.log['current_train_indices'], hc_samples_to_add)
        self.log['current_train_indices'] = np.asarray(list(set(self.log['current_train_indices'])))
        
        print("{} most confident samples have been added to the trainingsset".format(hc_samples_to_add.shape[0]))
        print("")

        if self.log['current_train_indices'].shape[0] == self.full_train_indices.shape[0]:
            print("All samples are now used for Training")
            self.training_finished = True

        labeled_samples = np.setdiff1d(self.log['current_train_indices'], self.current_pseudo_labeled_indices).shape[0]
        pseudo_labeled_samples = self.current_pseudo_labeled_indices.shape[0]
        perc = (self.log['current_train_indices'].shape[0]-self.current_pseudo_labeled_indices.shape[0])/self.full_train_indices.shape[0]
        print("Doing Training Cycle with {} labeled and {} pseudo-labeled samples ({}% of total dataset labeled)".format(labeled_samples,pseudo_labeled_samples, int(100*perc)))

    def _do_one_al_cycle(self, n_epoch,  nr_cycle, optim_args = {'lr': 0.0001}):

        self.log['current_al_cycle'] = nr_cycle
        self._add_samples()
        self._do_training(n_epoch, optim_args = optim_args)

        
    def do_active_learning(self,
                           save_checkpoints = True,
                           show_intermediate_plots = True,
                           save_intermediate_plots = True,
                           optim_args = {'lr': 0.0001}):

        self._print_intro()

        self._do_training(self.n_epochs[0], optim_args = optim_args)
        self.log['current_al_cycle'] = 0
        
        if save_checkpoints:
            self._save_model_checkpoint()
            self._save_log_checkpoint()

        if show_intermediate_plots:
            self._intermediate_plots()

        if self.al_until_full_train_set and self.low_certainty_threshold > 1:
            al_cycle = 0
            while not self.training_finished:

                print("")
                print("")
                print("{}. AL CYCLE".format(al_cycle + 1))
                print("")

                self._do_one_al_cycle(self.n_epochs[0], al_cycle + 1, optim_args=optim_args)

                if save_checkpoints:
                    self._save_model_checkpoint()
                    self._save_log_checkpoint()

                if show_intermediate_plots:
                    self._intermediate_plots(save_intermediate_plots)

                al_cycle +=1

        else:
            for al_cycle in range(len(self.n_epochs)-1):
                print("")
                print("")
                print("{}. AL CYCLE".format(al_cycle+1))
                print("")

                self._do_one_al_cycle(self.n_epochs[al_cycle+1], al_cycle+1, optim_args = optim_args)

                if save_checkpoints:
                    self._save_model_checkpoint()
                    self._save_log_checkpoint()

                if show_intermediate_plots:
                    self._intermediate_plots(save_intermediate_plots)

        print("")
        print("")
        print("PROCESS FINISHED. CONGRATS!")


    def _test_model(self, model, loader):

        print('Start testing model')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model = model.eval()
        torch.no_grad()
        acc = 0
        for t, data in enumerate(loader):
                
            inputsval, labelsval = data
            inputsval, labelsval = Variable(inputsval).to(device), Variable(labelsval).to(device)
            outputsval = model(inputsval)
            _, predicted = torch.max(outputsval.data, 1)  
            acc += (np.sum((predicted == labelsval).cpu().numpy())/(len(predicted)*len(loader)))
            update_progress((t+1)/len(loader))

        self.log['test_scores'].append(acc)
        print("Model scored {}% on {} Set".format(round(100*acc,2), "Validation"))



    def _save_model_checkpoint(self):

        now=datetime.datetime.now()
        path = os.path.abspath("./models/prototyping/al_checkpoints/{0}{1}{2}_cycle_{3}_LC_{4}_HC_{5}_MCDM_{6}_SMM_{7}_acc_{8}_model".format(
                       now.year,
                       now.month,
                       now.day,
                       self.log['current_al_cycle'],
                       self.low_certainty_method,
                       self.high_certainty_method,
                       self.mcd_uncertainty_method,
                       self.softmax_uncertainty_method,
                       self.log['test_scores'][-1]))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)


    def _intermediate_plots(self, save = True):

        if save:
            now=datetime.datetime.now()
            path = os.path.abspath("./models/prototyping/al_checkpoints/{0}{1}{2}_cycle_{3}_LC_{4}_HC_{5}_MCDM_{6}_SMM_{7}_acc_{8}_train_history.png".format(
                now.year,
                now.month,
                now.day,
                self.log['current_al_cycle'],
                self.low_certainty_method,
                self.high_certainty_method,
                self.mcd_uncertainty_method,
                self.softmax_uncertainty_method,
                self.log['test_scores'][-1]))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.solver.plot_training_stats(path=path)
        else:
            self.solver.plot_training_stats()

        
    def _save_log_checkpoint(self):

        now=datetime.datetime.now()
        path = os.path.abspath("./models/prototyping/al_checkpoints/{0}{1}{2}_cycle_{3}_LC_{4}_HC_{5}_MCDM_{6}_SMM_{7}_acc_{8}_log".format(
            now.year,
            now.month,
            now.day,
            self.log['current_al_cycle'],
            self.low_certainty_method,
            self.high_certainty_method,
            self.mcd_uncertainty_method,
            self.softmax_uncertainty_method,
            self.log['test_scores'][-1]))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        output = open(path, 'wb')
        pickle.dump(self.log, output)
        output.close()


    def load_model_checkpoint(self, path):

        #TODO
        return None


    def load_log_checkpoint(self, path):
        #TODO
        pkl_file = open('path', 'rb')
        self.log = pickle.load(pkl_file)
        pkl_file.close()


    def _print_intro(self):

        print("STARTING THE ACTIVE LEARNING PROCESS")
        print("")
        print("You are using {} for the high certainty estimation and {} for the low certainty estimation".format(
            self.high_certainty_method, self.low_certainty_method))
        if self.MCD:
            print("During the MCD evaluation {} separate runs will be performed for each sample".format(
                self.n_mcd_estimations))
              
        print("There will be {} active learning cycles".format(len(self.n_epochs)))
        
        print("")
        print("")
        print("INITIAL TRAINING with {} samples ({}% of total Dataset labeled)".format(
            self.log['current_train_indices'].shape[0],
            round((100*(self.log['current_train_indices'].shape[0]
                        -self.current_pseudo_labeled_indices.shape[0])/self.full_train_indices.shape[0]), 2)
        ))
        print("")
