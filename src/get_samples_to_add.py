from .utils import update_progress
from .utils import get_entropy_of_labels
from scipy.stats import entropy
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable

def get_samples_to_add(model,
                       al_val_loader,
                       low_certainty_threshold = 0.05,
                       high_certainty_threshold = 0,
                       low_certainty_method = 'MCD',
                       high_certainty_method = None,
                       n_mcd_estimations = 20,
                       mcd_uncertainty_method = 'entropy',
                       softmax_uncertainty_method = 'entropy',
                       RS = 42):
    
    mcd_predictions = None
    softmax_predictions = None
    
    # Check whether to run MCD_predict
    if low_certainty_method == 'MCD' or high_certainty_method == 'MCD':
        mcd_predictions = MCD_predict(model,
                                      al_val_loader,
                                      n_mcd_estimations,
                                      mcd_uncertainty_method)
    # Check whether to run softmax_predict
    if low_certainty_method == 'Softmax' or high_certainty_method == 'Softmax':
        softmax_predictions = softmax_predict(model,
                                              al_val_loader,
                                              softmax_uncertainty_method)

    if low_certainty_method == 'Random' or high_certainty_method == 'Random':
        full_al_val_indices = al_val_loader.sampler.indices


    if low_certainty_method == 'MCD':
        low_certainty_samples_to_add = get_low_certainty_samples(mcd_predictions, low_certainty_threshold)
    elif low_certainty_method == 'Softmax':
        low_certainty_samples_to_add = get_low_certainty_samples(softmax_predictions, low_certainty_threshold)
    elif low_certainty_method == 'Random':
        low_certainty_samples_to_add = get_random_samples(full_al_val_indices, low_certainty_threshold, RS)
    elif low_certainty_method == None:
        #print("Don't calculate low_certainty_samples")
        low_certainty_samples_to_add = np.asarray([], dtype = 'int')
    else:
        raise ValueError("low_certainty_method not defined! Possible values are MCD, Softmax, Random")

    if high_certainty_method == 'MCD':
        high_certainty_samples_to_add = get_high_certainty_samples(mcd_predictions, high_certainty_threshold)
    elif high_certainty_method == 'Softmax':
        high_certainty_samples_to_add = get_high_certainty_samples(softmax_predictions, high_certainty_threshold)
    elif high_certainty_method == 'Random':
        high_certainty_samples_to_add = get_random_samples(full_al_val_indices, low_certainty_threshold, RS)
    elif high_certainty_method == None:
        #print("Don't calculate high_certainty_samples")
        high_certainty_samples_to_add = np.asarray([], dtype = 'int')
    else:
        raise ValueError("high_certainty_method not defined! Possible values are MCD, Softmax")

    return low_certainty_samples_to_add, high_certainty_samples_to_add


def get_low_certainty_samples(df, low_certainty_threshold):

    "get the least confident samples based on percentile"

    if low_certainty_threshold > 1:
        if low_certainty_threshold > df.shape[0]:
            indices_to_add = df['indices'].to_numpy().astype('int')
        else:
            df = df.sort_values(by=['uncertainty'], ascending = False)
            df = df[:low_certainty_threshold]
            indices_to_add = df['indices'].to_numpy().astype('int')
    else:
        df = df[df['uncertainty'] >= df['uncertainty'].quantile(1-low_certainty_threshold)]
        indices_to_add = df['indices'].to_numpy().astype('int')

    return indices_to_add


def get_high_certainty_samples(df, high_certainty_threshold):

    "get the highest confident samples based on percentile"

    if high_certainty_threshold > 1:
        if high_certainty_threshold > df.shape[0]:
            indices_to_add = df['indices'].to_numpy().astype('int')
        else:
            df = df.sort_values(by=['uncertainty'], ascending = True)
            df = df[:high_certainty_threshold]
            indices_to_add = df['indices'].to_numpy().astype('int')
    else:
        df = df[df['uncertainty'] <= df['uncertainty'].quantile(high_certainty_threshold)]
        indices_to_add = df['indices'].to_numpy().astype('int')

    return indices_to_add


def get_random_samples(indices, threshold, RS = 42):

    "get random samples"

    if threshold > 1:
        np.random.seed(RS)
        np.random.shuffle(indices)
        indices_to_add = indices[:threshold]
    else:
        np.random.seed(RS)
        np.random.shuffle(indices)
        indices_to_add = indices[:int(indices.shape[0]*threshold)]

    return indices_to_add


def softmax_predict(model, loader, softmax_uncertainty_method = 'entropy'):
    
    """softmax_predict returns a pd.DataFrame with columns uncertainty, indices and labels"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = model.eval()
    torch.no_grad()
    predictions_dict = {'indices': loader.sampler.indices}
    print("Starting to predict for Softmax uncertainty measurement using", softmax_uncertainty_method)

    for t, data in enumerate(loader):
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
        outputs = torch.nn.Softmax(dim=1)(model(inputs))
        max_values, max_indices = torch.max(outputs.data, 1)
        if softmax_uncertainty_method == 'entropy':
            uncertainty = entropy(outputs.data.cpu().numpy().T)
        elif softmax_uncertainty_method == 'least_confidence':
            uncertainty = (torch.ones(max_values.shape)-max_values.cpu()).numpy()
        else:
            raise ValueError("softmax_uncertainty_method not defined! Can be entropy or least_confidence")
        if t == 0:
            predictions_dict['labels'] = labels.cpu()
            predictions_dict['uncertainty'] = uncertainty
        else:
            predictions_dict['labels'] = np.append(predictions_dict['labels'], labels.cpu())
            predictions_dict['uncertainty'] = np.append(predictions_dict['uncertainty'], uncertainty)
        update_progress((t+1)/len(loader))

    return pd.DataFrame.from_dict(predictions_dict)


def MCD_predict(model, loader, n_mcd_estimations = 20, mcd_uncertainty_method = 'entropy'):

    """MCD_predict runs model n_mcd_estimation times and returns a pd.DataFrame
    with length len(loader.sampler.indices)
    with columns: 'uncertainty', 'indices', 'labels'
    Note: the uncertainty is calculated using mcd_uncertainty_method"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if not model.MCD:
        raise ValueError("Model was not trained with MCD dropout use solver.predict instead")
    model = model.eval()
    for dol in model.dropout_layers:
        dol = dol.train()
    torch.no_grad()
    predictions_dict = {'uncertainty': None, 'labels': None, 'indices': loader.sampler.indices}
    print("Starting to predict using MCD with %d runs per sample." % n_mcd_estimations)

    for t, data in enumerate(loader):
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

        for i in range(n_mcd_estimations):
            outputs = model(inputs)
            max_values, max_indices = torch.max(outputs.data, 1)
            if i == 0:
                mc_predictions = np.expand_dims(max_indices.cpu().numpy(), axis = 1)
            else:
                mc_predictions = np.append(mc_predictions,
                                           np.expand_dims(max_indices.cpu().numpy(), axis = 1),
                                           axis = 1)

        if t == 0:
            predictions_dict['uncertainty'] = mc_predictions
            predictions_dict['labels'] = labels.cpu()
        else:
            predictions_dict['uncertainty'] = np.append(predictions_dict['uncertainty'],
                                                        mc_predictions,
                                                        axis = 0)
            predictions_dict['labels'] = np.append(predictions_dict['labels'], labels.cpu())
        update_progress((t+1)/len(loader))

    print("Starting calculating uncertainty for MCD estimations using", mcd_uncertainty_method)
    uncertainties = []
    if mcd_uncertainty_method == 'entropy':
        calculate_metric = get_entropy_of_labels
    else:
        raise ValueError("mcd_uncertainty_method not defined! Must be entropy")

    for i, pred in enumerate(predictions_dict['uncertainty']):
        uncertainties.append(calculate_metric(pred))
        update_progress((i+1)/len(predictions_dict['uncertainty']))

    predictions_dict['uncertainty'] = np.asarray(uncertainties)

    return pd.DataFrame.from_dict(predictions_dict)
