import matplotlib
import argparse
#matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import os
import copy
import sklearn
import pickle as pkl
from scipy.stats import pearsonr
from constants_and_util import *
import image_processing
import json
import sys
from image_processing import PytorchImagesDataset
from torch.utils.data import Dataset, DataLoader
from non_image_data_processing import NonImageData
import datetime
import cv2
#import mura_predict
import analysis
from scipy.ndimage.filters import gaussian_filter
from scipy.special import expit
from matplotlib import ticker
import gc
import json
import statsmodels.api as sm
#import my_modified_resnet

# https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')

def create_ses_weights(d, ses_col, covs, p_high_ses, use_propensity_scores):
    """
    Used for training preferentially on high or low SES people. If use_propensity_scores is True, uses propensity score matching on covs.

    Note: this samples from individual images, not from individual people. I think this is okay as long as we're clear about what's being done. If p_high_ses = 0 or 1, both sampling methods are equivalent. One reason to sample images rather than people is that if you use propensity score weighting, covs may change for people over time. 
    """
    assert p_high_ses >= 0 and p_high_ses <= 1

    high_ses_idxs = (d[ses_col] == True).values
    n_high_ses = high_ses_idxs.sum()
    n_low_ses = len(d) - n_high_ses
    assert pd.isnull(d[ses_col]).sum() == 0
    n_to_sample = min(n_high_ses, n_low_ses) # want to make sure train set size doesn't change as we change p_high_ses from 0 to 1 so can't have a train set size larger than either n_high_ses or n_low_ses
    n_high_ses_to_sample = int(p_high_ses * n_to_sample)
    n_low_ses_to_sample = n_to_sample - n_high_ses_to_sample
    all_idxs = np.arange(len(d))
    high_ses_samples = np.array(random.sample(list(all_idxs[high_ses_idxs]), n_high_ses_to_sample))
    low_ses_samples = np.array(random.sample(list(all_idxs[~high_ses_idxs]), n_low_ses_to_sample))
    print("%i high SES samples and %i low SES samples drawn with p_high_ses=%2.3f" % 
        (len(high_ses_samples), len(low_ses_samples), p_high_ses))

    # create weights. 
    weights = np.zeros(len(d)) 
    if len(high_ses_samples) > 0:
        weights[high_ses_samples] = 1.
    if len(low_ses_samples) > 0:
        weights[low_ses_samples] = 1.
    

    if not use_propensity_scores:
        assert covs is None
        weights = weights / weights.sum()
        return weights
    else:
        assert covs is not None

        # fit probability model
        propensity_model = sm.Logit.from_formula('%s ~ %s' % (ses_col, '+'.join(covs)), data=d).fit()
        print("Fit propensity model")
        print(propensity_model.summary())

        # compute inverse propensity weights.
        # "A subject's weight is equal to the inverse of the probability of receiving the treatment that the subject actually received"
        # The treatment here is whether they are high SES, 
        # and we are matching them on the other covariates. 
        high_ses_propensity_scores = propensity_model.predict(d).values
        high_ses_weights = 1 / high_ses_propensity_scores
        low_ses_weights = 1 / (1 - high_ses_propensity_scores)
        propensity_weights = np.zeros(len(d))
        propensity_weights[high_ses_idxs] = high_ses_weights[high_ses_idxs]
        propensity_weights[~high_ses_idxs] = low_ses_weights[~high_ses_idxs]

        assert np.isnan(propensity_weights).sum() == 0
        
        # multply indicator vector by propensity weights. 
        weights = weights * propensity_weights

        # normalize weights so that high and low SES sum to the right things. 
        print(n_high_ses_to_sample, n_low_ses_to_sample)
        if n_high_ses_to_sample > 0:
            weights[high_ses_idxs] = n_high_ses_to_sample * weights[high_ses_idxs] / weights[high_ses_idxs].sum()
        if n_low_ses_to_sample > 0:
            weights[~high_ses_idxs] = n_low_ses_to_sample * weights[~high_ses_idxs] / weights[~high_ses_idxs].sum()
        assert np.isnan(weights).sum() == 0
        
        # normalize whole vector, just to keep things clean
        weights = weights / weights.sum()

    return weights

def reweight_to_remove_correlation_between_pain_and_ses(d, ses_col, pain_col):
    """
    Robustness check: train on dataset where we've removed the correlation between pain and SES to verify that the model isn't just learning to predict SES. 
    """
    d = copy.deepcopy(d)
    high_ses_idxs = (d[ses_col] == True).values
    d['discretized_pain_score'] = analysis.cut_into_deciles(d[pain_col].values 
                                                            + .0001 * np.random.random(len(d)), # small hack to break ties
                                                            pain_col)

    predict_high_ses_given_pain = sm.Logit.from_formula('%s ~ C(discretized_pain_score)' % (
        ses_col), data=d).fit()
    high_ses_propensity_scores = predict_high_ses_given_pain.predict(d).values

    high_ses_weights = 1 / high_ses_propensity_scores
    low_ses_weights = 1 / (1 - high_ses_propensity_scores)
    propensity_weights = np.zeros(len(d))
    propensity_weights[high_ses_idxs] = high_ses_weights[high_ses_idxs]
    propensity_weights[~high_ses_idxs] = low_ses_weights[~high_ses_idxs]

    propensity_weights = propensity_weights / propensity_weights.sum()

    r, p = pearsonr(d[pain_col], d[ses_col])
    print("Original correlation between SES and pain: %2.3f" % r)
    samples = np.random.choice(range(len(d)), p=propensity_weights, size=[50000,])
    r, p = pearsonr(d[pain_col].iloc[samples], d[ses_col].iloc[samples])
    print("Correlation after inverse propensity weighting: %2.3f" % r)
    return propensity_weights

def load_real_data_in_transfer_learning_format(batch_size, 
    downsample_factor_on_reload, 
    normalization_method, 
    y_col, 
    max_horizontal_translation,
    max_vertical_translation,
    seed_to_further_shuffle_train_test_val_sets,
    additional_features_to_predict,
    crop_to_just_the_knee=False,
    show_both_knees_in_each_image=False,
    weighted_ses_sampler_kwargs=None,
    increase_diversity_kwargs=None,
    hold_out_one_imaging_site_kwargs=None,
    train_on_single_klg_kwargs=None,
    remove_correlation_between_pain_and_ses_kwargs=None,
    alter_train_set_size_sampler_kwargs=None,
    use_very_very_small_subset=False, 
    blur_filter=None):
    """
    Load dataset a couple images at a time using DataLoader class, as shown in pytorch dataset tutorial. 
    Checked. 
    """

    load_only_single_klg = None
    if (train_on_single_klg_kwargs is not None) and ('make_train_set_smaller' in train_on_single_klg_kwargs) and train_on_single_klg_kwargs['make_train_set_smaller']:
        raise Exception("Should not be using this option.")
        load_only_single_klg = train_on_single_klg_kwargs['klg_to_use']

    train_dataset = PytorchImagesDataset(dataset='train', 
                         downsample_factor_on_reload=downsample_factor_on_reload, 
                         normalization_method=normalization_method, 
                         show_both_knees_in_each_image=show_both_knees_in_each_image,
                         y_col=y_col, 
                         seed_to_further_shuffle_train_test_val_sets=seed_to_further_shuffle_train_test_val_sets,
                         transform='random_translation_and_then_random_horizontal_flip' if not show_both_knees_in_each_image else 'random_translation', 
                         additional_features_to_predict=additional_features_to_predict,
                         max_horizontal_translation=max_horizontal_translation,
                         max_vertical_translation=max_vertical_translation,
                         use_very_very_small_subset=use_very_very_small_subset, 
                         crop_to_just_the_knee=crop_to_just_the_knee, 
                         load_only_single_klg=load_only_single_klg, 
                         blur_filter=blur_filter)

    if weighted_ses_sampler_kwargs is not None:
        assert train_on_single_klg_kwargs is None
        assert alter_train_set_size_sampler_kwargs is None
        assert remove_correlation_between_pain_and_ses_kwargs is None
        ses_weights = create_ses_weights(copy.deepcopy(train_dataset.non_image_data), **weighted_ses_sampler_kwargs)
        print(ses_weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(ses_weights, len(ses_weights))  # https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
        shuffle = False        
        # per the pytorch documentation, 
        # "sampler defines the strategy to draw samples from the dataset. If specified, ``shuffle`` must be False."
        # Shuffling is already taken care of by the sampler. Cool.   
    elif hold_out_one_imaging_site_kwargs is not None:
        print("Hold out one train site kwargs are")
        print(hold_out_one_imaging_site_kwargs)
        weights = 1.*(train_dataset.non_image_data['v00site'].values != hold_out_one_imaging_site_kwargs['site_to_remove'])
        assert weights.mean() < 1
        print("After removing site %s, %i/%i train datapoints remaining" % (hold_out_one_imaging_site_kwargs['site_to_remove'], int(weights.sum()), len(weights)))
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        shuffle = False 
    elif increase_diversity_kwargs is not None:
        print("Increase diversity kwargs are")
        print(increase_diversity_kwargs)
        assert weighted_ses_sampler_kwargs is None
        assert remove_correlation_between_pain_and_ses_kwargs is None
        minority_idxs = (train_dataset.non_image_data[increase_diversity_kwargs['ses_col']].values == increase_diversity_kwargs['minority_val'])
        n_minority_people = len(set(train_dataset.non_image_data.loc[minority_idxs, 'id'].values))
        majority_ids = sorted(list(set(train_dataset.non_image_data.loc[~minority_idxs, 'id'].values)))
        n_majority_people = len(majority_ids)
        
        assert n_majority_people > n_minority_people
        if increase_diversity_kwargs['exclude_minority_group']:
            # remove all minorities. 
            weights = (~minority_idxs) * 1.
        else:
            # remove a random sample of majority people. 
            rng = random.Random(increase_diversity_kwargs['majority_group_seed'])
            majority_ids_to_keep = set(rng.sample(majority_ids, n_majority_people - n_minority_people))
            majority_idxs = train_dataset.non_image_data['id'].map(lambda x:x in majority_ids_to_keep).values
            assert ((majority_idxs == 1) & (minority_idxs == 1)).sum() == 0
            weights = ((minority_idxs == 1) | (majority_idxs == 1)) * 1.
        print("Number of people with %s=%i in train set: %i; number in majority set: %i; total number of points with nonzero weights %i; exclude minority group %s; random seed %s" % (
            increase_diversity_kwargs['ses_col'],
            increase_diversity_kwargs['minority_val'], 
            n_minority_people,
            n_majority_people,
            int(weights.sum()), 
            increase_diversity_kwargs['exclude_minority_group'], 
            increase_diversity_kwargs['majority_group_seed']))

        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        shuffle = False 
    elif remove_correlation_between_pain_and_ses_kwargs is not None:
        assert train_on_single_klg_kwargs is None
        assert alter_train_set_size_sampler_kwargs is None
        assert weighted_ses_sampler_kwargs is None
        weights = reweight_to_remove_correlation_between_pain_and_ses(copy.deepcopy(train_dataset.non_image_data), 
            **remove_correlation_between_pain_and_ses_kwargs)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        shuffle = False  
    elif train_on_single_klg_kwargs is not None:
        assert weighted_ses_sampler_kwargs is None
        assert remove_correlation_between_pain_and_ses_kwargs is None
        sample_weights = 1.*(train_dataset.non_image_data['xrkl'].values == train_on_single_klg_kwargs['klg_to_use'])
        # See note above for weighted_ses_sampler_kwargs
        if 'make_train_set_smaller' in train_on_single_klg_kwargs and train_on_single_klg_kwargs['make_train_set_smaller']:
            n_train_points = int(sample_weights.sum())
        else:
            n_train_points = len(sample_weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, n_train_points)
        shuffle = False
    elif alter_train_set_size_sampler_kwargs is not None:
        assert train_on_single_klg_kwargs is None
        assert weighted_ses_sampler_kwargs is None
        assert remove_correlation_between_pain_and_ses_kwargs is None
        train_set_frac = alter_train_set_size_sampler_kwargs['fraction_of_train_set_to_use']
        assert train_set_frac > 0 and train_set_frac <= 1
        all_train_ids = sorted(list(set(train_dataset.non_image_data['id'])))
        n_train_ids_to_use = int(len(all_train_ids) * train_set_frac)
        print("Total number of people in train set: %i. Taking fraction %2.3f of them (%i ids)" % 
            (len(all_train_ids), train_set_frac, n_train_ids_to_use))
        
        # ensure train set ids we take are always the same (we don't want to take best result across multiple datasets).
        rng = random.Random(42)
        rng.shuffle(all_train_ids)
        train_ids_to_use = set(all_train_ids[:n_train_ids_to_use])

        # Use WeightedRandomSampler exactly in analogy to SES weights above. This will make training slightly slower, but it seems more important to make the code consistent and functional. 
        sample_weights = 1.*train_dataset.non_image_data['id'].map(lambda x:x in train_ids_to_use).values
        
        # See note above for weighted_ses_sampler_kwargs
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    NUM_WORKERS_TO_USE = 8

    # Note: if you are using WeightedRandomSampler for train_sampler, and only selecting a small subset of the datapoints (eg, just those with KLG=4) you may 
    # quickly overtrain, since at each epoch you run through the dataset many times by sampling with replacement (replacement=True by default). This doesn't appear to be a major problem at present...tried modifying it to train on a single KLG, and it didn't improve results. 

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS_TO_USE, sampler=train_sampler)

    val_dataset = PytorchImagesDataset(dataset='val', 
                         downsample_factor_on_reload=downsample_factor_on_reload, 
                         normalization_method=normalization_method, 
                         show_both_knees_in_each_image=show_both_knees_in_each_image,
                         y_col=y_col, 
                         additional_features_to_predict=additional_features_to_predict,
                         seed_to_further_shuffle_train_test_val_sets=seed_to_further_shuffle_train_test_val_sets,
                         transform=None, 
                         crop_to_just_the_knee=crop_to_just_the_knee,
                         use_very_very_small_subset=use_very_very_small_subset, 
                         blur_filter=blur_filter)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS_TO_USE)

    test_dataset = PytorchImagesDataset(dataset='test', 
                         downsample_factor_on_reload=downsample_factor_on_reload, 
                         normalization_method=normalization_method, 
                         show_both_knees_in_each_image=show_both_knees_in_each_image,
                         y_col=y_col, 
                         additional_features_to_predict=additional_features_to_predict,
                         seed_to_further_shuffle_train_test_val_sets=seed_to_further_shuffle_train_test_val_sets,
                         transform=None, 
                         crop_to_just_the_knee=crop_to_just_the_knee,
                         use_very_very_small_subset=use_very_very_small_subset, 
                         blur_filter=blur_filter)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS_TO_USE)

    dataloaders = {'train':train_dataloader, 'val':val_dataloader, 'test':test_dataloader}
    if use_very_very_small_subset:
        dataset_sizes = {'train':500, 'val':500, 'test':500}
    else:
        dataset_sizes = {'train':len(train_dataset), 'val':len(val_dataset), 'test':len(test_dataset)}

    datasets = {'train':train_dataset, 'val':val_dataset, 'test':test_dataset}
    return dataloaders, datasets, dataset_sizes

class TransferLearningPytorchModel():
    """
    Load and fine-tune a pretrained pytorch model. 
    pretrained_model_name: one of the resnets or MURA. 
    binary_prediction: whether the prediction task is binary or continuous
    conv_layers_before_end_to_unfreeze: how many conv layers from the end we want to fine-tune. 
    optimizer_name, optimizer_kwargs: whether eg we're using Adam or SGD. 
    scheduler_kwargs: how we change the learning rate. 
    num_epochs: how many epochs to train for. 
    y_col: what we're trying to predict. 
    n_additional_image_features_to_predict: should be 0 or 19. Used for regularization. 
    additional_loss_weighting: how much we weight this additional loss. 
    mura_initialization_path: if training one of the MURA pretrained models, path to load weights from. 
    """
    def __init__(self, 
        pretrained_model_name, 
        binary_prediction, 
        conv_layers_before_end_to_unfreeze, 
        optimizer_name, 
        optimizer_kwargs, 
        scheduler_kwargs, 
        num_epochs, 
        y_col,
        where_to_add_klg,
        fully_connected_bias_initialization=None,
        n_additional_image_features_to_predict=0, 
        additional_loss_weighting=0, 
        mura_initialization_path=None):

        assert where_to_add_klg is None
        self.pretrained_model_name = pretrained_model_name
        self.binary_prediction = binary_prediction
        self.conv_layers_before_end_to_unfreeze = conv_layers_before_end_to_unfreeze
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.num_epochs = num_epochs
        self.y_col = y_col
        self.where_to_add_klg = where_to_add_klg
        self.fully_connected_bias_initialization = fully_connected_bias_initialization
        self.n_additional_image_features_to_predict = n_additional_image_features_to_predict # if we have additional features, they are just concatenated onto linear layer.  
        self.additional_loss_weighting = additional_loss_weighting
        if self.binary_prediction:
            self.metric_to_use_as_stopping_criterion = 'val_auc'
        else:
            self.metric_to_use_as_stopping_criterion = 'val_negative_rmse'
        assert (self.n_additional_image_features_to_predict > 0) == (self.additional_loss_weighting > 0)
        assert self.n_additional_image_features_to_predict in [0, 3, 19, 22]

        assert pretrained_model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'pretrained_mura_densenet']
         
        if self.where_to_add_klg in ['before_layer4', 'before_layer3', 'before_layer2']:
            resnet_source = my_modified_resnet # only use fancy resnet if we have to. 
        else:
            resnet_source = models # pytorch library. 

        if pretrained_model_name == 'resnet18':
            self.model = resnet_source.resnet18(pretrained=True)
            self.finalconv_name = 'layer4' # used for loading the final embedding for CAM. 
        elif pretrained_model_name == 'resnet34':
            self.model = resnet_source.resnet34(pretrained=True)
            self.finalconv_name = 'layer4'
        elif pretrained_model_name == 'resnet50':
            self.model = resnet_source.resnet50(pretrained=True)
            raise Exception("Not sure what final conv name is")
        elif pretrained_model_name == 'resnet101':
            self.model = resnet_source.resnet101(pretrained=True)
            raise Exception("Not sure what final conv name is")
        elif pretrained_model_name == 'resnet152':
            self.model = resnet_source.resnet152(pretrained=True)
            raise Exception("Not sure what final conv name is")
        elif pretrained_model_name == 'pretrained_mura_densenet':
            params = mura_predict.get_model_params()
            print("Mura params are", params.__dict__)
            assert mura_initialization_path is not None
            self.model = mura_predict.load_model(mura_initialization_path, 
                                 params=params, 
                                 use_gpu=True).model_ft
            
        else:
            raise Exception("Not a valid model name!")

        self.model.avgpool = nn.AdaptiveAvgPool2d(1)

        num_ftrs = self.model.fc.in_features

        if binary_prediction:
            self.model.fc = nn.Linear(in_features=num_ftrs, out_features=2 + n_additional_image_features_to_predict) # reset final fully connected layer for two-class prediction. If we have additional features, include those as outputs from the fully connected layer. 
            self.loss_criterion = nn.CrossEntropyLoss()
            assert self.fully_connected_bias_initialization is None

        else:
            self.model.fc = nn.Linear(in_features=num_ftrs, out_features=1 + n_additional_image_features_to_predict) # reset final fully connected layer and make it so it has a single output. 
            self.loss_criterion = nn.MSELoss()
            if self.fully_connected_bias_initialization is not None:
                # we do this for Koos pain subscore because otherwise the final layer ends up with all positive weights, and that's weird/hard to interpret.
                nn.init.constant(self.model.fc.bias.data[:1], self.fully_connected_bias_initialization)
                print("Bias has been initialized to")
                print(self.model.fc.bias)

        
        # loop over layers from beginning and freeze a couple. First we need to get the layers.
        def is_conv_layer(name):
            # If a layer is a conv layer, returns a substring which uniquely identifies it. Otherwise, returns None. 
            if pretrained_model_name == 'pretrained_mura_densenet':
                if 'conv' in name:
                    return name
            else:
                # this logic is probably more complex than it needs to be but it works. 
                if name[:5] == 'layer':
                    sublayer_substring = '.'.join(name.split('.')[:3])
                    if 'conv' in sublayer_substring:
                        return sublayer_substring
            return None

        param_idx = 0
        all_conv_layers = []
        for name, param in self.model.named_parameters():
            print("Param %i: %s" % (param_idx, name), param.data.shape)
            param_idx += 1
            conv_layer_substring = is_conv_layer(name)
            if conv_layer_substring is not None and conv_layer_substring not in all_conv_layers:
                all_conv_layers.append(conv_layer_substring)
        print("All conv layers", all_conv_layers)

        # now look conv_layers_before_end_to_unfreeze conv layers before the end, and unfreeze all layers after that. 
        start_unfreezing = False
        layers_modified_for_klg = 0
        assert conv_layers_before_end_to_unfreeze <= len(all_conv_layers)
        if conv_layers_before_end_to_unfreeze > 0:
            conv_layers_to_unfreeze = all_conv_layers[-conv_layers_before_end_to_unfreeze:]
        else:
            conv_layers_to_unfreeze = []

        for name, param in self.model.named_parameters():
            conv_layer_substring = is_conv_layer(name)
            if conv_layer_substring in conv_layers_to_unfreeze:
                start_unfreezing = True
            if name in ['fc.weight', 'fc.bias']:
                # we always unfreeze these layers. 
                start_unfreezing = True
            if start_unfreezing:                
                if self.where_to_add_klg in ['before_layer4', 'before_layer3', 'before_layer2']:
                    layer_to_modify = self.where_to_add_klg.replace('before_', '')
                    if name in ['%s.0.conv1.weight' % layer_to_modify, '%s.0.downsample.0.weight' % layer_to_modify]:
                        layers_modified_for_klg += 1
                        what_to_add = .1 * torch.randn(param.data.size()[0], 5, param.data.size()[2], param.data.size()[3])
                        param.data = torch.cat((param.data, what_to_add), 1)
                print("Param %s is UNFROZEN" % (name), param.data.shape)
            else:
                print("Param %s is FROZEN" % (name), param.data.shape)
                param.requires_grad = False

        if self.where_to_add_klg in ['before_layer4', 'before_layer3', 'before_layer2']:
            assert layers_modified_for_klg == 2 # make sure we unfroze the two layers we needed to unfreeze. 

        if self.where_to_add_klg == 'output':
            self.model.klg_fc = nn.Linear(in_features=5, out_features=1)

        self.model = self.model.cuda() # move model to GPU. 

        # https://github.com/pytorch/pytorch/issues/679
        if optimizer_name == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), **optimizer_kwargs)
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), **optimizer_kwargs)
        else:
            raise Exception("Not a valid optimizer")

        self.lr_scheduler_type = scheduler_kwargs['lr_scheduler_type']
        if self.lr_scheduler_type == 'decay':
            self.scheduler = lr_scheduler.StepLR(self.optimizer, 
                **scheduler_kwargs['additional_kwargs'])
        elif self.lr_scheduler_type == 'plateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                **scheduler_kwargs['additional_kwargs'])
        else:
            raise Exception("invalid scheduler")
        self.layer_magnitudes = {}
    
    def print_layer_magnitudes(self, epoch):
        # small helper method so we can make sure the right layers are being trained. 
        for name, param in self.model.named_parameters():
            magnitude = np.linalg.norm(param.data.cpu())
            if param not in self.layer_magnitudes:
                self.layer_magnitudes[param] = magnitude
                print("The magnitude of layer %s at epoch %i is %2.5f" % (name, epoch, magnitude))
            else:
                old_magnitude = self.layer_magnitudes[param]
                delta_magnitude = magnitude - old_magnitude
                print("The magnitude of layer %s at epoch %i is %2.5f (delta %2.5f from last epoch)" % (name, epoch, magnitude, delta_magnitude))
                self.layer_magnitudes[param] = magnitude

    def evaluate_on_dataset(self, dataloaders, dataset_sizes, phase, make_plot=False):
        """
        Given a model, data, and a phase (train/val/test) runs the model on the data and, if phase=train, trains the model. Checked.
        """
        print("Now we are evaluating on the %s dataset!" % phase)
        assert phase in ['train', 'val', 'test']
        use_gpu = torch.cuda.is_available()
        if phase == 'train':
            self.model.train(True)  # Set model to training mode
        else:
            self.model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        n_batches_loaded = 0
        start_time_for_100_images = time.time()

        # Iterate over data.
        # keep track of all labels + outputs to compute the final metrics. 
        concatenated_labels = []
        concatenated_outputs = []
        concatenated_binarized_education_graduated_college = []
        concatenated_binarized_income_at_least_50k = []
        concatenated_numerical_klg = []
        concatenated_site = []

        if self.n_additional_image_features_to_predict > 0:
            loss_additional_loss_ratios = [] # also keep track of how big the additional regularization loss is relative to the main loss. 
        for data in dataloaders[phase]:
            #print("We reached the beginning of the loop with %i images" % n_batches_loaded)
            n_batches_loaded += 1
            if n_batches_loaded % 100 == 0:
                print("Time taken to process 100 batches %2.3f seconds (total batches %i)" % (time.time() - start_time_for_100_images, len(dataloaders[phase])))
                start_time_for_100_images = time.time()

            # get the inputs
            inputs = data['image']
            labels = data['y']
            additional_features_to_predict = data['additional_features_to_predict']
            additional_features_are_not_nan = data['additional_features_are_not_nan']
            one_hot_klg = data['klg'] # Note that this is a matrix (one-hot). 
            assert one_hot_klg.size()[1] == 5
            numerical_klg = np.nonzero(np.array(one_hot_klg))[1]
            assert len(numerical_klg) == len(one_hot_klg)
            binarized_education_graduated_college = np.array(data['binarized_education_graduated_college'])
            binarized_income_at_least_50k = np.array(data['binarized_income_at_least_50k'])
            concatenated_site += list(np.array(data['site']))
            concatenated_binarized_education_graduated_college += list(binarized_education_graduated_college)
            concatenated_binarized_income_at_least_50k += list(binarized_income_at_least_50k)
            concatenated_numerical_klg += list(numerical_klg) 

            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.float().cuda())
                if self.n_additional_image_features_to_predict > 0:
                    additional_features_to_predict = Variable(additional_features_to_predict.float().cuda())
                    additional_features_are_not_nan = Variable(additional_features_are_not_nan.float().cuda())
                one_hot_klg = Variable(one_hot_klg.float().cuda())
                if self.binary_prediction: 
                    labels = Variable(labels.long().cuda())
                else:
                    labels = Variable(labels.float().cuda())
            else:
                raise Exception("Use a GPU, fool.")

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            if self.where_to_add_klg in ['before_layer4', 'before_layer3', 'before_layer2']:
                outputs = self.model(inputs, 
                    additional_input_features=one_hot_klg, 
                    where_to_add=self.where_to_add_klg)
            else:
                outputs = self.model(inputs)

            if self.where_to_add_klg == 'output':
                outputs = outputs + self.model.klg_fc(one_hot_klg)

            
            if self.n_additional_image_features_to_predict > 0:
                additional_feature_outputs = outputs[:, -self.n_additional_image_features_to_predict:]
                outputs = outputs[:, :-self.n_additional_image_features_to_predict]
                loss = self.loss_criterion(input=outputs, target=labels) 

                # basically, we only add to the additional feature loss if a feature is not NaN.  
                additional_feature_losses = ((additional_features_to_predict - additional_feature_outputs) ** 2) * additional_features_are_not_nan
                additional_loss = additional_feature_losses.sum(dim=1).mean(dim=0)

                original_loss_float = loss.data.cpu().numpy().flatten()
                additional_loss_float = additional_loss.data.cpu().numpy().flatten()
                loss_additional_loss_ratios.append(original_loss_float / (self.additional_loss_weighting * additional_loss_float))
                loss = loss + additional_loss * self.additional_loss_weighting
            else:
                loss = self.loss_criterion(input=outputs, target=labels)

            # keep track of everything for correlations
            concatenated_labels += list(labels.data.cpu().numpy().flatten())
            if self.binary_prediction: 
                # outputs are logits. Take softmax, class 1 prediction. 
                h_x = F.softmax(outputs, dim=1).data.squeeze()
                concatenated_outputs += list(h_x[:, 1].cpu().numpy().flatten())
            else:
                concatenated_outputs += list(outputs.data.cpu().numpy().flatten())

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                self.optimizer.step()
            # statistics
            running_loss += loss.data[0] * inputs.size(0)
        
        epoch_loss = running_loss / dataset_sizes[phase]
        metrics_for_epoch = {}
        if not self.binary_prediction: 
            concatenated_outputs = np.array(concatenated_outputs)
            concatenated_labels = np.array(concatenated_labels)

            if make_plot:
                plt.figure()
                plt.scatter(concatenated_outputs, concatenated_labels)
                plt.xlim([0, 100])
                plt.ylim([0, 100])
                plt.xlabel("Yhat")
                plt.ylabel("Y")
                plt.show()

            correlation_and_rmse = analysis.assess_performance(y=concatenated_labels, yhat=concatenated_outputs, binary_prediction=False)

            assert len(concatenated_outputs) == dataset_sizes[phase]
            print('%s epoch loss for %s: %2.6f; RMSE %2.6f; correlation %2.6f (n=%i)' % 
                (phase, self.y_col, epoch_loss, correlation_and_rmse['rmse'], correlation_and_rmse['r'], len(concatenated_labels)))
            metrics_for_epoch['%s_loss' % phase] = epoch_loss
            metrics_for_epoch['%s_rmse' % phase] = correlation_and_rmse['rmse']
            metrics_for_epoch['%s_negative_rmse' % phase] = correlation_and_rmse['negative_rmse']
            metrics_for_epoch['%s_r' % phase] = correlation_and_rmse['r']

            print("Correlation between binarized_education_graduated_college and labels: %2.3f" % pearsonr(concatenated_binarized_education_graduated_college, concatenated_labels)[0])
            print("Correlation between binarized_income_at_least_50k and labels: %2.3f" % pearsonr(concatenated_binarized_income_at_least_50k, concatenated_labels)[0])

            # if Koos score, also compute AUC + AUPRC for the binarized versions. 
            if self.y_col == 'koos_pain_subscore': 
                assert np.allclose(concatenated_labels.max(), 100)
                concatenated_binarized_labels = binarize_koos(concatenated_labels)
                concatenated_scores = -concatenated_outputs # lower predictions = more likely to be positive class
                binarized_auc_and_auprc = analysis.assess_performance(y=concatenated_binarized_labels, yhat=concatenated_scores, binary_prediction=True)
                metrics_for_epoch['%s_binarized_auc' % phase] = binarized_auc_and_auprc['auc']
                metrics_for_epoch['%s_binarized_auprc' % phase] = binarized_auc_and_auprc['auprc']
                
                metrics_for_epoch['%s_ses_betas' % phase] = {'binarized_education_graduated_college_betas':None, 
                'binarized_income_at_least_50k_betas':None}

                if phase == 'test':
                    # compute SES pain gaps for KLG >= 2. 
                    education_pain_gaps = analysis.compare_pain_levels_for_people_geq_klg_2(yhat=np.array(concatenated_outputs), 
                        y=np.array(concatenated_labels), 
                        klg=np.array(concatenated_numerical_klg), 
                        ses=np.array(concatenated_binarized_education_graduated_college), 
                        y_col=self.y_col)
                    income_pain_gaps = analysis.compare_pain_levels_for_people_geq_klg_2(yhat=np.array(concatenated_outputs), 
                        y=np.array(concatenated_labels), 
                        klg=np.array(concatenated_numerical_klg), 
                        ses=np.array(concatenated_binarized_income_at_least_50k), 
                        y_col=self.y_col)

                    metrics_for_epoch['%s_pain_gaps_klg_geq_2' % phase] = {'binarized_education_graduated_college':education_pain_gaps, 
                    'binarized_income_at_least_50k':income_pain_gaps}
                
                if phase == 'test' or phase == 'val':
                    # Stratify test performance by KLG. 
                    metrics_for_epoch['stratified_by_klg'] = {}
                    for klg_grade_to_use in range(5):
                        klg_idxs = np.array(concatenated_numerical_klg) == klg_grade_to_use
                        metrics_for_epoch['stratified_by_klg'][klg_grade_to_use] = analysis.assess_performance(
                            y=np.array(concatenated_labels)[klg_idxs], yhat=np.array(concatenated_outputs)[klg_idxs], binary_prediction=False)

                    # Stratify performance excluding one site at a time. 
                    metrics_for_epoch['stratified_by_site'] = {}
                    concatenated_site = np.array(concatenated_site)
                    for site_val in sorted(list(set(concatenated_site))):
                        exclude_site_idxs = concatenated_site != site_val
                        metrics_for_epoch['stratified_by_site']['every_site_but_%s' % site_val] = analysis.assess_performance(
                            y=np.array(concatenated_labels)[exclude_site_idxs], yhat=np.array(concatenated_outputs)[exclude_site_idxs], binary_prediction=False)


            else:
                metrics_for_epoch['%s_binarized_auc' % phase] = None
                metrics_for_epoch['%s_binarized_auprc' % phase] = None

            if self.n_additional_image_features_to_predict == 0:
                assert np.allclose(np.sqrt(epoch_loss), correlation_and_rmse['rmse'])
        else:
            concatenated_labels = np.array(concatenated_labels)
            concatenated_outputs = np.array(concatenated_outputs)
            auc_and_auprc = analysis.assess_performance(y=concatenated_labels, yhat=concatenated_outputs, binary_prediction=True)

            metrics_for_epoch['%s_loss' % phase] = epoch_loss
            metrics_for_epoch['%s_auc' % phase] = auc_and_auprc['auc']
            metrics_for_epoch['%s_auprc' % phase] = auc_and_auprc['auprc']
            print("%s AUC: %2.6f; AUPRC: %2.6f; loss: %2.6f" % (phase, auc_and_auprc['auc'], auc_and_auprc['auprc'], epoch_loss))
        if self.n_additional_image_features_to_predict > 0:
            print("Loss divided by additional loss is %2.3f (median ratio across batches)" % np.median(loss_additional_loss_ratios))
        metrics_for_epoch['%s_yhat' % phase] = concatenated_outputs
        metrics_for_epoch['%s_y' % phase] = concatenated_labels
        return metrics_for_epoch

    def train(self, dataloaders, dataset_sizes):
        """
        trains the model. dataloaders + dataset sizes should have keys train, val, and test. Checked. 
        """

        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_metric_val = -np.inf

        all_metrics = {}

        for epoch in range(self.num_epochs):
            epoch_t0 = time.time()

            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)
            metrics_for_epoch = {}
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                metrics_for_phase = self.evaluate_on_dataset(dataloaders, dataset_sizes, phase)

                # Change the learning rate. 
                if phase == 'val':
                    if self.lr_scheduler_type == 'decay':
                        self.scheduler.step()
                    elif self.lr_scheduler_type == 'plateau':
                        self.scheduler.step(
                            metrics_for_phase[self.metric_to_use_as_stopping_criterion])
                    else:
                        raise Exception("Not a valid scheduler type")
                    
                    print("Current learning rate after epoch %i is" % epoch)
                    # https://github.com/pytorch/pytorch/issues/2829 get learning rate. 
                    for param_group in self.optimizer.param_groups:
                        print(param_group['lr'])
                    # print(self.optimizer.state_dict())

                metrics_for_epoch.update(metrics_for_phase)
                # deep copy the model if the validation performance is better than what we've seen so far. 
                if phase == 'val' and metrics_for_phase[self.metric_to_use_as_stopping_criterion] > best_metric_val:
                    best_metric_val = metrics_for_phase[self.metric_to_use_as_stopping_criterion]
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            all_metrics[epoch] = metrics_for_epoch

            print("\n\n***\nPrinting layer magnitudes")
            self.print_layer_magnitudes(epoch)
            if self.where_to_add_klg == 'output':
                print("KLG weights are")
                print(self.model.klg_fc.weight)
            print("Total seconds taken for epoch: %2.3f" % (time.time() - epoch_t0))

        all_metrics['final_results'] = metrics_for_epoch
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        self.model.train(False)  # Set model to evaluate mode
        self.state_dict = best_model_wts

        # evaluate on test set. 
        all_metrics['total_seconds_to_train'] = time_elapsed
        all_metrics['test_set_results'] = self.evaluate_on_dataset(dataloaders, dataset_sizes, 'test')

        return all_metrics

    def get_fully_connected_layer(self, class_idx=None):
        if not self.binary_prediction:
            assert class_idx is None
            for name, param in self.model.named_parameters():
                if name == 'fc.weight':
                    return param.data[0, :].cpu().numpy().flatten() # we need to take zero-th index in case we have extra features. 
            raise Exception("No weight vector found")
        else:
            assert class_idx is not None
            for name, param in self.model.named_parameters():
                if name == 'fc.weight':
                    return param.data.cpu().numpy()[class_idx, :].flatten()
            raise Exception("No weight vector found")

def stratify_results_by_ses(y, yhat, high_ses_idxs, binary_prediction):
    """
    Report performance stratified by low and high SES. 
    """
    assert len(y) == len(yhat) 
    assert len(yhat) == len(high_ses_idxs)
    low_ses_results = analysis.assess_performance(y=y[~high_ses_idxs], 
        yhat=yhat[~high_ses_idxs], 
        binary_prediction=binary_prediction)
    high_ses_results = analysis.assess_performance(y=y[high_ses_idxs], 
        yhat=yhat[high_ses_idxs], 
        binary_prediction=binary_prediction)
    combined_results = {}
    for k in low_ses_results:
        combined_results['low_ses_%s' % k] = low_ses_results[k]
    for k in high_ses_results:
        combined_results['high_ses_%s' % k] = high_ses_results[k]
    return combined_results

def train_one_model(experiment_to_run):
    """
    Main method used for training one model. 
    experiment_to_run specifies our experimental condition. 
    """
    timestring = str(datetime.datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_').replace('-', '_')

    # load data. 
    if experiment_to_run == 'train_random_model':
        dataset_kwargs, model_kwargs = generate_random_config()
    elif experiment_to_run == 'predict_klg':
        dataset_kwargs, model_kwargs = generate_config_that_performs_well('koos_pain_subscore')
        # downweight additional loss by factor of roughly (std(koos_pain_score) / std(xrkl))^2
        # This is approximately 200. 
        model_kwargs['additional_loss_weighting'] = model_kwargs['additional_loss_weighting'] / 200.
        model_kwargs['y_col'] = 'xrkl'
        dataset_kwargs['y_col'] = 'xrkl'
    elif experiment_to_run == 'train_on_single_klg':
        dataset_kwargs, model_kwargs = generate_config_that_performs_well('koos_pain_subscore')
        dataset_kwargs['train_on_single_klg_kwargs'] = {'klg_to_use':random.choice([0, 1, 4]), 'make_train_set_smaller':False} # 1, 2, 3
        #model_kwargs['num_epochs'] = random.choice([25, 35, 50]) # 15
        #model_kwargs["scheduler_kwargs"]["additional_kwargs"]["factor"] = random.choice([0.5, 0.75, 0.9])
        #model_kwargs["scheduler_kwargs"]["additional_kwargs"]['patience'] = random.choice([1, 2])
    elif experiment_to_run == 'predict_residual':
        dataset_kwargs, model_kwargs = generate_config_that_performs_well('koos_pain_subscore')
        # don't downweight additional loss because residual is approximately the same scale. 
        model_kwargs['y_col'] = 'koos_pain_subscore_residual'
        dataset_kwargs['y_col'] = 'koos_pain_subscore_residual'
    elif experiment_to_run == 'train_best_model_binary':
        dataset_kwargs, model_kwargs = generate_config_that_performs_well('binarized_koos_pain_subscore')
    elif experiment_to_run == 'train_best_model_continuous':
        dataset_kwargs, model_kwargs = generate_config_that_performs_well('koos_pain_subscore')
    elif experiment_to_run == 'increase_diversity':
        dataset_kwargs, model_kwargs = generate_config_that_performs_well('koos_pain_subscore')
        ses_col = random.choice(['binarized_income_at_least_50k', 'binarized_education_graduated_college'])
        n_seeds_to_fit = 5
        if ses_col == 'race_black':
            minority_val = 1
        else:
            minority_val = 0
        if random.random() < 1./(n_seeds_to_fit + 1.):
            exclude_minority_group = True
        else:
            exclude_minority_group = False
        dataset_kwargs['increase_diversity_kwargs'] = {'ses_col':ses_col, 'minority_val':minority_val, 'exclude_minority_group':exclude_minority_group}
        if not dataset_kwargs['increase_diversity_kwargs']['exclude_minority_group']:
            dataset_kwargs['increase_diversity_kwargs']['majority_group_seed'] = random.choice(range(n_seeds_to_fit))
        else:
            dataset_kwargs['increase_diversity_kwargs']['majority_group_seed'] = None
    elif experiment_to_run == 'change_ses_weighting':
        dataset_kwargs, model_kwargs = generate_config_that_performs_well('koos_pain_subscore')
        ses_col = 'race_black'
        if ses_col == 'binarized_income_at_least_50k':
            p = random.choice([0, 1])
        elif ses_col == 'binarized_education_graduated_college':
            p = random.choice([0, 1])
        elif ses_col == 'race_black':
            p = 0 # remove minority group; can't remove majority group because minority is too small. 
        else:
            raise Exception("invalid ses col")

        dataset_kwargs['weighted_ses_sampler_kwargs'] = {'ses_col':ses_col, 
                                'covs':None,#DEMOGRAPHIC_CONTROLS + ['C(xrkl)'], 
                                'p_high_ses':p,
                                'use_propensity_scores':False}
    elif experiment_to_run == 'change_ses_weighting_with_propensity_matching':
        dataset_kwargs, model_kwargs = generate_config_that_performs_well('koos_pain_subscore')
        dataset_kwargs['weighted_ses_sampler_kwargs'] = {'ses_col':'binarized_income_at_least_50k', 
                                'covs':AGE_RACE_SEX_SITE + ['C(xrkl)'], 
                                'p_high_ses':random.choice([0., 0.5, 1.]),#random.choice([.1, .25, .5, .75, .9]), 
                                'use_propensity_scores':True}
    elif experiment_to_run == 'remove_correlation_between_pain_and_ses':
        dataset_kwargs, model_kwargs = generate_config_that_performs_well('koos_pain_subscore')
        dataset_kwargs['remove_correlation_between_pain_and_ses_kwargs'] = {'ses_col':random.choice(['binarized_income_at_least_50k', 
            'binarized_education_graduated_college', 'race_black']), 
        'pain_col':'koos_pain_subscore'}
    elif experiment_to_run == 'train_on_both_knees':
        dataset_kwargs, model_kwargs = generate_random_config()
        dataset_kwargs['show_both_knees_in_each_image'] = True
    elif experiment_to_run == 'alter_train_set_size':
        dataset_kwargs, model_kwargs = generate_random_config()
        dataset_kwargs['alter_train_set_size_sampler_kwargs'] = {'fraction_of_train_set_to_use':
        random.choice([.1, .2, .5, .75, .9, 1])}
    elif experiment_to_run == 'different_random_seeds':
        dataset_kwargs, model_kwargs = generate_config_that_performs_well('koos_pain_subscore')
        dataset_kwargs['seed_to_further_shuffle_train_test_val_sets'] = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    elif experiment_to_run == 'blur_image':
        dataset_kwargs, model_kwargs = generate_config_that_performs_well('koos_pain_subscore')
        dataset_kwargs['blur_filter'] = random.choice([1/2., 1/4., 1/8., 1/16., 1/32., 1/64.])#random.choice([0, 1, 2, 5, 8, 10, 15, 20, 50])
    elif experiment_to_run == 'hold_out_one_imaging_site':
        dataset_kwargs, model_kwargs = generate_config_that_performs_well('koos_pain_subscore')
        dataset_kwargs['hold_out_one_imaging_site_kwargs'] = {'site_to_remove':random.choice(['A', 'B', 'C', 'D', 'E'])}
    else:
        raise Exception("not a valid experiment")

    assert model_kwargs['y_col'] == dataset_kwargs['y_col']

    print('dataset kwargs', json.dumps(dataset_kwargs, indent=4))
    print('model kwargs', json.dumps(model_kwargs, indent=4))
    dataloaders, datasets, dataset_sizes = load_real_data_in_transfer_learning_format(**dataset_kwargs)

    # actually train model. 
    pytorch_model = TransferLearningPytorchModel(**model_kwargs)
    all_training_results = pytorch_model.train(dataloaders=dataloaders, dataset_sizes=dataset_sizes)

    # stratify test performance by SES. 
    high_ses_idxs = copy.deepcopy(datasets['test'].non_image_data['binarized_income_at_least_50k'] == True).values
    y = copy.deepcopy(all_training_results['test_set_results']['test_y'])
    yhat = copy.deepcopy(all_training_results['test_set_results']['test_yhat'])
    binary_prediction = model_kwargs['binary_prediction']
    ses_stratified_results = stratify_results_by_ses(y=y, 
        yhat=yhat, 
        high_ses_idxs=high_ses_idxs, 
        binary_prediction=binary_prediction)
    all_training_results['test_set_results'].update(ses_stratified_results)

    # Stratify test performance by KLG. 
    all_training_results['test_set_results']['stratified_by_klg'] = {}
    if experiment_to_run != 'predict_klg':
        for klg_grade_to_use in range(5):
            klg_idxs = copy.deepcopy(datasets['test'].non_image_data['xrkl'] == klg_grade_to_use).values
            all_training_results['test_set_results']['stratified_by_klg'][klg_grade_to_use] = analysis.assess_performance(y=y[klg_idxs], yhat=yhat[klg_idxs], binary_prediction=False)
            print("Test results for KLG=%i with %i points are" % (klg_grade_to_use, klg_idxs.sum()))
            print(all_training_results['test_set_results']['stratified_by_klg'][klg_grade_to_use])

    # save config. 
    print("Saving weights, config, and results at timestring %s" % timestring)
    config = {'dataset_kwargs':dataset_kwargs, 'model_kwargs':model_kwargs, 'experiment_to_run':experiment_to_run}
    config_path = os.path.join(FITTED_MODEL_DIR, 'configs', '%s_config.pkl' % timestring)
    pickle.dump(config, open(config_path, 'wb'))

    # save results
    results_path = os.path.join(FITTED_MODEL_DIR, 'results', '%s_results.pkl' % timestring)
    pickle.dump(all_training_results, open(results_path, 'wb'))

    # save model weights. 
    weights_path = os.path.join(FITTED_MODEL_DIR, 'model_weights', '%s_model_weights.pth' % timestring)
    torch.save(pytorch_model.model.state_dict(), weights_path)


def generate_random_config():
    """
    Generate a random config that specifies the dataset + model configuration. 
    Checked. 
    """
    #print("Random state at the beginning is", random.getstate())
    y_col = random.choice(['koos_pain_subscore'])#, 'binarized_koos_pain_subscore'])#random.choice(['binarized_education_graduated_college', 'binarized_income_at_least_50k', 'koos_pain_subscore_residual'])
    if y_col in ['binarized_koos_pain_subscore', 'binarized_education_graduated_college', 'binarized_income_at_least_50k']:
        binary_prediction = True
    elif y_col in ['koos_pain_subscore', 'xrkl', 'koos_pain_subscore_residual']:
        binary_prediction = False
    else:
        raise Exception("Not a valid y column")

    crop_to_just_the_knee = False

    if not crop_to_just_the_knee:
        show_both_knees_in_each_image = random.choice([True])# Seems to perform slightly better and also more interpretable., False])
        if show_both_knees_in_each_image:
            max_horizontal_translation = random.choice([0, .1, .2])
        else:
            max_horizontal_translation = random.choice([0, .25, .5, .75]) 
    else:
        show_both_knees_in_each_image = False
        max_horizontal_translation = random.choice([0, .1, .2])



    dataset_kwargs = {
    'y_col':y_col,
    'max_horizontal_translation': max_horizontal_translation,
    'max_vertical_translation':random.choice([0, .1, .2]), # Tried 0.5, but seems like a little too much, and doesn't improve performance. 
    'use_very_very_small_subset':False,
    'crop_to_just_the_knee':crop_to_just_the_knee,
    'show_both_knees_in_each_image':show_both_knees_in_each_image,
    'downsample_factor_on_reload':random.choice([None]) if not crop_to_just_the_knee else random.choice([0.5, None]),#random.choice([None, 0.7, 0.5, 0.3]), # Originally images were 512 x 512 and downsample factors were [None, 0.7]. Now images are 1024 by 1024. 
    'weighted_ses_sampler_kwargs':None, 
    'additional_features_to_predict':random.choice([None, CLINICAL_CONTROL_COLUMNS, OTHER_KOOS_PAIN_SUBSCORES, CLINICAL_CONTROL_COLUMNS+OTHER_KOOS_PAIN_SUBSCORES]),
    'seed_to_further_shuffle_train_test_val_sets':None}

    if (dataset_kwargs['downsample_factor_on_reload'] in [0.7, None]) and not crop_to_just_the_knee:
        dataset_kwargs['normalization_method'] = 'our_statistics' # we only moved these images to hyperion3, after verifying that they perform well. 
    else:
        dataset_kwargs['normalization_method'] = random.choice(['imagenet_statistics', 'our_statistics', 'zscore_individually'])



    if dataset_kwargs['downsample_factor_on_reload'] is not None and dataset_kwargs['downsample_factor_on_reload'] <= 0.5:
        dataset_kwargs['batch_size'] = random.choice([8, 16, 32]) # larger than 8 and we run out of memory for large resnets. 
    else:
        dataset_kwargs['batch_size'] = random.choice([8, 16]) # avoid memory problems for large batches. 

    pretrained_model_name = random.choice(['resnet18'])

    model_kwargs = {'pretrained_model_name':pretrained_model_name,#['resnet18']),#, deeper nets like resnet34 seem unnecessary; also tried 'resnet50', 'resnet101', 'resnet152']),
    'num_epochs':random.choice([10, 20, 30]) if 'mura' in pretrained_model_name else random.choice([10, 15]), # Resnet appears to do fine with 10 epochs; even if we use a decaying learning rate, doesn't really help. Also tried 20, 25 and 50. For mura pretrained model, longer training is better. 
    'fully_connected_bias_initialization':90 if y_col == 'koos_pain_subscore' else None,
    'n_additional_image_features_to_predict':0 if dataset_kwargs['additional_features_to_predict'] is None else len(dataset_kwargs['additional_features_to_predict']), # this improves results for binary prediction
    'binary_prediction':binary_prediction,
    # for resnet, conv_layers_before_end_to_unfreeze do better; < 4 does worse. 
    'conv_layers_before_end_to_unfreeze':random.choice(list(range(6, 17))) if 'mura' not in pretrained_model_name else random.choice([10, 15, 20, 25, 30] + list(range(5, 10))), 
    'optimizer_name':random.choice(['adam']), # also tried SGD -- ok, but not as good.  
    'y_col':y_col,
    'scheduler_kwargs':{'lr_scheduler_type':random.choice(['decay', 'plateau'])}
    }

    if model_kwargs['scheduler_kwargs']['lr_scheduler_type'] == 'decay':
        model_kwargs['scheduler_kwargs']['additional_kwargs'] = {
        'step_size':random.choice([10, 5, 2, 1]), 
        'gamma':random.choice([.5, .2, .1, .05])
        }
    elif model_kwargs['scheduler_kwargs']['lr_scheduler_type'] == 'plateau':
        model_kwargs['scheduler_kwargs']['additional_kwargs'] = {'mode':'max', 
        'factor':random.choice([.1, .2, .5]), 
        'patience':1, 
        'verbose':True
        }
    else:
        raise Exception("Invalid scheduler type")

    if 'koos_pain' in y_col:
        assert model_kwargs['pretrained_model_name'] == 'resnet18' # otherwise the layer counts will be off and we should change them. 
        if model_kwargs['conv_layers_before_end_to_unfreeze'] >= 12:
            model_kwargs['where_to_add_klg'] = None #random.choice([None, 'before_layer2', 'before_layer3', 'before_layer4', 'output'])
        elif model_kwargs['conv_layers_before_end_to_unfreeze'] >= 8:
            model_kwargs['where_to_add_klg'] = None # random.choice([None, 'before_layer3', 'before_layer4', 'output'])
        elif model_kwargs['conv_layers_before_end_to_unfreeze'] >= 4:
            model_kwargs['where_to_add_klg'] = None # random.choice([None, 'before_layer4', 'output'])
        else:
            model_kwargs['where_to_add_klg'] = None # random.choice([None, 'output'])
    else:
        model_kwargs['where_to_add_klg'] = None
    assert model_kwargs['where_to_add_klg'] in [None]#, 'before_layer4', 'before_layer3', 'before_layer2', 'output']
    
    if 'mura' in pretrained_model_name:
        all_mura_models = os.listdir(os.path.join(BASE_MURA_DIR, 'models'))
        model_kwargs['mura_initialization_path'] = os.path.join(BASE_MURA_DIR, 'models', random.choice(all_mura_models))

    if model_kwargs['n_additional_image_features_to_predict'] > 0:
        if binary_prediction:
            model_kwargs['additional_loss_weighting'] = random.choice([0.1, 0.5, 1, 2, 5, 10, 20]) # tried smaller values, didn't do as well. 
        elif y_col == 'koos_pain_subscore_residual':
            model_kwargs['additional_loss_weighting'] = random.choice([.1, 1, 5, 10, 50, 100])
        elif y_col == 'koos_pain_subscore':
            # need a higher weighting because this loss is on a larger scale (Koos ranges from 0 - 100).
            model_kwargs['additional_loss_weighting'] = random.choice([1, 5, 10, 50, 100, 500, 1000, 5000, 10000])
        elif y_col == 'xrkl':
            model_kwargs['additional_loss_weighting'] = random.choice([.2, .5, 1, 5, 10])
        else:
            raise Exception("Not a valid y column")
    else:
        model_kwargs['additional_loss_weighting'] = 0

    if model_kwargs['optimizer_name'] == 'adam':
        model_kwargs['optimizer_kwargs'] = {'lr':random.choice([0.005, 0.001, 0.0005, 0.0001, 0.00005]), 
        'betas':(random.choice([0.9, 0.95, 0.99]), random.choice([0.99, 0.999, 0.9999]))}
    elif model_kwargs['optimizer_name'] == 'sgd':
        model_kwargs['optimizer_kwargs'] = {'lr':random.choice([0.001, 0.0005, 0.0001, 0.00005]), 
        'momentum':random.choice([.99, .95, .9])}
    
    return dataset_kwargs, model_kwargs

def generate_config_that_performs_well(variable_to_predict):
    
    if variable_to_predict == 'koos_pain_subscore':
        dataset_kwargs = {
            "additional_features_to_predict": CLINICAL_CONTROL_COLUMNS,
            "batch_size": 8,
            "crop_to_just_the_knee": False,
            "downsample_factor_on_reload": None,
            "max_horizontal_translation": 0.1,
            "max_vertical_translation": 0.1,
            "normalization_method": "our_statistics",
            "seed_to_further_shuffle_train_test_val_sets": None,
            "show_both_knees_in_each_image": True,
            "use_very_very_small_subset": False,
            "weighted_ses_sampler_kwargs": None,
            "increase_diversity_kwargs":None,
            "hold_out_one_imaging_site_kwargs":None,
            "y_col": "koos_pain_subscore",
            "blur_filter":None
        }

        model_kwargs = {
            "additional_loss_weighting": 50,
            "binary_prediction": False,
            "conv_layers_before_end_to_unfreeze": 12,
            "fully_connected_bias_initialization": 90,
            "n_additional_image_features_to_predict": 19,
            "num_epochs": 15,
            "optimizer_kwargs": {
                "betas": [
                    0.9,
                    0.999
                ],
                "lr": 0.0005
            },
            "optimizer_name": "adam",
            "pretrained_model_name": "resnet18",
            "scheduler_kwargs": {
                "additional_kwargs": {
                    "factor": 0.5,
                    "mode": "max",
                    "patience": 1,
                    "verbose": True
                },
                "lr_scheduler_type": "plateau"
            },
            "where_to_add_klg": None,
            "y_col": "koos_pain_subscore"
        }
    elif variable_to_predict == 'binarized_koos_pain_subscore':
        raise Exception("This is deprecated and binary config should be updated.")
        # Loading model from timestring 2018_11_11_14_31_49_206053
        dataset_kwargs = {
            "batch_size": 8,
            "downsample_factor_on_reload": None,
            "max_horizontal_translation": 0.5,
            "max_vertical_translation": 0,
            "normalization_method": "our_statistics",
            "use_very_very_small_subset": False,
            "show_both_knees_in_each_image":False,
            "y_col": "binarized_koos_pain_subscore"
        }

        model_kwargs = {
            "additional_loss_weighting": 5,
            "binary_prediction": True,
            "conv_layers_before_end_to_unfreeze": 14,
            "fully_connected_bias_initialization": None,
            "n_additional_image_features_to_predict": 19,
            "num_epochs": 10,
            "optimizer_kwargs": {
                "betas": [
                    0.95,
                    0.9999
                ],
                "lr": 5e-05
            },
            "optimizer_name": "adam",
            "pretrained_model_name": "resnet18",
            "scheduler_kwargs": {
                "gamma": 0.2,
                "step_size": 10
            },
            "y_col": "binarized_koos_pain_subscore"
        }
    else:
        raise Exception("Not a valid variable")
    return dataset_kwargs, model_kwargs


def plot_training_curve(pytorch_model, previous_results):
    """
    plot a model's train and validation metrics by epoch. 
    Checked. 
    """
    previous_results = copy.deepcopy(previous_results)
    epochs = sorted([a for a in previous_results if type(a) is int])
    
    if not pytorch_model.binary_prediction:
        metrics_to_plot = ['r', 'rmse', 'loss']
    else:
        metrics_to_plot = ['auc', 'auprc', 'loss']
    plt.figure(figsize=[12, 4])
    for subplot_idx, metric in enumerate(metrics_to_plot):
        plt.subplot(1, len(metrics_to_plot), subplot_idx + 1)
        for dataset in ['train', 'val']:
            ys = [previous_results[epoch]['%s_%s' % (dataset, metric)] for epoch in epochs]
            plt.plot(epochs, ys, label=dataset)
        plt.legend()
        plt.title(metric)
    plt.show()


def load_model_and_data_from_timestring(timestring, compute_yhats, make_the_cam_plots=True, make_the_prediction_change_plots=True):
    """
    Given a timestring, loads the model and data. 
    If compute_yhats is true, recomputes model estimates on validation + test set to make sure we loaded it correctly. 
    """
    gc.collect()
    print("Loading model from timestring %s" % timestring)
    config_path = os.path.join(FITTED_MODEL_DIR, 'configs', '%s_config.pkl' % timestring)
    config = pickle.load(open(config_path, 'rb'))
    dataset_kwargs = config['dataset_kwargs']
    
    if 'y_col' not in config['model_kwargs']:
        config['model_kwargs']['y_col'] = dataset_kwargs['y_col']

    print("\ndataset kwargs")
    print(json.dumps(dataset_kwargs, sort_keys=True, indent=4))
    print("\nmodel kwargs")
    print(json.dumps(config['model_kwargs'], sort_keys=True, indent=4))
    
    print("Loading model")
    weights_path = os.path.join(FITTED_MODEL_DIR, 'model_weights', '%s_model_weights.pth' % timestring)
    pytorch_model = TransferLearningPytorchModel(**config['model_kwargs'])
    print("Loading model weights from %s" % weights_path)
    pytorch_model.model.load_state_dict(torch.load(weights_path))
    pytorch_model.model.train(False)

    results_path = os.path.join(FITTED_MODEL_DIR, 'results', '%s_results.pkl' % timestring)
    previous_results = pickle.load(open(results_path, 'rb'))
    plot_training_curve(pytorch_model, previous_results)

    best_epoch = sorted([a for a in previous_results.keys() if type(a) is int], 
        key=lambda epoch:previous_results[epoch][pytorch_model.metric_to_use_as_stopping_criterion])[::-1][0]
    print("Previously saved results")
    if not pytorch_model.binary_prediction:
        print('val RMSE: %2.6f correlation %2.6f' % 
                (previous_results[best_epoch]['val_rmse'], previous_results[best_epoch]['val_r']))
        print('test RMSE: %2.6f correlation %2.6f' % 
                (previous_results['test_set_results']['test_rmse'], previous_results['test_set_results']['test_r']))
    else:
        print("val AUC: %2.6f; AUPRC: %2.6f; loss: %2.6f" % (
            previous_results[best_epoch]['val_auc'],
            previous_results[best_epoch]['val_auprc'],
            previous_results[best_epoch]['val_loss']))
        print("test AUC: %2.6f; AUPRC: %2.6f; loss: %2.6f" % (
            previous_results['test_set_results']['test_auc'],
            previous_results['test_set_results']['test_auprc'],
            previous_results['test_set_results']['test_loss']))



    print("loading datasets")
    dataloaders, datasets, dataset_sizes = load_real_data_in_transfer_learning_format(**dataset_kwargs)

    plt.figure(figsize=[4, 4])
    if dataset_kwargs['y_col'] == 'koos_pain_subscore':
        assert (np.abs(np.array(previous_results['test_set_results']['test_y']) - datasets['test'].non_image_data['koos_pain_subscore'].values) < 1e-4).all()
        rounded_y = pd.cut(datasets['test'].non_image_data['koos_pain_subscore'], 
        bins=range(50, 101, 10)).values
        plt.ylim([70, 100])
    elif dataset_kwargs['y_col'] == 'koos_pain_subscore_residual':
        rounded_y = pd.cut(datasets['test'].non_image_data['koos_pain_subscore_residual'], 
        bins=range(-20, 21, 5)).values
        plt.ylim([-10, 10])
    elif config['model_kwargs']['binary_prediction']:
        y_col_to_use = config['dataset_kwargs']['y_col']
        rounded_y = copy.deepcopy(datasets['test'].non_image_data[y_col_to_use].values)
        plt.ylim([0, 1])
    elif config['model_kwargs']['y_col'] == 'xrkl':
        y_col_to_use = config['dataset_kwargs']['y_col']
        rounded_y = copy.deepcopy(datasets['test'].non_image_data[y_col_to_use].values)
        plt.ylim([0, 4])
    else:
        raise Exception("This doesn't work yet")
    yhat = copy.deepcopy(previous_results['test_set_results']['test_yhat'])

    if not (len(yhat) == len(datasets['test'])):
        raise Exception("Length of predictions (%i) does not match length of dataset (%i); this is likely caused by a change in the dataset since you trained the model." % 
            (len(yhat), len(datasets['test'])))

    prediction_df_to_plot = pd.DataFrame({'Y':rounded_y, 'Yhat':yhat})
    print(prediction_df_to_plot.groupby('Y').agg(['mean', 'size']))
    # violin plot cuz scatterplot is ugly. 
    sns.violinplot(x=prediction_df_to_plot['Y'],
        y=prediction_df_to_plot['Yhat'])
    plt.show()

    plt.figure()
    # scatter plot for completeness. 
    plt.scatter(x=previous_results['test_set_results']['test_yhat'],
        y=previous_results['test_set_results']['test_y'])
    plt.title('test correlation: %2.3f' % pearsonr(previous_results['test_set_results']['test_yhat'], 
        previous_results['test_set_results']['test_y'])[0])
    plt.xlabel("Yhat")
    plt.ylabel("Y")
    plt.show()

    if compute_yhats:
        print("Evaluating reloaded model on same test set...")
        reconstituted_test_results = pytorch_model.evaluate_on_dataset(dataloaders, dataset_sizes, phase='test', make_plot=True)
        assert np.allclose(reconstituted_test_results['test_loss'], previous_results['test_set_results']['test_loss'], rtol=1e-4)
        print("Test losses are the same.")

        print("Evaluating reloaded model on same validation set...")
        reconstituted_val_results = pytorch_model.evaluate_on_dataset(dataloaders, dataset_sizes, phase='val', make_plot=True)
        assert np.allclose(reconstituted_val_results['val_loss'], previous_results[best_epoch]['val_loss'], rtol=1e-4)
        print("Validation losses are the same.")
        yhats = {'test_yhat':reconstituted_test_results['test_yhat'], 'val_yhat':reconstituted_val_results['val_yhat']}        
    else:
        yhats = {'test_yhat':copy.deepcopy(previous_results['test_set_results']['test_yhat']), 
        'val_yhat':None}
    gc.collect()

    random_plot_string = random.choice(list(range(100000)))
    rng = random.Random(42)
    image_idxs_for_interpretability_plots = rng.sample(range(len(yhats['test_yhat'])), 8)
    print("Interpretability images are", image_idxs_for_interpretability_plots)
    if make_the_cam_plots:
        # for some reason if you make the interpretability plots and THEN loop over the whole dataset to recompute yhats
        # you get out of memory errors. 
        make_cam_plots(pytorch_model, dataset_kwargs=dataset_kwargs, filename_string='test_cam_%i.png' % random_plot_string, plot_title='test', img_idxs=image_idxs_for_interpretability_plots)
        plt.show()

    if make_the_prediction_change_plots:
        make_prediction_change_plots(pytorch_model, figtitle='test_prediction_change_%i.png' % random_plot_string, dataset_kwargs=dataset_kwargs, n_images_to_plot=8, 
            img_idxs=image_idxs_for_interpretability_plots)
        
    
    gc.collect()



    return pytorch_model, dataloaders, datasets, dataset_sizes, yhats, dataset_kwargs


def returnCAM(feature_conv, weight_vector, size_to_upsample_to):
    """
    generate the class activation maps and upsample to size_to_upsample_to.
    Original source: https://github.com/metalbubble/CAM/blob/master/pytorch_CAM.py
    size_to_upsample_to should be a height, width. 
    Checked.
    """
    bz, nc, h, w = feature_conv.shape # (1, n_channels, height, width). For example, in resnet18, n_channels = 512.
    print("Dimensions of final CAM representation layer:", nc, h, w)
    assert nc == len(weight_vector)
    assert bz == 1

    cam = weight_vector.dot(feature_conv.reshape((nc, h*w))) # This upweights each filter by its weight in the fully connected layer. 
    cam = cam.reshape(h, w) 

    # just a sanity check to make sure this is doing what I think it's doing. 
    cam2 = np.zeros([h, w])
    for i in range(len(weight_vector)):
        cam2 += weight_vector[i] * feature_conv[0, i, :, :].squeeze()
    assert np.allclose(cam2, cam, atol=1e-5)

    # scale to 0 - 255. 
    #cam = cam - np.min(cam) 
    #cam_img = cam / np.max(cam) 
    #cam_img = np.uint8(255 * cam_img) # normalize to be on 0-255
    return cv2.resize(cam, size_to_upsample_to[::-1]) # upsample. We have to flip dimensions because cv2 uses width, height as opposed to height, width. 

def generate_mask_array(center_x, center_y, height, width, kernel_width):
    """
    Helper method to generate make_prediction_change plots. Create a mask centered at center_x, center_y. 
    This is approximate: we only compute nonzero values in vicinity of center_x, center_y to keep it from being really slow. 
    """
    mask_arr = np.zeros([height, width])
    for i in range(center_x - kernel_width * 2, center_x + kernel_width * 2 + 1):
        for j in range(center_y - kernel_width * 2, center_y + kernel_width * 2 + 1):
            if i >= 0 and j >= 0 and i < width and j < height:
                val = np.exp(-((i - center_x) ** 2 + (j - center_y) ** 2) / kernel_width ** 2)
                mask_arr[j][i] = val
    return mask_arr

def prediction_change_helper_get_pred_from_numpy_img(pytorch_model, img):
    # small helper method: make a prediction on a numpy image.
    img_tensor = Variable(torch.from_numpy(img).unsqueeze(0).float().cuda())
    pred = pytorch_model.model(img_tensor).cpu().data.numpy()[0]
    if pytorch_model.binary_prediction:
        pred = expit(pred[1])
    else:
        pred = pred[0]
    return pred
    
def make_prediction_change_plots(trained_model, dataset_kwargs, n_images_to_plot=8, dataset_to_use='test', figtitle=None, climit=None, random_seed=None, img_idxs=None):
    """
    Plot how much the predictions change at a point when we mask out a blob centered on that point. 
    trained_model should be either a model or a list of timestrings. 
    """
    all_trained_models = []
    if type(trained_model) is list:
        print("Averaging results from %i pretrained models" % len(trained_model))
        # list of timestrings
        for timestring in trained_model:
            all_trained_models.append(load_model_and_data_from_timestring(timestring,
                compute_yhats=False,
                make_the_cam_plots=False,
                make_the_prediction_change_plots=False)[0])
    else:
        all_trained_models = [trained_model]
    trained_model = None # avoid using this variable accidentally. 

    assert dataset_to_use in ['train', 'val', 'test']
    binary_prediction = all_trained_models[0].binary_prediction

    if img_idxs is not None:
        assert random_seed is None
        assert n_images_to_plot == len(img_idxs)

    # we just need to load dataset so we can get the non-image data. 
    image_dataset = PytorchImagesDataset(dataset=dataset_to_use, 
                         downsample_factor_on_reload=dataset_kwargs['downsample_factor_on_reload'], 
                         normalization_method=dataset_kwargs['normalization_method'], 
                         y_col=dataset_kwargs['y_col'], 
                         transform=None, 
                         show_both_knees_in_each_image=dataset_kwargs['show_both_knees_in_each_image'], 
                         additional_features_to_predict=dataset_kwargs['additional_features_to_predict'],
                         seed_to_further_shuffle_train_test_val_sets=dataset_kwargs['seed_to_further_shuffle_train_test_val_sets'], 
                         crop_to_just_the_knee=dataset_kwargs['crop_to_just_the_knee'], 
                         blur_filter=dataset_kwargs['blur_filter'] if 'blur_filter' in dataset_kwargs else None)

    if random_seed is not None:
        rng = random.Random(random_seed)


    fig = plt.figure(figsize=[10, 2.4 * n_images_to_plot])
     
    for i in range(n_images_to_plot):
        print("image %i" % i)
        if img_idxs is not None:
            random_img = img_idxs[i]
        else:
            if random_seed is not None:
                random_img = rng.choice(list(range(len(image_dataset))))
            else:
                random_img = random.choice(list(range(len(image_dataset))))
        datapoint = image_dataset[random_img]
        true_y = datapoint['y']
        img = datapoint['image']
        original_means = img.mean(axis=1).mean(axis=1)
        original_preds = []
        for trained_model in all_trained_models:
            original_preds.append(prediction_change_helper_get_pred_from_numpy_img(trained_model, img))

        all_pred_differences = [[] for a in all_trained_models]
        img_h = img.shape[1]
        img_w = img.shape[2]
        assert img_h >= img_w # height should be greater than or equal to width. 

        n_height_bins = 32
        n_width_bins = int(n_height_bins * 1.* img_w / img_h)
        assert n_height_bins == n_width_bins
        mask_radius = int(img_h / n_height_bins) # kernel width is approximately 1/n_height_bins of the height. 
        print("Number of width bins: %i; height bins: %i; mask radius %2.3f" % (n_width_bins, n_height_bins, mask_radius))

        # current we are using evenly spaced bins that are approximately mask_radius apart. 
        # bins start and end mask_radius/2 from the edge. 
        for model_idx, trained_model in enumerate(all_trained_models):
            for center_y in np.linspace(mask_radius/2., img_h - mask_radius/2., n_height_bins):
                assert np.allclose(np.linspace(mask_radius/2., img_h - mask_radius/2., n_height_bins), range(16, 1024 - 16 + 1, 32))
                all_pred_differences[model_idx].append([])
                center_y = int(center_y)
                for center_x in np.linspace(mask_radius/2., img_w - mask_radius/2., n_width_bins):
                    center_x = int(center_x)
                    mask_weight = generate_mask_array(center_x=center_x, 
                        center_y=center_y, 
                        height=img_h, 
                        width=img_w, 
                        kernel_width=int(mask_radius))
                    modified_img = copy.deepcopy(img)
                    for rgb_idx in range(3):
                        mask = np.ones([img_h, img_w]) * original_means[rgb_idx]
                        assert modified_img[rgb_idx, :, :].shape == mask_weight.shape
                        assert modified_img[rgb_idx, :, :].shape == mask.shape
                        modified_img[rgb_idx, :, :] = modified_img[rgb_idx, :, :] * (1 - mask_weight) + mask * mask_weight
                    
                    abs_pred_diff = np.abs(prediction_change_helper_get_pred_from_numpy_img(trained_model, modified_img) - original_preds[model_idx])
                    all_pred_differences[model_idx][-1].append(abs_pred_diff)

            all_pred_differences[model_idx] = np.array(all_pred_differences[model_idx])
        all_pred_differences = np.array(all_pred_differences)
        print(all_pred_differences.shape)
        all_pred_differences = np.mean(all_pred_differences, axis=0)
        print('Shape of prediction change plot prior to resizing (this tells you the number of bins)', all_pred_differences.shape)
        assert all_pred_differences.shape == tuple([n_width_bins, n_height_bins])
        all_pred_differences = cv2.resize(all_pred_differences, 
                                          tuple([img_w, img_h]))

        assert all_pred_differences.shape == img[0, :, :].shape

        if climit is None:
            climit_for_this_knee = all_pred_differences.max()
        else:
            climit_for_this_knee = climit

        zscored_image_to_plot = img[0, :, :].copy()
        zscored_image_to_plot = (zscored_image_to_plot - zscored_image_to_plot.mean())/zscored_image_to_plot.std(ddof=1)
        plt.subplot(n_images_to_plot, 3, 3 * i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(zscored_image_to_plot, cmap='bone', clim=[-3, 3])
        plt.title('y: %2.1f; yhat: %2.1f' % (true_y, np.mean(original_preds)))


        plt.subplot(n_images_to_plot, 3, 3 * i + 2)
        plt.imshow(all_pred_differences, clim=[0, climit_for_this_knee])
        plt.xticks([])
        plt.yticks([])

        plt.subplot(n_images_to_plot, 3, 3 * i + 3)
        plt.imshow(zscored_image_to_plot, cmap='bone', clim=[-3, 3])
        plt.imshow(all_pred_differences, alpha=.5, clim=[0, climit_for_this_knee])
        plt.xticks([])
        plt.yticks([])

        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=4)
        cb.locator = tick_locator
        cb.update_ticks()
    plt.subplots_adjust(wspace=.02, hspace=.3, top=.9)
    if figtitle is not None:
        plt.savefig(figtitle, dpi=300)
    plt.show()
    plt.close()


def make_cam_plots(trained_model, dataset_kwargs, filename_string, plot_title, dataset_to_use='test', img_idxs=None):
    """
    Given a trained model and a dataset, make CAM plots for it by sampling random images from the dataset of choice.

    If img_idxs is not None, make plots for those images specifically. 

    Returns a list of CAM maps for the images. 
    """
    assert dataset_to_use in ['train', 'val', 'test']
    binary_prediction = trained_model.binary_prediction

    # we just need to load dataset so we can get the non-image data. 
    image_dataset = PytorchImagesDataset(dataset=dataset_to_use, 
                         downsample_factor_on_reload=dataset_kwargs['downsample_factor_on_reload'], 
                         normalization_method=dataset_kwargs['normalization_method'], 
                         y_col=dataset_kwargs['y_col'], 
                         transform=None, 
                         show_both_knees_in_each_image=dataset_kwargs['show_both_knees_in_each_image'], 
                         additional_features_to_predict=dataset_kwargs['additional_features_to_predict'],
                         seed_to_further_shuffle_train_test_val_sets=dataset_kwargs['seed_to_further_shuffle_train_test_val_sets'],
                         crop_to_just_the_knee=dataset_kwargs['crop_to_just_the_knee'], 
                         blur_filter=dataset_kwargs['blur_filter'] if 'blur_filter' in dataset_kwargs else None)

    print("Number of images: %i" % len(image_dataset))

    # store the weight vectors we want to plot. Each of these weight vectors weight the channels of the final layer embedding. 
    # The final layer embedding is always positive because it is passed through a RELU. 
    if binary_prediction:
        weight_vectors_to_use = {'Pos Class':trained_model.get_fully_connected_layer(class_idx=1), 
                                 'Neg Class':trained_model.get_fully_connected_layer(class_idx=0)}
        order_to_plot_in = ['Pos Class', 'Neg Class']
    else:
        original_weight_vector = trained_model.get_fully_connected_layer()
        print("Minimum value for the weight vector is %2.3f; max is %2.3f" % (original_weight_vector.min(), original_weight_vector.max()))

        weight_vectors_to_use = {'Original W': original_weight_vector, 
                                 '|W|':np.abs(original_weight_vector), 
                                 'max(W, 0)':np.maximum(original_weight_vector, 0), # channels that push us UP. Chris suggests thresholding at 0. 
                                 'min(W, 0)':np.minimum(original_weight_vector, 0)}# channels that push us DOWN.

        order_to_plot_in = ['Original W', '|W|', 'min(W, 0)', 'max(W, 0)']


    n_to_plot = 8
    n_cols = len(weight_vectors_to_use) * 2 + 1
    plt.figure(figsize=[n_cols * 1.2 + 5, n_to_plot + 3])

    # sort images by y. 
    if img_idxs is None:
        img_idxs = random.sample(range(len(image_dataset)), n_to_plot)
    else:
        assert len(img_idxs) == n_to_plot
    ys = [image_dataset[img_idx]['y'] for img_idx in img_idxs]
    sorted_img_idxs = np.argsort(ys)
    img_idxs = [img_idxs[i] for i in sorted_img_idxs]
    CAM_weights = []

    for fig_idx in range(n_to_plot):
        CAM_weights.append([])
        
        img_idx = img_idxs[fig_idx]
        img = image_dataset[img_idx]['image']
        size_to_upsample_to = img[0, :, :].shape
        img_tensor = Variable(torch.from_numpy(img).unsqueeze(0).float().cuda())

        # hook the feature extractor. I don't totally understand the logic but basically, this function 
        # gets called every time we run the model on an image. And when it gets called it appends to the end of conv_layer_embedding. 
        # See https://github.com/metalbubble/CAM/blob/master/pytorch_CAM.py#L36. 
        # per the documentation for get_forward_hook: The hook will be called every time after forward() has computed an output. 
        # for resnet18, finalconv_name='layer4'
        conv_layer_embedding = []
        def copy_conv_layer(module, input, output):
            conv_layer_embedding.append(output.data.cpu().numpy())
        
        trained_model.model._modules.get(trained_model.finalconv_name).register_forward_hook(copy_conv_layer)

        # do a forward pass so we append the embedding to the end of conv_layer_embedding.
        trained_model.model(img_tensor)
        embedding = conv_layer_embedding[-1]
        assert embedding.min() >= 0

        zscored_image_to_plot = img[0, :, :].copy()
        zscored_image_to_plot = (zscored_image_to_plot - zscored_image_to_plot.mean()) / zscored_image_to_plot.std(ddof=1)

        plt.subplot(n_to_plot, n_cols, 1 + fig_idx * n_cols)
        plt.imshow(zscored_image_to_plot, cmap='bone', clim=[-3, 3])
        plt.title('y=%2.3f' % (float(image_dataset.non_image_data[dataset_kwargs['y_col']].iloc[img_idx])))
        plt.xticks([])
        plt.yticks([])

        col_idx = 2
        for weight_vector_name in order_to_plot_in:
            plt.subplot(n_to_plot, n_cols, col_idx + fig_idx * n_cols)
            output_cam = returnCAM(embedding, weight_vectors_to_use[weight_vector_name], size_to_upsample_to)
            CAM_weights[-1].append(output_cam)
            clim = np.abs(output_cam).max()
            plt.imshow(output_cam, clim=[-clim, clim], cmap='bwr')
            plt.title(weight_vector_name)
            plt.xticks([])
            plt.yticks([])
            cb = plt.colorbar()
            tick_locator = ticker.MaxNLocator(nbins=5)
            cb.locator = tick_locator
            cb.update_ticks()
            col_idx += 1

            # overlay on original image. 
            plt.subplot(n_to_plot, n_cols, col_idx + fig_idx * n_cols)
            plt.imshow(zscored_image_to_plot, cmap='bone', clim=[-3, 3])
            plt.imshow(output_cam, clim=[-clim, clim], cmap='bwr', alpha=.4)
            assert output_cam.size == zscored_image_to_plot.size
            plt.xticks([])
            plt.yticks([])
            plt.title(weight_vector_name)
            col_idx += 1

    plt.suptitle(plot_title)
    plt.subplots_adjust(wspace=.1, hspace=.3)
    if filename_string is not None:
        plt.savefig(filename_string, dpi=300)
    plt.show()
    plt.close()
    return CAM_weights

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_to_run', help='The name of the experiment to run')
    args = parser.parse_args()
    while True:
        train_one_model(args.experiment_to_run)



