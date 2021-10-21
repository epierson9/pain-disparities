import matplotlib
matplotlib.use('Agg')
import warnings 
import os
import pandas as pd
import copy
from scipy.stats import pearsonr
import random
from collections import Counter
import numpy as np
import pickle
import platform
import sys
import subprocess
import time
import platform
import getpass


node_name = platform.node().split('.')[0]
print("Running code on %s with Python version %s" % (node_name, sys.version.split()[0]))

pd.set_option('max_columns', 500)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="This call to matplotlib.use() has no effect because the backend has already been chosen")


USE_HELD_OUT_TEST_SET = True
MODEL_NAME = 'resnet18'
TOTAL_PEOPLE = 4796
N_BOOTSTRAPS = 1000

if getpass.getuser() == 'emmap1':
    # Do not modify this code; it is the original logic the authors used to process the images/run models, maintained for reproducibility.
    REPROCESS_RAW_DATA = True # set this to False if you just want to work with the processed data, and don't need to reprocess it. 
    assert node_name in ['hyperion', 'hyperion2', 'hyperion3', 'rambo', 'trinity', 'turing1', 'turing2']
    assert sys.version.split()[0] == '3.5.2'
    NODE_TO_USE_TO_STORE_IMAGES_FOR_GPU = 'hyperion3'
    assert NODE_TO_USE_TO_STORE_IMAGES_FOR_GPU in ['turing2', 'hyperion3']
    BASE_NON_IMAGE_DATA_DIR = '/dfs/dataset/tmp/20180910-OAI/data/emma_downloaded_oai_data_9112018/'
    BASE_IMAGE_DATA_DIR = '/dfs/dataset/tmp/20180910-OAI/data/'
    BASE_MURA_DIR = '/dfs/dataset/tmp/20180910-OAI/data/mura_pretrained_weights/'
    DFS_BASE_IMAGES_PATH = os.path.join(BASE_IMAGE_DATA_DIR, 'processed_image_data', 'individual_images')
    FITTED_MODEL_DIR = os.path.join(BASE_IMAGE_DATA_DIR, 'fitted_models')

    assert os.path.exists(DFS_BASE_IMAGES_PATH)
    if node_name in ['rambo', 'trinity', 'hyperion2', 'hyperion', 'turing1', 'turing2', 'turing3', 'turing4']:
        # if we are on rambo or trinity, we are reprocessing the images.
        INDIVIDUAL_IMAGES_PATH = DFS_BASE_IMAGES_PATH
        raise Exception("This is likely the wrong computer to be processing things on; it uses the wrong test set and you should be using hyperion3. Unless you are testing or regenerating data, something may be wrong.")
    else:
        if NODE_TO_USE_TO_STORE_IMAGES_FOR_GPU == 'hyperion3':
            if USE_HELD_OUT_TEST_SET:
                INDIVIDUAL_IMAGES_PATH = '/lfs/hyperion3/0/emmap1/oai_image_data/processed_image_data_new_with_held_out_test_set_april_2019/'
            else:
                INDIVIDUAL_IMAGES_PATH = '/lfs/hyperion3/0/emmap1/oai_image_data/processed_image_data/'
        elif NODE_TO_USE_TO_STORE_IMAGES_FOR_GPU == 'turing2':
            if USE_HELD_OUT_TEST_SET:
                INDIVIDUAL_IMAGES_PATH = '/lfs/turing2/0/emmap1/oai_image_data/processed_image_data_new_with_held_out_test_set_april_2019/'
            else:
                INDIVIDUAL_IMAGES_PATH = '/lfs/turing2/0/emmap1/oai_image_data/processed_image_data/'
        else:
            raise Exception("invalid place to store images for GPU")
else:
    # Please modify variables / paths here. 
    if sys.version.split()[0] != '3.5.2':
        print("Warning: running code with a Python version which differs from original Python version (3.5.2)")
    REPROCESS_RAW_DATA = False # set this to False if you just want to work with the processed data, and don't need to reprocess it. 
    
    # Please set these paths for your system. 
    INDIVIDUAL_IMAGES_PATH = 'THIS_IS_A_TEMPORARY_PATH_PLEASE_REPLACE_ME' # points to the directory which stores the processed data, so you should download the processed data into this folder. If you are reprocessing the raw data, the individual images will be stored in this folder. 
    FITTED_MODEL_DIR = 'THIS_IS_A_TEMPORARY_PATH_PLEASE_REPLACE_ME' # This is where you store the fitted models.  Please create three empty subdirectories in this directory: "configs", "results", and "model_weights". 
    
    # Only need to set these paths if you are reprocessing raw data. 
    BASE_NON_IMAGE_DATA_DIR = 'THIS_IS_A_TEMPORARY_PATH_PLEASE_REPLACE_ME' # Set this path to point to the directory where you downloaded the NON-IMAGE OAI data - eg, it should contain folders like "AllClinical_ASCII". 
    BASE_IMAGE_DATA_DIR = 'THIS_IS_A_TEMPORARY_PATH_PLEASE_REPLACE_ME' # Set this path to point to the directory where you downloaded the IMAGE OAI data - eg, it should contain folders like "00m" for each timepoint. 

assert os.path.exists(INDIVIDUAL_IMAGES_PATH), 'You need to set INDIVIDUAL_IMAGES_PATH; see "Please set these paths for your system" comment in constants_and_util.py'
assert os.path.exists(FITTED_MODEL_DIR), 'You need to set FITTED_MODEL_DIR; see "Please set these paths for your system" comment in constants_and_util.py. After setting this directory, please create empty subdirectories called "configs", "results", and "model_weights" within it'
assert os.path.exists(os.path.join(FITTED_MODEL_DIR, 'configs')) and os.path.exists(os.path.join(FITTED_MODEL_DIR, 'results')) and os.path.exists(os.path.join(FITTED_MODEL_DIR, 'model_weights')), 'Please create empty subdirectories called "configs","results", and "model_weights" within %s' % FITTED_MODEL_DIR

if REPROCESS_RAW_DATA:
    # these paths are primarily used in reprocessing data; they store the non-image data and image data. 
    assert os.path.exists(BASE_NON_IMAGE_DATA_DIR), 'If you are reprocessing raw data, you need to set BASE_NON_IMAGE_DATA_DIR; see "Please set these paths for your system" comment in constants_and_util.py'
    assert os.path.exists(BASE_IMAGE_DATA_DIR), 'If you are reprocessing raw data, you need to set BASE_IMAGE_DATA_DIR; see "Please set these paths for your system" comment in constants_and_util.py'

RESAMPLED_IMAGE_SIZE = [1024, 1024]
CROPPED_KNEE_RESAMPLED_IMAGE_SIZE = [int(0.5 * RESAMPLED_IMAGE_SIZE[0]), int(0.5 * RESAMPLED_IMAGE_SIZE[1])] # smaller because it's just the knee. 
                                     
assert RESAMPLED_IMAGE_SIZE[0] == RESAMPLED_IMAGE_SIZE[1]
IMAGE_DATASET_KWARGS = {'desired_image_type':'Bilateral PA Fixed Flexion Knee',
                'normalization_method':'our_statistics',
                'max_images_to_load':1000000000}
GAPS_OF_INTEREST_COLS = ['race_black', 'binarized_income_at_least_50k', 'binarized_education_graduated_college', 'is_male']
CLINICAL_CONTROL_COLUMNS = ['xrosfm', 'xrscfm','xrcyfm', 'xrjsm', 'xrchm','xrostm','xrsctm','xrcytm','xrattm','xrkl','xrosfl','xrscfl','xrcyfl', 'xrjsl','xrchl','xrostl','xrsctl','xrcytl','xrattl']
OTHER_KOOS_PAIN_SUBSCORES = ['koos_function_score', 'koos_quality_of_life_score', 'koos_symptoms_score']
MEDICATION_CODES = {'V00RXACTM':'Acetaminophen', 
                        'V00RXANALG':'Analgesic',
                        'V00RXASPRN':'Aspirin',
                        'V00RXBISPH':'Bisphosphonate',
                        'V00RXCHOND':'Chondroitin',
                        'V00RXCLCTN':'Calcitonin',
                        'V00RXCLCXB':'Celecoxib',
                        'V00RXCOX2':'COX_II',
                        'V00RXFLUOR':'Fluoride',
                        'V00RXGLCSM':'Glucosamine',
                        'V00RXIHYAL':'Injected_hyaluronic_acid',
                        'V00RXISTRD':'Injected_corticosteroid',
                        'V00RXMSM':'Methylsulfonylmethane',
                        'V00RXNARC':'Narcotic_analgesic',
                        'V00RXNSAID':'NSAID',
                        'V00RXNTRAT':'Nitrate',
                        'V00RXOSTRD':'Oral_corticosteroid',
                        'V00RXOTHAN':'Other_analgesic',
                        'V00RXRALOX':'Raloxifene',
                        'V00RXRFCXB':'Rofecoxib',
                        'V00RXSALIC':'Salicylate',
                        'V00RXSAME':'S_adenosylmethionine',
                        'V00RXTPRTD':'Teriparatide',
                        'V00RXVIT_D':'Vitamin_D',
                        'V00RXVLCXB':'Valdecoxib'
    }

# Variables associated with enrollment visit are prefixed V00, variables at 12-month follow-up are prefixed V01, variables at 18-month interim visit are prefixed V02, variables at 24-month follow-up are prefixed V03, variables at 30-month follow-up are prefixed V04, variables at 36-month follow-up are prefixed V05, variables at 48-month follow-up visit are prefixed V06, variables at 72-month follow-up visit are prefixed V08, and variables at 96-month follow- up visit are prefixed V10.
# Or see also the document ClinicalDataGettingStartedOverview.pdf
CLINICAL_WAVES_TO_FOLLOWUP = {'00':'00 month follow-up: Baseline',
'01':'12 month follow-up',
'03':'24 month follow-up',
'05':'36 month follow-up',
'06':'48 month follow-up',
'07':'60 month follow-up',
'08':'72 month follow-up',
'09':'84 month follow-up',
'10':'96 month follow-up',
'11':'108 month follow-up'}

TIMEPOINTS_TO_FILTER_FOR = ['12 month follow-up', 
                              '24 month follow-up', 
                              '36 month follow-up', 
                              '48 month follow-up', 
                              '00 month follow-up: Baseline']

WAVES_WE_ARE_USING = ['00', '01', '03', '05', '06']

assert set(TIMEPOINTS_TO_FILTER_FOR) == set([CLINICAL_WAVES_TO_FOLLOWUP[a] for a in WAVES_WE_ARE_USING])

TRAIN_VAL_TEST_HOLD_OUT_FRACTIONS = {'train_frac':(TOTAL_PEOPLE - 1500. - 1000.)/TOTAL_PEOPLE,
                                  'val_frac':(500. + 1e-3)/TOTAL_PEOPLE, #1e-3 is a small hack to make sure that train_frac + val_frac + test_frac doesn't get weirdly rounded. Sigh. 
                                  'test_frac':500./TOTAL_PEOPLE, 
                                  'hold_out_frac':1500./TOTAL_PEOPLE} 

assert TRAIN_VAL_TEST_HOLD_OUT_FRACTIONS['hold_out_frac'] <= 1500./TOTAL_PEOPLE # if you do change hold out set, must make it smaller. 


IMAGE_TIMEPOINT_DIRS_TO_FOLLOWUP = {'00m':'00 month follow-up: Baseline', 
'12m':'12 month follow-up', 
'18m':'18 month follow-up', 
'24m':'24 month follow-up', 
'30m':'30 month follow-up', 
'36m':'36 month follow-up', 
'48m':'48 month follow-up', 
'72m':'72 month follow-up', 
'96m':'96 month follow-up'}
KOOS_BINARIZATION_THRESH = 86.1
WOMAC_BINARIZATION_THRESH = 3.

AGE_RACE_SEX_SITE = ['C(age_at_visit)*C(p02sex)', 'C(p02hisp)', "C(p02race, Treatment(reference='1: White or Caucasian'))", 'C(v00site)']
AGE_SEX_SITE_NO_RACE = ['C(age_at_visit)*C(p02sex)', 'C(v00site)']

KNEE_INJURY_OR_SURGERY = ['C(knee_surgery)', 'C(knee_injury)']
MEDICAL_HISTORY = ['C(hrtat)', 
'C(hrtfail)', 'C(bypleg)','C(stroke)', 'C(asthma)', 
    'C(lung)', 'C(ulcer)', 'C(diab)', 'C(kidfxn)', 
    'C(ra)', 'C(polyrh)', 'C(livdam)', 'C(cancer)']
OTHER_PAIN = ['left_hip_pain_more_than_half_of_days',
   'right_hip_pain_more_than_half_of_days',
   'how_often_bothered_by_back_pain',
   'left_foot_pain_more_than_half_of_days',
   'right_foot_pain_more_than_half_of_days',
   'left_ankle_pain_more_than_half_of_days',
   'right_ankle_pain_more_than_half_of_days',
   'left_shoulder_pain_more_than_half_of_days',
   'right_shoulder_pain_more_than_half_of_days',
   'left_elbow_pain_more_than_half_of_days',
   'right_elbow_pain_more_than_half_of_days',
   'left_wrist_pain_more_than_half_of_days',
   'right_wrist_pain_more_than_half_of_days',
   'left_hand_pain_more_than_half_of_days',
   'right_hand_pain_more_than_half_of_days']
RISK_FACTORS = ['C(cigarette_smoker)', 'C(drinks_per_week)', 'C(v00maritst)']
BMI = ["C(current_bmi, Treatment(reference='18.5-25'))", "C(max_bmi, Treatment(reference='18.5-25'))"]
MRI = ['C(bml2plusl)', 'C(bml2plusm)', 'C(bml2pluspf)',
               'C(car11plusl)', 'C(car11plusm)', 'C(car11pluspf)', 
               'C(menextl)', 'C(menextm)', 'C(mentearl)', 'C(mentearm)']
FRACTURES_AND_FALLS = ['fractured_spine', 'fractured_hip', 'fractured_bone', 'fell_in_last_12_months']
TIMEPOINT_AND_SIDE = ["C(visit, Treatment(reference='00 month follow-up: Baseline'))", 'side', 'C(dominant_leg)']

def validate_folder_contents(path):
    """
    Make sure that the folder we're copying (of processed image data) has exactly the files we expect, and return the maximum image file number. 
    """
    all_filenames = os.listdir(path)
    all_filenames.remove('image_codes.pkl')
    all_filenames.remove('non_image_data.csv')
    image_numbers = sorted([int(a.replace('image_', '').replace('.npy', '')) for a in all_filenames])
    max_image_number = max(image_numbers)
    assert image_numbers == list(range(max_image_number + 1))
    return max_image_number


def rename_blinded_test_set_files(base_dir, inner_folder, running_for_real=False):
    """
    Take the four folders 'BLINDED_HOLD_OUT_DO_NOT_USE', 'test', 'train', 'val' 
    and combine them into three:
    test + train -> train
    val -> val
    BLINDED_HOLD_OUT_DO_NOT_USE -> test

    Sample usage: 
    rename_blinded_test_set_files('/lfs/hyperion3/0/emmap1/oai_image_data/processed_image_data_new_with_held_out_test_set_april_2019/', 
                                'show_both_knees_True_downsample_factor_None_normalization_method_our_statistics', 
                                running_for_real=True)

    """
    raise Exception("This method is destructive. Do you actually want to run it?")
    print("Relabeling folders in %s" % base_dir)
    expected_datasets = ['BLINDED_HOLD_OUT_DO_NOT_USE', 'test', 'train', 'val']
    assert sorted(os.listdir(base_dir)) == sorted(expected_datasets)
    for dataset in expected_datasets:
        full_folder_path = os.path.join(base_dir, dataset, inner_folder)
        assert os.path.exists(full_folder_path)
        max_image_number = validate_folder_contents(full_folder_path)
        print("Maximum image number in %s: %i (total images is this + 1)" % (dataset, max_image_number))

    # combine test + train set into train set. 
    full_train_path = os.path.join(base_dir, 'train', inner_folder)
    full_test_path = os.path.join(base_dir, 'test', inner_folder)
    max_train_image = validate_folder_contents(full_train_path)
    max_test_image = validate_folder_contents(full_test_path)

    train_non_image_data =  pd.read_csv(os.path.join(full_train_path, 'non_image_data.csv'), index_col=0)
    test_non_image_data =  pd.read_csv(os.path.join(full_test_path, 'non_image_data.csv'), index_col=0)
    combined_non_image_data = pd.concat([train_non_image_data, test_non_image_data])
    combined_non_image_data.index = range(len(combined_non_image_data))

    train_image_codes = pickle.load(open(os.path.join(full_train_path, 'image_codes.pkl'), 'rb'))
    test_image_codes = pickle.load(open(os.path.join(full_test_path, 'image_codes.pkl'), 'rb'))
    combined_image_codes = train_image_codes + test_image_codes

    ensure_barcodes_match(combined_non_image_data, combined_image_codes)
    assert len(combined_image_codes) == len(combined_non_image_data)
    assert len(combined_image_codes) == (max_train_image + 1) + (max_test_image + 1)

    print("Moving test images to train folder")
    for i in range(max_test_image + 1):
        old_path = os.path.join(full_test_path, 'image_%i.npy' % i)
        new_path = os.path.join(full_train_path, 'image_%i.npy' % (max_train_image + 1 + i))
        assert os.path.exists(old_path)
        assert not os.path.exists(new_path)
        cmd = 'mv %s %s' % (old_path, new_path)
        print(cmd)
        if running_for_real:
            os.system(cmd)

    print("Moving test non-image data to train folder")
    if running_for_real:
        combined_non_image_data.to_csv(os.path.join(full_train_path, 'non_image_data.csv'))
        pickle.dump(combined_image_codes, open(os.path.join(full_train_path, 'image_codes.pkl'), 'wb'))
        
    print("Renaming blinded held out set to test")
    full_blinded_held_out_path =  os.path.join(base_dir, 'BLINDED_HOLD_OUT_DO_NOT_USE', inner_folder)
    if running_for_real:
        os.system('rm -rf %s' % full_test_path)
        os.system('mv %s %s' % (full_blinded_held_out_path, full_test_path))
    expected_datasets = ['test', 'train', 'val']
    assert sorted(os.listdir(base_dir)) == sorted(expected_datasets)
    print("Done relabeling folders")


def binarize_koos(koos_arr):
    return 1.*(koos_arr <= KOOS_BINARIZATION_THRESH)

def binarize_womac(womac_arr):
    return 1.*(womac_arr > WOMAC_BINARIZATION_THRESH)

def get_all_ids():
    """
    Gets all the ids from the clinical file. Checked. 
    """
    full_path = os.path.join(BASE_NON_IMAGE_DATA_DIR, 'AllClinical_ASCII', 'AllClinical00.txt')
    d = pd.read_csv(full_path, sep='|')
    ids = sorted(list(d['ID'].values.astype(int)))
    assert len(set(ids)) == len(ids)
    assert len(ids) == TOTAL_PEOPLE
    return ids

def make_train_val_test_hold_out_set(seed_to_further_shuffle_train_test_val_sets):
    """
    Get the list of ids to have in the train/test/hold-out set. Checked. 
    If seed_to_further_shuffle_train_test_val_sets is None, returns the original data. 
    Otherwise, further shuffles the data as a robustness check 
    (so we see how much results vary across test splits)
    """

    if seed_to_further_shuffle_train_test_val_sets is not None:
        print("Attention: further shuffling with random seed %s" % str(seed_to_further_shuffle_train_test_val_sets))
    train_frac = TRAIN_VAL_TEST_HOLD_OUT_FRACTIONS['train_frac']
    val_frac = TRAIN_VAL_TEST_HOLD_OUT_FRACTIONS['val_frac']
    test_frac = TRAIN_VAL_TEST_HOLD_OUT_FRACTIONS['test_frac']
    hold_out_frac = TRAIN_VAL_TEST_HOLD_OUT_FRACTIONS['hold_out_frac']
    assert np.allclose(train_frac + val_frac + test_frac + hold_out_frac, 1)
    ids = get_all_ids()
    n = len(ids)
    random.Random(0).shuffle(ids)

    # make sure the ids are in the same order as before (random seeds are the same). 
    shuffled_id_path = os.path.join(BASE_NON_IMAGE_DATA_DIR, 'shuffled_ids.pkl')
    if os.path.exists(shuffled_id_path):
        previously_cached_ids = pickle.load(open(shuffled_id_path, 'rb'))
        assert ids == previously_cached_ids
    else:
        pickle.dump(ids, open(shuffled_id_path, 'wb'))

    if seed_to_further_shuffle_train_test_val_sets is not None:
        # if seed is not None, further shuffle everything but the blinded hold out set. 
        train_test_val_cutoff = int((train_frac + val_frac + test_frac)*n)
        train_test_val_ids = copy.deepcopy(ids[:train_test_val_cutoff])
        random.Random(seed_to_further_shuffle_train_test_val_sets).shuffle(train_test_val_ids)
        ids[:train_test_val_cutoff] = train_test_val_ids

    results = {'train_ids':ids[:int(train_frac*n)], 
    'val_ids':ids[int(train_frac*n):int((train_frac + val_frac)*n)], 
    'test_ids':ids[int((train_frac + val_frac)*n):int((train_frac + val_frac + test_frac)*n)], 
    'BLINDED_HOLD_OUT_DO_NOT_USE_ids':ids[int((train_frac + val_frac + test_frac)*n):]}

    assert sorted(results['train_ids'] + results['val_ids'] + results['test_ids'] + results['BLINDED_HOLD_OUT_DO_NOT_USE_ids']) == sorted(ids)
    
    blinded_hold_out_set_path = os.path.join(BASE_NON_IMAGE_DATA_DIR, 'blinded_hold_out_set_ids.pkl')
    if os.path.exists(blinded_hold_out_set_path):
        previously_cached_hold_out_set_ids = pickle.load(open(blinded_hold_out_set_path, 'rb'))
        assert results['BLINDED_HOLD_OUT_DO_NOT_USE_ids'] == previously_cached_hold_out_set_ids
    else:
        pickle.dump(results['BLINDED_HOLD_OUT_DO_NOT_USE_ids'], open(blinded_hold_out_set_path, 'wb'))
    for k in results:
        print("Number of ids in %s set: %i" % (k.replace('_ids', ''), len(results[k])))
    return results

def copy_data_from_hyperion_to_turing():
    assert node_name == 'turing2'
    os.system('scp -r emmap1@hyperion3:/lfs/hyperion3/0/emmap1/oai_image_data/processed_image_data/* /lfs/turing2/0/emmap1/oai_image_data/processed_image_data/')
    os.system('scp -r emmap1@hyperion3:/lfs/hyperion3/0/emmap1/oai_image_data/processed_image_data_new_with_held_out_test_set_april_2019/* /lfs/turing2/0/emmap1/oai_image_data/processed_image_data_new_with_held_out_test_set_april_2019/')
    print("Successfully copied images")

def copy_data_from_dfs_to_hyperion(substrings_to_copy, datasets_to_copy=['test', 'val', 'train', 'BLINDED_HOLD_OUT_DO_NOT_USE']):
    """
    Move processed image data from DFS to hyperion because it loads way faster. 
    """
    raise Exception("This is deprecated. You should update so it works with either turing or hyperion")
    original_dfs_folders = None
    assert USE_NEW_DATA_ON_HYPERION_WITH_HELD_OUT_TEST_SET

    raise Exception("Do not use this method lightly! It deletes files! Remove this exception if you really want to use it.")

    for dataset in datasets_to_copy:
        print("Removing data from %s" % os.path.join(HYPERION_BASE_IMAGES_PATH, dataset))
        print('rm -rf %s/*' % os.path.join(HYPERION_BASE_IMAGES_PATH, dataset))
        os.system('rm -rf %s/*' % os.path.join(HYPERION_BASE_IMAGES_PATH, dataset))
        dfs_folders = sorted(os.listdir(os.path.join(DFS_BASE_IMAGES_PATH, dataset)))
        

        dfs_folders = [a for a in dfs_folders 
                       if any([substring_to_copy in a for substring_to_copy in substrings_to_copy])
                       and 'random_seed' not in a]

        if original_dfs_folders is not None:
            assert original_dfs_folders == dfs_folders
        else:
            original_dfs_folders = dfs_folders

        for dfs_folder in dfs_folders:
            original_full_path = os.path.join(DFS_BASE_IMAGES_PATH, dataset, dfs_folder)
            new_full_path = os.path.join(HYPERION_BASE_IMAGES_PATH, dataset, dfs_folder)
            cmd = 'cp -r %s/ %s/' % (original_full_path, new_full_path)
            print(cmd)
            t0 = time.time()
            p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True) # https://stackoverflow.com/a/38956698/9477154
            p1.communicate()
            print("Command %s completed in %2.3f seconds" % (cmd, time.time() - t0))
    print("Successfully completed all copying.")

def get_combined_dataframe(non_image_dataset, clinical_assessments):
    """
    Returns a combined data with knee pain scores, semiquantitative scores, and demographic covariates. 
    Uses clinical_assessments as the original dataframe to select the subset of rows. 
    Each row should have a unique id, visit, and side. 
    Checked. 
    """
    combined_data = copy.deepcopy(clinical_assessments)
    print("Number of datapoints with clinical assessments: %i" % len(combined_data))
    # merge with pain scores. 
    combined_data = pd.merge(combined_data, 
                           non_image_dataset.processed_dataframes['all_knee_pain_scores'], 
                           how='inner', 
                           on=['id', 'visit', 'side'])
    assert len(combined_data.dropna(subset=['koos_pain_subscore', 'womac_pain_subscore'])) == len(combined_data)
    old_len = len(combined_data)

    # Now merge with a lot of control dataframes. 
    original_order = copy.deepcopy(combined_data[['id', 'visit', 'side']]) # debugging sanity check: sometimes the merge changes the join order. 
    for control_dataframe in sorted(list(non_image_dataset.processed_dataframes.keys())):
        if control_dataframe in ['kxr_sq_bu', 'all_knee_pain_scores']:
            continue
        df_to_merge_with = copy.deepcopy(non_image_dataset.processed_dataframes[control_dataframe])
        cols_to_merge_on = [a for a in ['id', 'visit', 'side'] if a in df_to_merge_with.columns] # changing this doesn't make a difference.         
        if control_dataframe == 'david_mri_data':
            join_type = 'left' # we are lacking rows for some images for MRI data, and we don't want to cut these out. 
        else:
            join_type = 'inner'

        print("Performing a %s join with %s using columns %s" % (join_type, control_dataframe, cols_to_merge_on))
        combined_data = pd.merge(combined_data, 
                             df_to_merge_with, 
                             how=join_type, 
                             on=cols_to_merge_on)
        assert len(combined_data) == old_len
        assert len(combined_data[['id', 'visit', 'side']].drop_duplicates()) == len(combined_data)
        if not original_order.equals(combined_data[['id', 'visit', 'side']]):
            print("Alert! Order of dataframe changed after merge. Old order:")
            print(original_order.head())
            print("new order:")
            print(combined_data[['id', 'visit', 'side']].head())
            original_order = copy.deepcopy(combined_data[['id', 'visit', 'side']])
             


    pd.set_option('max_rows', 500)
    print("Prior to dropping people missing socioeconomic status data, %i rows" % len(combined_data))
    combined_data = combined_data.dropna(
        subset=['binarized_education_graduated_college', 'binarized_income_at_least_50k'])
    print("After dropping people missing socioeconomic status data, %i rows" % len(combined_data))
    combined_data = combined_data.dropna(subset=['p02hisp', 'p02race', 'p02sex', 'age_at_visit'])
    print("After dropping people missing age/race/sex data, %i rows" % len(combined_data))

    missing_data_fracs_by_col = []
    for c in combined_data.columns:
        missing_data_fracs_by_col.append({
            'col':c, 
            'missing_data':pd.isnull(combined_data[c]).mean()})

    missing_data_fracs_by_col = pd.DataFrame(missing_data_fracs_by_col) 
    print(missing_data_fracs_by_col.sort_values(by='missing_data')[::-1])

    return combined_data

def find_image_barcodes_that_pass_qc(non_image_dataset):
    """
    Get the list of image barcodes which pass QC. Note: this returns the LONG barcodes (12 characters). 
    """
    all_good_barcodes = set()
    for k in sorted(non_image_dataset.original_dataframes):
        if 'xray' in k:
            visit_id = k.replace('xray', '')
            assert visit_id in CLINICAL_WAVES_TO_FOLLOWUP
            df = copy.deepcopy(non_image_dataset.original_dataframes[k])
            passed_qc_vals = ["'Y': QCd and found to be acceptable", 
                               "'YD': Not QCd and accepted by default"]
            all_vals = passed_qc_vals + ["'NR': QCd unacceptable, chosen for release", "'NA': QCd unacceptable, no better available", "P"] # P is a very rare value
            assert (df['v%saccept' % visit_id] == 'P').sum() < 10
            print("Warning: %i values in xray dataset %s are P, a value which should occur rarely" % ((df['v%saccept' % visit_id] == 'P').sum(), k))
            assert df['v%saccept' % visit_id].dropna().map(lambda x:x in all_vals).all()
            passed_qc = df['v%saccept' % visit_id].map(lambda x:x in passed_qc_vals)
            good_barcodes_for_visit = df['v%sxrbarcd' % visit_id].loc[passed_qc].values
            assert len(set(good_barcodes_for_visit)) == len(good_barcodes_for_visit)
            good_barcodes_for_visit = set(good_barcodes_for_visit)
            assert len(good_barcodes_for_visit.intersection(all_good_barcodes)) == 0
            all_good_barcodes = all_good_barcodes.union(good_barcodes_for_visit)
    all_good_barcodes = ['0' + str(int(a)) for a in all_good_barcodes] # first digit is truncated; it's a 0 -- so we add it back in. 
    assert all([len(a) == 12 for a in all_good_barcodes])
    all_good_barcodes = set(all_good_barcodes)
    return all_good_barcodes

def ensure_barcodes_match(combined_df, image_codes):
    """
    Sanity check: make sure non-image data matches image data. 
    """
    print("Ensuring that barcodes line up.")
    assert len(combined_df) == len(image_codes)
    for idx in range(len(combined_df)):
        barcode = str(combined_df.iloc[idx]['barcdbu'])
        if len(barcode) == 11:
            barcode = '0' + barcode
        side = str(combined_df.iloc[idx]['side'])
        code_in_df = barcode + '*' + side

        if image_codes[idx] != code_in_df:
            raise Exception("Barcode mismatch at index %i, %s != %s" % (idx, image_codes[idx], code_in_df))
    print("All %i barcodes line up." % len(combined_df))

def match_image_dataset_to_non_image_dataset(image_dataset, non_image_dataset, swap_left_and_right=False):
    """
    Given an image dataset + a non-image dataset, returns
    a) a dataframe of clinical ratings and 
    b) a list of images which correspond to the clinical ratings
    There should be no missing data in either. 
    Checked. 
    """

    # Filter for clinical assessments for images that pass QC.
    clinical_assessments = copy.deepcopy(non_image_dataset.processed_dataframes['kxr_sq_bu'])
    assert clinical_assessments['barcdbu'].map(lambda x:len(x) == 12).all()
    print(clinical_assessments.head())
    print("Prior to filtering for images that pass QC, %i images" % len(clinical_assessments))
    acceptable_barcodes = find_image_barcodes_that_pass_qc(non_image_dataset)
    clinical_assessments = clinical_assessments.loc[clinical_assessments['barcdbu'].map(lambda x:x in acceptable_barcodes)]
    print("After filtering for images that pass QC, %i images" % len(clinical_assessments)) # this doesn't filter out a lot of clinical assessments, even though a lot of values in the xray01 etc datasets are NA, because those values are already filtered out of the kxr_sq_bu -- you can't assign image scores to an image which isn't available. 
    
    combined_df = get_combined_dataframe(non_image_dataset, clinical_assessments)
    non_image_keys = list(combined_df['barcdbu'].map(str) + '*' + combined_df['side'])
    non_image_keys = dict(zip(non_image_keys, range(len(non_image_keys))))
    matched_images = [None for i in range(len(combined_df))]
    image_codes = [None for i in range(len(combined_df))]
    for i in range(len(image_dataset.images)):
        if i % 1000 == 0:
            print('Image %i/%i' % (i, len(image_dataset.images)))
        image = image_dataset.images[i]
        if not swap_left_and_right:
            left_key = str(image['barcode']) + '*left'
            right_key = str(image['barcode'])  + '*right'
        else:
            right_key = str(image['barcode']) + '*left'
            left_key = str(image['barcode'])  + '*right'
        if left_key in non_image_keys: 
            idx = non_image_keys[left_key]
            assert matched_images[idx] is None
            matched_images[idx] = image['left_knee'].copy()
            image_codes[idx] = left_key
        if right_key in non_image_keys:
            idx = non_image_keys[right_key]
            assert matched_images[idx] is None
            matched_images[idx] = image['right_knee'].copy()
            image_codes[idx] = right_key
    combined_df['has_matched_image'] = [a is not None for a in matched_images]
    print("Fraction of clinical x-ray ratings with matched images")
    print(combined_df[['has_matched_image', 'visit', 'side']].groupby(['visit', 'side']).agg(['mean', 'sum']))
    idxs_to_keep = []
    for i in range(len(combined_df)):
        if combined_df['has_matched_image'].values[i]:
            idxs_to_keep.append(i)
    combined_df = combined_df.iloc[idxs_to_keep]
    combined_df.index = range(len(combined_df))
    matched_images = [matched_images[i] for i in idxs_to_keep]
    image_codes = [image_codes[i] for i in idxs_to_keep]
    ensure_barcodes_match(combined_df, image_codes)
    print("Total number of images matched to clinical ratings: %i" % len(matched_images))
    assert all([a is not None for a in matched_images])
    assert combined_df['has_matched_image'].all()
    return combined_df, matched_images, image_codes




