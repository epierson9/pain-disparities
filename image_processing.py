import os

import time
from traceback import print_exc
import cv2
import numpy as np
from constants_and_util import *
import matplotlib.pyplot as plt

import non_image_data_processing
from scipy.stats import spearmanr
from traceback import print_exc
import random
import pickle
from sklearn.linear_model import Lasso
import seaborn as sns
import datetime
import sys
import statsmodels.api as sm
from scipy.ndimage.filters import gaussian_filter
import gc

import torch
import pydicom
from pydicom.data import get_testdata_files
from torchvision import datasets, models, transforms
import torchsummary
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image



def get_directories(path):
    """ 
    Small helper method: list the directories along a given path.
    Checked. 
    """
    return sorted([a for a in os.listdir(path) if os.path.isdir(os.path.join(path, a))])

def is_valid_date(s):
    """
    asserts that s is in fact a date. 
    Should be an 8-character string in yyyymmdd format. 
    Checked. 
    """
    if not len(s) == 8:
        print(s)
        return False
    year = int(s[:4])
    month = int(s[4:6])
    day = int(s[6:])
    try:
        datetime.datetime(year, month, day)
    except:
        print_exc()
        return False
    if year > 2017:
        return False
    return True

class XRayImageDataset:
    """
    Class for loading data.
    """
    def __init__(self, 
        desired_image_type, 
        normalization_method, 
        reprocess_all_images,
        show_both_knees_in_each_image,
        crop_to_just_the_knee,
        make_plot=False,
        max_images_to_load=None, 
        use_small_data=False, 
        downsample_factor_on_reload=None):
        """
        Creates the dataset. 
        desired_image_type: the type of x-ray part you want (for example, Bilateral PA Fixed Flexion Knee)
        normalization_method: specifies how to z-score each image. 
        reprocess_all_images: whether to rerun the whole pipeline or just load the processed pkl. 
        make_plot: whether to plot a random sample of images. 
        max_images_to_load: set to a small number to test. 
        downsample_factor_on_reload: how much we further downsample the (already downsampled) images when we reload them. This is a little messy. 
        Originally we save the images as 1024x1024; we can further downsample them. 

        The pipeline, then, is: 
        (before saving)
        1. Load all diacom images and downsample each image. 
        2. Scale each image to 0-1 and compute statistics of dataset. 
        3. Save all the images. 
        (after reloading saved images)
        4. Cut each image in half (or flip, or crop)
        5. If desired, further downsample each image. 
        6. Turn each image into RGB (ie, give it three channels) and normalize the images (z-score etc). 
        """

        self.images = []
        self.desired_image_type = desired_image_type
        assert self.desired_image_type == 'Bilateral PA Fixed Flexion Knee' # not sure pipeline will work for other body parts 
        self.normalization_method = normalization_method
        self.make_plot = make_plot
        self.reprocess_all_images = reprocess_all_images
        self.downsample_factor_on_reload = downsample_factor_on_reload
        self.show_both_knees_in_each_image = show_both_knees_in_each_image
        if crop_to_just_the_knee:
            sys.path.append('KneeLocalizer/oulukneeloc/')
            from detector import KneeLocalizer
            self.knee_localizer = KneeLocalizer()
        self.crop_to_just_the_knee = crop_to_just_the_knee

        if use_small_data:
            self.processed_image_path = os.path.join(BASE_IMAGE_DATA_DIR, 'processed_image_data', 'small_data.pkl')
        else:
            self.processed_image_path = os.path.join(BASE_IMAGE_DATA_DIR, 'processed_image_data', 'data.pkl')

        self.extra_margin_for_each_image = 1.1 # how much extra margin to give the left/right images. 

        if max_images_to_load is not None:
            self.max_images_to_load = max_images_to_load
        else:
            self.max_images_to_load = 99999999999
        if self.reprocess_all_images:
            print("Reprocessing all images from scratch")
            self.load_all_images() # load images into numpy arrays from dicom

            # put images on 0-1 scale. Do this separately for the cropped knee images and the full images. 
            # Note: it is important to do this for cropped knees separately because they are not on the same scale. 
            # The external package that we uses loads them as 8-bit rather than 16-bit or something. 
            self.diacom_image_statistics = {}
            self.compute_dataset_image_statistics_and_divide_by_max(just_normalize_cropped_knees=False)
            self.compute_dataset_image_statistics_and_divide_by_max(just_normalize_cropped_knees=True)

            for i in range(len(self.images)):
                # don't save extra images
                self.images[i]['unnormalized_image_array'] = None
            print("Number of images: %i" % len(self.images))
            pickle.dump({'images':self.images, 'image_statistics':self.diacom_image_statistics}, open(self.processed_image_path, 'wb'))
            print("Successfully processed and saved images")
        else:
            print("loading images from %s" % self.processed_image_path)
            reloaded_data = pickle.load(open(self.processed_image_path, 'rb'))
            self.images = reloaded_data['images']
            if not self.crop_to_just_the_knee:
                if not self.show_both_knees_in_each_image:
                    self.cut_images_in_two() # cut into left + right images. 
                else:
                    self.flip_right_images() # if you want both knees in one image, flip the right images so knees are on same side. 
            else:
                for i in range(len(self.images)):
                    assert self.images[i]['cropped_left_knee'].max() <= 1
                    assert self.images[i]['cropped_right_knee'].max() <= 1
                    assert self.images[i]['cropped_left_knee'].min() >= 0
                    assert self.images[i]['cropped_right_knee'].min() >= 0

                    self.images[i]['left_knee_scaled_to_zero_one'] = self.images[i]['cropped_left_knee'].copy()
                    self.images[i]['right_knee_scaled_to_zero_one'] = self.images[i]['cropped_right_knee'][:, ::-1].copy()
                    self.images[i]['cropped_left_knee'] = None
                    self.images[i]['cropped_right_knee'] = None


            if self.downsample_factor_on_reload is not None:
                for i in range(len(self.images)):
                    for side in ['left', 'right']:
                        orig_shape = self.images[i]['%s_knee_scaled_to_zero_one' % side].shape
                        assert len(orig_shape) == 2
                        new_shape = (int(orig_shape[0] * self.downsample_factor_on_reload), 
                            int(orig_shape[1] * self.downsample_factor_on_reload))

                        # https://stackoverflow.com/questions/21248245/opencv-image-resize-flips-dimensions
                        # confusing: open cv resize flips image dimensions, so if image is not a square we have to flip the shape we want. 
                        new_shape = new_shape[::-1] 
                        self.images[i]['%s_knee_scaled_to_zero_one' % side] = cv2.resize(self.images[i]['%s_knee_scaled_to_zero_one' % side],
                         dsize=tuple(new_shape))
            self.diacom_image_statistics = reloaded_data['image_statistics']
            print("Image statistics are", reloaded_data['image_statistics'])
            self.make_images_RGB_and_zscore() # z-score. The reason we do this AFTER processing is that we don't want to save the image 3x. 
            #self.plot_pipeline_examples(25) # make sanity check plots
            print("Successfully loaded %i images" % len(self.images))

    def crop_to_knee(self, dicom_image_path):
        results = self.knee_localizer.predict(dicom_image_path)
        if results is None:
            print("Warning: was not able to identify bounding boxes for this knee.")
            return None, None
        bounding_boxes, image = results
        l_bounding_box, r_bounding_box = bounding_boxes
        # IMPORTANT AND CONFUSING: THE IMAGE ON THE LEFT IS THE RIGHT KNEE.
        # Per email: "Confusingly, the knee on the right of the image is the patient's left knee."
        assert l_bounding_box[0] > r_bounding_box[0]
        assert l_bounding_box[2] > r_bounding_box[2]
        left_knee = image[l_bounding_box[1]:l_bounding_box[3], l_bounding_box[0]:l_bounding_box[2]]
        right_knee = image[r_bounding_box[1]:r_bounding_box[3], r_bounding_box[0]:r_bounding_box[2]] 
        print("Size of left knee prior to resizing is", left_knee.shape)
        print("Size of right knee prior to resizing is", right_knee.shape)
        if min(left_knee.shape) == 0 or min(right_knee.shape) == 0:
            print("Warning: was not able to identify bounding boxes for this knee.")
            return None, None

        left_knee = self.resize_image(left_knee, CROPPED_KNEE_RESAMPLED_IMAGE_SIZE)
        right_knee = self.resize_image(right_knee, CROPPED_KNEE_RESAMPLED_IMAGE_SIZE)

        print("Size of left knee after resizing is", left_knee.shape)
        print("Size of right knee after resizing is", right_knee.shape)
        return left_knee, right_knee
        
    def load_all_images(self):
        """
        loop over the nested subfolders + load images. 
        """
        for timepoint_dir in get_directories(BASE_IMAGE_DATA_DIR):
            if timepoint_dir not in IMAGE_TIMEPOINT_DIRS_TO_FOLLOWUP:
                continue
            # confirmed that this set of directories is consistent with website that provides information about data. 
            base_dir_for_timepoint = os.path.join(BASE_IMAGE_DATA_DIR, timepoint_dir)
            # for some reason some directories are nested -- /dfs/dataset/tmp/20180910-OAI/data/48m/48m/48m -- 
            while timepoint_dir in get_directories(base_dir_for_timepoint):
                print("%s directory is found in %s; concatenating and looking in the nested directory" % (timepoint_dir, base_dir_for_timepoint))
                base_dir_for_timepoint = os.path.join(base_dir_for_timepoint, timepoint_dir)
            for cohort_folder in get_directories(base_dir_for_timepoint):
                # A value of “C” for letter [X] indicates that the images are from participants are in the initial 2686 participants in Group C of the OAI cohort, 
                # and a value of “E” represents the remaining 2110 participants from the cohort.
                print(cohort_folder)
                if timepoint_dir in ['18m']:
                    assert cohort_folder.split('.')[1] in ['D']
                    assert len(get_directories(base_dir_for_timepoint)) == 1
                elif timepoint_dir in ['30m']:
                    assert cohort_folder.split('.')[1] in ['G']
                    assert len(get_directories(base_dir_for_timepoint)) == 1
                else:
                    assert cohort_folder.split('.')[1] in ['C', 'E']
                    assert len(get_directories(base_dir_for_timepoint)) == 2
                participants = get_directories(os.path.join(base_dir_for_timepoint, 
                                                            cohort_folder))
                for participant in participants:
                    participant_path = os.path.join(base_dir_for_timepoint, 
                                               cohort_folder, 
                                               participant)
                    dates = get_directories(participant_path)
                    # Each individual participant’s folder contains subfolders for each date on which a participant had images 
                    # (format of folder name is yyyymmdd).
                    for date in dates:
                        assert is_valid_date(date)
                        date_path = os.path.join(base_dir_for_timepoint, 
                                               cohort_folder, 
                                               participant, 
                                               date)
                        # There is one more level of sub- folders below this level: 
                        # one sub-folder for each image series acquired on that date. 
                        # These sub-folders have unique 8-digit identifiers that are assigned 
                        # to the image series in the central OAI imaging database maintained 
                        # at Synarc, Inc. 
                        # If the 8-digit identifier begins with 0 then the folder contains x-ray images, 
                        # and if it starts with 1, then the folder contains MR images.
                        all_image_series = get_directories(date_path)
                        assert all([a[0] in ['0', '1'] for a in all_image_series])
                        for image_series in all_image_series:
                            is_xray = image_series[0] == '0'
                            image_series_dir = os.path.join(date_path, 
                                                    image_series)
                            if is_xray:
                                if len(self.images) >= self.max_images_to_load:
                                    print("Loaded the maximum number of images: %i" % len(self.images))
                                    return
                                assert os.listdir(image_series_dir) == ['001']
                                image_path = os.path.join(image_series_dir, '001')
                                diacom_image = self.load_diacom_file(image_path, 
                                    desired_image_type=self.desired_image_type)

                                
                                if diacom_image is not None:
                                    if self.crop_to_just_the_knee:
                                        cropped_left_knee, cropped_right_knee = self.crop_to_knee(image_path)
                                        if (cropped_left_knee is None) or (cropped_right_knee is None):
                                            print("Warning: unable to crop knee image.")
                                    else:
                                        cropped_left_knee = None
                                        cropped_right_knee = None
                                    image_array = self.get_resized_pixel_array_from_dicom_image(diacom_image)
                                    self.images.append({'timepoint_dir':timepoint_dir, 
                                        'full_path':image_path,
                                        'cohort_folder':cohort_folder, 
                                        'visit':diacom_image.ClinicalTrialTimePointDescription,
                                        'id':int(participant), 
                                        'date':date, 
                                        'image_series':image_series, 
                                        'body_part':diacom_image.BodyPartExamined, 
                                        'series_description':diacom_image.SeriesDescription,
                                        'unnormalized_image_array':image_array, 
                                        'cropped_left_knee':cropped_left_knee, 
                                        'cropped_right_knee':cropped_right_knee,
                                        # Users may also want to identify the specific image that was assessed to generate the data for an anatomic site and time point and merge the image assessment data with meta-data about that image (please see Appendix D for example SAS code). Individual images (radiographs, MRI series) are identified by a unique barcode. The barcode is recorded in the AccessionNumber in the DICOM header of the image.
                                        'barcode':diacom_image.AccessionNumber
                                        })
    def plot_pipeline_examples(self, n_examples):
        """
        plot n_examples random images to make sure pipeline looks ok. 
        Checked. 
        """
        print("Plotting pipeline examples")
        for i in range(n_examples):
            idx = random.choice(range(len(self.images)))
            plt.figure(figsize=[15, 5])

            original_diacom_image = self.load_diacom_file(self.images[idx]['full_path'], self.images[idx]['series_description'])
            plt.subplot(131)
            plt.imshow(original_diacom_image.pixel_array, cmap='bone')
            plt.colorbar()
            
            zscore_range = 2
            plt.subplot(132)
            plt.imshow(self.images[idx]['left_knee'][0, :, :], cmap='bone', clim=[-zscore_range, zscore_range])
            plt.title("Left knee")
            plt.colorbar()

            plt.subplot(133)
            plt.imshow(self.images[idx]['right_knee'][0, :, :], cmap='bone', clim=[-zscore_range, zscore_range])
            plt.title("Right knee")
            plt.colorbar()

            plt.subplots_adjust(wspace=.3, hspace=.3)
            plt.savefig('example_images/pipeline_example_%i.png' % i, dpi=300)
            plt.show()
    
    def cut_image_in_half(self, image_arr):
        """
        Cut the image into left + right knees. 
        Checked. 
        """
        half_image = RESAMPLED_IMAGE_SIZE[1] / 2.
       
        border_of_image_on_the_left = int(half_image * self.extra_margin_for_each_image)
        border_of_image_on_the_right = RESAMPLED_IMAGE_SIZE[1] - int(half_image * self.extra_margin_for_each_image)

        image_on_the_left = image_arr[:, :border_of_image_on_the_left].copy()
        image_on_the_right = image_arr[:, border_of_image_on_the_right:].copy()

        # flip left image so symmetric
        image_on_the_left = image_on_the_left[:, ::-1]
        assert image_on_the_left.shape == image_on_the_right.shape

        # IMPORTANT AND CONFUSING: THE IMAGE ON THE LEFT IS THE RIGHT KNEE.
        # Per email: "Confusingly, the knee on the right of the image is the patient's left knee."
        right_knee = image_on_the_left
        left_knee = image_on_the_right

        return left_knee, right_knee

    def cut_images_in_two(self):
        """
        Loop over all images and cut each in two. 
        """
        for i in range(len(self.images)):
            self.images[i]['left_knee_scaled_to_zero_one'], self.images[i]['right_knee_scaled_to_zero_one'] = self.cut_image_in_half(self.images[i]['image_array_scaled_to_zero_one'])
            self.images[i]['image_array_scaled_to_zero_one'] = None

    def flip_right_images(self):
        for i in range(len(self.images)):
            self.images[i]['left_knee_scaled_to_zero_one'] = self.images[i]['image_array_scaled_to_zero_one'].copy()
            self.images[i]['right_knee_scaled_to_zero_one'] = self.images[i]['image_array_scaled_to_zero_one'][:, ::-1].copy()
            self.images[i]['image_array_scaled_to_zero_one'] = None


    def resize_image(self, original_array, new_size):
        """
        resample the image to new_size. Checked. 
        """
        assert len(original_array.shape) == 2
        print("Resizing image from %s to %s" % (original_array.shape, new_size))
        new_array = cv2.resize(original_array, dsize=tuple(new_size), interpolation=cv2.INTER_CUBIC)
        return new_array

    def load_diacom_file(self, filename, desired_image_type):
        """
        load a matplotlib array from the pydicom file filename. Checked. 
        Drawn heavily from this documentation example: 
        https://pydicom.github.io/pydicom/stable/auto_examples/input_output/plot_read_dicom.html#sphx-glr-auto-examples-input-output-plot-read-dicom-py
        """
        dataset = pydicom.dcmread(filename)
        
        if dataset.SeriesDescription != desired_image_type:
            return None

        print("Image %i" % len(self.images))
        print("Filename.........:", filename)
        pat_name = dataset.PatientName
        display_name = pat_name.family_name + ", " + pat_name.given_name
        print("Patient's name...: %s" % display_name)
        print("Patient id.......: %s" % dataset.PatientID)
        print("Modality.........: %s" % dataset.Modality)
        print("Study Date.......: %s" % dataset.StudyDate)
        print("Body part examined: %s" % dataset.BodyPartExamined)
        print("Series description: %s" % dataset.SeriesDescription) # eg, Bilateral PA Fixed Flexion Knee
        print("Accession number: %s" % dataset.AccessionNumber) # this is the barcode. 
        print("ClinicalTrialTimePointDescription: %s" % dataset.ClinicalTrialTimePointDescription)
        print("ClinicalTrialTimePointID: %s" % dataset.ClinicalTrialTimePointID)

        if 'PixelData' in dataset:
            rows = int(dataset.Rows)
            cols = int(dataset.Columns)
            print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
                rows=rows, cols=cols, size=len(dataset.PixelData)))
            if 'PixelSpacing' in dataset:
                print("Pixel spacing....:", dataset.PixelSpacing)

        return dataset

    def get_resized_pixel_array_from_dicom_image(self, diacom_image):
        """
        Extract pydicom pixel array and resize. Checked. 
        Per documentation, "The pixel_array property returns a NumPy array"
        """
        arr = self.resize_image(diacom_image.pixel_array, RESAMPLED_IMAGE_SIZE) * 1.0
        assert len(arr.shape) == 2
        return arr

    def compute_dataset_image_statistics_and_divide_by_max(self, just_normalize_cropped_knees):
        """
        Put images into the zero-one range by dividing by the maximum value. 
        Also compute statistics of the images: mean and std. 

        Note: it is important to do this for cropped knees separately because they are not on the same scale. 
        The external package that we uses loads them as 8-bit rather than 16-bit or something. 

        Checked. 
        """
        print("\n\nNow computing overall dataset statistics")
        print("Just analyze cropped knees: %s" % just_normalize_cropped_knees)

        all_pixel_arrays = []
        for i in range(len(self.images)):
            if just_normalize_cropped_knees:
                if self.images[i]['cropped_right_knee'] is not None:
                    all_pixel_arrays.append(self.images[i]['cropped_right_knee'])
                    all_pixel_arrays.append(self.images[i]['cropped_left_knee'])
            else:
                all_pixel_arrays.append(self.images[i]['unnormalized_image_array'])
                
        all_pixel_arrays = np.array(all_pixel_arrays)
        arr_max =  np.max(all_pixel_arrays)
        assert np.min(all_pixel_arrays) >= 0
        
        if just_normalize_cropped_knees:
            suffix = 'cropped_knee_only'
        else:
            suffix = 'full_image'

        self.diacom_image_statistics['max_%s' % suffix] = 1.0*arr_max

        for i in range(len(self.images)):
            if just_normalize_cropped_knees:
                if self.images[i]['cropped_right_knee'] is not None:
                    self.images[i]['cropped_right_knee'] = self.images[i]['cropped_right_knee'] / arr_max
                    self.images[i]['cropped_left_knee'] = self.images[i]['cropped_left_knee'] / arr_max
            else:
                self.images[i]['image_array_scaled_to_zero_one'] = self.images[i]['unnormalized_image_array'] / arr_max
        self.diacom_image_statistics['mean_of_zero_one_data_%s' % suffix] = np.mean(all_pixel_arrays) / arr_max
        self.diacom_image_statistics['std_of_zero_one_data_%s' % suffix] = np.std(all_pixel_arrays) / arr_max
        for k in self.diacom_image_statistics.keys():
            print(k, self.diacom_image_statistics[k])
    
    def make_images_RGB_and_zscore(self):
        """
        Normalize each image by z-scoring. 
        Checked. 
        """
        print("Computing normalized images")
        assert self.normalization_method in ['imagenet_statistics', 'our_statistics', 'zscore_individually']
        
        def normalize_array(arr, mean_to_use, std_to_use):
            assert len(mean_to_use) == 3
            assert len(std_to_use) == 3
            new_arr = arr.copy()
            for k in range(3):
                new_arr[k, :, :] = (new_arr[k, :, :] - mean_to_use[k]) / std_to_use[k]
            return new_arr

        for i in range(len(self.images)):
            for side in ['left', 'right']:
                original_image = self.images[i]['%s_knee_scaled_to_zero_one' % side]

                rgb_image = np.array([original_image, original_image, original_image])
                
                # determine what the size of the image ought to be. 

                if self.crop_to_just_the_knee:
                    original_reloaded_image_size = CROPPED_KNEE_RESAMPLED_IMAGE_SIZE[0]
                else:
                    original_reloaded_image_size = RESAMPLED_IMAGE_SIZE[0]

                if self.downsample_factor_on_reload is not None:
                    downsampled_size = int(original_reloaded_image_size * self.downsample_factor_on_reload)
                else:
                    downsampled_size = original_reloaded_image_size

                if self.show_both_knees_in_each_image or self.crop_to_just_the_knee:
                    assert rgb_image.shape == tuple([3, downsampled_size, downsampled_size])
                else:
                    assert rgb_image.shape == tuple([3, downsampled_size, int(downsampled_size * self.extra_margin_for_each_image / 2.)])
                if self.normalization_method == 'imagenet_statistics':
                    mean_to_use = [0.485, 0.456, 0.406]
                    std_to_use = [0.229, 0.224, 0.225]
                elif self.normalization_method == 'our_statistics':
                    if self.crop_to_just_the_knee:
                        mean_to_use = [self.diacom_image_statistics['mean_of_zero_one_data_cropped_knee_only']] * 3
                        std_to_use = [self.diacom_image_statistics['std_of_zero_one_data_cropped_knee_only']] * 3
                    else:
                        mean_to_use = [self.diacom_image_statistics['mean_of_zero_one_data_full_image']] * 3
                        std_to_use = [self.diacom_image_statistics['std_of_zero_one_data_full_image']] * 3
                elif self.normalization_method == 'zscore_individually':
                    mean_to_use = [original_image.mean()] * 3
                    std_to_use = [original_image.std()] * 3
                else:
                    raise Exception("invalid image normalization method")

                self.images[i]['%s_knee' % side] = normalize_array(
                    rgb_image, 
                    mean_to_use, 
                    std_to_use)
                self.images[i]['%s_knee_scaled_to_zero_one' % side] = None

def compare_contents_files_to_loaded_images(image_dataset, series_description):
    """
    Sanity check: make sure the images we loaded are the images which are supposed to be there
    according to the contents file. 
    """
    barcodes_in_image_dataset = [a['barcode'][5:] for a in image_dataset.images]
    assert all([len(a) == 7 for a in barcodes_in_image_dataset])
    # Every x-ray image has a unique 12 digit barcode associated with it and the first 5 digits are always 01660.
    # so we look at the last 7 digits. 
    assert len(barcodes_in_image_dataset) == len(set(barcodes_in_image_dataset))
    barcodes_in_image_dataset = set(barcodes_in_image_dataset)
    print("Total number of barcodes in image dataset: %i" % len(barcodes_in_image_dataset))
    all_barcodes_in_contents_dir = set()                     
    for image_timepoint_dir in sorted(IMAGE_TIMEPOINT_DIRS_TO_FOLLOWUP):
        content_filename = os.path.join(BASE_IMAGE_DATA_DIR, image_timepoint_dir, 'contents.csv')
        d = pd.read_csv(content_filename, dtype={'Barcode':str})
        
        d['SeriesDescription'] = d['SeriesDescription'].map(lambda x:x.strip())
        d = d.loc[d['SeriesDescription'] == series_description]
        # several contents files are, unfortunately, inconsistently formatted. 
        if 'Barcode' not in d.columns:
            d['Barcode'] = d['AccessionNumber'].map(lambda x:str(x)[4:])
        elif image_timepoint_dir == '72m':
            d['Barcode'] = d['Barcode'].map(lambda x:str(x)[4:])
        else:
            needs_leading_0 = d['Barcode'].map(lambda x:len(x) == 6)
            d.loc[needs_leading_0, 'Barcode'] = '0' + d.loc[needs_leading_0, 'Barcode'] 
        if len(d) > 0:
            assert d['Barcode'].map(lambda x:len(x) == 7).all()
            assert len(set(d['Barcode'])) == len(d)
        all_barcodes_in_contents_dir = all_barcodes_in_contents_dir.union(set(d['Barcode']))
        n_properly_loaded = d['Barcode'].map(lambda x:x in barcodes_in_image_dataset).sum()
        
        print("%-5i/%-5i images in %s match to our dataset" % (n_properly_loaded,
                                                      len(d),
                                                      content_filename))

    print("Warning: The following images have barcodes in our dataset but do not appear in contents file")
    print("This appears to be due to barcodes that differ by 1 in a very small number of images")
    print([a for a in barcodes_in_image_dataset if a not in all_barcodes_in_contents_dir])
    assert sum([a not in all_barcodes_in_contents_dir for a in barcodes_in_image_dataset]) <= 5

def check_consistency_with_enrollees_table(image_dataset, non_image_dataset):
    """
    Check consistency between the images we have and the images the enrollees table thinks we should have. 
    THIS IS NOT CURRENTLY WORKING AND WE ARE NOT USING IT.
    """
    raise Exception("Not using at present because the enrollees data is weird and the image data shows good concordance with other files. If you use this, check it.")
    print(Counter([a['visit'] for a in image_dataset.images]))
    for timepoint in ['00', '01', '03', '05', '06', '08']:
        df = copy.deepcopy(non_image_dataset.original_dataframes['enrollees'])
        all_ids_in_enrollees_table = set(df['id'])
        def has_knee_xray(s):
            
            assert s in {'0: No', 
                         '2: Yes, Knee Xray only', 
                         '1: Yes, Knee MR only', 
                         '.: Missing Form/Incomplete Workbook', 
                         '3: Yes, Knee MR and knee xray'}
            return s in ['2: Yes, Knee Xray only', '3: Yes, Knee MR and knee xray']
        df['has_knee_xray'] = (df['v%simagesc' % timepoint].map(has_knee_xray) | 
                               df['v%simagese' % timepoint].map(has_knee_xray))
        people_who_should_have_xrays = set(list(df['id'].loc[df['has_knee_xray']].map(int)))
        
        # now figure out who actually does. 
        people_who_actually_have_xrays = set()
        timepoints_to_visit_names = {'00':'Screening Visit', 
        '01':'12 month Annual Visit', 
        '03':'24 month Annual Visit', 
        '05':'36 month Annual Visit', 
        '06':'48 month Annual Visit', 
        '08':'72 month Annual Visit'}
        for image in image_dataset.images:
            if (image['visit'] == timepoints_to_visit_names[timepoint] and 
                image['id'] in all_ids_in_enrollees_table):
                people_who_actually_have_xrays.add(image['id'])
        print("%i/%i who should have knee xrays at timepoint %s actually do" % (
            len([a for a in people_who_should_have_xrays if a in people_who_actually_have_xrays]),
            len(people_who_should_have_xrays),
            timepoint))
        have_ids_and_not_in_enrollees_table = [a for a in people_who_actually_have_xrays if a not in people_who_should_have_xrays]
        if len(have_ids_and_not_in_enrollees_table) > 0:
            print("Warning: %i people in our dataset has x-rays and does not appear in enrollees table as someone who should" % 
                len(have_ids_and_not_in_enrollees_table))

class PretrainedTorchModel:
    """
    class for loading pretrained Torch models.
    Checked.  
    """
    def __init__(self, model_name, layer_of_interest_name, use_adaptive_pooling):
        assert model_name in ['resnet18', 
        'resnet34', 'resnet50', 'resnet101', 'resnet152']
        self.model_name = model_name
        self.layer_of_interest_name = layer_of_interest_name
        self.use_adaptive_pooling = use_adaptive_pooling
        if 'resnet' in model_name:
            assert self.layer_of_interest_name in ['avgpool'] # could also try something like "layer3"
            if model_name == 'resnet18':
                self.model = models.resnet18(pretrained=True)
                self.embedding_size = [512]
            elif model_name == 'resnet34':
                self.model = models.resnet34(pretrained=True)
                self.embedding_size = [512]
            elif model_name == 'resnet50':
                self.model = models.resnet50(pretrained=True)
                self.embedding_size = [2048]
            elif model_name == 'resnet101':
                self.model = models.resnet101(pretrained=True)
                self.embedding_size = [2048]
            elif model_name == 'resnet152':
                self.model = models.resnet152(pretrained=True)
                self.embedding_size = [2048]
            else:
                raise Exception("%s is not a valid model" % model_name)
            if self.use_adaptive_pooling:
                print("Using adaptive pooling")
                self.model.avgpool = nn.AdaptiveAvgPool2d(1) # see eg http://forums.fast.ai/t/ideas-behind-adaptive-max-pooling/12634. Basically this automatically computes the appropriate size for the window. 
            self.model.cuda()
        else:
            raise Exception("%s is not a valid model" % model_name)

        # Use the model object to select the desired layer
        self.layer_of_interest = self.model._modules.get(self.layer_of_interest_name)

        self.model.eval()
        print("model")
        print(self.model)
        

    def get_embedding(self, input_data):
        # Load the pretrained model
        # https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
        # 1. Create a vector of zeros that will hold our feature vector
        my_embedding = torch.zeros(*self.embedding_size)

        # 2. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            my_embedding.copy_(o.data.squeeze())

        # 3. Attach that function to our selected layer
        h = self.layer_of_interest.register_forward_hook(copy_data)

        # 4. Run the model on our transformed image
        self.model(input_data)

        # 5. Detach our copy function from the layer
        h.remove()

        # 6. Return the feature vector, 
        # converted to numpy and flattened. 
        return my_embedding.numpy().flatten()

def convert_to_torch_tensor(arr):
    """
    convert to torch tensor.
    Checked. 
    """
    input_data = torch.from_numpy(arr).float()
    input_data = input_data.unsqueeze(0)
    input_data = torch.autograd.Variable(input_data).cuda()
    return input_data

def generate_embeddings_for_images_from_pretrained_model(images, 
    torch_model_name, 
    model_layer):
    """
    Given a list of images, generates embeddings for the images using a pretrained neural net. 
    Two different embedding methods: use_adaptive_pooling, which modifies the neural net to work with different image sizes
    and rescale, which resamples the image. 

    Checked. 
    """
    assert torch_model_name in ['resnet18', 
        'resnet34', 'resnet50', 'resnet101', 'resnet152']
    embedding_method_to_embeddings = {}
    for embedding_method in ['use_adaptive_pooling', 'rescale']:
        embedding_method_to_embeddings[embedding_method] = []
        print("Embedding method: %s" % embedding_method)
        assert embedding_method in ['use_adaptive_pooling', 'rescale']
        use_adaptive_pooling = embedding_method == 'use_adaptive_pooling'
        torch_model = PretrainedTorchModel(model_name=torch_model_name, 
                                       layer_of_interest_name=model_layer, 
                                       use_adaptive_pooling=use_adaptive_pooling)      
        for idx, image in enumerate(images):
            if idx % 1000 == 0:
                print(idx, len(images))
            if embedding_method == 'rescale':
                resized_images = []
                for k in range(3):
                    resized_images.append(cv2.resize(image[k, :, :], (224,224)))
                image = np.array(resized_images)
            torch_tensor = convert_to_torch_tensor(image)
            embedding = torch_model.get_embedding(torch_tensor)
            embedding_method_to_embeddings[embedding_method].append(embedding)
        embedding_method_to_embeddings[embedding_method] = np.array(embedding_method_to_embeddings[embedding_method])
        print("Size of image embeddings is", embedding_method_to_embeddings[embedding_method].shape)
    return embedding_method_to_embeddings

def predict_yhat_from_embeddings(all_train_embeddings, 
    all_test_embeddings, 
    train_combined_df, 
    test_combined_df):
    """
    Given train + test embeddings, and train and test datasets which include pain scores
    Comes up with train and test predictions using lasso. 
    Checked. 
    """
    assert list(all_train_embeddings.keys()) == list(all_test_embeddings.keys())
    all_yhat = []
    for y_col in ['koos_pain_subscore', 'womac_pain_subscore']:
        for alpha in [10 ** a for a in np.arange(-3, 4, .5)]:
            for embedding_method in all_train_embeddings.keys():
                print("Embedding method %s" % embedding_method)
                train_Y = copy.deepcopy(train_combined_df[y_col].values)
                test_Y = copy.deepcopy(test_combined_df[y_col].values)
                train_X = copy.deepcopy(all_train_embeddings[embedding_method])
                test_X = copy.deepcopy(all_test_embeddings[embedding_method])
                linear_model = Lasso(alpha=alpha)
                linear_model.fit(train_X, train_Y)
                num_nnz_coefficients = (np.abs(linear_model.coef_) > 1e-6).sum()
                print("Number of nonzero coefficients: %i" % num_nnz_coefficients)
                if num_nnz_coefficients == 0:
                    continue

                train_yhat = linear_model.predict(train_X)
                test_yhat = linear_model.predict(test_X)
                train_r, train_p = pearsonr(train_yhat, train_Y)
                test_r, test_p = pearsonr(test_yhat, test_Y)
                
                all_yhat.append({'train_yhat':train_yhat, 
                                'test_yhat':test_yhat, 
                                'train_r':train_r, 
                                'test_r':test_r, 
                                'train_p':train_p, 
                                'test_p':test_p, 
                                'alpha':alpha, 
                                'embedding_method':embedding_method, 
                                'y_col':y_col
                                })
                print("\n\n**Embedding method %s, alpha=%2.3f; train r: %2.3f (p=%2.3e); test r: %2.3f; (p=%2.3e)" % (embedding_method, 
                                                                                                                      alpha, 
                                                                                                                      train_r, 
                                                                                                                      train_p, 
                                                                                                                      test_r, 
                                                                                                                      test_p))

                # quick plot to give a sense of results. 
                plt.figure(figsize=[8, 8])
                sns.regplot(test_Y, test_yhat, x_jitter=.2)
                plt.xlabel("Test Y")
                plt.ylabel("Test Yhat")
                if y_col == 'womac_pain_subscore':
                    plt.ylim([0, 20])
                    plt.xlim([0, 20])
                else:
                    plt.ylim([0, 100])
                    plt.xlim([0, 100])
                plt.show()

                # are results driven by only a single visit or a single side? 
                for visit in sorted(list(set(test_combined_df['visit']))):
                    idxs = (test_combined_df['visit'] == visit).values
                    r, p = pearsonr(test_yhat[idxs], test_Y[idxs])
                    print("Visit %s, test r %2.3f (n = %i)" % (visit, r, idxs.sum()))
                for side in ['left', 'right']:
                    idxs = (test_combined_df['side'] == side).values
                    r, p = pearsonr(test_yhat[idxs], test_Y[idxs])
                    print("Side %s, test r %2.3f (n = %i)" % (side, r, idxs.sum()))
    all_yhat = pd.DataFrame(all_yhat)
    return all_yhat

def delete_old_images_from_dfs():
    """
    remove the old image files when we regenerate images so we don't have any old stuff lying around. 
    This command takes a while to run. 
    """
    raise Exception("Do not use this method lightly! It deletes files! Remove this exception if you really want to use it.")
    assert node_name in ['rambo', 'trinity']
    for dataset in ['train', 'val', 'test', 'BLINDED_HOLD_OUT_DO_NOT_USE']:
        base_path_to_delete = os.path.join(INDIVIDUAL_IMAGES_PATH, dataset)
        if os.path.exists(base_path_to_delete):
            cmd = 'rm -rf %s/' % base_path_to_delete
            print("Deleting all files from directory %s" % base_path_to_delete)
            os.system(cmd)
        # make a new folder, because we've deleted the old folder. The reason we have to do it this way is 
        # if we don't delete the folder but only the files within it, 
        # we get an error during the deletion command because there are too many image files. 
        cmd = 'mkdir %s' % base_path_to_delete
        os.system(cmd)

def get_base_dir_for_individual_image(dataset, 
    show_both_knees_in_each_image, 
    downsample_factor_on_reload, 
    normalization_method, 
    seed_to_further_shuffle_train_test_val_sets, 
    crop_to_just_the_knee):
    """
    Get the path for an image. 
    """
    assert seed_to_further_shuffle_train_test_val_sets is None # this is deprecated; don't let us use it accidentally. 
    assert dataset in ['train', 'val', 'test', 'BLINDED_HOLD_OUT_DO_NOT_USE']
    assert show_both_knees_in_each_image in [True, False]
    assert downsample_factor_on_reload in [None, 0.7, 0.5, 0.3]
    assert normalization_method in ['imagenet_statistics', 'our_statistics', 'zscore_individually']
    assert crop_to_just_the_knee in [True, False]

    if show_both_knees_in_each_image:
        assert not crop_to_just_the_knee


    if seed_to_further_shuffle_train_test_val_sets is None:
        random_seed_suffix = ''
    else:
        random_seed_suffix = '_random_seed_%i' % seed_to_further_shuffle_train_test_val_sets
    
    if not crop_to_just_the_knee:
        base_dir = os.path.join(INDIVIDUAL_IMAGES_PATH, 
                dataset, 
                'show_both_knees_%s_downsample_factor_%s_normalization_method_%s%s'  % (
                    show_both_knees_in_each_image, 
                    downsample_factor_on_reload, 
                    normalization_method, 
                    random_seed_suffix))
    else:
        base_dir = os.path.join(INDIVIDUAL_IMAGES_PATH, 
                dataset, 
                'crop_to_just_the_knee_downsample_factor_%s_normalization_method_%s%s'  % (
                    downsample_factor_on_reload, 
                    normalization_method, 
                    random_seed_suffix))

    return base_dir
        

def write_out_individual_images_for_one_dataset(write_out_image_data, 
    normalization_method, 
    show_both_knees_in_each_image, 
    downsample_factor_on_reload, 
    seed_to_further_shuffle_train_test_val_sets, 
    crop_to_just_the_knee):
    """
    If we actually want to train several neural nets simultaneously, the entire image dataset is too large to fit in memory. 
    So, after loading the whole image dataset, we also write out each image into a separate file. 
    We save the images several different ways -- with different preprocessing and downsampling sizes. 
    Checked. 
    """    
    image_dataset_kwargs = copy.deepcopy(IMAGE_DATASET_KWARGS)
    image_dataset_kwargs['reprocess_all_images'] = False
    image_dataset_kwargs['use_small_data'] = False
    image_dataset_kwargs['normalization_method'] = normalization_method
    image_dataset_kwargs['downsample_factor_on_reload'] = downsample_factor_on_reload
    image_dataset_kwargs['show_both_knees_in_each_image'] = show_both_knees_in_each_image
    image_dataset_kwargs['crop_to_just_the_knee'] = crop_to_just_the_knee
    image_dataset = XRayImageDataset(**image_dataset_kwargs)
    for dataset in ['train', 'val', 'test', 'BLINDED_HOLD_OUT_DO_NOT_USE']:
        print("Writing out individual images for %s" % dataset)
        base_path = get_base_dir_for_individual_image(dataset=dataset, 
                                                      show_both_knees_in_each_image=show_both_knees_in_each_image, 
                                                      downsample_factor_on_reload=downsample_factor_on_reload, 
                                                      normalization_method=normalization_method, 
                                                      seed_to_further_shuffle_train_test_val_sets=seed_to_further_shuffle_train_test_val_sets, 
                                                      crop_to_just_the_knee=crop_to_just_the_knee)
        if os.path.exists(base_path):
            raise Exception('base path %s should not exist' % base_path)
        time.sleep(3)

        while not os.path.exists(base_path):
            # for some reason this command occasionally fails; make it more robust. 
            os.system('mkdir %s' % base_path)
            time.sleep(10)

        if dataset == 'BLINDED_HOLD_OUT_DO_NOT_USE':
            i_promise_i_really_want_to_use_the_blinded_hold_out_set = True
        else:
            i_promise_i_really_want_to_use_the_blinded_hold_out_set = False

        non_image_dataset = non_image_data_processing.NonImageData(what_dataset_to_use=dataset, 
                                                                   timepoints_to_filter_for=TIMEPOINTS_TO_FILTER_FOR, 
                                                                   seed_to_further_shuffle_train_test_val_sets=seed_to_further_shuffle_train_test_val_sets, 
                                                                   i_promise_i_really_want_to_use_the_blinded_hold_out_set=i_promise_i_really_want_to_use_the_blinded_hold_out_set)
        combined_df, matched_images, image_codes = match_image_dataset_to_non_image_dataset(image_dataset, non_image_dataset)
        ensure_barcodes_match(combined_df, image_codes)
        assert combined_df['visit'].map(lambda x:x in TIMEPOINTS_TO_FILTER_FOR).all()
        
        non_image_csv_outfile = os.path.join(base_path, 'non_image_data.csv')
        combined_df.to_csv(non_image_csv_outfile)
        if write_out_image_data:
            ensure_barcodes_match(combined_df, image_codes)
            pickle.dump(image_codes, open(os.path.join(base_path, 'image_codes.pkl'), 'wb'))
            for i in range(len(combined_df)):
                image_path = os.path.join(base_path, 'image_%i.npy' % i)
                np.save(image_path, matched_images[i])
                print("%s image %i/%i written out to %s" % (dataset, i + 1, len(combined_df), image_path))
    print("Successfully wrote out all images.")

def write_out_image_datasets_in_parallel():
    """
    Parallelize the writing out of images since it takes a while. This can be run on rambo. 
    Each job writes out the images for one normalization_method,show_both_knees_in_each_image,downsample_factor_on_reload. 
    This undoubtedly is not the CPU or memory-efficient way to do it, but whatever. 

    This does not write out the cropped-knee datasets or different random seed datasets; I wrote separate methods to do taht. 
    """
    ataset_idx = 1
    n_currently_running = 0
    for normalization_method in ['imagenet_statistics', 'our_statistics', 'zscore_individually']:
        for show_both_knees_in_each_image in [True]:
            for downsample_factor_on_reload in [None, 0.7, 0.5, 0.3]:
                for crop_to_just_the_knee in [False]:
                    cmd = 'nohup python -u image_processing.py --normalization_method %s --show_both_knees_in_each_image %s --downsample_factor_on_reload %s --write_out_image_data True --seed_to_further_shuffle_train_test_val_sets None --crop_to_just_the_knee %s > processing_outfiles/image_processing_dataset_%i.out &' % (
                        normalization_method, 
                        show_both_knees_in_each_image, 
                        downsample_factor_on_reload, 
                        crop_to_just_the_knee, 
                        dataset_idx)

                    print("Now running command %s" % cmd)
                    dataset_idx += 1
                    n_currently_running += 1
                    os.system(cmd)
                    if n_currently_running >= 4:
                        time.sleep(6 * 3600)
                        n_currently_running = 0


def write_out_datasets_shuffled_with_different_random_seed():
    """
    Write out a couple additional shuffled datasets. Robustness check to make sure our main results are consistent across train sets. 
    """
    raise Exception("This is deprecated; we now can just reshuffle the train/test/val sets using the original dataset.")
    dataset_idxs = [int(a.split('_')[-1].replace('.out', '')) for a in os.listdir('processing_outfiles')]
    dataset_idx = max(dataset_idxs) + 1
    n_currently_running = 0
    for normalization_method in ['our_statistics']:
        for show_both_knees_in_each_image in [True]:
            for downsample_factor_on_reload in [None]:
                for random_seed in range(1, 5):
                
                    cmd = 'nohup python -u image_processing.py --normalization_method %s --show_both_knees_in_each_image %s --downsample_factor_on_reload %s --write_out_image_data True --seed_to_further_shuffle_train_test_val_sets %i --crop_to_just_the_knee False > processing_outfiles/image_processing_dataset_%i.out &' % (
                        normalization_method, 
                        show_both_knees_in_each_image, 
                        downsample_factor_on_reload, 
                        random_seed,
                        dataset_idx)
                    print("Now running command %s" % cmd)
                    dataset_idx += 1
                    n_currently_running += 1
                    os.system(cmd)
                    if n_currently_running >= 1:
                        time.sleep(6 * 3600)
                        n_currently_running = 0

def write_out_datasets_cropped_to_just_the_knee():
    """
    Write out cropped knee datasets. 
    """
    dataset_idx = 1
    for normalization_method in ['imagenet_statistics', 'our_statistics', 'zscore_individually']:
        for downsample_factor_on_reload in [None, 0.5]:
            cmd = 'nohup python -u image_processing.py --normalization_method %s --show_both_knees_in_each_image False --downsample_factor_on_reload %s --write_out_image_data True --seed_to_further_shuffle_train_test_val_sets None --crop_to_just_the_knee True > processing_outfiles/image_processing_dataset_%i.out &' % (
                        normalization_method, 
                        downsample_factor_on_reload, 
                        dataset_idx)
            print("Now running command %s" % cmd)
            dataset_idx += 1
            os.system(cmd)




def random_horizontal_vertical_translation(img, max_horizontal_translation, max_vertical_translation):
    """
    Translates the image horizontally/vertically by a fraction of its width/length. 
    To keep the image the same size + scale, we add a background color to fill in any space created. 
    """
    assert max_horizontal_translation >= 0 and max_horizontal_translation <= 1
    assert max_vertical_translation >= 0 and max_vertical_translation <= 1
    if max_horizontal_translation == 0 and max_vertical_translation == 0:
        return img

    img = img.copy()

    assert len(img.shape) == 3
    assert img.shape[0] == 3
    assert img.shape[1] >= img.shape[2]
    
    height = img.shape[1]
    width = img.shape[2]

    translated_img = img
    horizontal_translation = int((random.random() - .5) * max_horizontal_translation * width)
    vertical_translation = int((random.random() - .5) * max_vertical_translation * height)
    background_color = img[:, -10:, -10:].mean(axis=1).mean(axis=1)

    # first we translate the image. 
    if horizontal_translation != 0:
        if horizontal_translation > 0:
            translated_img = translated_img[:, :, horizontal_translation:] # this cuts off pixels on the left of the image
        else:
            translated_img = translated_img[:, :, :horizontal_translation] # this cuts off pixels on the right of the image

    if vertical_translation != 0:
        if vertical_translation > 0:
            translated_img = translated_img[:, vertical_translation:, :] # this cuts off pixels on the top of the image
        else:
            translated_img = translated_img[:, :vertical_translation, :] # this cuts off pixels on the bottom of the image. 

    # then we keep the dimensions the same. 
    new_height = translated_img.shape[1]
    new_width = translated_img.shape[2]
    new_image = []
    for i in range(3): # loop over RGB
        background_square = np.ones([height, width]) * background_color[i]
        if horizontal_translation < 0:
            if vertical_translation < 0:
                # I don't really know if the signs here matter all that much -- it's just whether we're putting the translated 
                # images on the left or right. 
                background_square[-new_height:, -new_width:] = translated_img[i, :, :]
            else:
                background_square[:new_height, -new_width:] = translated_img[i, :, :]
        else:
            if vertical_translation < 0:
                background_square[-new_height:, :new_width] = translated_img[i, :, :]
            else:
                background_square[:new_height, :new_width] = translated_img[i, :, :]
        new_image.append(background_square)
    new_image = np.array(new_image)

    return new_image

class PytorchImagesDataset(Dataset):
    """
    A class for loading in images one at a time. 
    Follows pytorch dataset tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, 
        dataset, 
        downsample_factor_on_reload, 
        normalization_method, 
        show_both_knees_in_each_image,
        y_col, 
        transform, 
        seed_to_further_shuffle_train_test_val_sets,
        crop_to_just_the_knee,
        max_horizontal_translation=None, 
        max_vertical_translation=None, 
        additional_features_to_predict=None,
        use_very_very_small_subset=False, 
        load_only_single_klg=None, 
        blur_filter=None):
        """
        Args:
            dataset: train, val, or test. 
            downsample_factor_on_reload, normalization_method -- same as in image processing. 
            y_col: what we're trying to predict. 
            transform: how to augment the loaded images. 
        """
        assert dataset in ['train', 'val', 'test', 'BLINDED_HOLD_OUT_DO_NOT_USE']
        assert downsample_factor_on_reload in [None, 0.7, 0.5, 0.3]
        assert y_col in ['koos_pain_subscore', 
        'womac_pain_subscore', 
        'binarized_koos_pain_subscore', 
        'binarized_womac_pain_subscore',
        'xrkl', 
        'koos_pain_subscore_residual',
        'binarized_education_graduated_college', 
        'binarized_income_at_least_50k']

        assert normalization_method in ['imagenet_statistics', 'our_statistics', 'zscore_individually']
        assert transform in [None, 'random_translation', 'random_translation_and_then_random_horizontal_flip']
        assert (max_horizontal_translation is None) == (transform is None)
        assert (max_vertical_translation is None) == (transform is None)
        if show_both_knees_in_each_image == True:
            assert transform != 'random_translation_and_then_random_horizontal_flip'
             
        self.dataset = dataset
        self.downsample_factor_on_reload = downsample_factor_on_reload
        self.normalization_method = normalization_method
        self.show_both_knees_in_each_image = show_both_knees_in_each_image
        self.crop_to_just_the_knee = crop_to_just_the_knee
        self.use_very_very_small_subset = use_very_very_small_subset
        self.max_horizontal_translation = max_horizontal_translation
        self.max_vertical_translation = max_vertical_translation
        self.seed_to_further_shuffle_train_test_val_sets = seed_to_further_shuffle_train_test_val_sets
        self.clinical_control_columns = CLINICAL_CONTROL_COLUMNS
        self.additional_features_to_predict = additional_features_to_predict
        self.load_only_single_klg = load_only_single_klg
        self.blur_filter = blur_filter

        if seed_to_further_shuffle_train_test_val_sets is None:
            self.base_dir_for_images = get_base_dir_for_individual_image(dataset=self.dataset,
                                                                        show_both_knees_in_each_image=self.show_both_knees_in_each_image,
                                                                        downsample_factor_on_reload=self.downsample_factor_on_reload,
                                                                        normalization_method=self.normalization_method, 
                                                seed_to_further_shuffle_train_test_val_sets=self.seed_to_further_shuffle_train_test_val_sets, 
                                                crop_to_just_the_knee=self.crop_to_just_the_knee)
            self.image_codes = pickle.load(open(os.path.join(self.base_dir_for_images, 'image_codes.pkl'), 'rb'))
            self.non_image_data = pd.read_csv(os.path.join(self.base_dir_for_images, 'non_image_data.csv'), index_col=0)
        else:
            # We need to (somewhat hackily) paste the train, val, and test sets together. 
            print("Alert! Random seed is %i" % self.seed_to_further_shuffle_train_test_val_sets)
            assert dataset in ['train', 'val', 'test']
            shuffled_ids = make_train_val_test_hold_out_set(self.seed_to_further_shuffle_train_test_val_sets)
            ids_we_are_using = set(shuffled_ids[dataset + '_ids'])
            self.non_image_data = []
            self.image_codes = []
            self.new_image_paths = []
            for dataset_2 in ['train', 'val', 'test']:
                base_dir_for_dataset = get_base_dir_for_individual_image(dataset=dataset_2,
                                                                        show_both_knees_in_each_image=self.show_both_knees_in_each_image,
                                                                        downsample_factor_on_reload=self.downsample_factor_on_reload,
                                                                        normalization_method=self.normalization_method, 
                                                seed_to_further_shuffle_train_test_val_sets=None, 
                                                crop_to_just_the_knee=self.crop_to_just_the_knee)
                non_image_data_from_dataset = pd.read_csv(os.path.join(base_dir_for_dataset, 'non_image_data.csv'), index_col=0)
                image_codes_from_this_dataset = pickle.load(open(os.path.join(base_dir_for_dataset, 'image_codes.pkl'), 'rb'))
                idxs_from_this_dataset = non_image_data_from_dataset['id'].map(lambda x:x in ids_we_are_using).values
                self.non_image_data.append(non_image_data_from_dataset.loc[idxs_from_this_dataset])
                self.image_codes += list(np.array(image_codes_from_this_dataset)[idxs_from_this_dataset])

                # for loading individual images, we just create a new data structure, image paths, which has a list of paths that you need. 
                image_numbers_for_dataset = np.arange(len(non_image_data_from_dataset))[idxs_from_this_dataset]
                image_paths_for_dataset = [os.path.join(base_dir_for_dataset, 'image_%i.npy' % i) for i in image_numbers_for_dataset]
                self.new_image_paths += image_paths_for_dataset
                assert len(image_paths_for_dataset) == len(image_numbers_for_dataset) == idxs_from_this_dataset.sum()
                print("Number of new images added to dataset from original %s dataset: %i; IDs: %i" % (
                    dataset_2, len(image_paths_for_dataset), len(set(non_image_data_from_dataset.loc[idxs_from_this_dataset, 'id'].values))))
            self.non_image_data = pd.concat(self.non_image_data)
            self.non_image_data.index = range(len(self.non_image_data))
            assert set(self.non_image_data['id']) - ids_we_are_using == set([])
            print("Reconstructed dataset %s with %i rows and %i IDs" % (dataset, len(self.non_image_data), len(set(self.non_image_data['id']))))

        if self.additional_features_to_predict is not None:
            self.additional_feature_array = copy.deepcopy(self.non_image_data[self.additional_features_to_predict].values)
            for i in range(len(self.additional_features_to_predict)):
                not_nan = ~np.isnan(self.additional_feature_array[:, i])
                std = np.std(self.additional_feature_array[not_nan, i], ddof=1)
                mu = np.mean(self.additional_feature_array[not_nan, i])
                print("Z-scoring additional feature %s with mean %2.3f and std %2.3f" % (
                    self.additional_features_to_predict[i], mu, std))
                self.additional_feature_array[:, i] = (self.additional_feature_array[:, i] - mu) / std


        if 'binarized_' in y_col:
            if 'koos' in y_col:
                assert y_col not in list(self.non_image_data.columns)
                self.non_image_data[y_col] = binarize_koos(self.non_image_data['koos_pain_subscore'].values)
                print("Using binary column %s as y_col, a fraction %2.3f are positive (high pain) examples <= threshold %2.3f" % 
                (y_col, self.non_image_data[y_col].mean(), KOOS_BINARIZATION_THRESH))
            elif 'womac' in y_col:
                assert y_col not in list(self.non_image_data.columns)
                self.non_image_data[y_col] = binarize_womac(self.non_image_data['womac_pain_subscore'].values)
                print("Using binary column %s as y_col, a fraction %2.3f are positive (high pain) examples > threshold %2.3f" % 
                (y_col, self.non_image_data[y_col].mean(), WOMAC_BINARIZATION_THRESH))

        # add column with residual. 
        if y_col == 'koos_pain_subscore_residual':
            assert len(self.non_image_data[['koos_pain_subscore', 'xrkl']].dropna()) == len(self.non_image_data)
            pain_kl_model = sm.OLS.from_formula('koos_pain_subscore ~ C(xrkl)', data=self.non_image_data).fit()
            assert 'koos_pain_subscore_residual' not in self.non_image_data.columns
            self.non_image_data['koos_pain_subscore_residual'] = self.non_image_data['koos_pain_subscore'].values - pain_kl_model.predict(self.non_image_data).values
            print(pain_kl_model.summary())
            
        self.y_col = y_col
        self.transform = transform
        print("Dataset %s has %i rows" % (dataset, len(self.non_image_data)))
        ensure_barcodes_match(self.non_image_data, self.image_codes)
        if self.show_both_knees_in_each_image:
            self.spot_check_ensure_original_images_match()
        
            
    
    def spot_check_ensure_original_images_match(self):
        """
        Sanity check: make sure we're loading the right images, as measured by a high correlation between the processed images and the original dicom images. 
        Can only do this if there's been relatively little preprocessing -- eg, no dramatic cropping of images. 

        Images are not necessarily identical because we have done some preprocessing (eg, smoothing or downsampling) but should be very highly correlated. 
        """
        necessary_path = os.path.join(BASE_IMAGE_DATA_DIR, '00m')
        if not os.path.exists(necessary_path):
            print("Warning: not spot-checking that images match original raw data because necessary path %s does not exist" % necessary_path)
            print("If you want to spot-check, you need to download the raw data and store it at this path")
            return

        print("Spot checking that images match.")
        contents_df = pd.read_csv(os.path.join(BASE_IMAGE_DATA_DIR, '00m/contents.csv'))
        idxs_to_sample = [a for a in range(len(self.non_image_data)) if self.non_image_data.iloc[a]['visit'] in ['00 month follow-up: Baseline']]
        all_correlations = []
        for random_idx in random.sample(idxs_to_sample, 10):
            row = self.non_image_data.iloc[random_idx][['id', 'side', 'barcdbu', 'visit']]
            #print(row)
            barcode = int(row['barcdbu'].astype(str)[-7:])
            folder = str(contents_df.loc[(contents_df['SeriesDescription'] == 'Bilateral PA Fixed Flexion Knee') & 
                    (contents_df['Barcode'] == barcode)].iloc[0]['Folder'])
            original_image_path = os.path.join(BASE_IMAGE_DATA_DIR, '00m', folder, '001')
            original_image = pydicom.dcmread(original_image_path)
            if self.seed_to_further_shuffle_train_test_val_sets is None:
                our_image_path = os.path.join(self.base_dir_for_images, 'image_%i.npy' % random_idx)
            else:
                our_image_path = self.new_image_paths[random_idx]
            our_test_image = np.load(our_image_path)[0, :, :].squeeze()
            original_image = cv2.resize(original_image.pixel_array, dsize=tuple(our_test_image.shape)[::-1], interpolation=cv2.INTER_CUBIC)
            if row['side'] == 'right':
                original_image = original_image[:, ::-1]
            all_correlations.append(spearmanr(original_image.flatten(), our_test_image.flatten())[0])
            print("Correlation between original and reloaded image is", all_correlations[-1])
        assert np.median(all_correlations) >= .99
        assert np.mean(all_correlations) >= .97
        print("Image spot check image passed.")

    def __len__(self):
        if self.use_very_very_small_subset:
            return 500
        if self.load_only_single_klg is not None:
            raise Exception("This is not an option you should be using.")
            return (self.non_image_data['xrkl'].values == self.load_only_single_klg).sum()
        return len(self.non_image_data)

    def __getitem__(self, idx):
        if self.seed_to_further_shuffle_train_test_val_sets is None:
            image_path = os.path.join(self.base_dir_for_images, 'image_%i.npy' % idx)
        else:
            image_path = self.new_image_paths[idx]
        image = np.load(image_path)
        if self.transform:
            assert self.transform in ['random_translation_and_then_random_horizontal_flip', 'random_translation']
            image = random_horizontal_vertical_translation(image, self.max_horizontal_translation, self.max_vertical_translation)
            if self.transform == 'random_translation_and_then_random_horizontal_flip':
                if random.random() < 0.5:
                    image = image[:, :, ::-1].copy()
        if self.blur_filter is not None:
            assert self.blur_filter > 0 and self.blur_filter < 1 # this argument is the downsample fraction
            downsample_frac = self.blur_filter
            new_image = []
            for i in range(3):
                img = image[i, :, :].copy()
                original_size = img.shape # note have to reverse arguments for cv2. 
                img2 = cv2.resize(img, (int(original_size[1] * downsample_frac), int(original_size[0] * downsample_frac)))
                new_image.append(cv2.resize(img2, tuple(original_size[::-1])))
                #image[i, :, :] = gaussian_filter(image[i, :, :], sigma=self.gaussian_blur_filter)
            new_image = np.array(new_image)
            assert new_image.shape == image.shape
            image = new_image
        if self.additional_features_to_predict is not None:
            additional_features = self.additional_feature_array[idx, :]
            additional_features_are_not_nan = ~np.isnan(additional_features)
            additional_features[~additional_features_are_not_nan] = 0
            additional_features_are_not_nan = additional_features_are_not_nan * 1.
        else:
            additional_features = []
            additional_features_are_not_nan = []

        yval = self.non_image_data[self.y_col].iloc[idx]
        
        klg = self.non_image_data['xrkl'].iloc[idx]
        assert klg in [0, 1, 2, 3, 4]
        klg_coding = np.array([0., 0., 0., 0., 0.])
        klg_coding[int(klg)] = 1.
        klg = klg_coding

        binarized_education_graduated_college = self.non_image_data['binarized_education_graduated_college'].iloc[idx]
        assert binarized_education_graduated_college in [0, 1]

        binarized_income_at_least_50k = self.non_image_data['binarized_income_at_least_50k'].iloc[idx]
        assert binarized_income_at_least_50k in [0, 1]

        site = self.non_image_data['v00site'].iloc[idx]
        assert site in ['A', 'B', 'C', 'D', 'E']

        assert ~np.isnan(yval)

        sample = {'image': image, 
        'y':yval, 
        'klg':klg,
        'binarized_education_graduated_college':binarized_education_graduated_college,
        'binarized_income_at_least_50k':binarized_income_at_least_50k,
        'additional_features_to_predict':additional_features, 
        'additional_features_are_not_nan':additional_features_are_not_nan, 
        'site':site}
        return sample

if __name__ == '__main__':
    from traceback import print_exc
    import argparse

    parser = argparse.ArgumentParser()
    args = sys.argv

    def str2bool(x):
        assert x in ['True', 'False']
        return x == 'True'

    if len(sys.argv) > 1:
        parser.add_argument('--write_out_image_data', type=str)
        parser.add_argument('--normalization_method', type=str)
        parser.add_argument('--show_both_knees_in_each_image', type=str)
        parser.add_argument('--downsample_factor_on_reload', type=str)
        parser.add_argument('--seed_to_further_shuffle_train_test_val_sets', type=str)
        parser.add_argument('--crop_to_just_the_knee', type=str)
        args = parser.parse_args()

        downsample_factor_on_reload = None if args.downsample_factor_on_reload == 'None' else float(args.downsample_factor_on_reload)
        seed_to_further_shuffle_train_test_val_sets = None if args.seed_to_further_shuffle_train_test_val_sets == 'None' else int(args.seed_to_further_shuffle_train_test_val_sets)

        write_out_individual_images_for_one_dataset(write_out_image_data=str2bool(args.write_out_image_data), 
                        normalization_method=args.normalization_method, 
                        show_both_knees_in_each_image=str2bool(args.show_both_knees_in_each_image), 
                        downsample_factor_on_reload=downsample_factor_on_reload, 
                        seed_to_further_shuffle_train_test_val_sets=seed_to_further_shuffle_train_test_val_sets, 
                        crop_to_just_the_knee=str2bool(args.crop_to_just_the_knee))
    else:
        image_dataset = XRayImageDataset(reprocess_all_images=True, show_both_knees_in_each_image=True, crop_to_just_the_knee=False, **IMAGE_DATASET_KWARGS)
        # DEPRECATED COMMENTS. 
        # Step 1: clear out old images on /dfs.
        #delete_old_images_from_dfs()
        # Step 2: reprocess the original DICOM images into a pkl.
        # image_dataset = XRayImageDataset(reprocess_all_images=True, show_both_knees_in_each_image=True, crop_to_just_the_knee=False, **IMAGE_DATASET_KWARGS)
        # Step 3: write out individual images on /dfs. 
        #write_out_image_datasets_in_parallel()
        #time.sleep(6 * 3600)
        # Step 4: (somewhat optional) write out images cropped to the knee. 
        #time.sleep(6 * 3600)
        #write_out_datasets_cropped_to_just_the_knee()        
        #time.sleep(8 * 3600)
        
        #
        
    #compare_contents_files_to_loaded_images(image_dataset, IMAGE_DATASET_KWARGS['desired_image_type'])

    


