TODO: Check readme and any changes to code made while creating README. 
TODO: replace any XXs in README. 

Code to generate results in "[XX](XX)" by XX. Please contact emmap1@cs.stanford.edu with any questions. 

## Regenerating results

1. **Setting up virtualenv**. Our code is run in a virtual environment using Python 3.5.2. You can set up the environment by using `virtualenv -p python3.5 YOUR_PATH_TO_VIRTUALENV`, activating the virtualenv via `source YOUR_PATH_TO_VIRTUALENV/bin/activate`, and then installing packages via `pip install -r requirements.txt`. Make sure the virtual environment is activated prior to running any of the steps below.  If you want to run `main_results_for_public_repo.ipynb` you will additionally need to run `python -m ipykernel install --user --name=knee` to install a kernel for the IPython notebook; make sure to use this kernel when running the notebook. 

2. **Obtaining processed data**. The fastest and easiest way to reproduce our results is to obtain the processed image and non-image data, available at XX. This is faster and easier because the original OAI data is multiple terabytes, taking a while to download, and also requires substantial compute and storage to process. Data was processed on a computer with several terabytes of RAM and hundreds of cores. However, for completeness, we also provide the code needed to regenerate the processed data from raw data (which can be downloaded at [https://nda.nih.gov/oai/](https://nda.nih.gov/oai/)).

    - a. Process the original DICOM files into a pickle of numpy arrays. This can be done by running `python image_processing.py`. (We recommend running this in a screen session or similar because it takes a while). 
    - b. Write out the individual images as separate files because the original pickle is too large. This can be done by running 
        `python image_processing.py --normalization_method our_statistics --show_both_knees_in_each_image True --downsample_factor_on_reload None --write_out_image_data True --seed_to_further_shuffle_train_test_val_sets None --crop_to_just_the_knee False`. Again, we recommend running this in a screen session. Note this actually writes out four datasets, not three - train, val, test, and a blinded hold out set. As described in the paper, all exploratory analysis on the paper was performed using only the train, val, and test sets. However, for the final analysis, we retrained models on the train+test sets and evaluated on the blinded hold out set. The four datasets can be combined into three using the method `rename_blinded_test_set_files` in `constants_and_util.py`. 

3. **Set paths.** You will need to set paths suitable for your system in constants_and_util.py. Please see the "Please set these paths for your system" comment in `constants_and_util.py`, and the associated capitalized variables. 

4. **Training models.** Neural network experiments are performed using `python train_models.py EXPERIMENT_NAME`. (For valid experiment names, see the `train_one_model` method.) Running this script will train neural net models indefinitely (after a model is trained and saved, training for a new one begins) which is useful for ensembling models. Models in the paper were trained using four Nvidia XP GPUs. Specific experiments discussed in the results are: 

    - `train_best_model_continuous`: Trains models to predict pain using the best-performing config. 

    - `hold_out_one_imaging_site`: Trains the models using data from all but one imaging site to confirm results generalize across sites. 

    - `predict_klg`: Train the models to predict KLG rather than pain (using same config as in `train_best_model_continuous`) and show that our results are comparable to previous ones. 

    - `increase_diversity`: Assess the effect of altering the racial or socioeconomic diversity in the train dataset while keeping dataset size constant. 

5. **Analyzing models and generating figures for paper**. Once models have been run, figures and results in the paper can be reproduced by running `main_results_for_public_repo.ipynb`. Running this notebook takes a while (about a day) because of the number of bootstrap iterations, so we recommend running it in a screen session using, eg, `jupyter nbconvert --execute --ExecutePreprocessor.timeout=-1 --to notebook main_results_for_public_repo.ipynb`. A much faster approach is to run only the cells you need to reproduce the results of interest; alternately, you can reduce the number of bootstrap iterations. Note that running cells which call the class `non_image_data_processing.NonImageData` will require downloading the original non-image data from the OAI (but these files are much smaller and faster to process than the image files). 

## Files

**constants_and_util.py**: Constants and general utility methods. 

**non_image_data_processing.py**: Processes non-image data.

**image_processing.py**: Processes image data and combines with non-image data. 

**train_models.py**: Trains the neural network models used in analysis. 

**analysis.py**: Helper methods for analysis used in the paper. 

**main_results_for_public_repo.ipynb**: Generates the figures and numerical results in the paper using the methods in `analysis.py`. 

**requirements.txt**: Packages used in the virtualenv. 