from constants_and_util import *
import os
import pandas as pd
import copy
from scipy.stats import pearsonr
from collections import Counter
import datetime

class NonImageData():
    """
    Class for loading the non-image data. 
    Requires an argument to specify train val test or BLINDED_HOLD_OUT_SET. 
    """
    def __init__(self, 
        what_dataset_to_use, 
        timepoints_to_filter_for, 
        seed_to_further_shuffle_train_test_val_sets=None,
        i_promise_i_really_want_to_use_the_blinded_hold_out_set=False, 
        filter_out_special_values_in_mri_data=False):
        """
        Load raw data, turn it into processed data, and do some validations. Checked. 
        Raw data was downloaded from https://ndar.nih.gov/oai/full_downloads.html

        Minor note: this method raises a "DtypeWarning: Columns (5) have mixed types." warning. This is caused by a file in a column we do not use in a timepoint we do not use. It could be fixed by using 

        pd.read_csv('/dfs/dataset/tmp/20180910-OAI/data/emma_downloaded_oai_data_9112018/MRI MetaAnalysis_ASCII/MRI10.txt', 
                sep='|', 
                dtype={'V10MQCCMNT':str})
        """
        assert what_dataset_to_use in ['train', 'val', 'test', 'BLINDED_HOLD_OUT_DO_NOT_USE', 'all']
        if not i_promise_i_really_want_to_use_the_blinded_hold_out_set:
            assert what_dataset_to_use not in ['BLINDED_HOLD_OUT_DO_NOT_USE', 'all'] # just a sanity check to make sure we don't accidentally use these. 
        self.seed_to_further_shuffle_train_test_val_sets = seed_to_further_shuffle_train_test_val_sets
        self.what_dataset_to_use = what_dataset_to_use
        self.clinical_base_dir = os.path.join(BASE_NON_IMAGE_DATA_DIR, 'AllClinical_ASCII')
        self.semiquantitative_xray_dir = os.path.join(BASE_NON_IMAGE_DATA_DIR, 
            'X-Ray Image Assessments_ASCII', 
            'Semi-Quant Scoring_ASCII')
        self.semiquantitative_mri_dir = os.path.join(BASE_NON_IMAGE_DATA_DIR, 
            'MR Image Assessment_ASCII', 
            'Semi-Quant Scoring_ASCII')
        self.xray_metadata_dir = os.path.join(BASE_NON_IMAGE_DATA_DIR, 'X-Ray MetaAnalysis_ASCII')
        self.mri_metadata_dir = os.path.join(BASE_NON_IMAGE_DATA_DIR, 'MRI MetaAnalysis_ASCII')
        self.original_dataframes = {} # store the original CSVs
        self.processed_dataframes = {} # store the processed data
        self.col_mappings = {}
        self.missing_data_val = '.: Missing Form/Incomplete Workbook'
        self.filter_out_special_values_in_mri_data = filter_out_special_values_in_mri_data
        # From the OAI quantitative x-ray notes: 
        # The variable SIDE denotes whether the row of data is for a right side image (SIDE=1) or a left side image (SIDE=2)
        self.side_mappings = {1:'right', 2:'left'}
        if timepoints_to_filter_for is None:
            self.timepoints_to_filter_for = TIMEPOINTS_TO_FILTER_FOR
            print("Set timepoints to filter for to", TIMEPOINTS_TO_FILTER_FOR)
        else:
            self.timepoints_to_filter_for = timepoints_to_filter_for
            
        # load various dataframes 
        self.load_clinical_data()
        self.load_semiquantitative_xray_data()
        self.load_xray_metadata()
        self.load_semiquantitative_mri_data()
        self.load_mri_metadata()

        # make processed dataframes. 
        
        self.make_nonstandard_interventions_dataframe()
        self.make_medications_dataframe()
        self.make_400m_walk_dataframe()
        self.make_redundant_knee_xray_variable_dataframe()
        self.make_knee_pain_dataframe()
        self.make_other_koos_subscores_dataframe()
        self.make_per_person_controls_dataframe()
        self.make_previous_injury_dataframe()
        self.make_previous_surgery_dataframe()
        self.make_previous_knee_replacement_dataframe()
        self.make_bmi_dataframe()
        self.make_drinking_and_smoking_dataframe()
        self.make_medical_history_dataframe()
        self.make_pain_dataframe_for_all_other_types_of_pain()
        self.make_age_dataframe()
        self.make_dominant_leg_dataframe()
        self.make_previous_fracture_or_fall_dataframe()
        self.make_processed_mri_data()

        # some validation. 
        self.validate_processed_data()
        self.validate_ids()
        
        if self.what_dataset_to_use != 'all':
            self.filter_for_correct_set()
        self.filter_out_timepoints()
        self.filter_out_visits_too_far_from_xray_imaging()
        print("Successfully loaded non-image data.")



    def filter_out_timepoints(self):
        """
        Remove datapoints from processed dataframes if they're not in timepoints_to_filter_for.
        """
        print("Filtering for timepoints", self.timepoints_to_filter_for)
        for k in sorted(self.processed_dataframes.keys()):
            if 'visit' in self.processed_dataframes[k].columns:
                print("\nLength of %s prior to filtering: %i" % (k, len(self.processed_dataframes[k])))
                assert pd.isnull(self.processed_dataframes[k]['visit']).sum() == 0
                print("Values of visit prior to filtering", sorted(list(set(self.processed_dataframes[k]['visit']))))
                if not all([a in list(set(self.processed_dataframes[k]['visit'].dropna())) for a in self.timepoints_to_filter_for]):
                    raise Exception("There is a problem with the visit column in %s: not all the timepoints we want are present." % k)
                if not all([a in CLINICAL_WAVES_TO_FOLLOWUP.values() for a in list(set(self.processed_dataframes[k]['visit'].dropna()))]):
                    raise Exception("There is a problem with the visit column in %s: not all values in the column are valid visits." % k)

                self.processed_dataframes[k] = self.processed_dataframes[k].loc[self.processed_dataframes[k]['visit'].map(
                    (lambda x:x in self.timepoints_to_filter_for))]

                self.processed_dataframes[k].index = range(len(self.processed_dataframes[k]))
                print("Length of %s after filtering: %i" % (k, len(self.processed_dataframes[k])))
                print("Values of visit after filtering", sorted(list(set(self.processed_dataframes[k]['visit']))))
            else:
                print("Not filtering for visit for dataframe %s because no visit column" % k)


    def filter_for_correct_set(self):
        """
        Make sure our dataset contains only the right dataset (eg, train set etc). Checked. 
        """
        print("Filtering for %s set." % self.what_dataset_to_use)
        ids = make_train_val_test_hold_out_set(seed_to_further_shuffle_train_test_val_sets=self.seed_to_further_shuffle_train_test_val_sets)
        ids = ids[self.what_dataset_to_use + '_ids']
        self.all_ids = sorted(ids)
        id_set = set(ids)
        print('****Filtering unprocessed data for %s set.' % self.what_dataset_to_use)
        for k in sorted(self.original_dataframes.keys()):
            assert 'ID' not in self.original_dataframes[k].columns
            if 'id' in self.original_dataframes[k].columns:
                orig_length = len(self.original_dataframes[k])
                self.original_dataframes[k] = self.original_dataframes[k].loc[self.original_dataframes[k]['id'].map(lambda x:x in id_set)]
                print("After filtering, number of rows in %s goes from %i -> %i" % (k, orig_length, len(self.original_dataframes[k])))
                assert orig_length != len(self.original_dataframes[k])
        print('\n****Filtering processed data for %s set.' % self.what_dataset_to_use)
        for k in sorted(self.processed_dataframes.keys()):
            assert 'ID' not in self.processed_dataframes[k].columns
            if 'id' in self.processed_dataframes[k].columns:
                orig_length = len(self.processed_dataframes[k])
                self.processed_dataframes[k] = self.processed_dataframes[k].loc[self.processed_dataframes[k]['id'].map(lambda x:x in id_set)]
                print("After filtering, number of rows in %s goes from %i -> %i" % (k, orig_length, len(self.processed_dataframes[k])))
                assert orig_length != len(self.processed_dataframes[k])

    def validate_processed_data(self):
        """
        Make sure there are no missing data values in the processed data. Checked. 
        """
        for k in self.processed_dataframes:
            assert 'id' in self.processed_dataframes[k].columns
            print("Validating id column for %s" % k)
            assert pd.isnull(self.processed_dataframes[k]['id']).sum() == 0
            if 'visit' in self.processed_dataframes[k].columns:
                print("Validating visit column for %s" % k)
                assert pd.isnull(self.processed_dataframes[k]['visit']).sum() == 0
                assert self.processed_dataframes[k]['visit'].map(lambda x:x in CLINICAL_WAVES_TO_FOLLOWUP.values()).all()
            if 'side' in self.processed_dataframes[k].columns:
                print("Validating side column for %s" % k)
                assert pd.isnull(self.processed_dataframes[k]['side']).sum() == 0
                assert self.processed_dataframes[k]['side'].map(lambda x:x in ['left', 'right']).all()
            for c in self.processed_dataframes[k].columns:
                assert self.processed_dataframes[k][c].map(lambda x:str(x) == self.missing_data_val).sum() == 0

    def load_all_text_files_in_directory(self, base_dir, datasets_to_skip):
        """
        Given a base directory, and datasets to skip, loads in the relevant datasets to self.original_dataframes.
        Column names + dataset names are stored in lowercase. 
        Checked. 
        """
        print("Base directory: %s" % base_dir)
        skipped_datasets = [] # make sure we actually skipped all the datasets we want to skip. 
        for filename in sorted(os.listdir(base_dir)):
            if filename[-4:] == '.txt':
                dataset_name = filename.replace('.txt', '').lower()
                if dataset_name in datasets_to_skip:
                    skipped_datasets.append(dataset_name)
                    continue
                full_path = os.path.join(base_dir, filename)
                d = pd.read_csv(full_path, sep='|')
                d.columns = d.columns.map(lambda x:x.lower())
                assert len(d.columns) == len(set(d.columns))
                print("%s has %i columns, %i rows" % (filename, len(d.columns), len(d)))
                assert dataset_name not in self.original_dataframes # don't add same dataset twice. 
                self.original_dataframes[dataset_name] = d
                self.col_mappings[dataset_name] = {} # in case we want to map column names to anything else, this is a data dictionary. 
        assert sorted(datasets_to_skip) == sorted(skipped_datasets) 

    def concatenate_dataframes_from_multiple_timepoints(self, dataset_substring, columns_to_subset_on=None, visit_numbers_to_skip=None):
        """
        Takes all datasets in original_dataframes that contain dataset_substring, takes the columns in columns_to_subset_on, 
        and adds a column called "visit" which denotes which visit it is. 
        Checked. 
        """
        print('Combining dataframes with substring %s' % dataset_substring)
        dataframes_to_concatenate = []
        expected_columns = None
        for dataset_name in sorted(self.original_dataframes):
            if dataset_substring in dataset_name:
                visit_number = dataset_name.replace(dataset_substring, '') # this should be something like 00. 
                if visit_numbers_to_skip is not None and visit_number in visit_numbers_to_skip:
                    continue
                visit = CLINICAL_WAVES_TO_FOLLOWUP[visit_number]
                print("Adding visit=%s to dataframe %s" % (visit, dataset_name))                
                dataset_copy = copy.deepcopy(self.original_dataframes[dataset_name])
                # make sure each field has a consistent prefix (eg, v00) indicating that it comes from the right timepoint. 
                # there are some exceptions: fields like id, and fields with p01 or p02, which indicate pre-enrollment measurements. 
                assert all(['v%s' % visit_number in a for a in dataset_copy.columns if a not in ['id', 'side', 'readprj', 'version'] and a[:3] not in ['p01', 'p02']])
                dataset_copy.columns = dataset_copy.columns.map(lambda x:x.replace('v%s' % visit_number, ''))
                # if desired, subset the columns. 
                if columns_to_subset_on is not None:
                    dataset_copy = dataset_copy[columns_to_subset_on]

                # make sure columns stay consistent. 
                if expected_columns is None:
                    expected_columns = list(dataset_copy.columns)
                else:
                    assert expected_columns == list(dataset_copy.columns)
                dataset_copy['visit'] = visit
                dataframes_to_concatenate.append(dataset_copy)
        combined_data = pd.concat(dataframes_to_concatenate)
        combined_data.index = range(len(combined_data))
        print("Number of rows in combined data: %i" % len(combined_data))
        return combined_data

    def load_clinical_data(self):
        print("\n***Loading all clinical data.")
        # skip allclinical02 and allclinical04 because they have very little data.
        self.load_all_text_files_in_directory(self.clinical_base_dir, datasets_to_skip=['allclinical02', 'allclinical04'])

    def map_to_date(self, x):
        # sometimes X-ray dates are missing because, as documentation notes
        # "In addition, x-ray date and all QC variables have been set to missing .A for numeric variables, 
        # blank for text variables) when an x-ray was acquired, but is not available."
        # So this date is fairly often NA. But that's okay, because that only occurs (confirmed this) 
        # if the ACCEPT variable is NA anyway, so the data gets filtered out subsequently in find_image_barcodes_that_pass_qc
        if x is not None and str(x) != 'nan':
            return datetime.datetime.strptime(x, '%m/%d/%Y')
        return None

    def filter_out_visits_too_far_from_xray_imaging(self):
        print("\n\n***Filtering out visits too far from x-rays.")
        THRESHOLD_IN_DAYS = 90
        visits_to_bad_ids = {}
        for visit_substring in ['00', '01', '03', '05', '06']:
            allclinical_df = copy.deepcopy(self.original_dataframes['allclinical%s' % visit_substring])
            xray_df = copy.deepcopy(self.original_dataframes['xray%s' % visit_substring])
            xray_df = xray_df.loc[xray_df['v%sexamtp' % visit_substring] == 'Bilateral PA Fixed Flexion Knee']
            xray_date_dict = dict(zip(xray_df['id'].values, 
                                     xray_df['v%sxrdate' % visit_substring].values))

            def return_date_in_dict_if_possible(xray_date_dict, x):
                if x in xray_date_dict:
                    return xray_date_dict[x]
                else:
                    print("Warning! ID %i not in dict." % x) # this happens only once. 
                    return '01/01/1900'
            allclinical_df['v%sxrdate' % visit_substring] = allclinical_df['id'].map(lambda x:return_date_in_dict_if_possible(xray_date_dict, x))

            # xrdate: Date x-ray completed (calc). 
            # p01svdate: Date Screening Visit completed. 
            # v00evdate: Date Enrollment Visit completed. 
            # v01fvdate: Follow-up visit date. 

            if visit_substring == '00':
                all_date_cols = ['p01svdate', 'v00evdate', 'v00xrdate']
            else:
                all_date_cols = ['v%sfvdate' % visit_substring, 'v%sxrdate' % visit_substring]
            
            print("\n\n%s visit" % CLINICAL_WAVES_TO_FOLLOWUP[visit_substring])

            # At followup, there are some people missing dates for x-rays because they didn't have them. 
            # We don't filter them out at this stage because they are filtered out subsequently. 
            # We do verify that a) very few people are missing any date data at the initial timepoint (00) and 
            # b) everyone missing allclinical data is also missing x-ray data, so should be filtered out. 
            if visit_substring != '00':
                xr_missing_date = pd.isnull(allclinical_df['v%sxrdate' % visit_substring].map(lambda x:self.map_to_date(x)))
                allclinical_missing_date = pd.isnull(allclinical_df['v%sfvdate' % visit_substring].map(lambda x:self.map_to_date(x)))
                assert (allclinical_missing_date & (~xr_missing_date)).sum() == 0 # make sure there's no one who has x-rays without coming in for followup in allclinical. 
            else:
                for k in all_date_cols:
                    assert pd.isnull(allclinical_df[k].map(lambda x:self.map_to_date(x))).mean() < .005
            
            bad_ids = None
            
            assert len(set(allclinical_df['id'])) == len(allclinical_df)
            for i in range(len(all_date_cols)):
                print("Fraction of date column %s which cannot be mapped to a date: %2.3f" % 
                    (all_date_cols[i], 
                    pd.isnull(allclinical_df[all_date_cols[i]].map(lambda x:self.map_to_date(x))).mean()))
                for j in range(i):
                    print('***gaps between %s and %s' % (all_date_cols[i], all_date_cols[j]))

                    days_between = np.abs((allclinical_df[all_date_cols[i]].map(lambda x:self.map_to_date(x)) - 
                                    allclinical_df[all_date_cols[j]].map(lambda x:self.map_to_date(x))).map(lambda x:x.days))
                    print("Mean: %2.3f; median %2.3f; greater than 30 days %2.3f; greater than 60 days %2.3f; greater than 90 days %2.5f; missing data %2.5f" % (
                                                                                                        days_between.mean(), 
                                                                                                        days_between.median(), 
                                                                                                        (days_between > 30).mean(), 
                                                                                                        (days_between > 60).mean(), 
                                                                                                        (days_between > 90).mean(), 
                                                                                                        np.isnan(days_between).mean()))
                    if bad_ids is None:
                        bad_ids = set(allclinical_df.loc[days_between > THRESHOLD_IN_DAYS, 'id'].values)
                    else:
                        bad_ids = bad_ids.union(set(allclinical_df.loc[days_between > THRESHOLD_IN_DAYS, 'id'].values))
                    
            visits_to_bad_ids[visit_substring] = bad_ids
            print("Total number of IDs filtered out for visit: %i/%i" % (len(bad_ids), len(allclinical_df)))
        self.visits_too_far_from_xray_screening = visits_to_bad_ids

        for k in self.processed_dataframes:
            if 'visit' in self.processed_dataframes[k].columns:
                rows_to_filter_out = None
                for visit in self.visits_too_far_from_xray_screening:
                    bad_rows_for_visit = (self.processed_dataframes[k]['id'].map(lambda x:x in self.visits_too_far_from_xray_screening[visit]) & 
                                          (self.processed_dataframes[k]['visit'] == CLINICAL_WAVES_TO_FOLLOWUP[visit]))
                    if rows_to_filter_out is None:
                        rows_to_filter_out = bad_rows_for_visit
                    else:
                        rows_to_filter_out = rows_to_filter_out | bad_rows_for_visit
                self.processed_dataframes[k] = self.processed_dataframes[k].loc[~rows_to_filter_out]
                print("For dataframe %s, filtered out %i/%i rows as too far from x-ray date" % (k, rows_to_filter_out.sum(), len(rows_to_filter_out)))

    def make_drinking_and_smoking_dataframe(self):
        """
        Risk factors at baseline. 
        """
        df = copy.deepcopy(self.original_dataframes['allclinical00'])
        
        # cigarette smoking. 
        df['cigarette_smoker'] = df['v00smoker']
        df.loc[df['cigarette_smoker'] == '3: Current, but never regular', 'cigarette_smoker'] = '1: Current'
        df.loc[df['cigarette_smoker'] == self.missing_data_val, 'cigarette_smoker'] = None
        print('Cigarette smoker: ', Counter(df['cigarette_smoker']))
        
        # drinks per week
        df['drinks_per_week'] = df['v00drnkamt']
        df.loc[df['drinks_per_week'] == self.missing_data_val, 'drinks_per_week'] = None
        print('Drinks per week: ', Counter(df['drinks_per_week']))
        
        self.processed_dataframes['drinking_and_smoking'] = df[['id', 'drinks_per_week', 'cigarette_smoker']]

    def make_medical_history_dataframe(self):
        """
        Used to replicate David's regressions as a sanity check, but not actually for any analysis in the paper. 
        
        Currently someone is defined as a 1 if they report having a disease prior to the timepoint
        Defined as missing if they are missing disease data at baseline and don't report having it subsequently. 
        Defined as false otherwise. 
        
        Not entirely sure this is the right way to do this. There's a lot of missing data for RA at baseline. Regarding RA: people are supposed to be excluded if they have it for sure. But I guess v00ra may or may not indicate RA, as defined by the study -- perhaps they think some people are giving unreliable answers, and that accounts for the missing data? 

        "Participants who report that a doctor has told them they have RA, SLE, psoriatic arthritis, ankylosing spondylitis or another inflammatory arthritis will be asked about use of specific medications that are used primarily for RA and other forms of inflammatory arthritis: e.g. gold, methotrexate, etanercept, infliximab, leflunamide, plaquenil, etc. If the person has ever used any of these medications, they will be excluded. If the participant reports having RA or inflammatory arthritis but none of these medications have been used, they will be asked about symptoms of RA and excluded if the responses are suggestive of RA"

        This includes a couple of other covariates David actually doesn't use in his regression. 
        """
        print("\n\n***Making dataframe of medical history.")
        all_dfs = []
        medical_conditions = ['hrtat', 'hrtfail', 'bypleg', 'stroke', 'asthma', 'lung', 
                                        'ulcer', 'diab', 'kidfxn', 'ra', 'polyrh', 'livdam', 'cancer']
        
        # we omit ALZDZ even though it's in david's script because it doessn't appear to be in our data. 
        
        all_ids = list(self.original_dataframes['allclinical00']['id'])
        has_disease = {}
        nas_at_baseline = {}
        for condition in medical_conditions:
            has_disease[condition] = set([])
            nas_at_baseline[condition] = set([])
        for visit in WAVES_WE_ARE_USING:
            df = copy.deepcopy(self.original_dataframes['allclinical%s' % visit])
            for condition in medical_conditions:
                if visit == '00':
                    has_disease_idxs = df['v%s%s' % (visit, condition)] == '1: Yes'
                    self.validate_col(df['v%s%s' % (visit, condition)], ['1: Yes', '0: No', self.missing_data_val])
                    nas_at_baseline_idxs = df['v%s%s' % (visit, condition)] == self.missing_data_val
                    nas_at_baseline[condition] = set(df.loc[nas_at_baseline_idxs, 'id'])
                    print('Proportion missing data for %-10s at visit 00: %2.3f' % (condition, nas_at_baseline_idxs.mean()))
                elif visit in ['03', '06']:
                    has_disease_idxs = df['v%s%s' % (visit, condition)] == 1.0
                    self.validate_col(df['v%s%s' % (visit, condition)], [0, 1])
                    print("Proportion missing data for %-10s at visit %s: %2.3f" % (condition, visit, pd.isnull(df['v%s%s' % (visit, condition)]).mean()))
                else:
                    # unfortunately, don't appear to have data for these visits. 
                    continue
                has_disease_ids = set(df.loc[has_disease_idxs, 'id'])
                has_disease[condition] = has_disease[condition].union(has_disease_ids)
                    
            df_for_visit = pd.DataFrame({'id':all_ids, 'visit':CLINICAL_WAVES_TO_FOLLOWUP[visit]})
            for condition in medical_conditions:
                has_disease_idxs = df_for_visit['id'].map(lambda x:x in has_disease[condition])
                df_for_visit[condition] = has_disease_idxs.values * 1.
                nas_at_baseline_idxs = df_for_visit['id'].map(lambda x:x in nas_at_baseline[condition])
                df_for_visit.loc[nas_at_baseline_idxs & (~has_disease_idxs), condition] = None
            all_dfs.append(df_for_visit)
        combined_df = pd.concat(all_dfs)
        combined_df.index = range(len(combined_df))
        print(combined_df.groupby('visit').mean())
        self.processed_dataframes['medical_history'] = combined_df

    def make_previous_fracture_or_fall_dataframe(self):
        """
        Fractures are cumulatively defined: currently someone is defined as a 1 if they report having a fracture prior to the timepoint. 
        Defined as missing if they are missing data at baseline and don't report having it subsequently. 
        Defined as false otherwise. 
        
        Falls occur in the last 12 months and are thus not cumulatively defined. 
        """
        print("Making fracture and fall dataframe!")
        all_ids = list(self.original_dataframes['allclinical00']['id'])
        have_fracture = {}
        nas_at_baseline = {}
        all_dfs = []
        for condition in ['fractured_bone', 'fractured_hip', 'fractured_spine']:
            have_fracture[condition] = set([])
            nas_at_baseline[condition] = set([])
            
        for visit in WAVES_WE_ARE_USING:
            # get the DF we need data from
            df = copy.deepcopy(self.original_dataframes['allclinical%s' % visit])
            
            # construct df for visit. 
            df_for_visit = pd.DataFrame({'id':all_ids})
            df_for_visit['visit'] = CLINICAL_WAVES_TO_FOLLOWUP[visit]
            
            # Do falls. This is different from fractures because it's non-cumulative. 
            fall_col = 'v%sfall' % visit
            if visit in ['00', '01']:
                self.validate_col(df[fall_col], ['1: Yes', '0: No', self.missing_data_val])
                fell_ids = set(df.loc[df[fall_col] == '1: Yes', 'id'].values)
                fall_missing_data_ids = set(df.loc[df[fall_col] == self.missing_data_val, 'id'].values)
            else:
                fell_ids = set(df.loc[df[fall_col] == 1.0, 'id'].values)
                self.validate_col(df[fall_col], [0, 1])
                fall_missing_data_ids = set(df.loc[pd.isnull(df[fall_col]), 'id'].values)
            df_for_visit['fell_in_last_12_months'] = df_for_visit['id'].map(lambda x:x in fell_ids)
            df_for_visit.loc[df_for_visit['id'].map(lambda x:x in fall_missing_data_ids), 'fell_in_last_12_months'] = None
            
            
            # Do fractures. 
            got_fracture_at_timepoint = {}
            for condition in have_fracture.keys():
                got_fracture_at_timepoint[condition] = set([])
                if condition == 'fractured_bone':

                    if visit == '00':
                        col = 'v00bonefx'
                    else:
                        col = 'v%sbonfx' % visit
                    if visit in ['01', '00']:
                        got_fracture_at_timepoint[condition] = df.loc[df[col] == '1: Yes', 'id'].values
                        self.validate_col(df[col], ['1: Yes', '0: No', self.missing_data_val])
                    else:
                        got_fracture_at_timepoint[condition] = df.loc[df[col] == 1.0, 'id'].values
                        self.validate_col(df[col], [0, 1])
                    if visit == '00':
                        nas_at_baseline[condition] = df.loc[df[col] == self.missing_data_val, 'id'].values
                elif condition == 'fractured_hip':
                    if visit == '00':
                        col = 'v00hipfx'
                        got_fracture_at_timepoint[condition] = df.loc[df[col] == '1: Yes', 'id'].values
                        nas_at_baseline[condition] = df.loc[df[col] == self.missing_data_val, 'id'].values
                        self.validate_col(df[col], ['1: Yes', '0: No', self.missing_data_val])
                    else:
                        # can't find hip fracture data at subsequent timepoints. 
                        continue
                elif condition == 'fractured_spine':
                    if visit == '00':
                        col = 'v00spnfx' 
                    else:
                        col = 'v%sbonfx6' % visit
                    if visit in ['01', '00']:
                        got_fracture_at_timepoint[condition] = df.loc[df[col] == '1: Yes', 'id'].values
                        self.validate_col(df[col], ['1: Yes', '0: No', self.missing_data_val])
                    else:
                        got_fracture_at_timepoint[condition] = df.loc[df[col] == 1.0, 'id'].values
                        self.validate_col(df[col], [0, 1])
                    if visit == '00':
                        nas_at_baseline[condition] = df.loc[df[col] == self.missing_data_val, 'id'].values
                else:
                    raise Exception("not a valid disease")
                    
            for condition in have_fracture.keys():
                have_fracture[condition] = have_fracture[condition].union(got_fracture_at_timepoint[condition])
                df_for_visit[condition] = df_for_visit['id'].map(lambda x:x in have_fracture[condition])
                na_idxs = df_for_visit['id'].map(lambda x:x in nas_at_baseline[condition] )
                df_for_visit.loc[na_idxs & (~df_for_visit[condition]), condition] = None
                
            
            all_dfs.append(df_for_visit)
        combined_df = pd.concat(all_dfs)
        combined_df.index = range(len(combined_df))
        print("Average values by visit")
        print(combined_df[[a for a in combined_df.columns if a != 'id']].groupby('visit').mean())
        print("NAs by visit")
        print(combined_df[[a for a in combined_df.columns if a != 'id']].groupby('visit').agg(lambda x:np.mean(pd.isnull(x))))
        self.processed_dataframes['fractures_and_falls'] = combined_df  

    def make_400m_walk_dataframe(self):
        """
        Stats about how quickly they can walk. Only have data for three timepoints.
        """
        walk_cols = ['400mtr', '400excl', '400mcmp', '400mtim']
        walk_df = self.concatenate_dataframes_from_multiple_timepoints(dataset_substring='allclinical', 
                                                            columns_to_subset_on=['id'] + walk_cols, 
                                                            visit_numbers_to_skip=['01', '05', '07', '08', '09','10', '11'])
        ids = sorted(list(set(walk_df['id'])))
        
        print(Counter(walk_df['400excl'].dropna()))
        print(Counter(walk_df['400mcmp'].dropna()))
        walk_df['400excl'] = walk_df['400excl'].map(lambda x:str(x) not in ['0.0', '0: Not excluded'])
        walk_df['400mcmp'] = walk_df['400mcmp'].map(lambda x:str(x) in ['1.0', '1: Completed test without stopping'])

        print("After processing")
        print(Counter(walk_df['400excl'].dropna()))
        print(Counter(walk_df['400mcmp'].dropna()))
        for c in walk_df.columns:
            assert (walk_df[c].astype(str) == self.missing_data_val).sum() == 0
        print(walk_df.head())


        # Add timepoints for '01' and '05' for consistency with other processing (just fill out other columns with None). 
        for timepoint in ['01', '05']:
            timepoint_df = pd.DataFrame({'id':ids, 'visit':CLINICAL_WAVES_TO_FOLLOWUP[timepoint]})
            for col in walk_cols:
                timepoint_df[col] = None
            timepoint_df = timepoint_df[walk_df.columns]
            walk_df = pd.concat([walk_df, timepoint_df])
        self.processed_dataframes['400m_walk'] = walk_df



    def make_redundant_knee_xray_variable_dataframe(self):
        """
        A couple extra variables that Sendhil noticed at baseline and wanted to pull just in case. 
        """
        cols = ['P01SV%sKOST', 'P01SV%sKJSL', 'P01SV%sKJSM']
        new_col_names = ['knee_osteophytes', 
                         'knee_lateral_joint_space_narrowing', 
                         'knee_medial_joint_space_narrowing']
                         
        cols = [col.lower() for col in cols]
        left_cols = [col % 'l' for col in cols]
        right_cols = [col % 'r' for col in cols]
        
        left_df = self.original_dataframes['allclinical00'][['id'] + left_cols].copy()
        right_df = self.original_dataframes['allclinical00'][['id'] + right_cols].copy()
        
        left_df.columns = ['id'] + new_col_names
        right_df.columns = ['id'] + new_col_names
                             
        left_df['side'] = 'left'
        right_df['side'] = 'right'
        
        redundant_knee_xray_clinical_features = pd.concat([left_df, right_df])
        redundant_knee_xray_clinical_features.index = range(len(redundant_knee_xray_clinical_features))
        for c in new_col_names:
            if c == 'id':
                continue
            print(c)
            assert pd.isnull(redundant_knee_xray_clinical_features[c]).sum() == 0
            redundant_knee_xray_clinical_features.loc[
                redundant_knee_xray_clinical_features[c] == self.missing_data_val, 
                c] = None
            print(redundant_knee_xray_clinical_features[c].value_counts())
            print("Missing data fraction: %2.3f" % pd.isnull(redundant_knee_xray_clinical_features[c]).mean())
        
        self.processed_dataframes['redundant_knee_xray_clinical_features'] = redundant_knee_xray_clinical_features
    def make_dominant_leg_dataframe(self):
        """
        Checked. 
        Don’t use timepoint info (ie, we define this using allclinical00 only) because lots of missing data at 
        subsequent timepoints and seems like there are causality problems.  
        """
        print("\n\n***Making dominant leg dataframe")
        right_leg_df = copy.deepcopy(self.original_dataframes['allclinical00'][['id', 'v00kikball']])
        right_leg_df.columns = ['id', 'dominant_leg']
        missing_data_idxs = (right_leg_df['dominant_leg'] == self.missing_data_val).values
        left_leg_df = copy.deepcopy(right_leg_df)
        
        right_leg_df['dominant_leg'] = right_leg_df['dominant_leg'].map(lambda x:'right' in x.lower())
        left_leg_df['dominant_leg'] = left_leg_df['dominant_leg'].map(lambda x:'left' in x.lower())
        
        left_leg_df.loc[missing_data_idxs, 'dominant_leg'] = None
        right_leg_df.loc[missing_data_idxs, 'dominant_leg'] = None
        
        left_leg_df['side'] = 'left'
        right_leg_df['side'] = 'right'
        
        combined_df = pd.concat([left_leg_df, right_leg_df])
        combined_df.index = range(len(combined_df))
        
        print(combined_df[['side', 'dominant_leg']].groupby('side').agg(['mean', 'size']))
        print("Missing data: %2.3f" % pd.isnull(combined_df['dominant_leg']).mean())
        
        self.processed_dataframes['dominant_leg'] = combined_df

    def make_bmi_dataframe(self):
        """
        Computes current and max BMI as categorical variables. Only uses baseline numbers. 
        Checked. 
        """
        print("\n\nComputing current amd max BMI.")

        current_weight_col = 'p01weight'
        max_weight_col = 'v00wtmaxkg'
        current_height_col = 'p01height'
        desired_cols = ['id'] + [current_weight_col, max_weight_col, current_height_col]
        bmi_df = copy.deepcopy(self.original_dataframes['allclinical00'][desired_cols])
        
        bmi_df['current_bmi'] = bmi_df[current_weight_col] / ((bmi_df[current_height_col] / 1000.) ** 2)
        bmi_df['max_bmi'] = bmi_df[max_weight_col] / ((bmi_df[current_height_col] / 1000.) ** 2)
        bmi_df = bmi_df[['id', 'current_bmi', 'max_bmi']]    
        def map_bmi_to_david_cats(x):
            if x < 18.5:
                return '<18.5'
            elif x < 25:
                return '18.5-25'
            elif x < 30:
                return '25-30'
            elif x < 35:
                return '30-35'
            elif x >= 35:
                return '>=35'
            else:
                return None
            
        bmi_not_nan = (~pd.isnull(bmi_df['current_bmi'])) & (~pd.isnull(bmi_df['max_bmi']))
        bmi_max_smaller_than_current = bmi_not_nan & (bmi_df['current_bmi'] > bmi_df['max_bmi'])
        print('Warning: proportion %2.3f of rows have current BMI > max BMI. Setting max to current.' % 
              bmi_max_smaller_than_current.mean()) # this is likely caused by fact that max BMI is self-reported, while current BMI I assume is weighed at the site. 
        bmi_df.loc[bmi_max_smaller_than_current, 'max_bmi'] = bmi_df.loc[bmi_max_smaller_than_current, 'current_bmi'].values
        assert (bmi_not_nan & (bmi_df['current_bmi'] > bmi_df['max_bmi'])).sum() == 0

        print(bmi_df[['current_bmi', 'max_bmi']].describe())
        bmi_df['current_bmi'] = bmi_df['current_bmi'].map(map_bmi_to_david_cats)
        bmi_df['max_bmi'] = bmi_df['max_bmi'].map(map_bmi_to_david_cats)
        
        print('Counts of values for current BMI are', Counter(bmi_df['current_bmi']))
        print('Counts of values for max BMI are', Counter(bmi_df['max_bmi']))
        self.processed_dataframes['bmi'] = bmi_df

    def make_previous_knee_replacement_dataframe(self):
        print("\n\nComputing previous knee replacements/arthroplasties")
        # "ever have replacement where all or part of joint was replaced"
        self.processed_dataframes['knee_replacement'] = self.make_previous_injury_or_surgery_dataframe(
            baseline_substring='krs', 
            followup_substring='krs',
            col_name='knee_replacement', 
            set_missing_baseline_to_0=True, 
            waves_to_skip='06'
            )
        df_to_concat = self.processed_dataframes['knee_replacement'].loc[self.processed_dataframes['knee_replacement']['visit'] == '36 month follow-up'].copy()
        df_to_concat['visit'] = '48 month follow-up'
        self.processed_dataframes['knee_replacement'] = pd.concat([self.processed_dataframes['knee_replacement'], df_to_concat])
        self.processed_dataframes['knee_replacement'].index = range(len(self.processed_dataframes['knee_replacement']))

    def make_previous_injury_dataframe(self):
        print("\n\nComputing previous injuries to knees!")
        self.processed_dataframes['knee_injury'] = self.make_previous_injury_or_surgery_dataframe(
            baseline_substring='inj', 
            followup_substring='inj',
            col_name='knee_injury')

    def make_previous_surgery_dataframe(self):
        print("\n\nComputing previous surgeries to knees!")
        self.processed_dataframes['knee_surgery'] = self.make_previous_injury_or_surgery_dataframe(
            baseline_substring='ksurg', 
            followup_substring='ksrg',
            col_name='knee_surgery')

    def make_age_dataframe(self):
        print("\n\n***Creating combined age dataframe")
        combined_df = []
        for visit in WAVES_WE_ARE_USING:
            age_df = copy.deepcopy(self.original_dataframes['allclinical%s' % visit][['id', 'v%sage' % visit]])
            age_df.columns = ['id', 'age_at_visit']
            age_df['visit'] = CLINICAL_WAVES_TO_FOLLOWUP[visit]
            combined_df.append(age_df)
        
        def convert_age_to_categorical_variable(age):
            assert not (age < 45)
            assert not (age > 85)
            if age < 50 and age >= 45:
                return '45-49'
            if age < 55:
                return '50-54'
            if age < 60:
                return '55-59'
            if age < 65:
                return '60-64'
            if age < 70:
                return '65-69'
            if age < 75:
                return '70-74'
            if age < 80:
                return '75-79'
            if age < 85:
                return '80-84'
            assert np.isnan(age)
            return None
            
        combined_df = pd.concat(combined_df)
        combined_df['age_at_visit'] = combined_df['age_at_visit'].map(convert_age_to_categorical_variable)
        print(Counter(combined_df['age_at_visit']))
        self.processed_dataframes['age_at_visit'] = combined_df

    def make_other_pain_dataframe(self, type_of_pain):
        """
        Helper method to make the combined pain dataframe. 
        Returns things as strings. 
        """
        assert type_of_pain in ['hip', 'back', 
                                'foot', 'ankle', 'shoulder', 'elbow', 'wrist', 'hand']
        
        combined_df = []
        for visit in WAVES_WE_ARE_USING:        
            # first have to identify cols of interest. 
            if type_of_pain == 'hip':
                if visit == '00':
                    cols_of_interest = ['p01hp%s12cv' % side for side in ['l', 'r']]
                else:
                    cols_of_interest = ['v%shp%s12cv' % (visit, side) for side in ['l', 'r']]
                col_names_to_use = ['id', 
                                    'left_hip_pain_more_than_half_of_days', 
                                    'right_hip_pain_more_than_half_of_days']
            elif type_of_pain == 'back':
                if visit == '00':
                    cols_of_interest = ['p01bp30oft']
                else: 
                    cols_of_interest = ['v%sbp30oft' % visit]
                col_names_to_use = ['id', 'how_often_bothered_by_back_pain']
            elif type_of_pain in ['foot', 'ankle', 'shoulder', 'elbow', 'wrist', 'hand']:
                pain_abbrv = type_of_pain[0]
                if visit == '00':
                    cols_of_interest = ['p01ojpn%s%s' % (side, pain_abbrv) for side in ['l', 'r']]
                else:
                    cols_of_interest = ['v%sojpn%s%s' % (visit, side, pain_abbrv) for side in ['l', 'r']]
                col_names_to_use = ['id', 
                                    'left_%s_pain_more_than_half_of_days' % type_of_pain, 
                                    'right_%s_pain_more_than_half_of_days' % type_of_pain]
            else:
                raise Exception("Your pain is invalid :(")
           
            # select columns. 
            pain_df = copy.deepcopy(self.original_dataframes['allclinical%s' % visit][['id'] + cols_of_interest])
            
            # do mapping. 
            if type_of_pain == 'hip':
                if visit == '00' or visit == '01':
                    for col in cols_of_interest:
                        self.validate_col(pain_df[col], ['1: Yes', '0: No', self.missing_data_val])
                else:
                    for col in cols_of_interest:
                        self.validate_col(pain_df[col], [0, 1])
                        pain_df[col] = pain_df[col].replace({np.nan:self.missing_data_val, 
                                                             1:'1: Yes',
                                                             0:'0: No'}).astype(str)
                for col in cols_of_interest:
                    self.validate_col(pain_df[col], [self.missing_data_val, '1: Yes', '0: No'])

            elif type_of_pain == 'back':
                if visit == '00' or visit == '01':
                    for col in cols_of_interest:
                        self.validate_col(pain_df[col], ['1: Some of the time', '0: Rarely', 
                            '2: Most of the time', '3: All of the time', self.missing_data_val])
                else:
                    for col in cols_of_interest:
                        self.validate_col(pain_df[col], [0, 1, 2, 3])
                        pain_df[col] = pain_df[col].replace({1:'1: Some of the time', 
                                                             0:'0: Rarely', 
                                                             2:'2: Most of the time', 
                                                             3:'3: All of the time', 
                                                            np.nan:self.missing_data_val}).astype(str)
                for col in cols_of_interest:
                    self.validate_col(pain_df[col], ['0: Rarely', '1: Some of the time', '2: Most of the time', '3: All of the time', self.missing_data_val])
                
                
            elif type_of_pain in ['foot', 'ankle', 'shoulder', 'elbow', 'wrist', 'hand']:
                if visit == '00' or visit == '01':
                    for col in cols_of_interest:
                        self.validate_col(pain_df[col], ['1: Yes', '0: No', self.missing_data_val])
                else:
                    for col in cols_of_interest:
                        self.validate_col(pain_df[col], [0, 1])
                        pain_df[col] = pain_df[col].replace({None:self.missing_data_val, 
                                                            1:'1: Yes'}).astype(str)
                for col in cols_of_interest:        
                    self.validate_col(pain_df[col], [self.missing_data_val, '1: Yes'])
                
            pain_df.columns = col_names_to_use
            pain_df['visit'] = CLINICAL_WAVES_TO_FOLLOWUP[visit]
            combined_df.append(pain_df)

        combined_df = pd.concat(combined_df)
        combined_df.index = range(len(combined_df))

        # Set missing values to None for consistency with the rest of data processing. 
        for col in combined_df.columns:
            if col == 'visit' or col == 'id':
                continue
            assert type(combined_df[col].iloc[0]) is str
            assert pd.isnull(pain_df[col]).sum() == 0
            print("Setting values of %s in column %s to None" % (self.missing_data_val, col))
            combined_df.loc[combined_df[col] == self.missing_data_val, col] = None

        return combined_df

    def make_nonstandard_interventions_dataframe(self):
        """
        Make dataframe of 0-1 indicators whether someone has had other interventions for pain 
        which are not standard in medical practice. 
        """
        print("Processing interventions data")
        interventions = ["V00ACUTCV", "V00ACUSCV", "V00CHELCV", "V00CHIRCV", 
                       "V00FOLKCV", "V00HOMECV", "V00MASSCV", "V00DIETCV", 
                       "V00VITMCV", "V00RUBCV", "V00CAPSNCV", "V00BRACCV", 
                       "V00YOGACV", "V00HERBCV", "V00RELACV", "V00SPIRCV", 
                       "V00OTHCAMC", "V00OTHCAM"]
        cols = ['id'] + [a.lower() for a in interventions]
        df = self.original_dataframes['allclinical00'][cols].copy()
        
        for c in df.columns:
            if c != 'id':
                self.validate_col(df[c], ['0: No', '1: Yes', self.missing_data_val])
                
                nan_idxs = df[c].map(lambda x:x in self.missing_data_val).values
                intervention_idxs = df[c] == '1: Yes'
                df[c] = 0.
                df.loc[intervention_idxs, c] = 1.
                df.loc[nan_idxs, c] = None
        print("Missing data")
        print(df.agg(lambda x:np.mean(pd.isnull(x))))
        print("Fraction with other interventions")
        print(df.mean())

        self.processed_dataframes['nonstandard_interventions'] = df
        
    def make_medications_dataframe(self):
        """
        Make dataframe of 0-1 indicators whether someone is taking medication. 
        """
        print("Processing medications data")
        medications = ["V00RXACTM", "V00RXANALG", "V00RXASPRN", "V00RXBISPH", 
                       "V00RXCHOND", "V00RXCLCTN", "V00RXCLCXB", "V00RXCOX2", 
                       "V00RXFLUOR", "V00RXGLCSM", "V00RXIHYAL", "V00RXISTRD", 
                       "V00RXMSM", "V00RXNARC", "V00RXNSAID", "V00RXNTRAT", 
                       "V00RXOSTRD", "V00RXOTHAN", "V00RXRALOX", "V00RXRFCXB", 
                       "V00RXSALIC", "V00RXSAME", "V00RXTPRTD", "V00RXVIT_D", "V00RXVLCXB"]
        medications = [a.replace('V00', '').lower() for a in medications]
        med_df = self.concatenate_dataframes_from_multiple_timepoints(dataset_substring='allclinical', 
                                                                columns_to_subset_on=['id'] + medications, 
                                                                visit_numbers_to_skip=['07', '08', '09', '10', '11'])
        for c in med_df.columns:
            if c != 'id' and c != 'visit':
                self.validate_col(med_df[c].map(lambda x:str(x)), ['1.0', '0.0', 
                                                                   '0: Not used in last 30 days', 
                                                                   '1: Used in last 30 days', 
                                                                   self.missing_data_val, 
                                                                   'nan'])
                nan_idxs = med_df[c].map(lambda x:str(x) in [self.missing_data_val, 'nan']).values
                took_idxs = med_df[c].map(lambda x:str(x) in ['1: Used in last 30 days', '1.0']).values
                med_df[c] = 0.
                med_df.loc[took_idxs, c] = 1.
                med_df.loc[nan_idxs, c] = None
        print("Missing data")
        print(med_df.groupby('visit').agg(lambda x:np.mean(pd.isnull(x))))
        print("Fraction taking medication")
        print(med_df.groupby('visit').mean())
            
        self.processed_dataframes['medications'] = med_df
        
    def make_pain_dataframe_for_all_other_types_of_pain(self):
        print("\n\n\n***Creating dataframe for all other types of pain")
        for i, other_type_of_pain in enumerate(['hip', 'back', 
                                'foot', 'ankle', 'shoulder', 'elbow', 'wrist', 'hand']):
            if i == 0:
                combined_pain_df = self.make_other_pain_dataframe(other_type_of_pain)
                original_len = len(combined_pain_df)
            else:
                combined_pain_df = pd.merge(combined_pain_df, 
                                            self.make_other_pain_dataframe(other_type_of_pain), 
                                            how='inner', 
                                            on=['id', 'visit'])
                assert len(combined_pain_df) == original_len
                assert len(combined_pain_df[['id', 'visit']].drop_duplicates() == original_len)

        print("Missing data by timepoint")
        print(combined_pain_df.groupby('visit').agg(lambda x:np.mean(pd.isnull(x))))
                
        self.processed_dataframes['other_pain'] = combined_pain_df

    def validate_col(self, col, expected_values):
        if not (col.dropna().map(lambda x:x not in expected_values).sum() == 0):
            print("Error: unexpected value in column. Expected values:")
            print(expected_values)
            print("Actual values")
            print(sorted(list(set(col.dropna()))))
            assert False

    def make_previous_injury_or_surgery_dataframe(self, baseline_substring, followup_substring, col_name, set_missing_baseline_to_0=False, waves_to_skip=None):
        """
        While the code in this method refers to "injury", we actually use it to define both injuries + surgeries. 
        baseline_substring identifies the column used in allclinical00
        followup_substring identifies the column in subsequent clinical dataframes
        col_name is the name we want to give the column. 

        Set someone to True if they report an injury at any previous timepoint. 
        Set them to NA if they don't report an injury and are missing data for the first timepoint 
        Set them to False otherwise. 
        (some followup people are missing data, so we might have a few false negatives who didn't report an injury, but it should be small). 
        Checked. 
        """
        
        ids_who_report_injury_at_any_timepoint = {'left':set([]), 'right':set([])}
        ids_with_nas_at_first_timepoint = {'left':set([]), 'right':set([])}
        all_dfs = []
        if waves_to_skip is None:
            waves_to_skip = []

        for visit in WAVES_WE_ARE_USING:
            if visit in waves_to_skip:
                continue
            if visit == '00':
                left_col = 'p01%sl' % baseline_substring
                right_col = 'p01%sr' % baseline_substring
            else:
                left_col = 'v%s%sl12' % (visit, followup_substring)
                right_col = 'v%s%sr12' % (visit, followup_substring)
            df_to_use = copy.deepcopy(self.original_dataframes['allclinical%s' % visit][['id', left_col, right_col]])
            df_to_use.columns = ['id', 'left_side', 'right_side']
            assert len(set(df_to_use['id'])) == len(df_to_use)
            df_to_use['visit'] = CLINICAL_WAVES_TO_FOLLOWUP[visit]
            if visit == '00':
                all_ids = set(df_to_use['id'])
            else:
                assert set(df_to_use['id']) == all_ids

            dfs_by_knee = {}
            for side in ['left', 'right']:
                dfs_by_knee[side] = copy.deepcopy(df_to_use[['id', 'visit', '%s_side' % side]])
                dfs_by_knee[side].columns = ['id', 'visit', col_name]
                dfs_by_knee[side]['side'] = side
                
                # map to bools.
                if visit == '00' or visit == '01':
                    self.validate_col(dfs_by_knee[side][col_name], ['1: Yes', '0: No', self.missing_data_val])
                    knee_injury_at_this_timepoint = set(dfs_by_knee[side]['id'].loc[
                        dfs_by_knee[side][col_name] == '1: Yes'])
                    
                else:
                    knee_injury_at_this_timepoint = set(dfs_by_knee[side]['id'].loc[
                        dfs_by_knee[side][col_name] == 1])
                    self.validate_col(dfs_by_knee[side][col_name], [0, 1])
                if visit == '00':
                    na_ids = set(dfs_by_knee[side]['id'].loc[dfs_by_knee[side][col_name] == self.missing_data_val])
                    if set_missing_baseline_to_0:
                        ids_with_nas_at_first_timepoint[side] = set([])
                        print("Warning: setting %i missing datapoints for baseline to 0" % len(na_ids))
                    else:
                        ids_with_nas_at_first_timepoint[side] = na_ids
                
                # update list of people who report an injury. 
                ids_who_report_injury_at_any_timepoint[side] = ids_who_report_injury_at_any_timepoint[side].union(knee_injury_at_this_timepoint)
                
                # set people to True if report injury at any timepoint. 
                dfs_by_knee[side][col_name] = dfs_by_knee[side]['id'].map(lambda x:x in ids_who_report_injury_at_any_timepoint[side])
                # set people to NA if False and missing data at initial timepoint 
                dfs_by_knee[side].loc[dfs_by_knee[side]['id'].map(lambda x:(x in ids_with_nas_at_first_timepoint[side]) & 
                                                                  (x not in ids_who_report_injury_at_any_timepoint[side])),
                                      col_name] = None
                
                
                dfs_by_knee[side].index = range(len(dfs_by_knee[side]))
                all_dfs.append(dfs_by_knee[side].copy())
                print("At timepoint %s, rate for %s leg: %i=1, %i=0, %i are missing" % (CLINICAL_WAVES_TO_FOLLOWUP[visit],
                                                                                      side, 
                                                                                      (dfs_by_knee[side][col_name] == 1).sum(), 
                                                                                      (dfs_by_knee[side][col_name] == 0).sum(),
                                                                                      pd.isnull(dfs_by_knee[side][col_name]).sum()))
            
        combined_df = pd.concat(all_dfs)
        combined_df.index = range(len(combined_df))
        assert len(combined_df[['id', 'visit', 'side']].drop_duplicates()) == len(combined_df)
        print("Average values")
        print(combined_df[[col_name, 'visit', 'side']].groupby(['side', 'visit']).agg(['mean', 'size']))
        print("Missing data")
        print(combined_df[[col_name, 'visit', 'side']].groupby(['side', 'visit']).agg(lambda x:np.mean(pd.isnull(x))))

        return combined_df

    def make_other_koos_subscores_dataframe(self):
        """
        Make dataframe of other Koos pain subscores. 
        Each row is one visit for one side for one id. 
        Other koos_symptoms_score is knee specific. Everything else is the same for both. 
        """
        print("Making other koos subscores dataframe")
        
        base_cols = {'koosfsr':'koos_function_score', 
        'koosqol':'koos_quality_of_life_score', 
        'koosym':'koos_symptoms_score'}

        left_cols = copy.deepcopy(base_cols)
        right_cols = copy.deepcopy(base_cols)

        left_cols['koosyml'] = left_cols['koosym']
        right_cols['koosymr'] = right_cols['koosym']
        del left_cols['koosym']
        del right_cols['koosym']

        dfs_to_concat = []
        for side in ['left', 'right']:
            if side == 'left':
                cols_to_use = left_cols
            else:
                cols_to_use = right_cols

            old_col_names = sorted(cols_to_use.keys())
            new_col_names = [cols_to_use[a] for a in old_col_names]
            all_koos_scores_for_side = self.concatenate_dataframes_from_multiple_timepoints(dataset_substring='allclinical', 
                columns_to_subset_on=['id'] + old_col_names)
            assert list(all_koos_scores_for_side.columns) == ['id'] + old_col_names + ['visit']
            all_koos_scores_for_side.columns = ['id'] + new_col_names + ['visit']
            all_koos_scores_for_side['side'] = side
            dfs_to_concat.append(all_koos_scores_for_side)
        final_df = pd.concat(dfs_to_concat)
        final_df.index = range(len(final_df))
        
        def map_blank_strings_to_none(x):
            # small helper method: empty strings become none, otherwise cast to float. 
            if len(str(x).strip()) == 0:
                return None
            return float(x)

        for c in sorted(base_cols.values()):
            final_df[c] = final_df[c].map(map_blank_strings_to_none)

        print('means by column and visit')
        print(final_df[['visit', 'side'] + list(base_cols.values())].groupby(['visit', 'side']).mean())
        for c in base_cols.values():
            print('missing data fraction for %s is %2.3f' % (c, pd.isnull(final_df[c]).mean()))
        for c1 in base_cols.values():
            for c2 in base_cols.values():
                if c1 > c2:
                    good_idxs = ~(pd.isnull(final_df[c1]) | pd.isnull(final_df[c2]))
                    print("Correlation between %s and %s: %2.3f" % (
                        c1, 
                        c2, 
                        pearsonr(final_df.loc[good_idxs, c1], final_df.loc[good_idxs, c2])[0]))


        self.processed_dataframes['other_koos_subscores'] = final_df



    def make_knee_pain_dataframe(self):
        """
        Extract Koos and Womac knee pain scores 
        Koos scores are transformed to a 0–100 scale, with zero representing extreme knee problems and 100 representing no knee problems as is common in orthopaedic assessment scales and generic measures. 
        http://www.koos.nu/koosfaq.html
        Womac scores: Higher scores on the WOMAC indicate worse pain, stiffness, and functional limitations. 
        https://www.physio-pedia.com/WOMAC_Osteoarthritis_Index
        Checked. 
        """
        all_left_knee_pain_scores = self.concatenate_dataframes_from_multiple_timepoints(dataset_substring='allclinical', 
            columns_to_subset_on=['id', 'kooskpl', 'womkpl'])
        assert list(all_left_knee_pain_scores.columns) == ['id', 'kooskpl', 'womkpl', 'visit']
        all_left_knee_pain_scores.columns = ['id', 'koos_pain_subscore', 'womac_pain_subscore', 'visit']
        all_left_knee_pain_scores['side'] = 'left'

        all_right_knee_pain_scores = self.concatenate_dataframes_from_multiple_timepoints(dataset_substring='allclinical', 
            columns_to_subset_on=['id', 'kooskpr', 'womkpr'])
        assert list(all_right_knee_pain_scores.columns) == ['id', 'kooskpr', 'womkpr', 'visit']
        all_right_knee_pain_scores.columns = ['id', 'koos_pain_subscore', 'womac_pain_subscore', 'visit']
        all_right_knee_pain_scores['side'] = 'right'
        all_knee_pain_scores = pd.concat([all_left_knee_pain_scores, all_right_knee_pain_scores])
        for k in ['koos_pain_subscore', 'womac_pain_subscore']:
            all_knee_pain_scores[k] = all_knee_pain_scores[k].map(lambda x:float(x) if len(str(x).strip()) > 0 else None)
        print("Number of knee pain scores: %i" % len(all_knee_pain_scores))
        print("Womac scores not missing data: %i; koos not missing data: %i" % (len(all_knee_pain_scores['koos_pain_subscore'].dropna()), 
            len(all_knee_pain_scores['womac_pain_subscore'].dropna())))
        for timepoint in sorted(list(set(all_knee_pain_scores['visit']))):
            df_for_timepoint = copy.deepcopy(all_knee_pain_scores.loc[all_knee_pain_scores['visit'] == timepoint])
            print("Timepoint %s, fraction womac scores complete: %2.3f; koos scores complete %2.3f" % (timepoint, 
                1 - pd.isnull(df_for_timepoint['womac_pain_subscore']).mean(), 
                1 - pd.isnull(df_for_timepoint['koos_pain_subscore']).mean()))

        all_knee_pain_scores = all_knee_pain_scores.dropna()
        print("Number of knee pain scores not missing data: %i" % len(all_knee_pain_scores))
        print("Correlation between KOOS and WOMAC scores is %2.3f" % pearsonr(all_knee_pain_scores['koos_pain_subscore'], 
            all_knee_pain_scores['womac_pain_subscore'])[0])
        self.processed_dataframes['all_knee_pain_scores'] = all_knee_pain_scores

    def make_per_person_controls_dataframe(self):
        """
        Extract covariates which are person-specific (eg, income). 
        Checked.
        """
        print("\n***Making dataset of per-person controls.")
        missing_data_val = self.missing_data_val

        # Income, education, marital status. Each row is one person. 
        all_clinical00_d = copy.deepcopy(self.original_dataframes['allclinical00'][['id', 'v00income', 'v00edcv', 'v00maritst']])
        for c in ['v00income', 'v00edcv']:
            val_counts = Counter(all_clinical00_d[c])
            for val in sorted(val_counts.keys()):
                print('%-50s %2.1f%%' % (val, 100.*val_counts[val] / len(all_clinical00_d)))
            missing_data_idxs = all_clinical00_d[c] == missing_data_val
            if c == 'v00edcv':
                col_name = 'binarized_education_graduated_college'
                all_clinical00_d[col_name] = (all_clinical00_d[c] >= '3: College graduate') * 1.
            elif c == 'v00income':
                col_name = 'binarized_income_at_least_50k'
                all_clinical00_d[col_name] = (all_clinical00_d[c] >= '4: $50K to < $100K') * 1.
            all_clinical00_d.loc[missing_data_idxs, col_name] = None
            all_clinical00_d.loc[missing_data_idxs, c] = None
            print("Binarizing into column %s with mean %2.3f and %2.3f missing data" % (col_name, 
                all_clinical00_d[col_name].mean(), 
                pd.isnull(all_clinical00_d[col_name]).mean()))

        all_clinical00_d.loc[all_clinical00_d['v00maritst'] == missing_data_val, 'v00maritst'] = None


        # Gender + race + site. 
        enrollees_path = os.path.join(BASE_NON_IMAGE_DATA_DIR, 'General_ASCII')
        self.load_all_text_files_in_directory(enrollees_path, datasets_to_skip=[])
        race_sex_site = copy.deepcopy(self.original_dataframes['enrollees'][['id', 'p02hisp', 'p02race', 'p02sex', 'v00site']])


        for c in race_sex_site.columns:
            if c == 'id':
                continue
            missing_data_idxs = race_sex_site[c] == missing_data_val
            race_sex_site.loc[missing_data_idxs, c] = None

        race_sex_site['race_black'] = (race_sex_site['p02race'] == '2: Black or African American') * 1.
        race_sex_site.loc[pd.isnull(race_sex_site['p02race']), 'race_black'] = None
        print("Proportion of missing data for race (this will be dropped): %2.3f; proportion black: %2.3f" % 
            (pd.isnull(race_sex_site['race_black']).mean(), 
             race_sex_site['race_black'].mean()))

        assert len(race_sex_site) == TOTAL_PEOPLE
        assert len(all_clinical00_d) == TOTAL_PEOPLE
        assert len(set(race_sex_site['id'])) == len(race_sex_site)
        assert len(set(all_clinical00_d['id'])) == len(all_clinical00_d)
        assert sorted(list(race_sex_site['id'])) == sorted(list(all_clinical00_d['id']))

        d = pd.merge(race_sex_site, all_clinical00_d, on='id', how='inner')
        assert len(d) == TOTAL_PEOPLE
        assert len(set(d['id'])) == len(d)

        print("All columns in per-person dataframe")
        for c in d.columns:
            if c == 'id':
                continue
            print("\nSummary stats for column %s" % c)
            print("Missing data: %2.1f%%" % (pd.isnull(d[c]).mean() * 100))
            val_counts = Counter(d[c].dropna())
            for val in sorted(val_counts.keys()):
                print('%-50s %2.1f%%' % (val, 100.*val_counts[val] / len(d[c].dropna())))

        self.processed_dataframes['per_person_covariates'] = d

    def make_processed_mri_data(self):
        """
        Process MRI data, roughly following David's methodology. 
        Essentially, to get each processed column, we take the max of a bunch of raw columns, then threshold that max. (So the processed variable is binary.)

        Various data peculiarities: 
        1. Appears that most patients are actually lacking the MOAKS data. Asked David about this, seems fine. 
        2. "For pooling MOAKS readings from different reading projects please read the documentation for the kMRI_SQ_MOAKS_BICLxx datasets very carefully." Took another look, seems fine. 
        3. what about special values of 0.5 or -0.5? These values occur quite rarely. Verified that they don't change our results. 
        4. Asymmetry in which knees are rated (some projects only rated one knee...) -- this seems unavoidable. 
        """
        print("Processing MRI data as David did!")

        concatenated_mri = self.concatenate_dataframes_from_multiple_timepoints('kmri_sq_moaks_bicl')
        
        processed_cols = {'car11plusm':{'cols':['mcmfmc', 'mcmfmp', 'mcmtma', 'mcmtmc', 'mcmtmp'], 
                                     'thresh':1.1}, 
                          'car11plusl':{'cols':['mcmflc', 'mcmflp', 'mcmtla','mcmtlc','mcmtlp'], 
                                        'thresh':1.1},
                          'car11pluspf':{'cols':['mcmfma', 'mcmfla','mcmpm', 'mcmpl'], 
                                         'thresh':1.1}, 
                          'bml2plusm':{'cols':['mbmsfmc', 'mbmsfmp', 'mbmstma', 'mbmstmc', 'mbmstmp'], 
                                       'thresh':2.0}, 
                          'bml2plusl':{'cols':['mbmsflc', 'mbmsflp', 'mbmstla', 'mbmstlc', 'mbmstlp'], 
                                       'thresh':2.0},
                          'bml2pluspf':{'cols':['mbmsfma','mbmsfla','mbmspm','mbmspl'], 
                                       'thresh':2.0},
                          'mentearm':{'cols':['mmtma', 'mmtmb', 'mmtmp'], 
                                      'thresh':2.0},
                          'mentearl':{'cols':['mmtla', 'mmtlb', 'mmtlp'], 
                                      'thresh':2.0},
                          'menextm':{'cols':['mmxmm', 'mmxma'], 
                                      'thresh':2.0},
                          'menextl':{'cols':['mmxll', 'mmxla'], 
                                      'thresh':2.0}
                         }
        side_mappings = {'2: Left':'left', '1: Right':'right', 1:'right', 2:'left'}
        concatenated_mri['side'] = concatenated_mri['side'].map(lambda x:side_mappings[x])
        print('Side variable for MRI', Counter(concatenated_mri['side']))
        self.validate_col(concatenated_mri['side'], ['right', 'left'])

        # we have multiple readings for each knee. Sort by number of missing values, keep the duplicate with fewest missing values. 
        all_necessary_cols = []
        for col in processed_cols:
             all_necessary_cols += processed_cols[col]['cols']


        def map_mri_to_float(x):
            if x == self.missing_data_val:
                return None
            if str(x) == 'nan':
                return None
            if type(x) is float:
                return x
            return float(x.split(':')[0])

        if self.filter_out_special_values_in_mri_data:
            # just a sanity check which we do not use by default in main processing. 
            # Basically, I was uncertain of whether we wanted to simply threshold all values, as is done in a previous analysis
            # even though values of 0.5 and -0.5 indicate change over time. So I wrote code so we could filter these rows out
            # and verify that it didn't change results. 
            special_values = np.array([False for a in range(len(concatenated_mri))])
            for col in all_necessary_cols:
                values_in_col = concatenated_mri[col].map(lambda x:map_mri_to_float(x))
                special_values_in_col = concatenated_mri[col].map(lambda x:map_mri_to_float(x) in [0.5, -0.5, -1]).values
                
                print(Counter(values_in_col[~np.isnan(values_in_col)]))
                special_values = special_values | special_values_in_col
                print("Fraction of special values in %s: %2.3f (n=%i); cumulative fraction %2.3f" % (col, 
                    special_values_in_col.mean(), 
                    special_values_in_col.sum(), 
                    special_values.mean()))

            print("Fraction of special values in MRI data: %2.3f." % special_values.mean())
            concatenated_mri = concatenated_mri.loc[~special_values]
            concatenated_mri.index = range(len(concatenated_mri))

        missing_data = ((concatenated_mri[all_necessary_cols] == self.missing_data_val).sum(axis=1) +
                        pd.isnull(concatenated_mri[all_necessary_cols]).sum(axis=1))
        concatenated_mri['num_missing_fields'] = missing_data.values
        concatenated_mri = concatenated_mri.sort_values(by='num_missing_fields')
        print("Prior to dropping duplicate readings for same side, person, and timepoint, %i rows" % 
              len(concatenated_mri))
        concatenated_mri = concatenated_mri.drop_duplicates(subset=['id', 'side', 'visit'], keep='first')
        print("After dropping duplicate readings for same side, person, and timepoint, %i rows" % 
              len(concatenated_mri))
                                      
        

        original_cols_already_used = set([]) # sanity check: make sure we're not accidentally using raw columns in two different processed columns. 
        for processed_col_name in processed_cols:
            original_cols = processed_cols[processed_col_name]['cols']
            processed_col_vals = []
            for c in original_cols:
                assert c not in original_cols_already_used
                original_cols_already_used.add(c)
                concatenated_mri[c] = concatenated_mri[c].map(map_mri_to_float).astype(float)
                print(concatenated_mri[c].value_counts(dropna=False)/len(concatenated_mri))
            for i in range(len(concatenated_mri)):
                vals_to_max = concatenated_mri.iloc[i][original_cols].values
                not_null = ~pd.isnull(vals_to_max)
                if not_null.sum() > 0:
                    max_val = np.max(vals_to_max[not_null])
                    processed_col_vals.append(max_val >= processed_cols[processed_col_name]['thresh'])
                else:
                    processed_col_vals.append(None)
                
            concatenated_mri[processed_col_name] = processed_col_vals
            concatenated_mri[processed_col_name] = concatenated_mri[processed_col_name].astype('float')
        concatenated_mri = concatenated_mri[['id', 'side', 'visit', 'readprj'] + sorted(list(processed_cols.keys()))]
        print("Average values")
        print(concatenated_mri.groupby(['visit', 'side']).mean())
        print("missing data")
        print(concatenated_mri.groupby(['visit', 'side']).agg(lambda x:np.mean(pd.isnull(x))))
        concatenated_mri.index = range(len(concatenated_mri))
        self.processed_dataframes['david_mri_data'] = concatenated_mri

    def load_semiquantitative_xray_data(self):
        """
        Load in all the semiquantitative x-ray ratings. 
        Checked.
        """
        print("\n***Loading all semi-quantitative x-ray data.")
        dataset_substring = 'kxr_sq_bu'
        datasets_to_skip = [a.replace('.txt', '') for a in os.listdir(self.semiquantitative_xray_dir) if dataset_substring not in a and '.txt' in a]
        self.load_all_text_files_in_directory(self.semiquantitative_xray_dir, datasets_to_skip=datasets_to_skip)
        
        for dataset_name in sorted(self.original_dataframes):
            if dataset_substring in dataset_name:
                # From the OAI notes: 
                # Please note that although some participants are coded READPRJ=42, they are in fact participants in Project 37. Users should recode these participants from READPRJ=42 to READPRJ=37.
                miscoded_project_idxs = self.original_dataframes[dataset_name]['readprj'] == 42
                self.original_dataframes[dataset_name].loc[miscoded_project_idxs, 'readprj'] = 37
                self.original_dataframes[dataset_name]['side'] = self.original_dataframes[dataset_name]['side'].map(lambda x:self.side_mappings[x])
        combined_data = self.concatenate_dataframes_from_multiple_timepoints(dataset_substring)
        
        # drop a very small number of rows with weird barcodes. 
        print("prior to dropping semiquantitative data missing a barcode, %i rows" % len(combined_data))
        combined_data = combined_data.dropna(subset=['barcdbu'])
        combined_data = combined_data.loc[combined_data['barcdbu'] != 'T']
        combined_data['barcdbu'] = combined_data['barcdbu'].map(lambda x:'0'+str(int(x)))
        assert (combined_data['barcdbu'].map(len) == 12).all()
        assert (combined_data['barcdbu'].map(lambda x:x[:4] == '0166')).all()
        print("After dropping, %i rows" % len(combined_data))
        
        # From the notes: "the variables uniquely identifying a record in these datasets are ID, SIDE, and READPRJ"
        assert len(combined_data.drop_duplicates(subset=['id', 'side', 'visit', 'readprj'])) == len(combined_data)

        # but we don't actually want multiple readings (from different projects) for a given knee and timepoint;
        # it appears that each timepoint is pretty exclusively read by a single project, so we just use the 
        # predominant project at each timepoint. 
        filtered_for_project = []

        def timepoint_a_less_than_or_equal_to_b(a, b):
            valid_timepoints = ['00 month follow-up: Baseline', 
            '12 month follow-up', 
            '24 month follow-up', 
            '36 month follow-up', 
            '48 month follow-up', 
            '72 month follow-up',
            '96 month follow-up']
            assert (a in valid_timepoints) and (b in valid_timepoints)
            a_idx = valid_timepoints.index(a)
            b_idx = valid_timepoints.index(b)
            return a_idx <= b_idx

        for timepoint in sorted(list(set(combined_data['visit']))):
            if timepoint == '72 month follow-up':
                print("Skipping %s because not sure how to fill in missing data; there is lots of missing data even for people with KLG >= 2" % timepoint)
                continue
            timepoint_idxs = combined_data['visit'] == timepoint
            df_for_timepoint = combined_data.loc[timepoint_idxs]
            readings_for_15 = set(df_for_timepoint.loc[df_for_timepoint['readprj'] == 15, 'id'])
            readings_for_37 = set(df_for_timepoint.loc[df_for_timepoint['readprj'] == 37, 'id'])
            # This illustrates that it is safe to take one project or the other for each timepoint. 
            # Many people do have readings for both projects. But I think it is cleaner to be consistent in the project used for timepoints 0 - 48m. 
            # Project 37 is done only on  a weird sample of people, so attempting to merge somehow would lead to an inconsistent definition of image variables
            # on a non-random subset of the population. However, note that this means that our definitions of some image variables don't quite line up 
            # with the definitions of image variables in allclinical00: eg, their knee lateral joint space narrowing appears to be some kind of max of the two projects. This is fine, because we don't use those variables for analysis.
            print("%s: %i people had readings for 15 but not 37; %i had readings for 37 but not 15; %i had readings for both" % (
                timepoint, 
                len(readings_for_15 - readings_for_37), 
                len(readings_for_37 - readings_for_15), 
                len(readings_for_37.intersection(readings_for_15))))
            if timepoint in ['00 month follow-up: Baseline', 
            '12 month follow-up', 
            '24 month follow-up', 
            '36 month follow-up', 
            '48 month follow-up']:
                df_for_timepoint = df_for_timepoint.loc[df_for_timepoint['readprj'] == 15]
            elif timepoint in ['72 month follow-up', '96 month follow-up']:
                df_for_timepoint = df_for_timepoint.loc[df_for_timepoint['readprj'] == 37]
            else:
                raise Exception("invalid timepoint")

            print("Filling in missing values for %s as 0" % timepoint)
            # Fill in missing data.
            # KLG and OARSI JSN grades are available for all participants in this project at all available time points. Scores for other IRFs (osteophytes, subchondral sclerosis, cysts and attrition) are available only in participants with definite radiographic OA at least one knee at one (or more) of the time points.
            # Following this, we say you should have data if you have had KLG >= 2 at this timepoint or earlier.
            participants_who_have_had_definite_radiographic_oa = set(combined_data['id'].loc[
                    combined_data['visit'].map(lambda x:timepoint_a_less_than_or_equal_to_b(x, timepoint)) & 
                    (combined_data['xrkl'] >= 2)])

            people_who_are_missing_data_but_should_have_data = None
            for c in df_for_timepoint.columns:
                missing_data_idxs = pd.isnull(df_for_timepoint[c]).values
                people_who_should_have_data = df_for_timepoint['id'].map(lambda x:x in participants_who_have_had_definite_radiographic_oa).values
                if c[0] == 'x':
                    if c not in ['xrjsl', 'xrjsm', 'xrkl']:
                        print("Filling in missing data for %i values in column %s" % (missing_data_idxs.sum(), c))
                        # fill in data as 0 for those we don't expect to have it. 
                        df_for_timepoint.loc[missing_data_idxs & (~people_who_should_have_data), c] = 0

                        # keep track of those who are missing data but shouldn't be, so we can drop them later.
                        if people_who_are_missing_data_but_should_have_data is None:
                            people_who_are_missing_data_but_should_have_data = (missing_data_idxs & people_who_should_have_data)
                        else:
                            people_who_are_missing_data_but_should_have_data = (missing_data_idxs & people_who_should_have_data) | people_who_are_missing_data_but_should_have_data

                    else:
                        print("NOT filling in missing data for %i values in column %s" % (missing_data_idxs.sum(), c))
                    print("Fraction of missing data %2.3f; non-missing values:" % pd.isnull(df_for_timepoint[c]).mean(), Counter(df_for_timepoint[c].dropna()))
                if c in ['id', 'side', 'readprj', 'version']:
                    assert missing_data_idxs.sum() == 0
            print("Prior to dropping missing data in x-ray image scoring for %s, %i points" % (timepoint, len(df_for_timepoint)))
            df_for_timepoint = df_for_timepoint.loc[~people_who_are_missing_data_but_should_have_data]
            # In total, this line drops about 1% of values for timepoints baseline - 48 m, which isn't the end of the world. 
            print("After dropping people who should be scored for other attributes but aren't, %i timepoints (%2.1f%% of values are bad)" % (len(df_for_timepoint), people_who_are_missing_data_but_should_have_data.mean() * 100))
            df_for_timepoint = df_for_timepoint.dropna(subset=['xrkl'])
            print("After dropping missing data in xrkl for %s, %i points" % (timepoint, len(df_for_timepoint)))
            
            filtered_for_project.append(df_for_timepoint)
        combined_data = pd.concat(filtered_for_project)
        combined_data.index = range(len(combined_data))
        assert len(combined_data.drop_duplicates(subset=['id', 'side', 'visit'])) == len(combined_data)
        assert len(combined_data.drop_duplicates(subset=['barcdbu', 'side'])) == len(combined_data)


        

        for timepoint in sorted(list(set(combined_data['visit']))):
            print(timepoint,
                Counter(combined_data.loc[(combined_data['visit'] == timepoint) & (combined_data['side'] == 'left'), 
                                        'readprj']))
        self.processed_dataframes[dataset_substring] = combined_data
        self.clinical_xray_semiquantitative_cols = [a for a in self.processed_dataframes['kxr_sq_bu'] if a[0] == 'x']

    def load_xray_metadata(self):
        # Load raw x-ray metadata. Checked. Not being used at present. 
        print("\n***Loading all x-ray metadata.")
        self.load_all_text_files_in_directory(self.xray_metadata_dir, datasets_to_skip=[])
    
    def load_semiquantitative_mri_data(self):
        # Load raw semiquantitative MRI data. Checked. Not being used at present. 
        print("\n***Loading all semi-quantitative MRI data.")
        self.load_all_text_files_in_directory(self.semiquantitative_mri_dir, datasets_to_skip=[])

    def load_mri_metadata(self):
        # Load raw MRI metadata. Checked. Not being used at present. 
        print("\n***Loading all MRI metadata.")
        self.load_all_text_files_in_directory(self.mri_metadata_dir, datasets_to_skip=[])

    def map_str_column_to_float(self, dataset_name, column):
        raise Exception("If you actually use this you need to check it.")
        col_dtype = str(self.original_dataframes[dataset_name][column].dtype)
        if 'float' in col_dtype:
            raise Exception("%s in %s is not a string column, it is a float column" % (column, dataset_name))
        #assert self.original_dataframes[dataset_name][column].dtype is str
        #self.original_dataframes[dataset_name][column] = self.original_dataframes[dataset_name][column].astype(str)
        nan_idxs = pd.isnull(self.original_dataframes[dataset_name][column])
        nan_value = self.missing_data_val
        #self.original_dataframes[dataset_name].loc[nan_idxs, column] = nan_value
        nan_idxs = pd.isnull(self.original_dataframes[dataset_name][column])
        assert nan_idxs.sum() == 0

        unique_vals = sorted(list(set(self.original_dataframes[dataset_name][column])))
        codebook = {}
        for original_val in unique_vals:
            assert ': ' in original_val
            if original_val == nan_value:
                shortened_val = None
            else:
                shortened_val = float(original_val.split(':')[0])
            codebook[original_val] = shortened_val
        self.original_dataframes[dataset_name][column] = self.original_dataframes[dataset_name][column].map(lambda x:codebook[x])
        p_missing = pd.isnull(self.original_dataframes[dataset_name][column]).mean()
        print("After mapping, column %s in dataset %s has proportion %2.3f missing data" % (column, dataset_name, p_missing))
        self.col_mappings[dataset_name][column] = codebook

    def validate_ids(self):
        """
        Make sure IDs are consistent across datasets they should be consistent in. 
        """
        print("\n***Validating that IDs look kosher")
        self.all_ids = sorted(list(copy.deepcopy(self.original_dataframes['allclinical00']['id'])))
        assert len(self.all_ids) == TOTAL_PEOPLE
        assert sorted(self.all_ids) == sorted(get_all_ids())
        assert len(set(self.all_ids)) == len(self.all_ids)

        for k in self.original_dataframes:
            if (('allclinical' in k)
                or ('baseclin' in k) 
                or ('enrollees' in k) 
                or ('enrshort' in k)
                or ('outcomes99' in k) 
                or ('outshort' in k)):
                print("Validating ids in %s" % k)
                assert len(self.original_dataframes[k]) == TOTAL_PEOPLE
                ids_in_dataframe = sorted(self.original_dataframes[k]['id'])
                assert len(set(ids_in_dataframe)) == len(ids_in_dataframe)
                assert ids_in_dataframe == self.all_ids
        
