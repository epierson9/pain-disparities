from constants_and_util import *
from scipy.stats import norm, pearsonr, spearmanr
import pandas as pd
import copy
import numpy as np
import random
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ttest_ind, rankdata
import non_image_data_processing
import patsy
import os
import math
import sklearn
import json
import seaborn as sns
from scipy.stats import scoreatpercentile
import statsmodels
from sklearn.kernel_ridge import KernelRidge
import scipy
from scipy.stats import scoreatpercentile, linregress, ttest_rel
from statsmodels.iolib.summary2 import summary_col

"""
Code to perform analyses on the fitted models. We note two potentially confusing naming conventions in the analysis code. 
First, much of the code was written during preliminary analyses looking just at SES; later, we broadened the analysis to look at pain gaps by sex, race, etc. 
Hence, many of the variable names/comments contain "ses", but in general, these refer to all three binary variables we consider in the final paper (capturing education and race, not just income). 
Second, while the paper refers to "training", "development", and "validation" sets, those correspond in the code to the "train", "val", and "test" sets, respectively. 

"""

def make_simple_histogram_of_pain(y, binary_vector_to_use, positive_class_label, negative_class_label, plot_filename):
    """
    Make a simple histogram of pain versus binary class (eg, pain for black vs non-black patients).
    Checked. 
    """
    sns.set_style()
    bins = np.arange(0, 101, 10)
    plt.figure(figsize=[4, 4])
    hist_weights = np.ones((binary_vector_to_use == False).sum())/float((binary_vector_to_use == False).sum()) # https://stackoverflow.com/a/16399202/9477154
    plt.hist(y[binary_vector_to_use == False], weights=hist_weights, alpha=1, bins=bins, label=negative_class_label, orientation='horizontal')
    hist_weights = np.ones((binary_vector_to_use == True).sum())/float((binary_vector_to_use == True).sum())
    plt.hist(y[binary_vector_to_use == True], weights=hist_weights, alpha=.7, bins=bins, label=positive_class_label, orientation='horizontal')
    plt.ylim([0, 100])
    plt.yticks([0, 20, 40, 60, 80, 100], fontsize=12)
    plt.xlabel("")
    plt.legend(loc=4, fontsize=12)
    plt.xticks([])
    plt.savefig(plot_filename)

def compare_to_mri_features(datasets, y, yhat, all_ses_vars, ids, df_for_filtering_out_special_values, also_include_xray_features, use_random_forest, mri_features):
    """
    Show that yhat still outperforms a predictor which uses MRI features. 

    df_for_filtering_out_special_values: this is a dataframe for MRI features only which only has rows if there are no 0.5/-0.5 values. 
    Just a sanity check (those values are rare) because I'm not sure whether binarizing really makes sense for those values. 
    """
    datasets = copy.deepcopy(datasets)

    idxs_with_mris = {}
    dfs_to_use_in_regression = {}

    for dataset in ['train', 'val', 'test']:
        idxs_with_mris[dataset] = np.isnan(datasets[dataset].non_image_data[mri_features].values).sum(axis=1) == 0
        
        if df_for_filtering_out_special_values is not None:
            
            dfs_to_use_in_regression[dataset] = pd.merge(datasets[dataset].non_image_data,
                                         df_for_filtering_out_special_values,
                                         how='left', 
                                         on=['id', 'side', 'visit'], 
                                         validate='one_to_one')
            no_special_values = ~pd.isnull(dfs_to_use_in_regression[dataset]['no_special_values']).values
            idxs_with_mris[dataset] = (idxs_with_mris[dataset]) & (no_special_values)
        else:
            dfs_to_use_in_regression[dataset] = datasets[dataset].non_image_data.copy()
            
        


    
    if also_include_xray_features:
        mri_features_to_use = ['C(%s)' % a for a in mri_features + CLINICAL_CONTROL_COLUMNS]
    else:
        mri_features_to_use = ['C(%s)' % a for a in mri_features]
    print("\n\n\n\n********Predicting pain from MRI features; including Xray clinical features=%s; using random forest %s; filtering out special values %s" % 
          (also_include_xray_features, use_random_forest, df_for_filtering_out_special_values is not None))



    yhat_from_mri = compare_to_clinical_performance(
        train_df=dfs_to_use_in_regression['train'].loc[idxs_with_mris['train']],
        val_df=dfs_to_use_in_regression['val'].loc[idxs_with_mris['val']],
        test_df=dfs_to_use_in_regression['test'].loc[idxs_with_mris['test']],
        y_col='koos_pain_subscore', 
        features_to_use=mri_features_to_use,
        binary_prediction=False,
        use_nonlinear_model=use_random_forest,
        do_ols_sanity_check=True)

    print("Compare to yhat performance")
    yhat_performance = assess_performance(y=y[idxs_with_mris['test']], 
                                    yhat=yhat[idxs_with_mris['test']], 
                                    binary_prediction=False)
    for k in yhat_performance:
        print('%s: %2.3f' % (k, yhat_performance[k]))

    mri_ses_vars = {}
    for k in all_ses_vars:
        mri_ses_vars[k] = all_ses_vars[k][idxs_with_mris['test']]

    print(quantify_pain_gap_reduction_vs_rival(yhat=yhat[idxs_with_mris['test']], 
                                                  y=y[idxs_with_mris['test']], 
                                                  rival_severity_measure=yhat_from_mri, 
                                                  all_ses_vars=mri_ses_vars, 
                                                 ids=ids[idxs_with_mris['test']]))

def sig_star(p):
    assert p >= 0 and p <= 1
    if p < .001:
        return '***'
    elif p < .01:
        return '**'
    elif p < .05:
        return '*'
    return ''

def get_pvalue_on_binary_vector_mean_diff(yhat_vector, klg_vector, ids):
    """
    Assess whether yhat_vector and KLG_vector are assigning different fractions of people to surgery. 
    Basically does a paired t-test on the binary vector accounting for clustering. 
    Used for surgery analysis. 
    """
    assert len(yhat_vector) == len(klg_vector) == len(ids)
    check_is_array(yhat_vector)
    check_is_array(klg_vector)
    diff_df = pd.DataFrame({'diff':1.*yhat_vector - 1.*klg_vector, 'id':ids})
    clustered_diff_model = sm.OLS.from_formula('diff ~ 1', data=diff_df).fit(cov_type='cluster', cov_kwds={'groups':diff_df['id']})
    assert np.allclose(clustered_diff_model.params['Intercept'], yhat_vector.mean() - klg_vector.mean())
    return clustered_diff_model.pvalues['Intercept']

def get_ci_on_binary_vector(vector, ids):
    """
    Compute standard error on a binary vector's mean, accounting for clustering. 
    Used for surgery analysis. 
    """
    assert len(vector) == len(ids)
    check_is_array(vector)
    check_is_array(ids)
    df = pd.DataFrame({'val':1.*vector, 'id':ids})
    cluster_model = sm.OLS.from_formula('val ~ 1', data=df).fit(cov_type='cluster', cov_kwds={'groups':df['id']})
    assert np.allclose(cluster_model.params['Intercept'], vector.mean())
    return '(%2.5f, %2.5f)' % (cluster_model.conf_int().loc['Intercept', 0], cluster_model.conf_int().loc['Intercept', 1])


def do_surgery_analysis_ziad_style(yhat, y, klg, all_ses_vars, baseline_idxs, have_actually_had_surgery, df_to_use, ids):
    """
    Hopefully the final surgery analysis. Does a couple things: 

    1. Uses a criterion for allocating surgery based on the prior literature: KLG >= 3 and high pain, as defined using pain threshold from prior literature. 
        - to compare to yhat we do it two ways: discretize yhat, use discretized_yhat >= 3
        - and, to make sure we allocate the same number of surgeries, take all high pain people and just count down by yhat until we have the same number that KLG allocates. 
    2. Then examines: 
        - What fraction of people are eligible for surgery both overall and in our racial/SES groups? 
        - What fraction of people are in a lot of pain but aren't eligible for surgery both overall and in our racial/SES groups? 
        - Is painkiller use correlated with yhat among those who don't receive surgery? 
    As a robustness check, all these analyses are repeated on both baseline + overall dataset, both excluding and including those who have already had surgery. 
    """
    check_is_array(yhat)
    check_is_array(y)
    check_is_array(klg)
    check_is_array(ids)
    check_is_array(baseline_idxs)
    check_is_array(have_actually_had_surgery)

    pd.set_option('precision', 6)
    pd.set_option('display.width', 1000)
    df_to_use = df_to_use.copy()
    in_high_pain = binarize_koos(y) == True
    discretized_yhat = discretize_yhat_like_kl_grade(yhat_arr=yhat, kl_grade_arr=klg, y_col='koos_pain_subscore')
    klg_cutoff = 3
    fit_surgery_criteria_under_klg = (in_high_pain == True) & (klg >= klg_cutoff)
    fit_surgery_criteria_under_discretized_yhat = (in_high_pain == True) & (discretized_yhat >= klg_cutoff)
    
    for just_use_baseline in [True, False]:
        for exclude_those_who_have_surgery in [True, False]:
            idxs_to_use = np.ones(baseline_idxs.shape) == 1
            if just_use_baseline:
                idxs_to_use = idxs_to_use & (baseline_idxs == 1)
            if exclude_those_who_have_surgery: 
                idxs_to_use = idxs_to_use & (have_actually_had_surgery == 0)
            print("\n\n\n\n****Just use baseline: %s; exclude those who have had surgery: %s; analyzing %i knees" % 
                (just_use_baseline, exclude_those_who_have_surgery, idxs_to_use.sum()))

            n_surgeries_under_klg = int(fit_surgery_criteria_under_klg[idxs_to_use].sum())
            # Alternate yhat criterion: assign exactly same number of people surgery under yhat as under KLG. 
            # Do this by taking people with the lowest yhat values subject to being in high pain. 
            # Compute this independently for each group specified by idxs_to_use. 
            lowest_yhat_idxs = np.argsort(yhat)
            yhat_match_n_surgeries = np.array([False for a in range(len(fit_surgery_criteria_under_discretized_yhat))])
            for idx in lowest_yhat_idxs:
                if yhat_match_n_surgeries.sum() < n_surgeries_under_klg:
                    if (in_high_pain[idx] == 1) & (idxs_to_use[idx] == 1):
                        yhat_match_n_surgeries[idx] = True
            assert yhat[yhat_match_n_surgeries == True].mean() < yhat[yhat_match_n_surgeries == False].mean()
            assert np.allclose(yhat_match_n_surgeries[idxs_to_use].mean(), fit_surgery_criteria_under_klg[idxs_to_use].mean())

            fracs_eligible_for_surgery = []
            fracs_eligible_for_surgery.append({'group':'Overall', 
                'klg':fit_surgery_criteria_under_klg[idxs_to_use].mean(),
                'klg_ci':get_ci_on_binary_vector(fit_surgery_criteria_under_klg[idxs_to_use], ids[idxs_to_use]), 
                'yhat':fit_surgery_criteria_under_discretized_yhat[idxs_to_use].mean(), 
                'yhat_ci':get_ci_on_binary_vector(fit_surgery_criteria_under_discretized_yhat[idxs_to_use], ids[idxs_to_use]),
                'yhat_match_surgeries':yhat_match_n_surgeries[idxs_to_use].mean(), 
                'yhat_klg_p':get_pvalue_on_binary_vector_mean_diff(yhat_vector=fit_surgery_criteria_under_discretized_yhat[idxs_to_use], 
                                                                   klg_vector=fit_surgery_criteria_under_klg[idxs_to_use], 
                                                                   ids=ids[idxs_to_use])})
            for ses_var in all_ses_vars:
                fracs_eligible_for_surgery.append({'group':ses_var,
                    'yhat':fit_surgery_criteria_under_discretized_yhat[(all_ses_vars[ses_var] == True) & idxs_to_use].mean(),
                    'yhat_ci':get_ci_on_binary_vector(fit_surgery_criteria_under_discretized_yhat[(all_ses_vars[ses_var] == True) & idxs_to_use], 
                                                      ids[(all_ses_vars[ses_var] == True) & idxs_to_use]),
                    'klg':fit_surgery_criteria_under_klg[(all_ses_vars[ses_var] == True) & idxs_to_use].mean(), 
                    'klg_ci':get_ci_on_binary_vector(fit_surgery_criteria_under_klg[(all_ses_vars[ses_var] == True) & idxs_to_use], 
                                                      ids[(all_ses_vars[ses_var] == True) & idxs_to_use]),
                    'yhat_match_surgeries':yhat_match_n_surgeries[(all_ses_vars[ses_var] == True) & idxs_to_use].mean(), 
                    'yhat_klg_p':get_pvalue_on_binary_vector_mean_diff(yhat_vector=fit_surgery_criteria_under_discretized_yhat[(all_ses_vars[ses_var] == True) & idxs_to_use], klg_vector=fit_surgery_criteria_under_klg[(all_ses_vars[ses_var] == True) & idxs_to_use], 
                        ids=ids[(all_ses_vars[ses_var] == True) & idxs_to_use])})
            fracs_eligible_for_surgery = pd.DataFrame(fracs_eligible_for_surgery)
            fracs_eligible_for_surgery['yhat/klg'] = fracs_eligible_for_surgery['yhat'] / fracs_eligible_for_surgery['klg']
            fracs_eligible_for_surgery['yhat_match_surgeries/klg'] = fracs_eligible_for_surgery['yhat_match_surgeries'] / fracs_eligible_for_surgery['klg']
            print("Fraction eligible for surgery")
            print(fracs_eligible_for_surgery[['group', 'klg', 'klg_ci', 'yhat', 'yhat_ci', 'yhat/klg', 'yhat_klg_p']])
            assert (fracs_eligible_for_surgery['yhat/klg'] > 1).all()
            assert (fracs_eligible_for_surgery['yhat_match_surgeries/klg'] >= 1).all()
            for check in ['klg', 'yhat']:
                # check CIs. 
                assert np.allclose(
                fracs_eligible_for_surgery[check].values - fracs_eligible_for_surgery['%s_ci' % check].map(lambda x:float(x.split()[0].replace(',', '').replace('(', ''))), 
                fracs_eligible_for_surgery['%s_ci' % check].map(lambda x:float(x.split()[1].replace(',', '').replace(')', ''))) - fracs_eligible_for_surgery[check].values, 
                atol=1e-5)

            #  for each population we calculate both under the current regime and under our counterfactual surgery assignment: the rate of people who do not receive surgery and are in pain. 
            do_not_receive_surgery_and_are_in_pain = []
            print("Do not receive surgery and are in pain")
            do_not_receive_surgery_and_are_in_pain.append({'group':'Overall', 
                'klg':((fit_surgery_criteria_under_klg == 0) & in_high_pain)[idxs_to_use].mean(), 
                'klg_ci':get_ci_on_binary_vector(((fit_surgery_criteria_under_klg == 0) & in_high_pain)[idxs_to_use], ids[idxs_to_use]),
                'yhat':((fit_surgery_criteria_under_discretized_yhat == 0) & in_high_pain)[idxs_to_use].mean(), 
                'yhat_ci':get_ci_on_binary_vector(((fit_surgery_criteria_under_discretized_yhat == 0) & in_high_pain)[idxs_to_use], ids[idxs_to_use]),
                'yhat_match_surgeries':((yhat_match_n_surgeries == 0) & in_high_pain)[idxs_to_use].mean(), 
                'yhat_klg_p':get_pvalue_on_binary_vector_mean_diff(yhat_vector=((fit_surgery_criteria_under_discretized_yhat == 0) & in_high_pain)[idxs_to_use], 
                                                                   klg_vector=((fit_surgery_criteria_under_klg == 0) & in_high_pain)[idxs_to_use], 
                                                                   ids=ids[idxs_to_use])})
            for ses_var in all_ses_vars:
                do_not_receive_surgery_and_are_in_pain.append({'group':ses_var,
                'klg':((fit_surgery_criteria_under_klg == 0) & in_high_pain)[(all_ses_vars[ses_var] == True) & idxs_to_use].mean(), 
                'klg_ci':get_ci_on_binary_vector(((fit_surgery_criteria_under_klg == 0) & in_high_pain)[idxs_to_use & (all_ses_vars[ses_var] == True)], ids[idxs_to_use & (all_ses_vars[ses_var] == True)]),
                'yhat':((fit_surgery_criteria_under_discretized_yhat == 0) & in_high_pain)[(all_ses_vars[ses_var] == True) & idxs_to_use].mean(), 
                'yhat_ci':get_ci_on_binary_vector(((fit_surgery_criteria_under_discretized_yhat == 0) & in_high_pain)[idxs_to_use & (all_ses_vars[ses_var] == True)], ids[idxs_to_use & (all_ses_vars[ses_var] == True)]),
                'yhat_match_surgeries':((yhat_match_n_surgeries == 0) & in_high_pain)[(all_ses_vars[ses_var] == True) & idxs_to_use].mean(), 
                'yhat_klg_p':get_pvalue_on_binary_vector_mean_diff(yhat_vector=((fit_surgery_criteria_under_discretized_yhat == 0) & in_high_pain)[(all_ses_vars[ses_var] == True) & idxs_to_use], 
                                                                   klg_vector=((fit_surgery_criteria_under_klg == 0) & in_high_pain)[(all_ses_vars[ses_var] == True) & idxs_to_use], 
                                                                   ids=ids[(all_ses_vars[ses_var] == True) & idxs_to_use])})
            do_not_receive_surgery_and_are_in_pain = pd.DataFrame(do_not_receive_surgery_and_are_in_pain)
            do_not_receive_surgery_and_are_in_pain['yhat/klg'] = do_not_receive_surgery_and_are_in_pain['yhat'] / do_not_receive_surgery_and_are_in_pain['klg']
            do_not_receive_surgery_and_are_in_pain['yhat_match_surgeries/klg'] = do_not_receive_surgery_and_are_in_pain['yhat_match_surgeries'] / do_not_receive_surgery_and_are_in_pain['klg']
            print(do_not_receive_surgery_and_are_in_pain[['group', 'klg', 'klg_ci', 'yhat', 'yhat_ci', 'yhat/klg', 'yhat_klg_p']])
            assert (do_not_receive_surgery_and_are_in_pain['yhat/klg'] < 1).all()
            assert (do_not_receive_surgery_and_are_in_pain['yhat_match_surgeries/klg'] <= 1).all()
            for check in ['klg', 'yhat']:
                # check CIs. 
                assert np.allclose(
                do_not_receive_surgery_and_are_in_pain[check].values - do_not_receive_surgery_and_are_in_pain['%s_ci' % check].map(lambda x:float(x.split()[0].replace(',', '').replace('(', ''))), 
                do_not_receive_surgery_and_are_in_pain['%s_ci' % check].map(lambda x:float(x.split()[1].replace(',', '').replace(')', ''))) - do_not_receive_surgery_and_are_in_pain[check].values, 
                atol=1e-5)

            # show in the non-surgical population the corrrelation between opioid use and y-hat
            predict_medication_results = []
            medications = ['rxactm', 'rxanalg', 'rxasprn', 'rxnarc', 'rxnsaid', 'rxothan']

            for surgery_criterion in ['yhat', 'yhat_match_surgeries', 'klg']:
                if surgery_criterion == 'yhat':
                    non_surgical_population = (fit_surgery_criteria_under_discretized_yhat == False) & idxs_to_use
                elif surgery_criterion == 'klg':
                    non_surgical_population = (fit_surgery_criteria_under_klg == False) & idxs_to_use
                elif surgery_criterion == 'yhat_match_surgeries':
                    non_surgical_population = (yhat_match_n_surgeries == False) & idxs_to_use

                for m in medications: 
                    df_for_regression = pd.DataFrame({'medication':df_to_use.loc[non_surgical_population, m].values, 
                        'yhat':yhat[non_surgical_population], 
                        'id':df_to_use.loc[non_surgical_population, 'id'].values})
                    df_for_regression = df_for_regression.dropna()
                    predict_on_medication_in_nonsurgical_population = sm.Logit.from_formula('medication ~ yhat', data=df_for_regression).fit(cov_type='cluster', cov_kwds={'groups':df_for_regression['id']})
                    predict_medication_results.append({'medication':MEDICATION_CODES[('v00' + m).upper()], 
                        'beta_yhat':predict_on_medication_in_nonsurgical_population.params['yhat'], 
                        'DV mean':df_for_regression['medication'].mean(),
                        'p_yhat':predict_on_medication_in_nonsurgical_population.pvalues['yhat'], 
                        'surgery_criterion':surgery_criterion, 
                        'n':predict_on_medication_in_nonsurgical_population.nobs})



            predict_medication_results = pd.DataFrame(predict_medication_results)[['surgery_criterion', 'medication', 'beta_yhat', 'p_yhat', 'DV mean', 'n']]
            predict_medication_results['sig'] = predict_medication_results['p_yhat'].map(sig_star)
            assert (predict_medication_results['sig'].map(lambda x:'*' in x) & (predict_medication_results['beta_yhat'] > 0)).sum() == 0 # make sure no significant associations in the wrong direction. 
            print(predict_medication_results.sort_values(by='medication'))

def extract_all_ses_vars(df):
    """
    Small helper method: return a dictionary of variables coded in the proper direction. 
    """
    for k in ['binarized_income_at_least_50k', 'binarized_education_graduated_college', 'race_black']:
        assert df[k].map(lambda x:x in [0, 1]).all()
        assert df[k].map(lambda x:x in [True, False]).all()
    income_at_least_50k = df['binarized_income_at_least_50k'].values == 1
    graduated_college = df['binarized_education_graduated_college'].values == 1
    race_black = df['race_black'].values == 1
    all_ses_vars = {'did_not_graduate_college':~(graduated_college == 1), 
                        'income_less_than_50k':~(income_at_least_50k == 1), 
                       'race_black':race_black == 1}
    return all_ses_vars, income_at_least_50k, graduated_college, race_black


def assess_treatment_gaps_controlling_for_klg(klg, all_ses_vars, baseline_idxs, df_to_use):
    """
    Regression: 

    treatment ~ SES + controls, where controls \in [KLG, none]. 
    """
    check_is_array(klg)
    check_is_array(baseline_idxs)
    pd.set_option('max_rows', 500)
    get_OR_and_CI = lambda m:'%2.2f (%2.2f, %2.2f)' % (np.exp(m.params['ses']), np.exp(m.conf_int().loc['ses', 0]), np.exp(m.conf_int().loc['ses', 1]))
    treatment_gaps_regression_results = []
    for treatment in ['knee_surgery', 'rxnarc', 'rxactm', 'rxanalg', 'rxasprn', 'rxnsaid', 'rxothan']:
        for just_use_baseline in [True, False]:
            idxs_to_use = np.ones(baseline_idxs.shape) == 1
            if just_use_baseline:
                idxs_to_use = idxs_to_use & (baseline_idxs == 1)
            for control_for_klg in [True, False]:
                for ses_var_name in all_ses_vars:
                    regression_df = pd.DataFrame({'ses':all_ses_vars[ses_var_name][idxs_to_use] * 1., 
                    'klg':klg[idxs_to_use], 
                    'treatment':df_to_use.loc[idxs_to_use, treatment].values, 
                    'id':df_to_use.loc[idxs_to_use, 'id'].values, 
                    'visit':df_to_use.loc[idxs_to_use, 'visit'].values}).dropna()

                    if control_for_klg:
                        formula = 'treatment ~ ses + C(klg)'
                    else:
                        formula = 'treatment ~ ses'

                    regression_model = sm.Logit.from_formula(formula, data=regression_df).fit(cov_type='cluster', cov_kwds={'groups':regression_df['id'].values})
                    treatment_gaps_regression_results.append({'n_obs':regression_model.nobs, 
                        'just_baseline':just_use_baseline, 
                        'klg_control':control_for_klg,
                        'treatment':MEDICATION_CODES[('v00' + treatment).upper()] if treatment != 'knee_surgery' else 'knee_surgery',
                        'ses_var':ses_var_name, 
                        'ses_OR':get_OR_and_CI(regression_model),
                        'DV mean':'%2.3f' % regression_df['treatment'].mean() ,
                        'sig':sig_star(regression_model.pvalues['ses'])})
    treatment_gaps_regression_results = pd.DataFrame(treatment_gaps_regression_results)[['just_baseline', 
            'klg_control',
            'treatment', 
            'ses_var', 
            'ses_OR', 
            'sig', 
            'DV mean', 
            'n_obs']]
    print(treatment_gaps_regression_results)

def study_effect_of_surgery(df_to_use, surgery_col_to_analyze):
    """
    The goal here was to show that people are in less pain after surgery, which is true for arthroplasty (not arthroscopy). 
    """
    pd.set_option('display.width', 500)
    df_to_use = df_to_use.copy()
    df_to_use['high_pain'] = binarize_koos(df_to_use['koos_pain_subscore'])
    print("Prior to dropping people with missing %s data, %i rows" % (surgery_col_to_analyze, len(df_to_use)))
    df_to_use = df_to_use.dropna(subset=[surgery_col_to_analyze])
    print("After dropping people with missing %s data, %i rows" % (surgery_col_to_analyze, len(df_to_use)))
    df_to_use['id_plus_side'] = df_to_use['id'].astype(str) + '*' + df_to_use['side'].astype(str)



    medications = ['rxactm', 'rxanalg', 'rxasprn', 'rxnarc', 'rxnsaid', 'rxothan']
    outcomes = ['koos_pain_subscore', 'high_pain'] + medications + ['all_pain_medications_combined']
    df_to_use['all_pain_medications_combined'] = False
    for k in medications:
        df_to_use['all_pain_medications_combined'] = (df_to_use['all_pain_medications_combined'] | (df_to_use[k] == 1))
    grouped_d = df_to_use.groupby('id_plus_side')
    outcomes_to_changes = {}
    for outcome in outcomes:
        outcomes_to_changes[outcome] = []
    outcomes_to_changes['pre_surgery_klg'] = []
    outcomes_to_changes['pre_surgery_discretized_yhat'] = []

    for group_id, small_d in grouped_d:
        small_d = small_d.copy().sort_values(by='visit')
        if small_d[surgery_col_to_analyze].sum() == 0:
            continue
        if small_d[surgery_col_to_analyze].iloc[0] == 1:
            continue
        small_d.index = range(len(small_d))
        before_surgery = small_d[surgery_col_to_analyze] == 0
        after_surgery = small_d[surgery_col_to_analyze] == 1
        assert before_surgery.sum() > 0
        assert after_surgery.sum() > 0
        outcomes_to_changes['pre_surgery_klg'].append(small_d.loc[before_surgery, 'xrkl'].dropna().mean())
        if 'discretized_yhat' in small_d.columns:
            outcomes_to_changes['pre_surgery_discretized_yhat'].append(small_d.loc[before_surgery, 'discretized_yhat'].dropna().mean())
        else:
            outcomes_to_changes['pre_surgery_discretized_yhat'].append(np.nan)
        for outcome in outcomes:
            if pd.isnull(small_d[outcome]).mean() > 0:
                continue
            before_surgery_mean = small_d.loc[before_surgery, outcome].mean()
            after_surgery_mean = small_d.loc[after_surgery, outcome].mean()
            outcomes_to_changes[outcome].append({'before_surgery':before_surgery_mean, 'after_surgery':after_surgery_mean})
        assert sorted(small_d[surgery_col_to_analyze].values) == list(small_d[surgery_col_to_analyze].values)
    
    outcomes_to_changes['pre_surgery_klg'] = np.array(outcomes_to_changes['pre_surgery_klg'])
    outcomes_to_changes['pre_surgery_discretized_yhat'] = np.array(outcomes_to_changes['pre_surgery_discretized_yhat'])
    if np.isnan(outcomes_to_changes['pre_surgery_discretized_yhat']).mean() < 1:
        assert (np.isnan(outcomes_to_changes['pre_surgery_klg']) == np.isnan(outcomes_to_changes['pre_surgery_discretized_yhat'])).all()
        for k in ['pre_surgery_klg', 'pre_surgery_discretized_yhat']:
            not_nan = ~np.isnan(outcomes_to_changes[k])
            print('Mean of %s prior to surgery in people who had surgery: %2.5f; median %2.5f' % (k, 
                outcomes_to_changes[k][not_nan].mean(), 
                np.median(outcomes_to_changes[k][not_nan])))

    results_df = []
    for outcome in outcomes:
        pre_surgery_values = np.array([a['before_surgery'] for a in outcomes_to_changes[outcome]])
        post_surgery_values = np.array([a['after_surgery'] for a in outcomes_to_changes[outcome]])

        t, p = ttest_rel(pre_surgery_values, post_surgery_values)
        pretty_outcome_name = MEDICATION_CODES['V00' + outcome.upper()] if 'V00' + outcome.upper() in MEDICATION_CODES else outcome
        results_df.append({'outcome':pretty_outcome_name, 
            'n':len(post_surgery_values), 
            'pre_surgery_larger':(pre_surgery_values > post_surgery_values).sum(), 
            'post_surgery_larger':(pre_surgery_values < post_surgery_values).sum(), 
            'no_change':(pre_surgery_values == post_surgery_values).sum(), 
            'pre_surgery_mean':pre_surgery_values.mean(), 
            'post_surgery_mean':post_surgery_values.mean(), 
            'p':p})

    if np.isnan(outcomes_to_changes['pre_surgery_discretized_yhat']).mean() < 1:
        # check whether yhat predicts surgical outcomes -- but this turns out to be pretty impossible due to small size o fhte test set. 
        for outcome in outcomes:
            print(outcome)
            pre_surgery_values = np.array([a['before_surgery'] for a in outcomes_to_changes[outcome]])
            post_surgery_values = np.array([a['after_surgery'] for a in outcomes_to_changes[outcome]])

            for k in ['pre_surgery_klg', 'pre_surgery_discretized_yhat']:
                not_nan = ~np.isnan(outcomes_to_changes[k])
                r, p = pearsonr(outcomes_to_changes[k][not_nan], post_surgery_values[not_nan] - pre_surgery_values[not_nan])
                print("Correlation between %s and post-surgery change: %2.3f, p=%2.3e; n=%i" % (k, r, p, not_nan.sum()))
    return pd.DataFrame(results_df)[['outcome', 'n', 'pre_surgery_larger', 'no_change', 'post_surgery_larger', 'pre_surgery_mean', 'post_surgery_mean', 'p']]


def analyze_performance_on_held_out_sites(all_site_generalization_results, yhat, y, yhat_from_klg, site_vector, all_ses_vars, ids, recalibrate_to_new_set):
    """
    Check how we do on held out data (ie, train just on 4 sites, validate+test on the fifth). 

    all_site_generalization_results is a dataframe with performance results. 
    If recalibrate_to_new_set is True, fits a model ax + b on the held out site (improves RMSE but leaves r^2 unchanged). 
    This seems like something you probably want to avoid doing. 
    """
    pd.set_option("display.width", 500)

    # Just a little bit of paranoia here to avoid accidental modification due to pass-by-reference. 
    yhat = yhat.copy()
    y = y.copy()
    yhat_from_klg = yhat_from_klg.copy()
    site_vector = site_vector.copy()
    all_ses_vars = copy.deepcopy(all_ses_vars)
    ids = ids.copy()

    # one way to combine performance across all 5 settings: stich together yhat/KLG for each held out site. But this is kind of weird. 
    stitched_together_held_out_yhat = np.nan * np.ones(yhat.shape)
    stitched_together_klg = np.nan * np.ones(yhat.shape)
    results_to_plot = []
    all_site_names = sorted(list(set(all_site_generalization_results['site_to_remove'])))
    concatenated_pain_gap_reductions = []
    for site in all_site_names:
        model_idxs = all_site_generalization_results['site_to_remove'] == site
        site_idxs = site_vector == site
        site_ensemble_results, ensemble_site_yhat = try_ensembling(all_site_generalization_results.loc[model_idxs], 
                                                                   5,
                                                                   binary_prediction=False)
        yhat_from_klg_to_use = yhat_from_klg.copy()
        
        ensemble_site_yhat[~site_idxs] = np.nan
        if recalibrate_to_new_set:
            # recalibrate yhat
            df_for_recalibration = pd.DataFrame({'yhat':ensemble_site_yhat[site_idxs], 'y':y[site_idxs]})
            recalibration_model = sm.OLS.from_formula('y ~ yhat', data=df_for_recalibration).fit()
            ensemble_site_yhat[site_idxs] = recalibration_model.predict(df_for_recalibration)
            
            # recalibrate KLG
            df_for_recalibration = pd.DataFrame({'yhat':yhat_from_klg_to_use[site_idxs], 'y':y[site_idxs]})
            recalibration_model = sm.OLS.from_formula('y ~ yhat', data=df_for_recalibration).fit()
            yhat_from_klg_to_use[site_idxs] = recalibration_model.predict(df_for_recalibration)
        
        stitched_together_held_out_yhat[site_idxs] = ensemble_site_yhat[site_idxs]
        stitched_together_klg[site_idxs] = yhat_from_klg_to_use[site_idxs]
        
        # KLG
        klg_results_for_site = assess_performance(
            yhat=yhat_from_klg_to_use[site_idxs],
            y=y[site_idxs],
            binary_prediction=False)
        klg_results_for_site['predictor'] = 'klg'
        
        # held out yhat
        held_out_yhat_results_for_site = assess_performance(
            yhat=ensemble_site_yhat[site_idxs],
            y=y[site_idxs],
            binary_prediction=False)
        held_out_yhat_results_for_site['predictor'] = 'held_out_yhat'
        
        # original performance results, restricted to site. 
        yhat_results_for_site = assess_performance(
            yhat=yhat[site_idxs],
            y=y[site_idxs],
            binary_prediction=False)
        yhat_results_for_site['predictor'] = 'yhat'
        
        results_for_site_compared = pd.DataFrame([yhat_results_for_site, held_out_yhat_results_for_site, klg_results_for_site])
        results_for_site_compared['n'] = site_idxs.sum()
        results_for_site_compared['n_models'] = model_idxs.sum()
        results_for_site_compared['site'] = site
        
        results_to_plot.append(results_for_site_compared)
        print(results_for_site_compared[['predictor', 'r^2', 'negative_rmse', 'spearman_r^2', 'n', 'n_models']])
        
        ses_vars_to_use = copy.deepcopy(all_ses_vars)
        for var_name in ses_vars_to_use:
            ses_vars_to_use[var_name] = all_ses_vars[var_name][site_idxs]
        print("Pain gap analysis (important point is that Rival red. vs nothing < red. vs nothing)")
        pain_gap_reduction = quantify_pain_gap_reduction_vs_rival(yhat=ensemble_site_yhat[site_idxs], 
                                                  y=y[site_idxs], 
                                                  rival_severity_measure=yhat_from_klg_to_use[site_idxs], 
                                                  all_ses_vars=ses_vars_to_use, 
                                                 ids=ids[site_idxs])
        concatenated_pain_gap_reductions.append(pain_gap_reduction)
        assert (pain_gap_reduction['No controls gap'] < 0).all()
        assert (pain_gap_reduction['Rival red. vs nothing'] < pain_gap_reduction['red. vs nothing']).all()
        print(pain_gap_reduction[['SES var', 
                                  'Rival red. vs nothing', 
                                  'red. vs nothing',
                                  'yhat/rival red. ratio', 
                                  'n_people',
                                  'n_obs']])
    results_to_plot = pd.concat(results_to_plot)
    print("\n\nUnweighted mean predictive performance across all 5 sites")
    print(results_to_plot.groupby('predictor').mean())

    print("Unweighted mean pain gap reduction performance across all 5 sites")
    concatenated_pain_gap_reductions = pd.concat(concatenated_pain_gap_reductions)
    print(concatenated_pain_gap_reductions.groupby('SES var').mean()[['Rival red. vs nothing',  'red. vs nothing']])

    plt.figure(figsize=[12, 4])
    for subplot_idx, col_to_plot in enumerate(['r^2', 'spearman_r^2', 'negative_rmse']):
        plt.subplot(1, 3, subplot_idx + 1)
        for predictor in ['held_out_yhat', 'klg']:
            predictor_idxs = results_to_plot['predictor'] == predictor
            plt.scatter(
                x=range(predictor_idxs.sum()),
                y=results_to_plot.loc[predictor_idxs, col_to_plot].values, 
                label=predictor)
        plt.xticks(range(predictor_idxs.sum()), results_to_plot.loc[predictor_idxs, 'site'].values)
        plt.title(col_to_plot)
        plt.legend()
    plt.show()     
                        
        
    print("Stitched together predictor across all sites! Not really using this at present")
    print("yhat")
    assert np.isnan(stitched_together_held_out_yhat).sum() == 0
    print(assess_performance(yhat=yhat, 
                                y=y,
                                binary_prediction=False))
    print("stitched together held out yhat")
    print(assess_performance(yhat=stitched_together_held_out_yhat, 
                                y=y,
                                binary_prediction=False))
    print("KLG")
    print(assess_performance(yhat=stitched_together_klg, 
                                y=y,
                                binary_prediction=False))
    print(quantify_pain_gap_reduction_vs_rival(yhat=stitched_together_held_out_yhat,
                                                  y=y, 
                                                  rival_severity_measure=stitched_together_klg, 
                                                  all_ses_vars=all_ses_vars, 
                                                 ids=ids)[['SES var', 
                                                          'Rival red. vs nothing', 
                                                          'red. vs nothing',
                                                          'yhat/rival red. ratio', 
                                                          'n_people',
                                                          'n_obs']])



def analyze_effect_of_diversity(all_diversity_results, all_ses_vars, y, yhat_from_klg, ids, n_bootstraps):
    """
    Look at the effect of training on all non-minority patients, as opposed to including some minority patients. 
    Checked. 
    """
    for ses_group in sorted(list(set(all_diversity_results['ses_col']))):
        print('\n\n\n\n%s' % ses_group)
        metrics_we_want = []
        if ses_group == 'race_black':
            minority_idxs = all_ses_vars['race_black']
        elif ses_group == 'binarized_education_graduated_college':
            minority_idxs = all_ses_vars['did_not_graduate_college']
        elif ses_group == 'binarized_income_at_least_50k':
            minority_idxs = all_ses_vars['income_less_than_50k']
        else:
            raise Exception("Invalid variable.")
        assert minority_idxs.mean() < .5
        vals_to_test_yhat = {}
        for val in sorted(list(set(all_diversity_results['majority_group_seed']))):
            # Note: all_diversity_results['majority_group_seed'] is None iff we exclude the minority group. 
            diversity_idxs = ((all_diversity_results['majority_group_seed'] == val) & 
                              (all_diversity_results['ses_col'] == ses_group))
            assert diversity_idxs.sum() >= 5
            ensemble_diversity_results, ensemble_test_diversity_yhat = try_ensembling(all_diversity_results.loc[diversity_idxs], 5, binary_prediction=False)

            vals_to_test_yhat[val] = ensemble_test_diversity_yhat
            # Predictive performance on full dataset. 
            ensemble_diversity_results = ensemble_diversity_results.loc[ensemble_diversity_results['model'] == 'ensemble']
            assert len(ensemble_diversity_results) == 1
            results_for_seed = {'majority_group_seed':val,
                                'r^2':ensemble_diversity_results['r^2'].iloc[-1], 
                                'spearman_r^2':ensemble_diversity_results['spearman_r^2'].iloc[-1],
                                'negative_rmse':ensemble_diversity_results['negative_rmse'].iloc[-1], 
                                'n_models':diversity_idxs.sum()}

            # predictive performance just on minority/majority. 
            just_minority_results = assess_performance(yhat=ensemble_test_diversity_yhat[minority_idxs], 
                                                         y=y[minority_idxs], 
                                                         binary_prediction=False)
            non_minority_results = assess_performance(yhat=ensemble_test_diversity_yhat[~minority_idxs], 
                                                         y=y[~minority_idxs], 
                                                         binary_prediction=False)
            for k in ['r^2', 'negative_rmse']:
                results_for_seed['Just minority %s' % k] = just_minority_results[k]
            for k in ['r^2', 'negative_rmse']:
                results_for_seed['Just non-minority %s' % k] = non_minority_results[k]

            # pain gap reduction. 
            diversity_pain_gap_reduction = quantify_pain_gap_reduction_vs_rival(yhat=ensemble_test_diversity_yhat, 
                                                      y=y, 
                                                      rival_severity_measure=yhat_from_klg, 
                                                      all_ses_vars=all_ses_vars, 
                                                     ids=ids)[['SES var', 'yhat/rival red. ratio']]
            for ses_var in all_ses_vars:
                results_for_seed['%s_pain gap reduction ratio' % ses_var] = diversity_pain_gap_reduction.loc[
                    diversity_pain_gap_reduction['SES var'] == ses_var, 'yhat/rival red. ratio'].iloc[0]
            metrics_we_want.append(results_for_seed)

        metrics_we_want = pd.DataFrame(metrics_we_want)  
        print(metrics_we_want)

        # CIs
        for val in ['0.0', '1.0', '2.0', '3.0', '4.0']:
            print("Comparing predictive performance for diverse dataset with seed %s to non-diverse dataset (note that 'KLG' here is the non-diverse dataset)" % val)
            bootstrap_CIs_on_model_performance(y=y,
                                            yhat=vals_to_test_yhat[val], 
                                            yhat_from_klg=vals_to_test_yhat['nan'], 
                                            yhat_from_clinical_image_features=None,
                                            ids=ids, 
                                            n_bootstraps=n_bootstraps)
            
        for val in ['0.0', '1.0', '2.0', '3.0', '4.0']:
            print("Comparing pain gap reduction for diverse dataset with seed %s to non-diverse dataset (note that 'KLG' here is the non-diverse dataset)" % val)
            bootstrap_CIs_on_pain_gap_reduction(y=y, 
                                                yhat=vals_to_test_yhat[val], 
                                                yhat_from_klg=vals_to_test_yhat['nan'],
                                                ids=ids, 
                                                all_ses_vars=all_ses_vars, 
                                                n_bootstraps=n_bootstraps, 
                                                quantities_of_interest=['yhat/rival red. ratio'])
        
            
        main_titles = {'race_black':'Race\ndiversity', 
        'binarized_education_graduated_college':'Education\ndiversity', 
        'binarized_income_at_least_50k':'Income\ndiversity'}
        plot_diversity_results(metrics_we_want,
            main_title=main_titles[ses_group],
            minority_idxs=minority_idxs,
            y=y,
            yhat_from_klg=yhat_from_klg)

def plot_diversity_results(metrics_we_want, main_title, minority_idxs, y, yhat_from_klg):
    """
    Plot blue dots for KLG baseline, red dots for no-diversity condition, black dots for diversity condition. 

    metrics_we_want is a dataframe with performance under different majority_group_seed conditions. 
    Checked. 
    """
    check_is_array(minority_idxs)
    check_is_array(y)
    check_is_array(yhat_from_klg)
    cols_to_plot = [#'negative_rmse', 
                    'r^2', 
                    #'spearman_r^2',
                    'did_not_graduate_college_pain gap reduction ratio', 
                    'income_less_than_50k_pain gap reduction ratio', 
                    'race_black_pain gap reduction ratio', 
                    #'Just minority r^2', 
                    #'Just non-minority r^2',
                    #'Just minority negative_rmse', 
                    #'Just non-minority negative_rmse']
                    ]

    col_pretty_names = {'r^2':'$r^2$', 
    'did_not_graduate_college_pain gap reduction ratio':'Reduction in education pain disparity\n(relative to KLG)', 
    'income_less_than_50k_pain gap reduction ratio':'Reduction in income pain disparity\n(relative to KLG)', 
    'race_black_pain gap reduction ratio':'Reduction in race pain disparity\n(relative to KLG)'}
    fontsize = 16

    plt.figure(figsize=[6 * len(cols_to_plot), 3])
    #plt.suptitle(main_title)
    for subplot_idx, col in enumerate(cols_to_plot):
        xlimits = None
        plt.subplot(1, len(cols_to_plot), subplot_idx + 1)

        assert sorted(list(set(metrics_we_want['majority_group_seed']))) == sorted(['0.0', '1.0', '2.0', '3.0', '4.0', 'nan'])
        assert len(metrics_we_want) == 6
        if 'pain gap reduction ratio' in col:
            plt.scatter([1], [1], color='blue', label='KLG')
            if col == 'did_not_graduate_college_pain gap reduction ratio':
                xlimits = [.9, 5.1]
                plt.xticks([1, 2, 3, 4, 5], ['1x', '2x', '3x', '4x', '5x'], fontsize=fontsize)
            elif col == 'income_less_than_50k_pain gap reduction ratio':
                xlimits = [.9, 3.1]
                plt.xticks([1, 2, 3], ['1x', '2x', '3x'], fontsize=fontsize)
            elif col == 'race_black_pain gap reduction ratio':
                xlimits = [.9, 6.1]
                plt.xticks([1, 2, 3, 4, 5, 6], ['1x', '2x', '3x', '4x', '5x', '6x'], fontsize=fontsize)
            else:
                raise Exception("Invalid column")
        elif 'Just minority' in col:
            klg_counterpart = assess_performance(y=y[minority_idxs], 
                                                          yhat=yhat_from_klg[minority_idxs], 
                                                          binary_prediction=False)[col.split()[-1]]
            plt.scatter([klg_counterpart], [1], color='blue', label='KLG')
        elif 'Just non-minority' in col:
            klg_counterpart = assess_performance(y=y[~minority_idxs], 
                                                          yhat=yhat_from_klg[~minority_idxs], 
                                                          binary_prediction=False)[col.split()[-1]]
            plt.scatter([klg_counterpart], [1], color='blue', label='KLG')
        else:
            if col == 'r^2':
                xlimits = [.09, .18]
                plt.xticks([.1, .12, .14, .16], fontsize=fontsize)
            klg_counterpart = assess_performance(y=y, yhat=yhat_from_klg, binary_prediction=False)[col]
            plt.scatter([klg_counterpart], [1], color='blue', label='KLG')

        plt.scatter([], [], label='') # this is just to make the legend spacing good. 

        # This is the non-diversity condition. One red dot. 
        plt.scatter(metrics_we_want.loc[metrics_we_want['majority_group_seed'] == 'nan', col].values,
                    [1], 
                    color='red', 
                    label='Non-diverse\ntrain set')

        # This is the diversity condition. 5 black dots. 
        plt.scatter(metrics_we_want.loc[metrics_we_want['majority_group_seed'] != 'nan', col].values, 
                    [1] * (len(metrics_we_want) - 1),
                    color='black', 
                    label='Diverse\ntrain set')

        if xlimits is not None:
            assert (metrics_we_want[col].values > xlimits[0]).all()
            assert (metrics_we_want[col].values < xlimits[1]).all()
            plt.xlim(xlimits)
        plt.yticks([])
        plt.xlabel(col_pretty_names[col] if col in col_pretty_names else col, fontsize=fontsize)
        if subplot_idx == 0:
            plt.ylabel(main_title, fontsize=fontsize)
            if 'race' in main_title.lower():
                plt.legend(ncol=3, fontsize=fontsize, labelspacing=0.2, columnspacing=.2, handletextpad=.1, loc=(.08, .6))
    plt.subplots_adjust(left=.05, right=.95, bottom=.3, wspace=.05)
    plt.savefig('diversity_%s.png' % main_title.replace(' ', '_').replace('\n', '_'), dpi=300)
    plt.show()

def print_out_paired_images(df, dataset, pair_idxs, title_fxn, directory_to_save):
    """
    Saves images in pairs so we can look for differences. 
    pair_idxs should be a list of image pairs (list of lists). Title_fxn takes in df and an index i and returns a title. 
    """
    for i in range(len(pair_idxs)):
        img_1 = dataset[pair_idxs[i][0]]['image'][0, :, :]
        img_1 = (img_1 - img_1.mean()) / img_1.std()
        img_2 = dataset[pair_idxs[i][1]]['image'][0, :, :]
        img_2 = (img_2 - img_2.mean()) / img_2.std()
        
        plt.figure()
        plt.subplot(121)
        plt.imshow(img_1, clim=[-3, 3], cmap='bone')
        plt.title(title_fxn(df, pair_idxs[i][0]))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(122)
        plt.imshow(img_2, clim=[-3, 3], cmap='bone')
        plt.title(title_fxn(df, pair_idxs[i][1]))
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(directory_to_save, 'pair_%i.png' % i), dpi=300)
        plt.show()

def generate_paired_images_to_inspect_three_different_ways(dataset_to_use, yhat):
    """
    Try pairing images by KLG; by all image features (basically we take the max over feature categories); and by invididual and side. In all cases, the 
    "high pain" image is on the right, although the way we define "high pain" changes. 
    """
    # 1. Pair images by KLG. Just use KLG 2. 
    df = dataset_to_use.non_image_data.copy()
    df['yhat'] = yhat
    klg_idxs = (df['xrkl'] == 2)
    bad_y_cutoff = scoreatpercentile(df.loc[klg_idxs, 'koos_pain_subscore'], 10)
    bad_yhat_cutoff = scoreatpercentile(df.loc[klg_idxs, 'yhat'], 10)
    print("Bad Y cutoff is %2.3f; yhat cutoff is %2.3f" % (bad_y_cutoff, bad_yhat_cutoff))
    good_y_cutoff = scoreatpercentile(df.loc[klg_idxs, 'koos_pain_subscore'], 90)
    good_yhat_cutoff = scoreatpercentile(df.loc[klg_idxs, 'yhat'], 90)
    print("Good Y cutoff is %2.3f; yhat cutoff is %2.3f" % (good_y_cutoff, good_yhat_cutoff))

    low_pain_candidates = np.array(range(len(df)))[klg_idxs & 
                                                   (df['yhat'] >= good_yhat_cutoff) & 
                                                   (df['koos_pain_subscore'] >= good_y_cutoff)]
    random.shuffle(low_pain_candidates)
    high_pain_candidates = np.array(range(len(df)))[klg_idxs & 
                                                   (df['yhat'] <= bad_yhat_cutoff) & 
                                                   (df['koos_pain_subscore'] <= bad_y_cutoff)]
    random.shuffle(high_pain_candidates)


    print("%i low pain candidates; %i high pain candidates" % (len(low_pain_candidates), 
                                                               len(high_pain_candidates)))
    n_images = min(len(high_pain_candidates), len(low_pain_candidates))

    paired_image_idxs = list(zip(low_pain_candidates[:n_images], high_pain_candidates[:n_images]))

    def title_fxn(df, idx):
        return 'KLG %i; yhat %2.1f; y %2.1f' % (df.iloc[idx]['xrkl'], 
                                                df.iloc[idx]['yhat'], 
                                                df.iloc[idx]['koos_pain_subscore'])
    print_out_paired_images(df=df, 
                        dataset=dataset_to_use, 
                        pair_idxs=paired_image_idxs,
                        title_fxn=title_fxn,
                        directory_to_save='paired_images/paired_by_KLG/')
    
    # pair images by image features. 
    # Set looser percentile cutoffs than in KLG pairing, to ensure we actually have pairs. 

    df = dataset_to_use.non_image_data.copy()
    df['yhat'] = yhat
    feature_groups_to_match_on = ['att', 'ch', 'cy', 'js', 'os', 'kl', 'sc']
    def title_fxn(df, idx):
        return '%s\nyhat %2.1f; y %2.1f' % (' '.join(['%s%1.0f' % (a, df.iloc[idx]['max_%s' % a]) for a in feature_groups_to_match_on]),
                                                df.iloc[idx]['yhat'], 
                                                df.iloc[idx]['koos_pain_subscore'])

    all_cols_used = []
    for feature_group in feature_groups_to_match_on:
        cols_to_use = sorted([a for a in CLINICAL_CONTROL_COLUMNS if feature_group in a])
        print("taking max of features", cols_to_use, 'for %s' % feature_group)
        df['max_%s' % feature_group] = df[cols_to_use].values.max(axis=1)
        assert pd.isnull(df['max_%s' % feature_group]).sum() == 0
        all_cols_used += cols_to_use
    assert sorted(all_cols_used) == sorted(CLINICAL_CONTROL_COLUMNS) # make sure we have a disjoint partition of the original column set. 
    grouped_d = df.groupby(['max_%s' % a for a in feature_groups_to_match_on])
    bad_y_cutoff = scoreatpercentile(df['koos_pain_subscore'], 40)
    bad_yhat_cutoff = scoreatpercentile(df['yhat'], 40)
    print("Y cutoff is %2.3f; yhat cutoff is %2.3f" % (bad_y_cutoff, bad_yhat_cutoff))
    good_y_cutoff = scoreatpercentile(df['koos_pain_subscore'], 60)
    good_yhat_cutoff = scoreatpercentile(df['yhat'], 60)
    print("Y cutoff is %2.3f; yhat cutoff is %2.3f" % (good_y_cutoff, good_yhat_cutoff))

    pair_idxs = []
    for feature_vals, small_d in grouped_d:
        if len(small_d) > 1:
            bad_idxs = ((small_d['koos_pain_subscore'] <= bad_y_cutoff) & 
                        (small_d['yhat'] <= bad_yhat_cutoff))
            good_idxs = ((small_d['koos_pain_subscore'] >= good_y_cutoff) & 
                        (small_d['yhat'] >= good_yhat_cutoff))
            if bad_idxs.sum() > 0 and good_idxs.sum() > 0:
                bad_small_d = small_d.loc[bad_idxs]
                good_small_d = small_d.loc[good_idxs]
                bad_idx = random.choice(bad_small_d.index)
                good_idx = random.choice(good_small_d.index)
                pair_idxs.append([good_idx, bad_idx])

    print_out_paired_images(df=df, 
                            dataset=dataset_to_use, 
                            pair_idxs=pair_idxs,
                            title_fxn=title_fxn,
                            directory_to_save='paired_images/paired_by_image_features/')
    
    # pair by person, side, and KLG. (So variation is just over time). Set a threshold on change in pain/yhat. 
    df = dataset_to_use.non_image_data.copy()
    df['yhat'] = yhat
    def title_fxn(df, idx):
        return '%s, %s side\nKLG %i\nyhat %2.1f; y %2.1f' % (
                                                df.iloc[idx]['visit'],
                                                df.iloc[idx]['side'],
                                                df.iloc[idx]['xrkl'],
                                                df.iloc[idx]['yhat'], 
                                                df.iloc[idx]['koos_pain_subscore'])


    grouped_d = df.groupby(['id', 'side', 'xrkl'])
    pair_idxs = []
    for feature_vals, small_d in grouped_d:
        if len(small_d) > 1:
            small_d = small_d.copy().sort_values(by='koos_pain_subscore')[::-1]
            koos_change = small_d.iloc[0]['koos_pain_subscore'] - small_d.iloc[-1]['koos_pain_subscore']
            yhat_change = small_d.iloc[0]['yhat'] - small_d.iloc[-1]['yhat']
            if koos_change > 5 and yhat_change > 5:

                pair_idxs.append([small_d.index[0], small_d.index[-1]])

    print_out_paired_images(df=df, 
                            dataset=dataset_to_use, 
                            pair_idxs=pair_idxs,
                            title_fxn=title_fxn,
                            directory_to_save='paired_images/paired_by_person_and_side/')



def stratify_performances(df, yhat, y, yhat_from_klg):
    """
    How do we do across subsets of the dataset relative to KLG? 
    """
    stratified_performances = []
    pd.set_option('max_rows', 500)

    for thing_to_stratify_by in ['v00site', 
                                 'side', 
                                 'visit', 
                                 'p02sex', 
                                 'age_at_visit',
                                 'current_bmi',
                                 'binarized_income_at_least_50k', 
                                 'binarized_education_graduated_college',
                                 'race_black']:
        for k in sorted(list(set(df[thing_to_stratify_by].dropna()))):

            substratification_idxs = df[thing_to_stratify_by].values == k
            if substratification_idxs.sum() < 100:
                # don't plot super-noisy groups. 
                continue
            yhat_performance = assess_performance(yhat=yhat[substratification_idxs], 
                                    y=y[substratification_idxs], 
                                    binary_prediction=False)
            klg_performance = assess_performance(yhat=yhat_from_klg[substratification_idxs], 
                                    y=y[substratification_idxs], 
                                    binary_prediction=False)
            stratified_performances.append(yhat_performance)
            stratified_performances[-1]['predictor'] = 'yhat'
            stratified_performances[-1]['substratification'] = thing_to_stratify_by + ' ' + str(k)
            stratified_performances[-1]['n'] = substratification_idxs.sum()
            
            stratified_performances.append(klg_performance)
            stratified_performances[-1]['predictor'] = 'klg'
            stratified_performances[-1]['substratification'] = thing_to_stratify_by + ' ' + str(k)
            stratified_performances[-1]['n'] = substratification_idxs.sum()
            
            for metric in ['r^2', 'negative_rmse']:
                if yhat_performance[metric] < klg_performance[metric]:
                    print('Warning: yhat %s (%2.3f) is less than KLGs (%2.3f) for %s' % 
                          (metric,
                           yhat_performance[metric], 
                           klg_performance[metric], 
                           k))
    print("All other metrics passed. If a few fail, and not by too much, probably noise.")
    stratified_performances = pd.DataFrame(stratified_performances)

    # plot performance across subsets. 
    sns.set_style('whitegrid')
    plt.figure(figsize=[15, 5])
    plt.subplot(121)
    labels = stratified_performances.loc[stratified_performances['predictor'] == 'klg', 'substratification'].values
    plt.scatter(range(len(labels)), stratified_performances.loc[stratified_performances['predictor'] == 'klg', 'r^2'].values, label='klg')
    plt.scatter(range(len(labels)), stratified_performances.loc[stratified_performances['predictor'] == 'yhat', 'r^2'].values, label='yhat')
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel("r^2")
    plt.legend()

    plt.subplot(122)
    plt.scatter(range(len(labels)), stratified_performances.loc[stratified_performances['predictor'] == 'klg', 'negative_rmse'].values, label='KLG')
    plt.scatter(range(len(labels)), stratified_performances.loc[stratified_performances['predictor'] == 'yhat', 'negative_rmse'].values, label='yhat')
    plt.legend()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel("Negative RMSE")
    plt.show()

    return stratified_performances

def get_CI_from_percentiles(bootstraps, alpha=0.05, make_plot=False):
    # given a list of bootstrapped values, compute the CIs. 
    assert alpha > 0 
    assert alpha < 1
    if make_plot:
        plt.figure()
        plt.hist(bootstraps, bins=50)
        plt.title("N bootstraps: %i; range %2.3f-%2.3f" % (len(bootstraps), np.min(bootstraps), np.max(bootstraps)))
        plt.show()

    return [scoreatpercentile(bootstraps, alpha * 100 / 2.), 
            scoreatpercentile(bootstraps, 100 - alpha * 100 / 2.)]

def bootstrap_CIs_on_pain_gap_reduction(y, yhat, yhat_from_klg, ids, all_ses_vars, n_bootstraps, quantities_of_interest=None):
    """
    Confidence intervals on how much we reduce the pain gap. 
    """
    if quantities_of_interest is None:
        quantities_of_interest = ['No controls gap', 'Rival gap', 'yhat gap', 'Rival red. vs nothing', 'red. vs nothing', 'yhat/rival red. ratio']
    def ci_fxn(bootstraps):
        print("Bootstrapped CIs")
        concat_bootstraps = pd.concat(bootstraps)
        for k in quantities_of_interest:
            print('Bootstrap CI on', k, get_CI_from_percentiles(concat_bootstraps[k].values, make_plot=True))
            
    for ses_var_name in all_ses_vars:
        print("Bootstrap on pain gap reduction for %s" % ses_var_name)
        get_bootstrapped_cis_on_quantity(df=pd.DataFrame({'id':ids, 
                                                          'y':y, 
                                                          'yhat':yhat, 
                                                          'rival':yhat_from_klg, 
                                                          ses_var_name:all_ses_vars[ses_var_name]}), 
                                              resample_points_within_cluster=False, 
                                              fxn=lambda x:quantify_pain_gap_reduction_vs_rival(yhat=x['yhat'].values, 
                                                                            y=x['y'].values, 
                                                                            rival_severity_measure=x['rival'].values, 
                                                                            all_ses_vars={ses_var_name:x[ses_var_name].values}, 
                                                                            ids=x['id'].values, 
                                                                            lower_bound_rival_reduction_at_0=True),
                                              n_bootstraps=n_bootstraps, 
                                              ci_fxn=ci_fxn)

def mean_absolute_error(y, yhat):
    return np.mean(np.abs(y - yhat))

def bootstrap_CIs_on_model_performance(y, yhat, yhat_from_klg, yhat_from_clinical_image_features, ids, n_bootstraps, binary_prediction=False, metrics=None):
    """
    Compare our r^2 and RMSE to KLGs. 
    """
    check_is_array(y)
    check_is_array(yhat)
    check_is_array(yhat_from_klg)
    if yhat_from_clinical_image_features is not None:
        check_is_array(yhat_from_clinical_image_features)
    check_is_array(ids)

    def compare_our_performance_to_rival(df, metric):
        if metric != 'mean_absolute_error':
            # coded mean_absolute_error separately because it was just for nature medicine R+R
            rival_performance = assess_performance(y=df['y'].values, yhat=df['rival'].values, binary_prediction=binary_prediction)[metric]
            our_performance = assess_performance(y=df['y'].values, yhat=df['yhat'].values, binary_prediction=binary_prediction)[metric]
        else:
            rival_performance = mean_absolute_error(y=df['y'].values, yhat=df['rival'].values) 
            our_performance = mean_absolute_error(y=df['y'].values, yhat=df['yhat'].values)

        return {'our_performance':our_performance, 
                'rival_performance':rival_performance, 
                'ratio':our_performance/rival_performance}

    def ci_fxn(bootstraps):
        print("Bootstrapped CIs")
        for k in bootstraps[0].keys():
            print('CI on', k, get_CI_from_percentiles([x[k] for x in bootstraps], make_plot=True))

    for rival_name in ['klg', 'all_image_features']:
        if rival_name == 'all_image_features' and yhat_from_clinical_image_features is None:
            continue
        if rival_name == 'klg':
            rival = yhat_from_klg
        elif rival_name == 'all_image_features':
            rival = yhat_from_clinical_image_features
        else:
            raise Exception("Invalid rival name")
        if metrics is None:
            if binary_prediction:
                metrics = ['auc', 'auprc']
            else:
                metrics = ['rmse', 'r^2', 'spearman_r^2']
            
        for metric in metrics:
            print("\n\nComputing CIs for metric %s, comparing yhat to %s" % (metric, rival_name))
            get_bootstrapped_cis_on_quantity(df=pd.DataFrame({'id':ids, 'y':y, 'yhat':yhat, 'rival':rival}), 
                                              resample_points_within_cluster=False, 
                                              fxn=lambda x:compare_our_performance_to_rival(x, metric=metric),
                                              n_bootstraps=n_bootstraps, 
                                              ci_fxn=ci_fxn)

def make_counterfactual_surgery_prediction(interventions_df, 
                                           yhat, 
                                           klg, 
                                           all_ses_vars):
    """
    Given an interventions dataframe (on which we fit the knee_surgery ~ xrkl model)
    and a counterfactual dataframe (on which we actually predict how things would change) 
    predict how surgery allocation to disadvantaged racial/SES groups would change under yhat rather than KLG. 
    """
    check_is_array(yhat)
    check_is_array(klg)
    assert len(yhat) == len(klg) 
    surgery_model = sm.Logit.from_formula('knee_surgery ~ C(xrkl)', data=interventions_df).fit()
    print(surgery_model.summary())

    print("Surgery gap in SINGLE KNEE surgery rates")
    for k in ['race_black', 'binarized_education_graduated_college', 'binarized_income_at_least_50k']:
        ses_var_1_mean = interventions_df.loc[interventions_df[k] == 1, 'knee_surgery'].mean()
        ses_var_0_mean = interventions_df.loc[interventions_df[k] == 0, 'knee_surgery'].mean()
        print("%s var=1 surgery rate: %2.3f; var=0 surgery rate: %2.3f; ratio: %2.5f; inverse ratio %2.5f" % (k, ses_var_1_mean, ses_var_0_mean, ses_var_1_mean/ses_var_0_mean, ses_var_0_mean/ses_var_1_mean))

    discretized_yhat = discretize_yhat_like_kl_grade(
        yhat_arr=yhat,
        kl_grade_arr=klg,
        y_col='koos_pain_subscore') # want to make sure the allocation of severity grades match up. 


    original_pred = surgery_model.predict({'xrkl':klg}).values
    counterfactual_pred = surgery_model.predict({'xrkl':discretized_yhat}).values
    assert (original_pred > 0).all()
    assert (counterfactual_pred > 0).all()
    assert (original_pred < 1).all()
    assert (counterfactual_pred < 1).all()
    print("Number of rows in counterfactual dataframe: %i"  % len(counterfactual_pred))

    # bar graph/histogram of fraction of disadvantaged groups assigned to each severity grade by yhat/KLG. 
    plt.figure(figsize=[12, 4])
    xs = sorted(list(set(klg)))
    assert xs == list(range(5))
    for subplot_idx, ses_var in enumerate(list(all_ses_vars.keys())):
        plt.subplot(1, 3, subplot_idx + 1)
        klg_props = []
        yhat_props = []
        total_count = all_ses_vars[ses_var].sum()
        klg_counts = Counter(klg[all_ses_vars[ses_var] == 1])
        discretized_yhat_counts = Counter(discretized_yhat[all_ses_vars[ses_var] == 1])
        for x in xs:
            klg_props.append(100. * klg_counts[x] / total_count)
            yhat_props.append(100. * discretized_yhat_counts[x] / total_count)
        print(ses_var)
        print(klg_props)
        print(yhat_props)
        
        barwidth = .3
        plt.bar(np.array(xs), klg_props, label='KLG', alpha=.7, width=barwidth)
        plt.bar(np.array(xs) + barwidth, yhat_props, label='$\hat y$', alpha=.7, width=barwidth)
        plt.title(ses_var)
        plt.legend()
        plt.xticks(range(5))
        plt.xlabel("Severity grade")
        plt.ylim([0, 50])
        plt.xlim([-.1 - barwidth / 2., 4 + 1.5 * barwidth + .1])
        if subplot_idx == 0:
            plt.ylabel("Probability of being assigned to that grade")
            plt.yticks(range(0, 60, 10), ['%i%%' % a for a in range(0, 60, 10)])
        else:
            plt.yticks([])
    plt.show()

    assert np.allclose(original_pred.mean(), counterfactual_pred.mean())
    pd.set_option('precision', 5)
    for var in all_ses_vars:
        ses_arr = all_ses_vars[var]
        check_is_array(ses_arr)
        ratio = counterfactual_pred[ses_arr].mean()/original_pred[ses_arr].mean()
        print("Frac of %s getting surgery under KLG: %2.3f; counterfactual %2.3f; ratio %2.3f" % (var, original_pred[ses_arr].mean(), counterfactual_pred[ses_arr].mean(), ratio))


def make_scatter_plot_showing_severity_reassignment_under_yhat(yhat, y, klg, all_ses_vars, idxs_to_use, interventions_df):
    """
    Compute how severity / surgery assignments change under yhat. Calls make_counterfactual_surgery_prediction. 
    We only compute reassignments for rows specified by idxs_to_use (used to, eg, choose only baseline values).
    """
    check_is_array(yhat)
    check_is_array(y)
    check_is_array(klg)
    check_is_array(idxs_to_use)

    yhat = yhat[idxs_to_use].copy()
    y = y[idxs_to_use].copy()
    klg = klg[idxs_to_use].copy()
    all_ses_vars = copy.deepcopy(all_ses_vars)
    for ses_var in all_ses_vars:
        all_ses_vars[ses_var] = all_ses_vars[ses_var][idxs_to_use]

    plt.figure(figsize=[5, 4])
    sns.set_style('white')
    for ses_var in all_ses_vars:
        check_is_array(all_ses_vars[ses_var])
        geq_klg_2_results = compare_pain_levels_for_people_geq_klg_2(yhat=yhat, 
                                                      y=y, 
                                                      klg=klg, 
                                                      ses=all_ses_vars[ses_var], 
                                                      y_col='koos_pain_subscore')
        if ses_var == 'race_black':
            pretty_label = 'Black'
        elif ses_var == 'did_not_graduate_college':
            pretty_label = "Didn't graduate college"
        elif ses_var == 'income_less_than_50k':
            pretty_label = 'Income < $50k'
        else:
            raise Exception("Invalid SES var")
        plt.scatter(range(5), 
                 geq_klg_2_results['all_klg_yhat_ratios'], 
                 label=pretty_label)
    plt.ylim([1, 1.7])
    plt.yticks([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
              ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%'], fontsize=14)
    plt.ylabel("How much likelier under $\hat y$", fontsize=14)
    plt.xticks([1, 2, 3, 4], 
              ['$\geq 1$', '$\geq 2$', '$\geq 3$', '$\geq 4$'], fontsize=14)
    plt.xlabel("Severity grade", fontsize=14)
    plt.xlim([.9, 4.1])
    plt.legend()
    plt.subplots_adjust(left=.3, bottom=.2)
    plt.savefig("reassign_under_yhat_plot.png", dpi=300)
    plt.show()
    make_counterfactual_surgery_prediction(interventions_df=interventions_df, 
                                           yhat=yhat, 
                                           klg=klg, 
                                           all_ses_vars=all_ses_vars)

def make_painkillers_and_surgery_frequency_bar_plot(interventions_df):
    """
    Plot the frequency of various medical interventions. 
    """
    interventions_df = interventions_df.copy()
    medications = ["V00RXACTM", "V00RXANALG", "V00RXASPRN", "V00RXBISPH", 
                       "V00RXCHOND", "V00RXCLCTN", "V00RXCLCXB", "V00RXCOX2", 
                       "V00RXFLUOR", "V00RXGLCSM", "V00RXIHYAL", "V00RXISTRD", 
                       "V00RXMSM", "V00RXNARC", "V00RXNSAID", "V00RXNTRAT", 
                       "V00RXOSTRD", "V00RXOTHAN", "V00RXRALOX", "V00RXRFCXB", 
                       "V00RXSALIC", "V00RXSAME", "V00RXTPRTD", "V00RXVIT_D", "V00RXVLCXB"] 

    



    sns.set_style('whitegrid')
    medications = [a.lower().replace('v00', '') for a in medications]
    other_cols = ["id", "knee_surgery", 
                  'binarized_income_at_least_50k', 'binarized_education_graduated_college', 'race_black', 
                  'xrkl', 'koos_pain_subscore']
    medication_df = interventions_df[medications + other_cols].copy()
    medication_df.columns = [MEDICATION_CODES[('v00' + a).upper()] for a in medications] + other_cols
    any_knee_surgery = set(medication_df.loc[medication_df['knee_surgery'] > 0, 'id'])
    medication_df['knee_surgery\neither_knee'] = medication_df['id'].map(lambda x:x in any_knee_surgery)
    intervention_cols_to_plot = ['Aspirin', 'Acetaminophen',
                                 'Narcotic_analgesic','NSAID',
                                 'Analgesic', 'knee_surgery\neither_knee']
    medication_df['in_high_pain'] = binarize_koos(medication_df['koos_pain_subscore'].values)
    for intervention in ['Aspirin', 'Acetaminophen', 'Narcotic_analgesic', 'NSAID', 'Analgesic']:
        df_to_use = medication_df.dropna(subset=['in_high_pain', intervention])
        medication_model = sm.Logit.from_formula('%s ~ in_high_pain' % intervention, data=df_to_use).fit(cov_type='cluster', cov_kwds={'groups':df_to_use['id'].values})
        print(medication_model.summary())

    for k in intervention_cols_to_plot + ['knee_surgery']:
        print("Fraction of missing data in %s: %2.3f out of %i points" % (k, pd.isnull(medication_df[k]).mean(), len(medication_df)))

    for col in intervention_cols_to_plot:
        # make sure that each person has a unique value. 
        assert len(medication_df.drop_duplicates('id')) == len(medication_df.drop_duplicates(['id', col]))
    medication_df = medication_df.drop_duplicates('id')

    make_intervention_frequencies_plot(medication_df, intervention_cols_to_plot, 'interventions_plot.png')

def make_rate_of_surgery_figure(interventions_df):
    """
    Fraction of surgery by KLG. 
    """
    interventions_df = interventions_df.copy()
    print(interventions_df[['xrkl', 'knee_surgery']].groupby('xrkl').agg(['mean', 'size']))

    xs = range(5)
    ys = []
    yerrs = []
    for x in xs:
        ys.append(interventions_df.loc[interventions_df['xrkl'] == x, 'knee_surgery'].mean())
    sns.set_style('white')
    plt.figure(figsize=[5, 4])
    plt.scatter(xs, ys)
    plt.xlabel("KLG", fontsize=14)
    plt.ylabel("Had knee surgery", fontsize=14)
    plt.xlim([-.1, 4.1])
    plt.xticks([0, 1, 2, 3, 4], fontsize=14)
    plt.yticks([0, .3, .6], ['0%', '30%', '60%'], fontsize=14)
    plt.ylim([0, .6])
    plt.subplots_adjust(bottom=.3, left=.3)
    plt.savefig('klg_surgery_plot.png', dpi=300)

def make_descriptive_stats_table(train_df, val_df, test_df):
    """
    Descriptive stats for table 1 in paper. 
    """
    # Need to load original data to get original BMI + age, which we render as categorical in final data. 
    all_clinical00 = pd.read_csv(os.path.join(BASE_NON_IMAGE_DATA_DIR, 'AllClinical_ASCII', 'AllClinical00.txt'), sep='|')
    all_clinical00.columns = all_clinical00.columns.map(lambda x:x.lower())
    assert len(all_clinical00.columns) == len(set(all_clinical00.columns))
    print("allclinical00 has %i columns, %i rows" % (len(all_clinical00.columns), len(all_clinical00)))
    
    all_clinical00['current_bmi'] = all_clinical00['p01weight'] / ((all_clinical00['p01height'] / 1000.) ** 2)
    all_clinical00 = all_clinical00[['id', 'current_bmi', 'v00age']]
    all_clinical00.index = all_clinical00['id']

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_plus_val_df = pd.concat([train_df, val_df])
    train_plus_val_df.index = range(len(train_plus_val_df))
    train_plus_val_plus_test_df = pd.concat([train_df, val_df, test_df])
    train_plus_val_plus_test_df.index = range(len(train_plus_val_plus_test_df))

    print("Sorted image features by how often they are nonzero (all three datasets combined)")

    how_often_not_zero = []
    for c in CLINICAL_CONTROL_COLUMNS:
        assert pd.isnull(train_plus_val_plus_test_df[c]).sum() == 0
        how_often_not_zero.append({'c':c, 
                                  'not_zero':(train_plus_val_plus_test_df[c] != 0).mean(), 
                                   'val_counts':Counter(train_plus_val_plus_test_df[c])})
    print(pd.DataFrame(how_often_not_zero).sort_values(by='not_zero')[::-1])

    dataset_names = ['train', 'val', 'train+val', 'test', 'train+val+test']

    for dataset_idx, descriptive_stats_df in enumerate([train_df, val_df, train_plus_val_df, test_df, train_plus_val_plus_test_df]):
        print("\n\n****%s" % dataset_names[dataset_idx])
        print("Points: %i total" % len(descriptive_stats_df))
        print("People: %i total" % len(set(descriptive_stats_df['id'])))

        descriptive_stats_df['is_female'] = (descriptive_stats_df['p02sex'] == '2: Female').values

        ids = list(set(descriptive_stats_df['id'].values))
        print(all_clinical00.loc[ids, ['current_bmi', 'v00age']].describe().loc[['mean', 'std']])
        assert pd.isnull(all_clinical00.loc[ids, 'v00age']).sum() == 0

        for k in ['binarized_income_at_least_50k', 
                  'binarized_education_graduated_college', 
                  'race_black', 
                  'is_female']:
            n_ids_in_cat = len(set(descriptive_stats_df.loc[descriptive_stats_df[k] == 1, 'id'].values))
            print('%s: %i/%i people, %2.5f '% (k, n_ids_in_cat, len(set(descriptive_stats_df['id'])), 1.*n_ids_in_cat/len(set(descriptive_stats_df['id']))))

        print(100 * descriptive_stats_df.drop_duplicates('id')['p02race'].value_counts(dropna=False)/len(descriptive_stats_df.drop_duplicates('id')))
        print('race + ethnicity')
        descriptive_stats_df['race+is_hispanic'] = descriptive_stats_df['p02race'] + ', hispanic ' + descriptive_stats_df['p02hisp']
        print(100 * descriptive_stats_df.drop_duplicates('id')['race+is_hispanic'].value_counts(dropna=False)/len(descriptive_stats_df.drop_duplicates('id')))
        # categorical baseline BMI/age. 
        baseline_idxs = descriptive_stats_df['visit'] == '00 month follow-up: Baseline'
        baseline_df = descriptive_stats_df.loc[baseline_idxs].copy()
        assert len(baseline_df.drop_duplicates('id')) == len(baseline_df[['id', 'current_bmi']].drop_duplicates())
        assert len(baseline_df.drop_duplicates('id')) == len(baseline_df[['id', 'age_at_visit']].drop_duplicates())
        baseline_df = baseline_df.drop_duplicates('id')
        print(baseline_df['current_bmi'].value_counts(dropna=False) / len(baseline_df))
        print(baseline_df['age_at_visit'].value_counts(dropna=False) / len(baseline_df))

        # fraction of people in high pain. 
        descriptive_stats_df['klg_geq_2'] = (descriptive_stats_df['xrkl'] >= 2).values
        descriptive_stats_df['high_pain'] = binarize_koos(descriptive_stats_df['koos_pain_subscore'].values)

        for outcome in ['klg_geq_2', 'high_pain']:
            print("\n\n***Outcome %s" % outcome)
            print("Overall fraction of knees %s: %2.5f" % (outcome, descriptive_stats_df[outcome].mean()))
            for k in ['binarized_income_at_least_50k', 'binarized_education_graduated_college', 'race_black']:
                mean_for_group_true = descriptive_stats_df.loc[descriptive_stats_df[k] == 1, outcome].mean()
                mean_for_group_false = descriptive_stats_df.loc[descriptive_stats_df[k] == 0, outcome].mean()
                print("Fraction for %-50s=1: %2.5f" % (k, mean_for_group_true))
                print("Fraction for %-50s=0: %2.5f" % (k, mean_for_group_false))
                # Compute p-value on difference. 
                df_for_regression = pd.DataFrame({'outcome':descriptive_stats_df[outcome].values * 1., 
                                                  'ses':descriptive_stats_df[k].values * 1., 
                                                  'id':descriptive_stats_df['id'].values})
                diff_p_value = (sm.OLS.from_formula('outcome ~ ses', data=df_for_regression).fit(cov_type='cluster', cov_kwds={'groups':df_for_regression['id']}))
                print('p-value for difference: %2.6f' % diff_p_value.pvalues['ses'])

        descriptive_stats_df['koos_pain_zscore'] = (descriptive_stats_df['koos_pain_subscore'] - descriptive_stats_df['koos_pain_subscore'].mean()) / descriptive_stats_df['koos_pain_subscore'].std(ddof=1)
        descriptive_stats_df['koos_pain_percentile'] = 100. * rankdata(descriptive_stats_df['koos_pain_subscore'].values)/len(descriptive_stats_df)
        pd.set_option('display.width', 500)
        for k in ['binarized_income_at_least_50k', 'binarized_education_graduated_college', 'race_black']:
            print("Continuous descriptive stats for pain and KLG")
            print(descriptive_stats_df[['xrkl', 'koos_pain_subscore', 'koos_pain_percentile', k]].groupby(k).agg(['mean', 'std']))
            absolute_pain_gap = np.abs(descriptive_stats_df.loc[descriptive_stats_df[k] == 1, 'koos_pain_subscore'].mean() - 
                              descriptive_stats_df.loc[descriptive_stats_df[k] == 0, 'koos_pain_subscore'].mean())
            print("Pain gap in stds: %2.3f" % (absolute_pain_gap / descriptive_stats_df['koos_pain_subscore'].std(ddof=1)))

            # Cohen's d, as defined by Wikipedia: https://en.wikipedia.org/wiki/Effect_size#Cohen%27s_d. This ends up being very similar to the effect size in sds. 
            n1 = (descriptive_stats_df[k] == 1).sum()
            n0 = (descriptive_stats_df[k] == 0).sum()
            var1 = descriptive_stats_df.loc[descriptive_stats_df[k] == 1, 'koos_pain_subscore'].std(ddof=1) ** 2
            var0 = descriptive_stats_df.loc[descriptive_stats_df[k] == 0, 'koos_pain_subscore'].std(ddof=1) ** 2
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2))
            print("Pain gap, cohen's d: %2.3f" % (absolute_pain_gap / pooled_std))

        print("\n\nComparing median to other distributions")
        for k in ['binarized_income_at_least_50k', 'binarized_education_graduated_college', 'race_black']:
            print(k)
            for ingroup in [0, 1]:
                ingroup_pain_median = descriptive_stats_df.loc[descriptive_stats_df[k] == ingroup, 'koos_pain_subscore'].median()
                outgroup_worse_pain = (descriptive_stats_df.loc[descriptive_stats_df[k] != ingroup, 'koos_pain_subscore'] < ingroup_pain_median).mean()
                outgroup_better_pain = (descriptive_stats_df.loc[descriptive_stats_df[k] != ingroup, 'koos_pain_subscore'] > ingroup_pain_median).mean()
                outgroup_same_pain = (descriptive_stats_df.loc[descriptive_stats_df[k] != ingroup, 'koos_pain_subscore'] == ingroup_pain_median).mean()
                print("var=%i: %2.1f%% of the other group has worse pain than median person in this group, %2.1f%% better, %2.1f%% the same" % (ingroup, 100*outgroup_worse_pain, 100*outgroup_better_pain, 100*outgroup_same_pain))
    
def make_intervention_frequencies_plot(medication_df, cols_to_plot, fig_filename=None):
    """
    Given a bunch of binary cols_to_plot, plot the frequency with which they occur in the data overall and in disadvantaged racial/SES groups. 

    Two plots: one of absolute risks, one of relative risks (relative to the outgroup). 

    Also run some regressions where we see whether pain or KLG better predicts if you'll receive a treatment. 
    """
    plt.figure(figsize=[10, 10])
    sns.set_style('whitegrid')
    assert len(set(medication_df['id'])) == len(medication_df) # should be person-level. 
    bar_width = .2
    for group_idx, group in enumerate(['income < $50k', "didn't graduate college", "black"]):
        bar_positions = []
        risks_for_each_group = [] # absolute risks
        relative_risks = [] # risk relative to outgroup. 
        labels = []
        current_pos = group_idx * bar_width
        for c in cols_to_plot:
            
            if c not in cols_to_plot:
                continue
            assert set(medication_df[c].dropna()).issubset(set([0, 1]))
            if medication_df[c].mean() < .01:
                continue
            if group == 'overall': # only need to do regressions once. 
                raise Exception("This is a bit sketchy because the plot is made using individual-level data, not side-individual level data, so KLG and koos are invalid")
                klg_rsquared = sm.OLS.from_formula('%s ~ C(xrkl)' % c, data=medication_df).fit().rsquared
                koos_rsquared = sm.OLS.from_formula('%s ~ koos_pain_subscore' % c, data=medication_df).fit().rsquared
                combined_model = sm.OLS.from_formula('%s ~ koos_pain_subscore + xrkl' % c, data=medication_df).fit()
                all_rsquareds.append({'intervention':c, 
                                      'koos r^2':koos_rsquared, 
                                      'klg r^2 with categorial KLG':klg_rsquared, 
                                      'koos_beta_in_combined_model with linear KLG':combined_model.params['koos_pain_subscore'], 
                                      'klg_beta_in_combined_model with linear KLG':combined_model.params['xrkl']})
            labels.append('%s\n(%1.0f%% overall)' % (c.replace('_', ' '), 100 * medication_df[c].mean()))
            bar_positions.append(current_pos)
            if group == 'overall':
                risks_for_each_group.append(medication_df[c].mean())
                relative_risks.append(1)
            elif group == 'black':
                risks_for_each_group.append(medication_df.loc[medication_df['race_black'] == 1, c].mean())
                relative_risks.append(medication_df.loc[medication_df['race_black'] == 1, c].mean()/
                                      medication_df.loc[medication_df['race_black'] == 0, c].mean())
            elif group == 'income < $50k':
                risks_for_each_group.append(medication_df.loc[medication_df['binarized_income_at_least_50k'] == 0, c].mean())
                relative_risks.append(medication_df.loc[medication_df['binarized_income_at_least_50k'] == 0, c].mean()/
                                      medication_df.loc[medication_df['binarized_income_at_least_50k'] == 1, c].mean())
            elif group == "didn't graduate college":
                risks_for_each_group.append(medication_df.loc[medication_df['binarized_education_graduated_college'] == 0, c].mean())
                relative_risks.append(medication_df.loc[medication_df['binarized_education_graduated_college'] == 0, c].mean()/
                                      medication_df.loc[medication_df['binarized_education_graduated_college'] == 1, c].mean())
            else:
                raise Exception("invalid col")
            print("%-30s: high SES people are %2.3fx as likely to get %s" % (group, 1/relative_risks[-1], c.replace('\n', ' ')))
            current_pos += 1
        #plt.subplot(121)
        #plt.barh(bar_positions, risks_for_each_group, label=group, height=bar_width)
        #plt.subplot(111)
        plt.barh(bar_positions, relative_risks, label=group, height=bar_width)
        

    #plt.subplot(121)
    #plt.yticks([a - bar_width for a in bar_positions], labels)
    #plt.legend()
    #plt.xlabel('Proportion of people reporting use')

    #plt.subplot(122)
    #plt.yticks([])
    fontsize = 18
    plt.yticks([a - bar_width for a in bar_positions], labels, fontsize=fontsize)
    plt.xlabel("Risk relative to outgroup", fontsize=fontsize)
    plt.xticks([0, 0.5, 1, 1.5, 2.0, 2.5], ['0x', '0.5x', '1x', '1.5x', '2.0x', '2.5x'], fontsize=fontsize)
    plt.plot([1, 1], [min(bar_positions) - 1, max(bar_positions) + 1], linestyle='-', color='black')
    plt.ylim(min(bar_positions) - .5, max(bar_positions) + bar_width/2)
    plt.legend(fontsize=fontsize - 2)
    plt.subplots_adjust(left=.3)
    if fig_filename is not None:
        plt.savefig(fig_filename, dpi=300)

def check_main_results_hold_when_controlling_for_things(df, yhat, rival_predictor, rival_name, all_controls):
    """
    Make sure that when we include controls, we still reduce the pain gap and we still outpredict rival_predictor. 
    """
    test_df = df.copy()
    check_is_array(yhat)
    check_is_array(rival_predictor)
    test_df['yhat'] = yhat
    test_df['rival'] = rival_predictor

    for control in all_controls:
        all_cols = control.split('*') # in case we have an interaction term. 
        for c in all_cols:
            col_name = c.replace('C(', '').replace(')', '')
            missing_data = pd.isnull(test_df[col_name])
            if missing_data.mean() > 0:
                print("warning! fraction %2.3f of control column %s is NULL; replacing with MISSING_DATA indicator" % (missing_data.mean(), col_name))
                if 'C(' in c:
                    test_df.loc[missing_data, col_name] = 'MISSING_DATA'
                else:
                    assert False

    control_string = '+'.join(all_controls)
    print("Testing whether we still outperform %s when controlling for %s" % (rival_name, ','.join(all_controls)))
    yhat_beta_with_no_controls_model = sm.OLS.from_formula('koos_pain_subscore ~ yhat', data=test_df).fit()
    yhat_beta_with_no_controls = yhat_beta_with_no_controls_model.params['yhat']
    yhat_beta_with_no_controls_model = yhat_beta_with_no_controls_model.get_robustcov_results(cov_type='cluster', groups=test_df['id'].astype(int))
    
    print("yhat beta with no controls: %2.3f" % yhat_beta_with_no_controls)
    our_model = sm.OLS.from_formula('koos_pain_subscore ~ yhat + %s' % control_string, data=test_df).fit()
    yhat_beta = our_model.params['yhat']
    our_rsquared = our_model.rsquared
    our_model = our_model.get_robustcov_results(cov_type='cluster', groups=test_df['id'].astype(int))
    print(yhat_beta_with_no_controls_model.summary())
    print(our_model.summary())


    rival_rsquared = sm.OLS.from_formula('koos_pain_subscore ~ rival + %s' % control_string, data=test_df).fit().rsquared
    yhat_from_controls_model = sm.OLS.from_formula('yhat ~ %s' % control_string, data=test_df).fit()
    assert yhat_from_controls_model.nobs == len(test_df)
    yhat_rsquared_from_controls = yhat_from_controls_model.rsquared
    print("Our r^2: %2.3f; rival r^2: %2.3f; yhat beta %2.3f; fraction of variance in yhat that controls explain %2.3f" % (our_rsquared, rival_rsquared, yhat_beta, yhat_rsquared_from_controls))

    comparisons_with_controls = []
    for k in ['race_black', 
              'binarized_education_graduated_college', 
              'binarized_income_at_least_50k']:
        test_df[k] = test_df[k] * 1.
        controls_only_model = sm.OLS.from_formula('koos_pain_subscore ~ %s + %s' % (k, control_string), data=test_df).fit()
        yhat_model = sm.OLS.from_formula('koos_pain_subscore ~ %s + yhat + %s' % (k, control_string), data=test_df).fit()
        rival_model = sm.OLS.from_formula('koos_pain_subscore ~ %s + rival + %s' % (k, control_string), data=test_df).fit()
        our_reduction = 1 - yhat_model.params[k] / controls_only_model.params[k]
        rival_reduction = 1 - rival_model.params[k] / controls_only_model.params[k]
        comparisons_with_controls.append({'var':k, 
                                        'rival':rival_name,
                                        'controls':all_controls,
                                        'ses_beta_just_controls':controls_only_model.params[k],
                                         'our_reduction':'%2.0f%%' % (100 * our_reduction), 
                                          'rival_reduction':'%2.0f%%' % (100 * rival_reduction), 
                                          'ratio':our_reduction/rival_reduction})
    return pd.DataFrame(comparisons_with_controls)[['var', 'rival', 'controls', 'ses_beta_just_controls', 'our_reduction', 'rival_reduction', 'ratio']] 

def try_ensembling(all_results, n_to_ensemble, binary_prediction):
    """
    Given a dataframe of results with test_yhats in each result, take top n_to_ensemble results and average them together. 
    Dataframe must be sorted. 
    """
    print("Ensembling results! Warning: dataframe must be sorted so yhats you like best are FIRST.")
    previous_y = None
    assert not binary_prediction
    assert len(all_results) >= n_to_ensemble
    
    all_performances = []
    all_yhats = []
    for i in range(n_to_ensemble):
        yhat = all_results.iloc[i]['test_yhat'].copy()
        y = all_results.iloc[i]['test_y'].copy()
        all_yhats.append(yhat.copy())
        
        performance = assess_performance(yhat, y, binary_prediction=binary_prediction)
        performance['model'] = i
        all_performances.append(performance)
        if i == 0:
            ensemble_yhat = yhat
            previous_y = y
        else:
            ensemble_yhat = ensemble_yhat + yhat
            assert (previous_y == y).all()
            
    for i in range(len(all_yhats)):
        for j in range(i):
            print("Correlation between yhat %i and %i: %2.3f" % (i, j, pearsonr(all_yhats[i], all_yhats[j])[0]))
    ensemble_yhat = ensemble_yhat / n_to_ensemble

    performance = assess_performance(ensemble_yhat, y, binary_prediction=binary_prediction)
    performance['model'] = 'ensemble'
    all_performances.append(performance)
    return pd.DataFrame(all_performances), ensemble_yhat
    
def get_david_education_variable(x):
    """
    Code education the way David did, for purpose of replicating his results. 
    """
    #gen byte educat = 1 if v00edcv==0 | v00edcv==1
    #replace  educat = 2 if v00edcv==2
    #replace  educat = 3 if v00edcv>=3 & v00edcv<=5
    #label define educat 1 "<=HS" 2 "Some college" 3 "College grad"
    mapping = {'0: Less than high school graduate':'1:<=HS',
                '1: High school graduate':'1:<=HS', 
               '2: Some college':'2:Some college', 
               '3: College graduate':"3:College grad", 
               '4: Some graduate school':'3:College grad', 
               '5: Graduate degree':"3:College grad"}
    return mapping[x]

def replicate_david_regressions(non_image_data, remove_missing_data):
    """
    Qualitatively replicate David's regressions. 
    Verified regression coefficients look similar. 
    Not sure exactly how he dealt with missing data so try it both ways.  
    """
    non_image_data = copy.deepcopy(non_image_data)
    
    controls_david_used = (AGE_RACE_SEX_SITE + 
                           RISK_FACTORS + 
                           BMI + 
                           TIMEPOINT_AND_SIDE + 
                           FRACTURES_AND_FALLS + 
                           KNEE_INJURY_OR_SURGERY + 
                           OTHER_PAIN)
    
    for c in sorted(list(set(controls_david_used + MRI))):
        col_name_to_use = c.replace('C(', '').replace(')', '').split(', Treatment')[0]
        if '*' in col_name_to_use:
            # this indicates an interaction term, not a column. 
            continue
        missing_data_for_col = pd.isnull(non_image_data[col_name_to_use])
        if missing_data_for_col.sum() > 0:
            if not remove_missing_data or c in OTHER_PAIN + ['fractured_hip']:
                # we never filter out missing data for a few columns because it's so often missing.
                print("Filling in missing data for proportion %2.3f values of %s" % 
                    (missing_data_for_col.mean(), col_name_to_use))
                non_image_data.loc[missing_data_for_col, col_name_to_use] = 'MISSING'
            else:
                print("removing rows with missing data for proportion %2.3f values of %s" % 
                    (missing_data_for_col.mean(), col_name_to_use))
                non_image_data = non_image_data.loc[~missing_data_for_col]
    
    # Define IV of interest. 
    non_image_data['david_education_variable'] = non_image_data['v00edcv'].map(get_david_education_variable)

    # Run the three specifications David did. 
    sp1 = sm.OLS.from_formula('womac_pain_subscore ~ %s' % '+'.join(controls_david_used + ['david_education_variable']), data=non_image_data).fit()
    
    sp1 = sp1.get_robustcov_results(cov_type='cluster', groups=non_image_data['id'].astype(int))
    
    sp2 = sm.OLS.from_formula('womac_pain_subscore ~ %s' % 
                              '+'.join(controls_david_used + ['david_education_variable', 'C(xrkl)']), data=non_image_data).fit()
    sp2 = sp2.get_robustcov_results(cov_type='cluster', groups=non_image_data['id'].astype(int))
    
    sp3 = sm.OLS.from_formula('womac_pain_subscore ~ %s' % '+'.join(controls_david_used + MRI + ['david_education_variable', 'C(xrkl)']), data=non_image_data).fit()
    param_names = sp3.params.index
    sp3 = sp3.get_robustcov_results(cov_type='cluster', groups=non_image_data['id'].astype(int))
    
    regressor_order = ([a for a in param_names if 'david_education_variable' in a] + 
                       [a for a in param_names if 'xrkl' in a] + 
                       [a for a in param_names if any([b in a for b in MRI])] + 
                       [a for a in param_names if 'knee_injury' in a] + 
                       [a for a in param_names if 'knee_surgery' in a] + 
                       [a for a in param_names if 'current_bmi' in a] + 
                       [a for a  in param_names if 'max_bmi' in a] + 
                       [a for a in param_names if 'smoker' in a] + 
                       [a for a in param_names if 'drinks' in a] + 
                       [a for a in param_names if 'fractured' in a or 'fell' in a] + 
                       [a for a in param_names if any([b in a for b in OTHER_PAIN])] + 
                       [a for a in param_names if 'age_at_visit' in a or 'p02sex' in a] + 
                       [a for a in param_names if 'p02race' in a or 'p02hisp' in a] + 
                       [a for a in param_names if 'v00maritst' in a] + 
                       [a for a in param_names if 'dominant' in a] + 
                       [a for a in param_names if 'v00site' in a] + 
                       [a for a in param_names if 'visit' in a and 'age_at_visit' not in a])

    return summary_col([sp1, sp2, sp3],
                       stars=True, 
                       regressor_order=regressor_order, 
                       info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                             'R2':lambda x: "{:.3f}".format(x.rsquared)})

def assess_what_image_features_y_and_yhat_correlate_with(non_image_data, yhat, y):
    """
    Make a plot of how strongly the image features correlate with Y and with yhat.
    """
    check_is_array(yhat)
    check_is_array(y)
    non_image_data = copy.deepcopy(non_image_data)
    assert len(non_image_data) == len(yhat)
    negative_klg = copy.deepcopy(-non_image_data['xrkl'].values)

    print("Testing how well yhat is modeled by all image features")
    non_image_data['yhat'] = yhat
    non_image_data['y'] = y
    
    yhat_model = sm.OLS.from_formula('yhat ~ %s' % ('+'.join(['C(%s)' % a for a in CLINICAL_CONTROL_COLUMNS])), 
        data=non_image_data).fit()
    print('Predicting yhat using image features. r: %2.3f; r^2 %2.3f' % (np.sqrt(yhat_model.rsquared), yhat_model.rsquared))
    yhat_hat = yhat_model.predict(non_image_data)

    # assess how well yhat_hat predicts yhat, predicts y, etc, etc. 
    performances_to_compare = []
    performances_to_compare.append(assess_performance(yhat_hat.values, yhat, binary_prediction=False))
    performances_to_compare[-1]['comparison'] = 'yhat_hat to yhat'
    performances_to_compare.append(assess_performance(yhat_hat.values, y, binary_prediction=False))
    performances_to_compare[-1]['comparison'] = 'yhat_hat to y'
    performances_to_compare.append(assess_performance(yhat, y, binary_prediction=False))
    performances_to_compare[-1]['comparison'] = 'yhat to y'
    print(pd.DataFrame(performances_to_compare))


    print("\n\nSanity check: make sure we are not wildly overfitting yhat")
    yhat_sanity_checks = []
    y_sanity_checks = []
    for iterate in range(100):
        all_ids = sorted(list(set(non_image_data['id'])))
        random.shuffle(all_ids)
        n_ids = len(all_ids)

        random_train_ids = set(all_ids[:int(n_ids * 0.6)])
        random_val_ids = set(all_ids[int(n_ids * 0.6):int(n_ids * 0.8)])
        random_test_ids = set(all_ids[int(n_ids * 0.8):])

        random_train_idxs = non_image_data['id'].map(lambda x:x in random_train_ids).values
        random_val_idxs = non_image_data['id'].map(lambda x:x in random_val_ids).values
        random_test_idxs = non_image_data['id'].map(lambda x:x in random_test_ids).values

        # compare_to_clinical_performance(train_df, val_df, test_df, y_col, features_to_use, binary_prediction, use_nonlinear_model)
        overfitting_sanity_check_yhat_hat = compare_to_clinical_performance(train_df=non_image_data.loc[random_train_idxs],
                                        val_df=non_image_data.loc[random_val_idxs], 
                                        test_df=non_image_data.loc[random_test_idxs],
                                        y_col='yhat', 
                                        features_to_use=['C(%s)' % a for a in CLINICAL_CONTROL_COLUMNS],
                                        binary_prediction=False, 
                                        use_nonlinear_model=False, 
                                        verbose=False)
        y_sanity_checks.append(assess_performance(overfitting_sanity_check_yhat_hat, y[random_test_idxs], binary_prediction=False))
        yhat_sanity_checks.append(assess_performance(overfitting_sanity_check_yhat_hat, yhat[random_test_idxs], binary_prediction=False))

    print("Performance across 100 shuffled folds of test set")
    print("yhat_hat to y")
    print(pd.DataFrame(y_sanity_checks).agg(['mean', 'std']))
    print("yhat_hat to yhat")
    print(pd.DataFrame(yhat_sanity_checks).agg(['mean', 'std']))

    ##### END OF SANITY CHECKS
    model = sm.OLS.from_formula('y ~ yhat', data=non_image_data).fit()
    print("Yhat beta without controlling for image features: %2.3f" % model.params['yhat'])
    model = model.get_robustcov_results(cov_type='cluster', groups=non_image_data['id'].astype(int))
    print(model.summary())

    model = sm.OLS.from_formula('y ~ yhat + %s' % ('+'.join(['C(%s)' % a for a in CLINICAL_CONTROL_COLUMNS])), data=non_image_data).fit()
    print("Yhat beta controlling for image features: %2.3f" % model.params['yhat'])
    model = model.get_robustcov_results(cov_type='cluster', groups=non_image_data['id'].astype(int))
    print(model.summary())


    def extract_r_squared_for_categorical_image_feature(df, image_feature, y_col):
        # Small helper method. Since some image features are categorical, look at total amount of variance feature explains
        # when we use it as a categorical variable. 
        image_model = sm.OLS.from_formula('%s ~ C(%s)' % (y_col, image_feature), data=df).fit()
        return image_model.rsquared
    all_correlations = []
    for feature in CLINICAL_CONTROL_COLUMNS:
        y_rsquared = extract_r_squared_for_categorical_image_feature(non_image_data, feature, 'y')
        yhat_rsquared = extract_r_squared_for_categorical_image_feature(non_image_data, feature, 'yhat')
        negative_klg_rsquared = extract_r_squared_for_categorical_image_feature(non_image_data, feature, 'xrkl')
        all_correlations.append({'feature':feature, 
                                'yhat_rsquared':yhat_rsquared, 
                                'y_rsquared':y_rsquared, 
                                'negative_klg_rsquared':negative_klg_rsquared})
    all_correlations = pd.DataFrame(all_correlations)
    all_correlations['klg_rsquared - yhat_rsquared'] = all_correlations['negative_klg_rsquared'] - all_correlations['yhat_rsquared']
    print("Correlations sorted by klg_rsquared - yhat_rsquared")
    print(all_correlations.sort_values(by='klg_rsquared - yhat_rsquared'))
    for var in ['yhat', 'y', 'negative_klg']:
        print("Average r^2 between %s and features besides KLG: %2.3f" % 
        (var, all_correlations.loc[all_correlations['feature'] != 'xrkl', '%s_rsquared' % var].mean()))

    
    plt.figure(figsize=[8, 8])
    for i in range(len(all_correlations)):
        plt.annotate(all_correlations['feature'].iloc[i], 
                     [all_correlations['y_rsquared'].iloc[i], all_correlations['yhat_rsquared'].iloc[i]])
    plt.scatter(all_correlations['y_rsquared'], all_correlations['yhat_rsquared'])
    
    min_val = 0
    max_val = max(all_correlations['y_rsquared'].max(), all_correlations['yhat_rsquared'].max()) + .005
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--')
    plt.plot([min_val, max_val], [0, 0], color='black')
    plt.plot([0, 0], [min_val, max_val], color='black')
    plt.xlabel("Feature r^2 with y")
    plt.ylabel("Feature r^2 with yhat")
    plt.show()
    return pd.DataFrame(all_correlations.sort_values(by='y_rsquared'))

def fit_followup_regression(combined_df, col_is_categorical):    
    if col_is_categorical:
        t0_control = 'C(col_of_interest_t0)'
    else:
        t0_control = 'col_of_interest_t0'
    col_of_interest_model = sm.OLS.from_formula('col_of_interest_t1 ~ %s' % t0_control, data=combined_df).fit()
    yhat_model = sm.OLS.from_formula('col_of_interest_t1 ~ yhat_t0', data=combined_df).fit()
    
    combined_model = sm.OLS.from_formula('col_of_interest_t1 ~ yhat_t0 + %s' % t0_control, data=combined_df).fit()
    assert combined_model.nobs == yhat_model.nobs == col_of_interest_model.nobs

    clustered_combined_model = combined_model.get_robustcov_results(cov_type='cluster', 
                                                                    groups=combined_df['id'].astype(int))
    yhat_t0_index = list(combined_model.params.index).index('yhat_t0')
    yhat_t0_pval = clustered_combined_model.pvalues[yhat_t0_index]
    assert np.allclose(clustered_combined_model.params[yhat_t0_index], combined_model.params['yhat_t0'])
    return {'col_t1 ~ yhat_t0 r^2':yhat_model.rsquared, 
                'col_t1 ~ col_t0 r^2':col_of_interest_model.rsquared,
                'col_t1 ~ yhat_t0 + col_t0 r^2':combined_model.rsquared,
                'yhat beta':'%2.5f (%2.5f, %2.5f)' % (combined_model.params['yhat_t0'], 
                                                      combined_model.conf_int().loc['yhat_t0', 0], 
                                                      combined_model.conf_int().loc['yhat_t0', 1]),
                'yhat p':yhat_t0_pval,
                'n_obs':int(combined_model.nobs), 
                'n_people':len(set(combined_df['id']))}

def fit_followup_binary_regression(combined_df):
    """
    If col_of_interest is a binary variable, fit a logistic regression instead. 
    col_of_interest here is binarized pain. 
    """
    assert combined_df['col_of_interest_t0'].map(lambda x:x in [0, 1]).all()
    assert combined_df['col_of_interest_t1'].map(lambda x:x in [0, 1]).all()
    assert not combined_df['koos_pain_subscore_t0'].map(lambda x:x in [0, 1]).all()
    assert not combined_df['koos_pain_subscore_t1'].map(lambda x:x in [0, 1]).all()
    assert not (combined_df['col_of_interest_t1'] == combined_df['col_of_interest_t0']).all()

    # predict binary pain at followup without any controls. 
    yhat_model = sm.Logit.from_formula('col_of_interest_t1 ~ yhat_t0', data=combined_df).fit(
        cov_type='cluster', cov_kwds={'groups':combined_df['id'].astype(int).values})

    # predict binary pain at followup controlling for binary pain at t0. 
    combined_model_binary_control = sm.Logit.from_formula('col_of_interest_t1 ~ yhat_t0 + col_of_interest_t0', data=combined_df).fit(
    cov_type='cluster', cov_kwds={'groups':combined_df['id'].astype(int).values})

    # predict binary pain at followup controlling for CONTINUOUS pain at t0. 
    combined_model_continuous_control = sm.Logit.from_formula('col_of_interest_t1 ~ yhat_t0 + koos_pain_subscore_t0', data=combined_df).fit(
        cov_type='cluster', cov_kwds={'groups':combined_df['id'].astype(int).values})

    get_OR_and_CI = lambda m:'%2.3f (%2.3f, %2.3f)' % (np.exp(m.params['yhat_t0']), np.exp(m.conf_int().loc['yhat_t0', 0]), np.exp(m.conf_int().loc['yhat_t0', 1]))

    return {'OR (no control)':get_OR_and_CI(yhat_model), 
            'OR (binary control)':get_OR_and_CI(combined_model_binary_control),
            'OR (continuous control)':get_OR_and_CI(combined_model_continuous_control)}
    

def predict_kl_at_future_timepoints(non_image_data, yhat, use_binary_pain=False):
    """
    Given the non-image data and an array of yhats, check how well 
    yhat_t0 predicts followup values of col_of_interest at t1, where col_of_interest = ['xrkl', 'koos_pain_subscore']
    """
    check_is_array(yhat)
    non_image_data = copy.deepcopy(non_image_data)
    assert 'yhat' not in non_image_data.columns
    non_image_data['yhat'] = yhat
    if not use_binary_pain:
        cols_to_use = ['xrkl', 'koos_pain_subscore']
        statistics_to_return = ['col', 't0', 't1', 'col_t1 ~ yhat_t0 r^2', 'col_t1 ~ col_t0 r^2', 
        'col_t1 ~ yhat_t0 + col_t0 r^2', 'yhat beta', 'yhat p', 'n_obs', 'n_people']
    else:
        cols_to_use = ['binarized_koos_pain_subscore']
        non_image_data['binarized_koos_pain_subscore'] = binarize_koos(non_image_data['koos_pain_subscore'].values)
        statistics_to_return = ['col', 't0', 't1', 'OR (no control)', 'OR (binary control)', 'OR (continuous control)']

    t0 = '00 month follow-up: Baseline'
    all_results = []
    
    for col_of_interest in cols_to_use:
        pooled_dfs = []
        for t1 in ['12 month follow-up', '24 month follow-up', '36 month follow-up', '48 month follow-up']:
            if t1 <= t0:
                continue
            df_t0 = copy.deepcopy(non_image_data.loc[non_image_data['visit'] == t0, ['id', 'side', 'yhat', col_of_interest, 'koos_pain_subscore', 'xrkl']])
            df_t1 = copy.deepcopy(non_image_data.loc[non_image_data['visit'] == t1, ['id', 'side', 'yhat', col_of_interest, 'koos_pain_subscore', 'xrkl']])
            df_t0.columns = ['id', 'side', 'yhat', 'col_of_interest', 'koos_pain_subscore', 'xrkl']
            df_t1.columns = ['id', 'side', 'yhat', 'col_of_interest', 'koos_pain_subscore', 'xrkl']
            
            
            assert len(df_t0) > 0
            assert len(df_t1) > 0
            assert df_t0[['side', 'id']].duplicated().sum() == 0
            assert df_t1[['side', 'id']].duplicated().sum() == 0
            assert len(df_t0[['yhat', 'col_of_interest']].dropna()) == len(df_t0)
            assert len(df_t1[['yhat', 'col_of_interest']].dropna()) == len(df_t1)
            combined_df = pd.merge(df_t0, df_t1, how='inner', on=['id', 'side'], suffixes=['_t0', '_t1'])
            if use_binary_pain:
                regression_results = fit_followup_binary_regression(combined_df)
            else:
                regression_results = fit_followup_regression(combined_df, col_is_categorical=(col_of_interest == 'xrkl'))
            regression_results['col'] = col_of_interest
            regression_results['t0'] = t0
            regression_results['t1'] = t1
            all_results.append(regression_results)
            pooled_dfs.append(combined_df)
        if use_binary_pain:
            regression_results = fit_followup_binary_regression(pd.concat(pooled_dfs))
        else:
            regression_results = fit_followup_regression(pd.concat(pooled_dfs), col_is_categorical=(col_of_interest == 'xrkl'))

        regression_results['col'] = col_of_interest
        regression_results['t0'] = t0
        regression_results['t1'] = 'pooled'
        all_results.append(regression_results)

    
    return pd.DataFrame(all_results)[statistics_to_return]

def quantify_pain_gap_reduction_vs_rival(yhat, y, rival_severity_measure, all_ses_vars, ids, lower_bound_rival_reduction_at_0=False):
    """
    Given a rival_severity_measure, return a dataframe of how much yhat reduces the pain gap as compared to the rival. 
    """
    check_is_array(yhat)
    check_is_array(y)
    check_is_array(rival_severity_measure)
    check_is_array(ids)

    all_beta_ratios = []
    for ses_var_name in all_ses_vars:
        ses_arr = all_ses_vars[ses_var_name].copy() * 1.
        assert set(ses_arr) == set([0, 1])
        check_is_array(ses_arr)

        #print("\n\n****Stratification: %s" % ses_var_name)
        ses_betas = compare_ses_gaps(yhat=yhat, 
                                              y=y, 
                                              rival_severity_measure=rival_severity_measure, 
                                              ses=ses_arr, 
                                              verbose=False)
        ses_beta_ratio_yhat_rival = ses_betas['yhat_ses_beta'] / ses_betas['rival_ses_beta']
        ses_beta_ratio_rival_nothing = ses_betas['rival_ses_beta'] / ses_betas['no_controls_ses_beta'] # used to be called rival_nothing_ratio
        ses_beta_ratio_yhat_nothing = ses_betas['yhat_ses_beta'] / ses_betas['no_controls_ses_beta'] # used to be called no_controls_beta_ratio
        if lower_bound_rival_reduction_at_0 and (1 - ses_beta_ratio_rival_nothing) < 0.001:
            print("Warning: rival actually makes pain gap larger, setting yhat:rival reduction ratio to 100.")
            yhat_rival_red_ratio = 100
        else:
            yhat_rival_red_ratio = (1 - ses_beta_ratio_yhat_nothing)/(1 - ses_beta_ratio_rival_nothing)
        all_beta_ratios.append({'SES var':ses_var_name, 
                                'n_obs':'%i/%i' % (sum(ses_arr == 1), len(ses_arr)),
                                'n_people':'%i/%i' % (len(set(ids[ses_arr == 1])), len(set(ids))),
                                'No controls gap':ses_betas['no_controls_ses_beta'], 
                                'Rival gap':ses_betas['rival_ses_beta'],
                                'yhat gap':ses_betas['yhat_ses_beta'],
                                'Rival red. vs nothing':(1 - ses_beta_ratio_rival_nothing) * 100, # '%2.0f%%' % 
                                'red. vs rival': ((1 - ses_beta_ratio_yhat_rival) * 100),# '%2.0f%%' %
                                'red. vs nothing': ((1 - ses_beta_ratio_yhat_nothing) * 100),  # '%2.0f%%' 
                                'yhat/rival red. ratio': yhat_rival_red_ratio})# '%2.1f'
    return pd.DataFrame(all_beta_ratios)[['SES var', 'No controls gap', 'Rival gap', 'yhat gap', 
                                   'Rival red. vs nothing', 'red. vs nothing', 'yhat/rival red. ratio', 
                                   'n_obs', 'n_people']]

def compare_ses_gaps(yhat, y, rival_severity_measure, ses, verbose=True):
    """
    How big are the SES/racial gaps controlling for various measures of severity? 
    """
    check_is_array(yhat)
    check_is_array(y)
    check_is_array(ses)
    check_is_array(rival_severity_measure)
    assert set(ses) == set([0, 1])
    ses = ses.copy() * 1.

    df = pd.DataFrame({'yhat':yhat, 'y':y, 'rival_severity_measure':rival_severity_measure, 'ses':ses})
    
    if verbose:
        print("Comparing pain gaps")
    no_controls_model = sm.OLS.from_formula('y ~ ses', data=df).fit()
    rival_model = sm.OLS.from_formula('y ~ rival_severity_measure + ses', data=df).fit()
    yhat_model = sm.OLS.from_formula('y ~ yhat + ses', data=df).fit()
    no_controls_beta = float(no_controls_model.params['ses'])
    rival_ses_beta = float(rival_model.params['ses'])
    yhat_ses_beta = float(yhat_model.params['ses'])
    
    if verbose:
        print(klg_model.summary())
        print(yhat_model.summary())

    results = {'yhat_ses_beta': yhat_ses_beta, 
    'rival_ses_beta':rival_ses_beta, 
    'no_controls_ses_beta':no_controls_beta}

    return results

def compare_pain_levels_for_people_geq_klg_2(yhat, y, klg, ses, y_col):
    """
    compute various algorithmic-fairness-inspired metrics 
    based on binarizing both the risk score and the outcome. 
    """
    check_is_array(yhat)
    check_is_array(klg)
    check_is_array(ses)
    check_is_array(y)
    assert y_col == 'koos_pain_subscore'
    assert y.max() == 100

    # first compute our three measures of severity
    # 1. binarized KLG
    # 2. binarized yhat
    # 3. binarized y (oracle)

    binarized_klg_severity = klg >= 2
    print("%i/%i people with KLG >= 2" % (binarized_klg_severity.sum(), len(binarized_klg_severity)))

    discretized_yhat = discretize_yhat_like_kl_grade(yhat_arr=yhat, kl_grade_arr=klg, y_col=y_col)
    binarized_yhat_severity = discretized_yhat >= 2
    discretized_y = discretize_yhat_like_kl_grade(yhat_arr=y, kl_grade_arr=klg, y_col=y_col)
    binarized_oracle_severity = discretized_y >= 2
    severity_measures = {'klg':binarized_klg_severity, 
                         'yhat':binarized_yhat_severity,
                         'oracle':binarized_oracle_severity}

    assert binarized_oracle_severity.sum() == binarized_yhat_severity.sum()
    assert binarized_klg_severity.sum() == binarized_yhat_severity.sum()

    # cast a couple of things to bools
    high_pain = binarize_koos(y).astype(bool)
    assert (high_pain == (binarize_koos(y) == 1)).all() 
    assert set(ses) == set([0, 1])
    ses = ses == 1

    print("Overall fraction of people with SES var=1: %2.3f" % ses.mean())

    for severity_measure_name in ['klg', 'yhat', 'oracle']:
        print("p(SES var=1|high severity), severity measure %s: %2.3f" % (
            severity_measure_name, ses[severity_measures[severity_measure_name] == 1].mean()))


    # alternate computation as sanity check. 

    percentile_cutoff = 100. * (klg >= 2).sum() / len(klg)
    y_cutoff =  scoreatpercentile(y, percentile_cutoff) 
    yhat_cutoff =  scoreatpercentile(yhat, percentile_cutoff) 

    print('Alternate computation using KLG with %i people above severity cutoff: %2.3f' % 
        ((klg >= 2).sum(), ses[klg >= 2].mean()))
    print('Alternate computation using yhat with %i people above severity cutoff: %2.3f' % 
        ((yhat <= yhat_cutoff).sum(), ses[yhat <= yhat_cutoff].mean()))
    print('Alternate computation using oracle with %i people above severity cutoff: %2.3f' % 
        ((y <= y_cutoff).sum(), ses[y <= y_cutoff].mean()))

    all_klg_yhat_ratios = []
    for threshold in [0, 1, 2, 3, 4]:
        yhat_geq_thresh = discretized_yhat >= threshold
        klg_geq_thresh = klg >= threshold
        assert yhat_geq_thresh.sum() == klg_geq_thresh.sum()
        print("Threshold %i: %-5i >= threshold. p(SES var = 1 | >= threshold): %2.3f for yhat, %2.3f for KLG, ratio %2.5f" % 
                (threshold, 
                yhat_geq_thresh.sum(), 
                ses[yhat_geq_thresh].mean(), 
                ses[klg_geq_thresh].mean(), 
                ses[yhat_geq_thresh].mean() / ses[klg_geq_thresh].mean()))
        all_klg_yhat_ratios.append(ses[yhat_geq_thresh].mean() / ses[klg_geq_thresh].mean())

    for threshold in [0, 1, 2, 3, 4]:
        yhat_geq_thresh = discretized_yhat >= threshold
        klg_geq_thresh = klg >= threshold
        assert yhat_geq_thresh.sum() == klg_geq_thresh.sum()
        print("Threshold %i: %-5i above threshold. p(above threshold | SES var = 1): %2.3f for yhat, %2.3f for KLG, ratio %2.3f" % 
                (threshold, 
                yhat_geq_thresh.sum(), 
                yhat_geq_thresh[ses == 1].mean(), 
                klg_geq_thresh[ses == 1].mean(), 
                yhat_geq_thresh[ses == 1].mean() / klg_geq_thresh[ses == 1].mean()))
        assert np.allclose(yhat_geq_thresh[ses == 1].mean() / klg_geq_thresh[ses == 1].mean(), all_klg_yhat_ratios[threshold])



    # compute actual results
    results = {'all_klg_yhat_ratios':all_klg_yhat_ratios}
    for ses_val in [False, True, 'all']:
        ses_key = 'ses_var=%s' % ses_val
        results[ses_key] = {}
        if ses_val in [False, True]:
            ses_idxs = ses == ses_val
        elif ses_val == 'all':
            ses_idxs = np.array([True for k in range(len(ses))])
        for severity_measure_name in severity_measures:
            high_severity = severity_measures[severity_measure_name]
            assert set(high_severity) == set([0, 1])

            p_high_severity = high_severity[ses_idxs].mean()
            p_high_severity_given_high_pain = high_severity[ses_idxs & high_pain].mean()
            p_low_severity_given_low_pain = 1 - high_severity[ses_idxs & (~high_pain)].mean()
            p_high_pain_given_high_severity = high_pain[ses_idxs & high_severity].mean()
            p_low_pain_given_low_severity = 1 - high_pain[ses_idxs & (~high_severity)].mean()
            correct = high_severity == high_pain
            accuracy = correct[ses_idxs].mean()

            results[ses_key][severity_measure_name] = {
                'p_high_severity':p_high_severity, 
                'p_high_severity_given_high_pain':p_high_severity_given_high_pain, 
                'p_low_severity_given_low_pain':p_low_severity_given_low_pain, 
                'p_high_pain_given_high_severity':p_high_pain_given_high_severity, 
                'p_low_pain_given_low_severity':p_low_pain_given_low_severity, 
                'accuracy':accuracy}
    return results

def plot_geq_klg_results(geq_klg_2_results, plot_title, figname=None):
    """
    Makes a plot of the results from compare_pain_levels_for_people_geq_klg_2. 
    """
    def make_metric_name_pretty(a):
        a = a.replace('_given_', '|')
        if a[:2] == 'p_':
            a = 'p(' + a[2:] + ')'
        return a
    metrics_to_plot = sorted(geq_klg_2_results['ses_var=True']['klg'].keys())
    severity_measures = ['klg', 'yhat', 'oracle']
    plt.figure(figsize=[15, 10])
    for subplot_idx, ses_status in enumerate([True, False, 'all']):
        plt.subplot(3, 1, subplot_idx + 1)
        x_offset = 0
        bar_width = .2
        for severity_measure in severity_measures:
            xs = np.arange(len(metrics_to_plot)) + x_offset
            ys = [geq_klg_2_results['ses_var=%s' % ses_status][severity_measure][metric] 
                  for metric in metrics_to_plot]
            plt.bar(xs, ys, width=bar_width, label=severity_measure)
            x_offset += bar_width
            plt.ylim([0, 1])
        plt.xticks(np.arange(len(metrics_to_plot)) + bar_width, 
                   [make_metric_name_pretty(a) for a in metrics_to_plot])
        plt.legend()
        plt.title('SES var = %s' % ses_status)
    plt.suptitle(plot_title)
    if figname is not None:
        plt.savefig(figname, dpi=300)

def make_comparison_klg_yhat_plot(yhat, y, klg):
    """
    Compare how well yhat and KLG fit y by plotting distribution of y in each bin. 
    Make a box plot to show this and also do a simple line plot with the median of y.
    Checked. 
    """
    check_is_array(yhat)
    check_is_array(y)
    check_is_array(klg)
     
    discretized_yhat = discretize_yhat_like_kl_grade(yhat_arr=yhat,
                                                          kl_grade_arr=klg,
                                                          y_col='koos_pain_subscore')
    discretized_vals = range(5)
    
    assert set(klg) == set(discretized_vals)
    assert y.max() == 100
    assert set(discretized_yhat) == set(discretized_vals)
    print('pearson correlation between our original score and y %2.3f' % pearsonr(yhat, y)[0]) 
    print('pearson correlation between our discretized score and y %2.3f' % pearsonr(discretized_yhat, y)[0]) 
    print('pearson correlation between klg and y %2.3f' % pearsonr(klg, y)[0])

    # box plot. 
    plt.figure(figsize=[8, 4])
    ylimits = [0, 102]
    plt.subplot(121)
    sns.boxplot(x=discretized_yhat, y=y)
    plt.xlabel("Discretized $\hat y$")
    plt.ylabel('Koos pain score')
    plt.ylim(ylimits)
    plt.subplot(122)
    sns.boxplot(x=klg, y=y)
    plt.yticks([])
    plt.xlabel("KLG")
    plt.ylim(ylimits)
    plt.savefig('sendhil_plots/klg_yhat_comparison_boxplot.png', dpi=300)
    plt.show()
    
    # plot median value of y by each KLG/yhat bin. 
    klg_y_medians = []
    yhat_y_medians = []
    for val in discretized_vals:
        yhat_idxs = discretized_yhat == val
        klg_idxs = klg == val
        print("score %i: yhat and KLG means: %2.3f %2.3f; yhat and KLG medians %2.3f %2.3f" % 
              (val, y[yhat_idxs].mean(), y[klg_idxs].mean(), np.median(y[yhat_idxs]), np.median(y[klg_idxs])))
        klg_y_medians.append(np.median(y[klg_idxs]))
        yhat_y_medians.append(np.median(y[yhat_idxs]))
    plt.figure(figsize=[5, 4])
    plt.plot(discretized_vals, yhat_y_medians, label='Our model', color='green')
    plt.plot(discretized_vals, klg_y_medians, label='KLG', color='red')
    plt.legend()
    plt.xticks(discretized_vals)
    plt.ylabel("Median Koos pain score")
    plt.xlabel("Severity grade")
    #plt.savefig('sendhil_plots/klg_yhat_comparison_line_plot.png', dpi=300)
    plt.show()
    
    
def make_kernel_regression_plot(yhat, y):
    """
    Kernel regression plot of y on yhat. 
    Also plot linear trend for comparison. 
    Checked. 
    """

    check_is_array(yhat)
    check_is_array(y)
    # fit RBF kernel. 
    kernel_model = KernelRidge(kernel='rbf', gamma=1/(15.**2)) # bandwidth of 15. 
    kernel_model.fit(X=yhat.reshape(-1, 1), y=y)
    vals_to_predict_at = np.arange(yhat.min(), yhat.max(), 1).reshape(-1, 1)
    kernel_predictions = kernel_model.predict(vals_to_predict_at)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=yhat, y=y)
    
    # make plot. Both linear + kernel fit for comparison. 
    plt.figure(figsize=[4, 4])
    plt.scatter(yhat, y, s=1, alpha=.7)
    plt.plot(vals_to_predict_at, vals_to_predict_at*slope + intercept, color='red', linewidth=3, label='linear fit')
    plt.plot(vals_to_predict_at, kernel_predictions, color='black', linewidth=3, label='kernel fit')
    plt.legend()
    plt.xlabel('$\hat y$')
    plt.ylabel('Koos pain score')
    plot_limits = [45, 102]
    plt.xlim(plot_limits)
    plt.ylim(plot_limits)
    print("Warning: %2.5f%% of y and %2.5f%% of yhat vals are truncated by lower limit of %s" % 
          ((y < plot_limits[0]).mean(), (yhat < plot_limits[0]).mean(), plot_limits[0]))
    plt.subplots_adjust(left=.2)
    plt.savefig('sendhil_plots/kernel_regression.png', dpi=300)
    plt.show()
    
def make_violin_nonredundancy_plot(yhat, klg):
    """
    Make violin plot showing lack of correlation between yhat and KLG. 
    Checked. 
    """
    check_is_array(yhat)
    check_is_array(klg)
    plt.figure(figsize=[4, 4])
    sns.violinplot(x=klg, y=yhat)
    
    assert set(klg) == set(range(5))
    plt.xticks(range(5), ['0', '1', '2', '3', '4'])
    plt.ylim([50, 100])
    plt.ylabel("$\hat y$")
    plt.xlabel('KLG')
    plt.subplots_adjust(left=.2)
    plt.savefig('violin_nonredundancy_plot.png', dpi=300)
    plt.show()

def generate_correlated_variable(x, desired_r):
    """
    This generates a variable with a given level of correlation desired_r with x. 
    Checked. 
    """
    tolerance = .003
    max_iterations = 1000
    assert desired_r > 0
    assert desired_r < 1
    upper_noise = np.std(x) * 100
    lower_noise = 0
    n = len(x)
    n_iter = 0
    while True:
        middle_noise = (upper_noise + lower_noise) / 2
        y = x + np.random.randn(n) * middle_noise
        r, p = pearsonr(y, x)
        print("%2.3f - %2.3f: %2.3f" % (lower_noise, upper_noise, r))
        if r < desired_r - tolerance:
            upper_noise = middle_noise
        elif r > desired_r + tolerance:
            lower_noise = middle_noise
        else:
            print("Within tolerance: %2.3f" % np.abs(r - desired_r))
            return y
        n_iter += 1
        if n_iter > max_iterations:
            return y

def get_baseline_scores(df):
    """
    filter for left knee scores at baseline. 
    Checked. 
    """
    idxs = (df['side'] == 'left') & (df['visit'] == '00 month follow-up: Baseline')
    df = copy.deepcopy(df.loc[idxs])
    df.index = range(len(df))
    return df

def power_analysis_is_y_associated_with_yhat(non_image_dataset, pval_thresh, n_iterates):
    """
    power analysis: is y associated with yhat when we do/don't control for covariates? 
    Checked. 
    """
    clinical_controls = ['C(%s)' % col for col in non_image_dataset.clinical_xray_semiquantitative_cols]
    knee_pain_scores = non_image_dataset.processed_dataframes['all_knee_pain_scores']
    clinical_assessments = non_image_dataset.processed_dataframes['kxr_sq_bu']
    baseline_pain_scores = get_baseline_scores(get_combined_dataframe(non_image_dataset, clinical_assessments))

    pain_subscores = ['koos_pain_subscore', 'womac_pain_subscore']
    for k in pain_subscores:
        baseline_pain_scores[k] = (baseline_pain_scores[k] - baseline_pain_scores[k].mean()) / baseline_pain_scores[k].std()

    all_simulations = []
    for desired_r in [.4, .3, .2, .15, .1, .08, .05]:
        for k in pain_subscores:
            
            baseline_pain_scores['pain_hat'] = generate_correlated_variable(baseline_pain_scores[k], desired_r)
            r, p = pearsonr(baseline_pain_scores['pain_hat'], baseline_pain_scores[k])
            
            for subset_size in list(range(250, 2000, 250)):
                for _ in range(n_iterates):
                    individuals_to_sample = set(random.sample(non_image_dataset.all_ids, subset_size))
                    df_to_use = baseline_pain_scores.loc[baseline_pain_scores['id'].map(lambda x:x in individuals_to_sample)]
                    for rhs in ['just_pain_hat', 'pain_hat_plus_clinical_controls']:
                        if rhs == 'just_pain_hat':
                            covs_to_use = ['pain_hat']
                        elif rhs == 'pain_hat_plus_clinical_controls':
                            covs_to_use = ['pain_hat'] + clinical_controls
                        covs_to_use = [cov for cov in covs_to_use if len(set(df_to_use[cov].dropna())) > 1]
                        covs_to_use = '+'.join(covs_to_use)
                        model = sm.OLS.from_formula('%s ~ %s' % (k, covs_to_use), data=df_to_use).fit()
                        #print(model.summary())
                        model_pval = model.pvalues['pain_hat']
                        all_simulations.append({'pain_subscore':k, 
                                               'r':r, 
                                             'rhs':rhs,
                                               'subset_size':subset_size, 
                                               'model_pval':model_pval})

    # now make the plot. 
    all_simulations = pd.DataFrame(all_simulations)
    all_simulations['is_sig'] = all_simulations['model_pval'] < pval_thresh

    for pain_subscore in ['koos_pain_subscore', 'womac_pain_subscore']:
        for rhs in sorted(list(set(all_simulations['rhs']))):

            simulations_to_use = all_simulations.loc[(all_simulations['pain_subscore'] == pain_subscore) & 
                                                     (all_simulations['rhs'] == rhs)]
            plt.figure(figsize=[8, 4])
            for r in sorted(list(set(simulations_to_use['r']))):
                x = []
                y = []
                for subset_size in sorted(list(set(simulations_to_use['subset_size']))):
                    simulations_for_subset = simulations_to_use.loc[(simulations_to_use['subset_size'] == subset_size) & 
                                                      (simulations_to_use['r'] == r), 'is_sig']

                    assert len(simulations_for_subset) == n_iterates

                    x.append(subset_size)
                    y.append(simulations_for_subset.mean())
                plt.plot(x, y, label='$r=%2.2f$' % r)
            plt.title("DV: %s\nRHS: %s" % (pain_subscore, rhs))
            plt.xlabel("Test set size")
            plt.ylabel("Fraction of simulations\nwhich are significant @ $p = %2.2f$" % pval_thresh)
            plt.xlim([min(x), max(x)])
            plt.ylim([0, 1.02])
            plt.legend(bbox_to_anchor=(1.1, 1.05))
            plt.savefig('power_analysis_dv_%s_rhs_%s.png' % (pain_subscore, rhs), dpi=300)


def power_analysis_does_yhat_reduce_effect_of_income(non_image_dataset, pval_thresh, n_iterates, dv):
    """
    power analysis for "does yhat reduce effect of income/education"

    for the answer to "does the coefficient change", see
    Clogg, C. C., Petkova, E., & Haritou, A. (1995). Statistical methods for comparing regression coefficients between models. American Journal of Sociology, 100(5), 1261-1293. 
    Bottom of page 1269 gives an example which makes it look like this is what we want to do.
    See eq (15) and also the paragraph which begins "results begin in Table 1"

    Andrew Gelman fwiw is skeptical of this -- https://andrewgelman.com/2010/01/21/testing_for_sig/
    Checked.
    """

    all_results = []
    assert dv in ['koos_pain_subscore', 'womac_pain_subscore']
    knee_pain_scores = non_image_dataset.processed_dataframes['all_knee_pain_scores']
    clinical_ratings = non_image_dataset.processed_dataframes['kxr_sq_bu']
    df_to_use = get_baseline_scores(get_combined_dataframe(non_image_dataset, clinical_ratings))    
    print("Length of baseline data")
    print(len(df_to_use))
    iv = 'binarized_income_at_least_50k'

    pain_subscores = ['koos_pain_subscore', 'womac_pain_subscore']
    assert dv in pain_subscores
    for k in pain_subscores:
        df_to_use[k] = (df_to_use[k] - df_to_use[k].mean()) / df_to_use[k].std()

    clinical_controls = '+'.join(['C(%s)' % a for a in non_image_dataset.clinical_xray_semiquantitative_cols])

    for noise_param in [3, 5, 8, 10]:
        for disparity_param in [.2]:
            print("Noise param: %2.3f; disparity param: %2.3f" % (noise_param, disparity_param))
            # as disparity param increases in magnitude, yhat gets more correlated with SES. 
            # as noise_param increases in magnitude, yhat gets less correlated with y and SES. 

            if dv == 'womac_pain_subscore':
                # higher scores indicate worse pain on the womac
                # so if you have a higher SES we want you to have lower predicted Yhat. 
                disparity_param = -disparity_param
            df_to_use['yhat'] = df_to_use[dv] + df_to_use[iv] * disparity_param + noise_param * np.random.randn(len(df_to_use),)
            df_to_use = df_to_use.dropna(subset=[dv, iv])
            print(df_to_use[[iv, 'yhat']].groupby(iv).agg(['mean', 'std']))
            for subset_size in list(range(250, 2000, 250)):
                for _ in range(n_iterates):
                    people = set(random.sample(non_image_dataset.all_ids, subset_size))
                    people_idxs = df_to_use['id'].map(lambda x:x in people).values

                    model_without_yhat = sm.OLS.from_formula('%s ~ %s + %s' % (dv, iv, clinical_controls), df_to_use.loc[people_idxs]).fit()
                    model_with_yhat = sm.OLS.from_formula('%s ~ %s + %s + yhat' % (dv, iv, clinical_controls), df_to_use.loc[people_idxs]).fit()


                    change_in_iv_coef = model_with_yhat.params[iv] - model_without_yhat.params[iv]
                    # Note: 
                    ## To get estimate of noise variance for a model, the following 3 are all the same.
                    # this is sigma_hat SQUARED, not sigma_hat.
                    # 1. np.sum(model_without_yhat.resid ** 2) / model_without_yhat.df_resid)
                    # 2. model_without_yhat.mse_resid
                    # 3. model_without_yhat.scale

                    squared_error_on_change = (model_with_yhat.bse[iv] ** 2 - 
                                          model_without_yhat.bse[iv] ** 2 * model_with_yhat.scale / model_without_yhat.scale)
                    assert squared_error_on_change > 0
                    error_on_change = np.sqrt(squared_error_on_change)
                    zscore = change_in_iv_coef/error_on_change
                    if (model_with_yhat.params[iv] > 0) != (model_without_yhat.params[iv] > 0): 
                        # if the sign of the coefficient changes that is weird. It should just get smaller. 
                        print("Warning: coefficient changed sign from %2.3f to %2.3f" % (model_without_yhat.params[iv], model_with_yhat.params[iv]))
                    results = {'r2_with_yhat':model_with_yhat.rsquared, 
                               'r2_without_yhat':model_without_yhat.rsquared, 
                               'beta_with_yhat':model_with_yhat.params[iv], 
                               'beta_without_yhat':model_without_yhat.params[iv], 
                               'change_in_IV_coef':change_in_iv_coef, 
                               'error_on_change':error_on_change, 
                               'zscore':zscore, 
                               'p_change':2*(1 - norm.cdf(abs(zscore))), # two-tailed p-value. 
                               'yhat_iv_corr':pearsonr(df_to_use['yhat'], df_to_use[iv])[0], 
                               'yhat_dv_corr':pearsonr(df_to_use['yhat'], df_to_use[dv])[0], 
                               'subset_size':subset_size}
                    all_results.append(results)
    
    # now make plot. 
    all_results = pd.DataFrame(all_results)
    for iv_corr in sorted(list(set(all_results['yhat_iv_corr'])), key=lambda x:abs(x)):
        for dv_corr in sorted(list(set(all_results['yhat_dv_corr'])), key=lambda x:abs(x)):
            idxs = ((all_results['yhat_iv_corr'] == iv_corr) & 
                    (all_results['yhat_dv_corr'] == dv_corr))
            if idxs.sum() == 0:
                continue
            x = []
            y = []
            for subset_size in sorted(list(set(all_results['subset_size']))):
                x.append(subset_size)
                results = all_results.loc[idxs & (all_results['subset_size'] == subset_size), 
                                         'p_change'] < pval_thresh
                assert len(results) == n_iterates
                y.append(results.mean())
                
            plt.plot(x, y, label='IV (income/educ): r=%2.3f, DV (pain): r=%2.3f' % (abs(iv_corr), 
                                                                                    abs(dv_corr)))
        
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.xlabel("Test set size")
    plt.ylabel("Fraction of simulations\nwhich are significant @ $p = %2.2f$" % pval_thresh)
    plt.title("Significance test is for change in income/educ coef")
    plt.savefig('power_analysis_does_yhat_reduce_effect_of_ses.png')

def discretize_yhat_like_kl_grade(yhat_arr, kl_grade_arr, y_col):
    """
    Given an array of yhats and an arry of kl grades, create a discretized yhat with the same bin size. 
    y_col argument is just to make sure we are intending to recreate koos pain score. 
    In this case, highest yhats should have the lowest KL grades. 
    """
    check_is_array(yhat_arr)
    check_is_array(kl_grade_arr)
    yhat_arr = copy.deepcopy(yhat_arr)
    kl_grade_arr = copy.deepcopy(kl_grade_arr)

    yhat_arr = yhat_arr + np.random.random(len(yhat_arr)) * 1e-6 # break ties by adding a small amount of random noise. 

    assert y_col == 'koos_pain_subscore'
    kl_counts = Counter(kl_grade_arr)
    print("KL counts are", kl_counts)
    total_count = len(kl_grade_arr)
    assert len(kl_grade_arr) == len(yhat_arr)
    assert sorted(kl_counts.keys()) == list(range(5))
    if y_col == 'koos_pain_subscore':
        yhat_ranks = rankdata(yhat_arr)
        discretized_yhat = np.zeros(yhat_arr.shape)
        cutoff_n = kl_counts[0] - 1
        for klg in range(1, 5): 
            discretized_yhat[yhat_ranks < (total_count - cutoff_n)] = klg
            cutoff_n += kl_counts[klg]
        assert pearsonr(kl_grade_arr, discretized_yhat)[0] > 0
        if not Counter(kl_grade_arr) == Counter(discretized_yhat):
            print("KL grade and yhat should be equal in counts and are not")
            print(Counter(kl_grade_arr))
            print(Counter(discretized_yhat))
            assert False
    
    # check how well things agree. 
    for klg in list(range(5)):
        kl_grade_idxs = kl_grade_arr == klg
        our_grade_lower = discretized_yhat < klg
        our_grade_equal = discretized_yhat == klg
        our_grade_higher = discretized_yhat > klg
        print('Original KLG %i: %2.1f%% lower, %2.1f%% the same, %2.1f%% higher' % (klg,
            our_grade_lower[kl_grade_idxs].mean() * 100, 
            our_grade_equal[kl_grade_idxs].mean() * 100, 
            our_grade_higher[kl_grade_idxs].mean() * 100))
    print("Overall agreement: %2.1f%% of the time. Disagree by at least 2: %2.1f%% of the time." % 
        ((discretized_yhat == kl_grade_arr).mean() * 100, (np.abs(discretized_yhat - kl_grade_arr) >= 2).mean() * 100))

    return discretized_yhat

def bootstrap_dataframe_accounting_for_clustering(df, resample_points_within_cluster, random_seed):
    """
    Given a dataframe, draw a new sample by resampling IDs. 
    If resample_points_within_cluster, also resample points within IDs (ie, measurements for each person); 
    otherwise, just takes the original dataframe for the person. 

    Note: we do not resample points within cluster. 
    "A Practitioner’s Guide to Cluster-Robust Inference" does not do this either. 
    Similarly, "BOOTSTRAP-BASED IMPROVEMENTS FOR INFERENCE WITH CLUSTERED ERRORS" section 3.1 also makes no mention of resampling within cluster.
    They say this method is variously referred to as "cluster bootstrap, case bootstrap, non- parametric bootstrap, and nonoverlapping block bootstrap"

    """
    assert not resample_points_within_cluster
    assert 'id' in df.columns
    assert sum(pd.isnull(df['id'])) == 0
    df = copy.deepcopy(df)
    unique_ids = list(set(df['id']))

    reproducible_random_sampler = random.Random(random_seed)

    ids_sampled_with_replacement = [reproducible_random_sampler.choice(unique_ids) for a in range(len(unique_ids))]
    grouped_df = df.groupby('id')

    new_df = []
    for i, sampled_id in enumerate(ids_sampled_with_replacement):
        df_for_person = grouped_df.get_group(sampled_id)
        #df_for_person['id'] = i # pretend they all have unique ids. This doesn't actually matter when we're fitting regressions on bootstrapped samples (since we don't do clustering on ID or use it in any way). 
        if resample_points_within_cluster:
            within_cluster_idxs = [reproducible_random_sampler.choice(range(len(df_for_one_person))) for k in range(len(df_for_one_person))]
            df_for_person = df_for_person.iloc[within_cluster_idxs]
        new_df.append(df_for_person)
    new_df = pd.concat(new_df)
    new_df.index = range(len(new_df))
    return new_df

def get_bootstrapped_cis_on_quantity(df, resample_points_within_cluster, fxn, n_bootstraps, ci_fxn):
    """
    fxn should be a function we call on each bootstrap-resampled dataframe to compute some quantities.
    ci_fxn is how we aggregate the output of all the bootstrapped results. 
    """
    df = copy.deepcopy(df)
    t0 = time.time()
    original_estimate = fxn(df)
    bootstrapped_estimates = []
    for i in range(n_bootstraps):
        bootstrapped_estimates.append(fxn(bootstrap_dataframe_accounting_for_clustering(df, resample_points_within_cluster=resample_points_within_cluster, random_seed=i)))
    print("original estimate")
    print(original_estimate)
    ci_fxn_results = ci_fxn(bootstrapped_estimates)
    print("Computation time %2.3f seconds" % (time.time() - t0))
    return ci_fxn_results

def create_combined_df_with_yhat(original_df, yhat, visits_to_use, sides_to_use, discretize_yhat, y_col):
    # Join the datasets.
    combined_data = copy.deepcopy(original_df)
    assert type(yhat) is list
    if discretize_yhat:
        yhat = list(discretize_yhat_like_kl_grade(yhat_arr=np.array(yhat), 
            kl_grade_arr=combined_data['xrkl'].values, 
            y_col=y_col))

    combined_data['yhat'] = copy.deepcopy(yhat)
    assert combined_data['p02sex'].map(lambda x:x in ['1: Male', '2: Female']).all()
    combined_data['is_male'] = (combined_data['p02sex'] == '1: Male').values * 1.

    if visits_to_use != 'all':
        assert type(visits_to_use) is list
        combined_data = copy.deepcopy(combined_data.loc[combined_data['visit'].map(lambda x:x in visits_to_use)])
        print("%i datapoints remaining after filtering for visits in" % len(combined_data), visits_to_use)
    combined_data = copy.deepcopy(combined_data.loc[combined_data['side'].map(lambda x:x in sides_to_use)])
    print("%i datapoints remaining after filtering for sides in" % len(combined_data), sides_to_use)
    return combined_data

def bootstrap_ses_diff(original_df, yhat, y_col, visits_to_use, sides_to_use, discretize_yhat, n_bootstrap_iterates, alternate_thing_to_control_for):
    """
    Uses bootstrapping to answer: does using yhat making the SES coefficient significantly smaller than controlling for alternate_thing_to_control_for? 
    """
    assert alternate_thing_to_control_for in [None, 'C(xrkl)']

    combined_data = create_combined_df_with_yhat(original_df=original_df, 
        yhat=yhat, 
        visits_to_use=visits_to_use, 
        sides_to_use=sides_to_use, 
        discretize_yhat=discretize_yhat, 
        y_col=y_col)

    # What we're going to name the SES/race betas. 
    alternate_severity_col = 'ses_coef_controlling_for_%s' % alternate_thing_to_control_for
    yhat_col = 'ses_coef_controlling_for_yhat'

    # How does our SES/race coefficient compare to alternate_thing_to_control_for's? 
    for resample_points_within_cluster in [True, False]:
        for income_education_col in GAPS_OF_INTEREST_COLS:
            all_bootstrap_results = []

            # first iterate is the original estimate. The rest are bootstrap estimates. 
            for i in range(n_bootstrap_iterates + 1):
                betas = {} 
                if i == 0:
                    betas['estimate'] = 'original_dataset'
                    df_to_use = combined_data
                else:
                    betas['estimate'] = 'bootstrap'
                    df_to_use = bootstrap_dataframe_accounting_for_clustering(combined_data, resample_points_within_cluster=resample_points_within_cluster)
                
                for severity_measure in [alternate_thing_to_control_for, 'yhat']:
                    if severity_measure is None:
                        ols_model = sm.OLS.from_formula('%s ~ %s' % (y_col, income_education_col), data=df_to_use).fit()
                    else:
                        ols_model = sm.OLS.from_formula('%s ~ %s + %s' % (y_col, income_education_col, severity_measure), data=df_to_use).fit()
                    betas['ses_coef_controlling_for_%s' % severity_measure] = float(ols_model.params[income_education_col])
                betas['diff'] = (betas[alternate_severity_col] - betas[yhat_col])
                all_bootstrap_results.append(betas)

            all_bootstrap_results = pd.DataFrame(all_bootstrap_results) # we end up with a dataframe with columns "estimate",ses_coef_controlling_for_yhat,ses_coef_controlling_for_C(xrkl),diff. Each row is the estimate from one iterate. 

            # Now we make plots of the results. 
            plt.figure(figsize=[10, 5])
            ymax = len(all_bootstrap_results) * .3
            assert str(all_bootstrap_results.iloc[0]['estimate']) == 'original_dataset'
            original_KLG_estimate = all_bootstrap_results.iloc[0][alternate_severity_col]
            original_yhat_estimate = all_bootstrap_results.iloc[0][yhat_col]
            original_diff = all_bootstrap_results.iloc[0]['diff']
            bootstrap_idxs = all_bootstrap_results['estimate'] == 'bootstrap'

            # first plot: two bootstrapped histograms comparing the SES/race col when controlling for yhat and controlling for KLG/nothing. 
            plt.subplot(121)
            plt.hist(all_bootstrap_results.loc[bootstrap_idxs, alternate_severity_col].values, color='red', alpha=.7, label='Controls=%s' % alternate_thing_to_control_for)
            plt.hist(all_bootstrap_results.loc[bootstrap_idxs, yhat_col].values, color='blue', alpha=.7, label='Controls=yhat')
            plt.ylim([0, ymax])
            plt.plot([original_KLG_estimate, original_KLG_estimate], [0, ymax], color='red', linestyle='--')
            plt.plot([original_yhat_estimate, original_yhat_estimate], [0, ymax], color='blue', linestyle='--')
            plt.title("$beta_{SES}$ in a regression of pain on SES + controls")
            plt.legend()

            # second plot: bootstrapped difference. 
            plt.subplot(122)
            plt.hist(all_bootstrap_results.loc[bootstrap_idxs, 'diff'].values, color='black')
            plt.plot([original_diff, original_diff], [0, ymax], color='black', linestyle='--')
            plt.title("$beta_{SES}$ (controlling for %s) -\n $beta_{SES}$ (controlling for yhat))" % alternate_thing_to_control_for)
            plt.ylim([0, ymax])
            
            if all_bootstrap_results.loc[bootstrap_idxs, 'diff'].values.min() > 0:
                plt.xlim([0, all_bootstrap_results.loc[bootstrap_idxs, 'diff'].values.max() * 1.1])
            else:
                print("WARNING! BOOTSTRAPPED DIFFS OVERLAP 0")

            plt.suptitle("SES col: %s; resample points within cluster: %s\nratio of original estimates, %2.3f" % (income_education_col, resample_points_within_cluster, original_yhat_estimate/original_KLG_estimate))
            plt.subplots_adjust(top=.8)
            plt.show()

def check_is_array(arr):
    if type(arr) is not np.ndarray:
        raise Exception("Variable should be an array, not a %s" % type(arr))
    not_na = ~np.isnan(arr)
    if set(arr[not_na]).issubset(set([0, 1])) or set(arr[not_na]).issubset(set([0., 1.])) or set(arr[not_na]).issubset(set([True, False])):
        # this is my almost certainly overparanoid code to make sure nothing wonky happens when you convert our binary arrays from floats to bools or vice versa.
        assert (arr[not_na] == (arr[not_na] * 1.)).all()
        assert (arr[not_na] == (arr[not_na] == 1)).all()
        old_x = arr[not_na]
        new_x = old_x.copy()
        new_x = new_x == 1
        new_x = new_x * 1.
        new_x = new_x == True
        new_x = new_x * 1
        new_x = new_x == 1
        assert (new_x == old_x).all()
        #print("boolean check activated and passed.")

def bootstrap_ses_stratification_helper(y, severity_score, ses):
    # This function helps bootstrap CIs for make_ses_stratification_plot. 
    means_by_group = {}
    groups = None
    for ses_var_val in [True, False]:   
        grouped_d = pd.DataFrame({'y':y[ses == ses_var_val], 'x':severity_score[ses == ses_var_val]}).groupby('x').mean()
        means_by_group['high_income_%s' % ses_var_val] = grouped_d.values

        if groups is not None:
            if (len(groups) != len(grouped_d)) or not ((groups == grouped_d.index).all()):
                print("Warning: in this bootstrapped replicate, the bootstrapped dataframe did not have members of both groups for all severity categories. Returning None. This should happen extremely rarely in the final analysis; if it happens frequently, use parametric standard errors.")
                return None
        else:
            groups = grouped_d.index
    means_by_group['diff'] = means_by_group['high_income_True'] - means_by_group['high_income_False']
    return means_by_group

def make_ses_stratification_plot(ses, y, dict_of_severity_scores, severity_score_order, ses_var_one_label, ses_var_zero_label, fig_title=None, income_education_col=None, discretization_threshold=None, n_bootstraps=None, ids=None, compare_two_severity_scores_on_one_plot=False):
    """
    Given a binary variable (ses) denoting whether someone is high/low SES status
    y, the outcome y,
    and a dictionary with various severity scores (KLG, yhat, etc), where keys are the severity score names
    Make plots showing the SES pain gap -- ie, average y stratified by high/low SES. 
    severity_score_order gives the order in which you wish to plot the severity scores. 

    If compare_two_severity_scores_on_one_plot, plot an additional plot showing both severity scores and how they reduce the pain gap. 

    Note this plot is sort of imperfect, because it doesn't show one of the two reasons we reduce the pain gap -- we reassign disadvantaged groups to higher severity scores. 

    Checked. 
    """

    assert set(ses) == set([True, False])
    assert np.isnan(ses).sum() == 0
    assert y.max() == 100
    assert len(y) == len(ses)
    check_is_array(y)
    check_is_array(ses)

    if discretization_threshold is not None:
        y = y <= discretization_threshold
        ylabel = 'Proportion Koos pain subscore <= %2.1f' % discretization_threshold
    else:
        ylabel = "Mean Koos pain subscore"

    n_severity_scores = len(severity_score_order)
    assert n_severity_scores == 2
    if compare_two_severity_scores_on_one_plot:
        n_plot_cols = n_severity_scores + 1
        keep_xs_consistent_sanity_check = None # make sure we're plotting severity scores on the same scale (don't use eg 0-4 and 0-10 on the same plot)
    else:
        n_plot_cols = n_severity_scores
    
    assert len(severity_score_order) == len(dict_of_severity_scores)
    plt.figure(figsize=[4 * n_plot_cols, 8])
    for severity_score_idx, severity_score_name in enumerate(severity_score_order):
        severity_score = copy.deepcopy(dict_of_severity_scores[severity_score_name])
        assert len(severity_score) == len(y)
        check_is_array(severity_score)
        xticks_to_use = sorted(list(set(severity_score)))
        
        plt.subplot(2, n_plot_cols, severity_score_idx + 1) # make first subplot. Stratify by SES (so one line for high SES, one line for low SES people). 
        means_by_group = {}
        xs_plotted = None
        for ses_var_val in [True, False]:
            y_vs_severity = pd.DataFrame({'y':y[ses == ses_var_val],
                'x':severity_score[ses == ses_var_val]})
            grouped_d = y_vs_severity.groupby('x').agg(['mean', 'size'])
            grouped_d.columns = ['mean', 'size']
            means_by_group['high_income_%s' % ses_var_val] = copy.deepcopy(grouped_d['mean'].values)
            line_label = ses_var_one_label if ses_var_val else ses_var_zero_label
            plt.plot(grouped_d.index, grouped_d['mean'], label=line_label)

            # make sure the x-values stay the same for high/low SES people. 
            if xs_plotted is None:
                xs_plotted = grouped_d.index
            else:
                assert (grouped_d.index == xs_plotted).all()
        if n_bootstraps is not None:
            assert ids is not None
            def ci_fxn(bootstraps):
                # Each item in bootstraps (a list) is a dataframe whose columns are estimates for the high_income_True line, the high_income_False line, and the diff. 
                # We aggregate this into a dictionary with keys 'high_income_True', 'high_income_False', 'diff'
                # the values are arrays with the CIs (first column lower CI, second column upper CI). 
                print("Bootstrapped CIs")
                proper_length = len(bootstraps[0]['diff'])
                all_CIs = {}
                for col in ['high_income_True', 'high_income_False', 'diff']:
                    arr = []
                    expected_n_bootstraps = 0
                    for i in range(len(bootstraps)):
                        if bootstraps[i] is None:
                            continue
                        arr.append(bootstraps[i][col].flatten())
                        assert len(bootstraps[i][col]) == proper_length
                        expected_n_bootstraps += 1
                    arr = np.array(arr)
                    assert arr.shape[1] == proper_length
                    assert arr.shape[0] == expected_n_bootstraps
                    all_CIs[col] = []
                    for j in range(arr.shape[1]):
                        all_CIs[col].append(get_CI_from_percentiles(arr[:, j], make_plot=False))
                    all_CIs[col] = np.array(all_CIs[col])
                return all_CIs


            # sanity check: parametric estimates. 
            sanity_check_df = pd.DataFrame({'id':ids, 
                                            'y':y, 
                                            'severity_score':severity_score, 
                                            'ses':ses})
            low_ses_model = sm.OLS.from_formula('y ~ C(severity_score) - 1', data=sanity_check_df.loc[sanity_check_df['ses'] == 0]).fit()
            low_ses_model = low_ses_model.get_robustcov_results(cov_type='cluster', groups=sanity_check_df.loc[sanity_check_df['ses'] == 0, 'id'].astype(int))

            high_ses_model = sm.OLS.from_formula('y ~ C(severity_score) - 1', data=sanity_check_df.loc[sanity_check_df['ses'] == 1]).fit()
            high_ses_model = high_ses_model.get_robustcov_results(cov_type='cluster', groups=sanity_check_df.loc[sanity_check_df['ses'] == 1, 'id'].astype(int))
            
            
            all_CIs = get_bootstrapped_cis_on_quantity(df=pd.DataFrame({'id':ids, 
                                                               'y':y, 
                                                               'severity_score':severity_score, 
                                                               'ses':ses}),
                                              resample_points_within_cluster=False, 
                                              fxn=lambda x:bootstrap_ses_stratification_helper(y=x['y'].values, severity_score=x['severity_score'].values, ses=x['ses'].values),
                                              n_bootstraps=n_bootstraps, 
                                              ci_fxn=ci_fxn)
            print("low ses conf int")
            for i in range(len(low_ses_model.conf_int(alpha=0.05))):
                print('parametric', list(low_ses_model.conf_int()[i, :]), 'nonparametric', list(all_CIs['high_income_False'][i, :]))
            print("high ses conf int")
            for i in range(len(high_ses_model.conf_int(alpha=0.05))):
                print('parametric', list(high_ses_model.conf_int()[i, :]), 'nonparametric', list(all_CIs['high_income_True'][i, :]))

            plt.fill_between(grouped_d.index, all_CIs['high_income_True'][:, 0], all_CIs['high_income_True'][:, 1], alpha=.1)
            plt.fill_between(grouped_d.index, all_CIs['high_income_False'][:, 0], all_CIs['high_income_False'][:, 1], alpha=.1)

        if severity_score_idx == 0:
            plt.ylabel(ylabel, fontsize=14)
        if discretization_threshold is None:
            plt.ylim([50, 100])
        else:
            plt.ylim([0, 1])
        plt.xlim(grouped_d.index.min(), grouped_d.index.max())
        plt.xlabel(severity_score_name, fontsize=14)
        plt.xticks(xticks_to_use)
        if severity_score_idx == 0:
            plt.legend()

        # plot the SES gap (ie, the difference between the high SES and the low SES plot). 
        plt.subplot(2, n_plot_cols, severity_score_idx + 1 + n_plot_cols)
        plt.plot(xs_plotted, 
            means_by_group['high_income_True'] - means_by_group['high_income_False'], color='black')

        if discretization_threshold is None:
            plt.ylim([-20, 20])
        else:
            plt.ylim([-.3, .3])
        plt.xlim(xs_plotted.min(), xs_plotted.max())
        plt.xticks(xticks_to_use)
        plt.plot([xs_plotted.min(), xs_plotted.max()], [0, 0], linestyle='--', color='grey')
        if n_bootstraps is not None:
            plt.fill_between(grouped_d.index, all_CIs['diff'][:, 0], all_CIs['diff'][:, 1], alpha=.1, color='black')
        plt.xlabel(severity_score_name, fontsize=14)
        if severity_score_idx == 0:
            plt.ylabel("Pain gap between groups", fontsize=14)

        if compare_two_severity_scores_on_one_plot:
            # make a final plot (bottom right corner) showing how severity scores 
            if keep_xs_consistent_sanity_check is None:
                keep_xs_consistent_sanity_check = xs_plotted
                max_abs_yval = 0
            else:
                assert list(xs_plotted) == list(keep_xs_consistent_sanity_check)
            all_linestyles = ['-', '--']
            plt.subplot(2, n_plot_cols, 2 * n_plot_cols)
            plt.plot(xs_plotted, means_by_group['high_income_True'] - means_by_group['high_income_False'], label=severity_score_name, color='black', linestyle=all_linestyles[severity_score_idx])
            plt.legend()
            max_abs_yval = max(np.abs(means_by_group['high_income_True'] - means_by_group['high_income_False']).max(), max_abs_yval)
            plt.ylim([-1.05 * max_abs_yval, 1.05 * max_abs_yval])
            plt.xlim(xs_plotted.min(), xs_plotted.max())
            plt.xticks(xticks_to_use)
            plt.plot([xs_plotted.min(), xs_plotted.max()], [0, 0], linestyle='--', color='grey')
            if n_bootstraps is not None:
                plt.fill_between(grouped_d.index, all_CIs['diff'][:, 0], all_CIs['diff'][:, 1], alpha=.1, color='black')
            plt.xlabel("Severity score", fontsize=14)
            plt.ylabel("Pain disparity", fontsize=14)

    plt.subplots_adjust(hspace=.3)
    if compare_two_severity_scores_on_one_plot:
        plt.subplots_adjust(wspace=.3)
    if income_education_col is not None:
        plt.suptitle("SES variable: %s" % income_education_col, fontsize=14)
    if fig_title is not None:
        plt.savefig(fig_title, dpi=300)
    plt.show()

def cut_into_deciles(arr, y_col):
    """
    Checked. 
    Note that the first (lowest) decile here is 1, the highest is 10, and that 
    the lowest decile actually corresponds to the largest values of arr.
    This is so that when we use this to discretize yhat, low scores (high on Koos pain subscore) = less pain = low KLG. 
    """
    check_is_array(arr)
    assert y_col == 'koos_pain_subscore'
    arr = 100.*rankdata(-arr) / len(arr)
    arr = np.array([math.ceil(x / 10.) for x in arr])
    assert set(arr) == set(range(1, 11))
    return arr


def assess_yhat_correlations(original_df, clinical_controls, yhat, y_col, visits_to_use, sides_to_use, discretize_yhat=False):
    """
    Given pain scores + a non-image dataset
    and a yhat prediction, and a y_col to use as the y
    Come up with answers to our three questions: 
    1. does yhat correlate with y? 
    2. Does this persist when we control for other things? 
    3. Does including yhat change the coef on SES? 

    Filters for measurements with visit in visits_to_use, side in sides_to_use. 
    Checked. 

    Note: this function computes a lot of p-values, but they will only be valid if the independence assumptions are valid. 
    """

    assert y_col in ['koos_pain_subscore', 'womac_pain_subscore']
    
    # have to add in yhat. 
    combined_data = create_combined_df_with_yhat(original_df=original_df, 
        yhat=yhat, visits_to_use=visits_to_use, sides_to_use=sides_to_use, discretize_yhat=discretize_yhat, 
        y_col=y_col)



    for c in ALL_CONTROLS:
        col_name_to_use = c.replace('C(', '').replace(')', '').split(', Treatment')[0]
        if '*' in col_name_to_use:
            # this indicates an interaction term, not a column. 
            continue
        missing_data_for_col = pd.isnull(combined_data[col_name_to_use])
        if missing_data_for_col.sum() > 0:
            print("Filling in missing data for proportion %2.3f values of %s" % 
                (missing_data_for_col.mean(), col_name_to_use))
        combined_data.loc[missing_data_for_col, col_name_to_use] = 'MISSING'


    control_sets = {'None':[],
                    'yhat':['yhat'], 
                    'just clinical controls':clinical_controls, 
                    'just kl score':['C(xrkl)'],
                    'age/sex/race/site':AGE_RACE_SEX_SITE, 
                    'all_controls_but_other_pain':ALL_CONTROLS_BUT_OTHER_PAIN,
                    'all_controls':ALL_CONTROLS,
                    'all_controls_plus_yhat':ALL_CONTROLS + ['yhat'],
                    'all_controls_but_other_pain_plus_yhat':ALL_CONTROLS_BUT_OTHER_PAIN + ['yhat'],
                    'yhat,clinical controls':['yhat'] + clinical_controls, 
                    'yhat,kl score':['C(xrkl)', 'yhat'],
                    'yhat,clinical controls,age/sex/race/site':['yhat'] + clinical_controls + AGE_RACE_SEX_SITE, 
                    'yhat,clinical_controls,age/sex/race/site,timepoint,side':['yhat'] + clinical_controls + AGE_RACE_SEX_SITE + ['visit', 'side']}
    ordered_control_sets = ['yhat',
                            'age/sex/race/site', 
                            'just clinical controls', 
                            'yhat,clinical controls', 
                            'just kl score', 
                            'yhat,kl score',
                            'yhat,clinical controls,age/sex/race/site', 
                            'yhat,clinical_controls,age/sex/race/site,timepoint,side', 
                            'all_controls_but_other_pain',
                            'all_controls_but_other_pain_plus_yhat',
                            'all_controls',
                            'all_controls_plus_yhat']
    
    # first analysis: how well do various combinations of covariates predict yhat? 
    # Cluster standard errors to get errorbars on yhat coefficient (show it's still significant when controlling for other things). 
    yhat_correlation_results = []
    for control_set_name in ordered_control_sets:
        controls_to_use = control_sets[control_set_name]
        non_clustered_model = sm.OLS.from_formula('%s ~ %s' % (y_col, '+'.join(controls_to_use)), data=combined_data).fit()

        assert non_clustered_model.nobs == len(combined_data) # make sure no missing data. 
        cluster_model = non_clustered_model.get_robustcov_results(cov_type='cluster', groups=combined_data['id'].astype(int))

        if 'yhat' in controls_to_use:
            yhat_idx = list(non_clustered_model.params.index).index('yhat') # for some reason parameters are returned as arrays, not dfs, in the clustered model. 
            clustered_beta = cluster_model.params[yhat_idx]
            clustered_pval = cluster_model.pvalues[yhat_idx]
            nonclustered_beta = non_clustered_model.params['yhat']
            nonclustered_pval = non_clustered_model.pvalues['yhat']
            assert np.allclose(nonclustered_beta, clustered_beta) # make sure the estimate doesn't change when we cluster standard errors. 
            print(cluster_model.summary())

        else:
            clustered_beta = None
            clustered_pval = None
            nonclustered_beta = None
            nonclustered_pval = None

        yhat_correlation_results.append({'RHS':control_set_name,
                            '$b_{yhat}$':clustered_beta, 
                           '$r^2$':non_clustered_model.rsquared, 
                           '$r$':np.sqrt(non_clustered_model.rsquared),
                            '$p_{yhat}$':clustered_pval, 
                            '$b_{yhat,unclustered}$':nonclustered_beta, 
                            '$p_{yhat,unclustered}$':nonclustered_pval,
                            'n':int(non_clustered_model.nobs)})
        assert pd.isnull(combined_data[y_col]).sum() == 0
        assert pd.isnull(combined_data['yhat']).sum() == 0

        
        
    pd.set_option('max_colwidth', 100)
    yhat_correlation_results = pd.DataFrame(yhat_correlation_results)[['RHS', '$b_{yhat}$', '$b_{yhat,unclustered}$', '$p_{yhat}$', 
    '$p_{yhat,unclustered}$', '$r^2$', '$r$', 'n']]


    # Are we actually reordering people or is our yhat just a categorical pain score? 
    plt.figure(figsize=[4, 4])
    sns.violinplot(x=combined_data['xrkl'].values, y=combined_data['yhat'].values)
    
    spearman_r, spearman_p = spearmanr(combined_data['xrkl'].values + 1e-8*combined_data['yhat'].values, # small hack. This ensures we'll get perfect correlation if the orderings are the same modulo discretization.
        combined_data['yhat'].values)

    assert np.abs(1 - spearmanr(combined_data['yhat'].values, 
        (combined_data['yhat'].values > 85) + (combined_data['yhat'].values > 95) + (1e-8*combined_data['yhat'].values))[0]) < 1e-6 # make sure the hack actually works. 
    plt.xlabel("KL Grade")
    plt.ylabel("Our yhat")
    plt.title("Modified spearman correlation\n(will be 1 if orderings are identical modulo discretization): %2.3f" % spearman_r)
    plt.show()


    # Stratify income by various coefficients. 
    for income_education_col in GAPS_OF_INTEREST_COLS:
        decile_yhat = cut_into_deciles(combined_data['yhat'].values, y_col=y_col)
        make_ses_stratification_plot(ses=combined_data[income_education_col].values, 
            y=combined_data[y_col].values, 
            dict_of_severity_scores={'KLG':combined_data['xrkl'].values, 
                                    '$\hat y$ decile':decile_yhat},
            severity_score_order=['KLG','$\hat y$ decile'], 
            income_education_col=income_education_col)


    # does it affect SES coefficient? 
    all_income_results = []
    for income_education_col in GAPS_OF_INTEREST_COLS:
        # make a histogram of predicted yhat by SES. 
        plt.figure()
        bins = np.arange(combined_data['yhat'].min() - .01, combined_data['yhat'].max() + .01, (combined_data['yhat'].max() - combined_data['yhat'].min()) / 15)
        high_ses_vals = combined_data['yhat'].loc[combined_data[income_education_col] == 1].values
        low_ses_vals = combined_data['yhat'].loc[combined_data[income_education_col] == 0].values
        plt.hist(high_ses_vals, color='black', density=1, alpha=1, bins=bins, label='High SES')
        plt.hist(low_ses_vals, color='red', density=1, alpha=.7, bins=bins, label='Low SES')
        t, p_ses_diff = ttest_ind(high_ses_vals, low_ses_vals)
        plt.xlabel("yhat")
        plt.ylabel('density')
        plt.title("SES measure: %s\nPain score: %s\np=%2.3e" % (income_education_col, y_col, p_ses_diff))
        plt.legend()
        plt.show()

        # Now run regressions to determine how SES coefficient changes. 
        for control_set_name in ['None'] + ordered_control_sets:
            
            controls_to_use = control_sets[control_set_name]

            model = sm.OLS.from_formula('%s ~ %s' % (y_col, '+'.join(controls_to_use + [income_education_col])), 
                                                     data=combined_data).fit()
            clustered_model = model.get_robustcov_results(cov_type='cluster', groups=combined_data['id'].values)

            ses_idx = list(model.params.index).index(income_education_col)
            assert np.allclose(clustered_model.params[ses_idx], model.params[income_education_col]) # sanity check; make sure we're indexing correctly. 

            if (income_education_col == 'is_male') and 'sex' in control_set_name:
                # estimates are not valid because sex is on LHS + RHS. 
                all_income_results.append({'SES_col':income_education_col, 
                                       'controls':control_set_name, 
                                       '$b_{SES}$':None, 
                                       'lower CI':None, 
                                       'upper CI':None})
            else:
                all_income_results.append({'SES_col':income_education_col, 
                                           'controls':control_set_name, 
                                           '$b_{SES}$':clustered_model.params[ses_idx], 
                                           'lower CI':float(clustered_model.conf_int()[ses_idx, 0]), 
                                           'upper CI':float(clustered_model.conf_int()[ses_idx, 1]), 
                                           'unclustered lower CI':float(model.conf_int().loc[income_education_col, 0]), 
                                           'unclustered upper CI':float(model.conf_int().loc[income_education_col, 1])})
                                       
    all_income_results = pd.DataFrame(all_income_results)[['SES_col', 'controls', '$b_{SES}$', 'lower CI', 'upper CI', 'unclustered lower CI', 'unclustered upper CI']]
    return yhat_correlation_results, all_income_results

def make_plot_of_effect_on_ses_coefficient(all_income_results, plot_errorbars):
    """
    Given the all_income_results dataframe returned by assess_yhat_correlations, make a plot of the coefficients. 
    Checked. 
    """
    plt.figure(figsize=[4, .5 * len(all_income_results)])
    ytick_labels = None
    for plt_idx, SES_col in enumerate(GAPS_OF_INTEREST_COLS):
        plt.subplot(len(GAPS_OF_INTEREST_COLS), 1, plt_idx + 1)
        df_to_plot = all_income_results.loc[all_income_results['SES_col'] == SES_col]
        for i in range(len(df_to_plot)):
            beta = float(df_to_plot.iloc[i]['$b_{SES}$'])
            low_CI = float(df_to_plot.iloc[i]['lower CI'])
            high_CI = float(df_to_plot.iloc[i]['upper CI'])
            if plot_errorbars:
                xerrs = [[beta - low_CI], [high_CI - beta]] 
            else:
                xerrs = None
            plt.errorbar([beta], [i], xerr=xerrs,
                         color='black', 
                         capsize=3)
            plt.scatter([beta], [i], color='blue')
        pretty_names = {'binarized_income_at_least_50k':r'$\beta_{income\geq50k}$', 
                        'binarized_education_graduated_college':r'$\beta_{college grad}$', 
                        'is_male':r'$\beta_{male}$'}
        if ytick_labels is None:
            ytick_labels = list(df_to_plot['controls'])
        else:
            assert ytick_labels == list(df_to_plot['controls'])
        plt.yticks(range(len(df_to_plot)), ytick_labels)
        plt.xlim([0, 8])
        plt.title(pretty_names[SES_col])
        plt.ylabel("Other controls")
    plt.subplots_adjust(hspace=.5)
    plt.show()



def compare_two_datasets_for_balance(dataset1, dataset2):
    """
    Make sure that both datasets are reasonably close on relevant covariates (randomization check).
    Checked. 
    """
    min_plausible_p = 1e-3
    df = copy.deepcopy(dataset1.processed_dataframes['per_person_covariates'])
    val_df = copy.deepcopy(dataset2.processed_dataframes['per_person_covariates'])
    df['hispanic'] = df['p02hisp'] == '1: Yes'
    val_df['hispanic'] = val_df['p02hisp'] == '1: Yes'
    df['white'] = df['p02race'] == '1: White or Caucasian'
    val_df['white'] = val_df['p02race'] == '1: White or Caucasian'
    df['female'] = df['p02sex'] == '1: Male'
    val_df['female'] = val_df['p02sex'] == '1: Male'


    for k in ['v00age', 'binarized_income_at_least_50k', 'binarized_education_graduated_college', 'hispanic', 
              'white', 'female']:
        t, p = ttest_ind(df[k].dropna().values, 
                         val_df[k].dropna().values)
        assert np.isnan(p) or p > min_plausible_p
        star = '*' if p < .05 else ''
        print("%-50s %2.3e %2.3f %2.3f (n=%i,%i) %s" % (k, 
                                              p,
                                              df[k].dropna().mean(), 
                                              val_df[k].dropna().mean(),
                                              len(df[k].dropna()),
                                              len(val_df[k].dropna()),
                                              star))

    print("Checking cohort balance!")
    df = copy.deepcopy(dataset1.original_dataframes['enrollees'])
    val_df = copy.deepcopy(dataset2.original_dataframes['enrollees'])
    df['progression_cohort'] = df['v00cohort'] == '1: Progression'
    val_df['progression_cohort'] = val_df['v00cohort'] == '1: Progression'
    for k in ['progression_cohort']:
        t, p = ttest_ind(df[k].dropna().values, 
                             val_df[k].dropna().values)
        assert np.isnan(p) or p > min_plausible_p
        star = '*' if p < .05 else ''
        print("%-50s %2.3e %2.3f %2.3f (n=%i,%i) %s" % (k, 
                                              p,
                                              df[k].dropna().mean(), 
                                              val_df[k].dropna().mean(),
                                              len(df[k].dropna()),
                                              len(val_df[k].dropna()),
                                              star))




    for df_name in ['kxr_sq_bu', 'all_knee_pain_scores']:
        df = copy.deepcopy(dataset1.processed_dataframes[df_name])
        val_df = copy.deepcopy(dataset2.processed_dataframes[df_name])
        for visit in sorted(list(set(df['visit']))):
            for side in ['left', 'right']:
                print("\n\n==========\nChecking balance for %s,%s,%s" % (df_name, visit, side))
                idxs = (df['visit'] == visit) & (df['side'] == side)
                val_idxs = (val_df['visit'] == visit) & (val_df['side'] == side)

                print("Checking balance in proportion of dataset with this value")
                t, p = ttest_ind(idxs.values, val_idxs.values)
                assert np.isnan(p) or p > min_plausible_p
                star = '*' if p < .05 else ''
                print("%-50s %2.3e %2.3f %2.3f (n=%i,%i) %s" % ('proportion of dataset', 
                                                          p, 
                                                          idxs.mean(), 
                                                          val_idxs.mean(),
                                                          len(idxs), 
                                                          len(val_idxs),
                                                          star))


                for key in df.columns:
                    if key == 'side' or key == 'readprj' or key == 'barcdbu' or key == 'version' or key == 'visit':
                        continue
                    t, p = ttest_ind(df[key].loc[idxs].dropna().values, 
                                         val_df[key].loc[val_idxs].dropna().values)
                    assert np.isnan(p) or p > min_plausible_p
                    star = '*' if p < .05 else ''
                    print("%-50s %2.3e %2.3f %2.3f (n=%i,%i) %s" % (key, 
                                                          p, 
                                                          df[key].loc[idxs].mean(), 
                                                          val_df[key].loc[val_idxs].mean(),
                                                          len(df[key].loc[idxs].dropna()), 
                                                          len(val_df[key].loc[val_idxs].dropna()),
                                                          star))


def test_all_dfs_for_balance():
    """
    Make sure all the datasets are balanced (randomization check). Checked. 
    """
    non_image_train_dataset = non_image_data_processing.NonImageData(what_dataset_to_use='train')
    non_image_validation_dataset = non_image_data_processing.NonImageData(what_dataset_to_use='val')
    non_image_hold_out_dataset = non_image_data_processing.NonImageData(what_dataset_to_use='BLINDED_HOLD_OUT_DO_NOT_USE', i_promise_i_really_want_to_use_the_blinded_hold_out_set=True)
    non_image_test_dataset = non_image_data_processing.NonImageData(what_dataset_to_use='test')

    print("\n\n\n**************\nBalance in train and validation set. Stars denote p < .05 for difference")
    compare_two_datasets_for_balance(non_image_train_dataset, non_image_validation_dataset) 

    print("\n\n\n**************\nBalance in train and hold out datasets. Stars denote p < .05 for difference")
    compare_two_datasets_for_balance(non_image_train_dataset, non_image_hold_out_dataset) 

    print("\n\n\n**************\nBalance in train and test set. Stars denote p < .05 for difference")
    compare_two_datasets_for_balance(non_image_train_dataset, non_image_test_dataset) 

def assess_performance(y, yhat, binary_prediction, return_tpr_and_fpr=False):
    """
    Return standard metrics of performance give y and yhat. yhat should never be binary, even for binarized scores. Checked. 
    """
    if binary_prediction:
        assert set(list(y)) == set([0, 1])
        assert set(list(yhat)) != set([0, 1])
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=y, y_score=yhat)
        
        

        auc = sklearn.metrics.roc_auc_score(y_score=yhat, y_true=y)
        auprc = sklearn.metrics.average_precision_score(y_score=yhat, y_true=y)
        metrics = {'auc':auc, 'auprc':auprc}
        if return_tpr_and_fpr:
            metrics['tpr'] = tpr
            metrics['fpr'] = fpr
    else:
        assert set(list(y)) != set([0, 1])
        r = pearsonr(y, yhat)[0]
        spearman_r = spearmanr(y, yhat)[0]
        rmse = np.sqrt(np.mean((y - yhat) ** 2))
        metrics = {'r':r, 'rmse':rmse, 'negative_rmse':-rmse, 'r^2':r**2, 'spearman_r':spearman_r, 'spearman_r^2':spearman_r**2}
    for metric in metrics:
        if metric in ['tpr', 'fpr']:
            continue
        if np.isnan(metrics[metric]):
            raise Exception("%s is a nan, something is weird about your predictor" % metric)
    return metrics
    
def compare_to_clinical_performance(train_df, val_df, test_df, y_col, features_to_use, binary_prediction, use_nonlinear_model, do_ols_sanity_check=False, verbose=True):
    """
    Rigorously measure clinical performance in a way that can be compared to model performance. Just as we choose yhat, we: 

    train on the train set
    select model on the val set
    return test_yhat and assess on the test set. 

    if use_nonlinear_model (only implemented for continuous prediction tasks) we fit a Random Forest, conducting hyperparameter search; this takes a while. 
    """
    train_df = copy.deepcopy(train_df)
    val_df = copy.deepcopy(val_df)
    test_df = copy.deepcopy(test_df)
    
    if verbose:
        print("Predicting %s using" % y_col, features_to_use)
    
    if not binary_prediction: 
        # sanity check to make sure results are similar to our weird patsy thing. 
        ols_model = sm.OLS.from_formula('%s ~ %s' % (y_col, '+'.join(features_to_use)), 
                                        data=train_df).fit()
        #print(ols_model.summary())

    assert list(train_df.columns) == list(test_df.columns)

    if do_ols_sanity_check:
        print("Sanity check: just use OLS to predict %s (these results should be very similar to performance on the test set)" % y_col)
        ols_model = sm.OLS.from_formula('%s ~ %s' % (y_col, '+'.join(features_to_use)), train_df).fit()
        ols_pred_df = copy.deepcopy(test_df)
        for col_name in features_to_use:
            if 'C(' in col_name:
                col_name = col_name.replace('C(', '').replace(')', '')
                for unique_val in sorted(list(set(ols_pred_df[col_name]))):
                    if unique_val not in train_df[col_name].values:
                        bad_idxs = ols_pred_df[col_name] == unique_val
                        print("Warning: Setting %i values in column %s to 0" % (bad_idxs.sum(), col_name))
                        ols_pred_df.loc[bad_idxs, col_name] = 0.
        ols_preds = ols_model.predict(ols_pred_df).values
        ols_performance_results = assess_performance(yhat=ols_preds, y=test_df[y_col].values, binary_prediction=False)
        for metric in ols_performance_results.keys():
            if metric == 'test_yhat':
                continue
            print("%s: %2.3f" % (metric, ols_performance_results[metric]))

    # concatenate both train + test matrices and encode them in patsy together so we make sure to use a consistent encoding. 
    combined_df = pd.concat([train_df, val_df, test_df])
    combined_y, combined_X = patsy.dmatrices('%s ~ %s' % (y_col, '+'.join(features_to_use)), 
                                       data=combined_df, return_type='dataframe')
    n_train_points = len(train_df)
    n_val_points = len(val_df)

    # remove intercept column because it's included in sklearn fit. The reason we don't do this with -1 in a patsy formula 
    # is -1 just recodes categorical variables; it doesn't actually reduce the number of columns. 
    assert 'Intercept' in combined_X.columns
    combined_X = combined_X[[a for a in combined_X.columns if a != 'Intercept']]

    train_y = combined_y.iloc[:n_train_points]
    train_X = combined_X.iloc[:n_train_points]

    val_y = combined_y.iloc[n_train_points:(n_train_points + n_val_points)]
    val_X = combined_X.iloc[n_train_points:(n_train_points + n_val_points)]

    test_y = combined_y.iloc[(n_train_points + n_val_points):]
    test_X = combined_X.iloc[(n_train_points + n_val_points):]

    train_y = np.array(train_y)
    train_X = np.array(train_X)
    val_y = np.array(val_y)
    val_X = np.array(val_X)
    test_y = np.array(test_y)
    test_X = np.array(test_X)

    for arr in [train_y, train_X, val_y, val_X, test_y, test_X]:
        assert np.isnan(arr).sum() == 0

    train_y = train_y.flatten()
    val_y = val_y.flatten()
    test_y = test_y.flatten()
    
    if verbose:
        print("Original length of train data: %i; after dropping missing data, %i" % (len(train_df), len(train_X)))

    if binary_prediction:
        assert 'binarized' in y_col
        sparsity_params = [10**a for a in np.arange(-5, 3, .5)]
        metric_to_sort_by = 'auc'
        if verbose:
            print("prediction task is BINARY")
    else:
        if not use_nonlinear_model:
            sparsity_params = [10**a for a in np.arange(-3, 3, .5)]
        else:
            # Fit a random forest model. Optimize over a bunch of hyperparameters, as suggested in
            # https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d 
            sparsity_params = []
            rng = random.Random(42)
            
            all_possible_param_sets = []
            for n_estimators in [50, 100, 500]:
                for max_depth in [5, 10, 50, 100, None]:
                    for max_features in ['auto', 'sqrt', 'log2', .3]:
                        for min_samples_split in [2, 5, 10]:
                            for min_samples_leaf in [1, 2, 4, 6, 8, 10, 20]:
                                all_possible_param_sets.append({'n_estimators':n_estimators, 
                                                                'max_depth':max_depth,
                                                                'max_features':max_features,
                                                                'min_samples_split':min_samples_split, 
                                                                'min_samples_leaf':min_samples_leaf})
            sparsity_params = rng.sample(all_possible_param_sets, 250)
        assert 'binarized' not in y_col
        metric_to_sort_by = 'negative_rmse'
        if verbose:
            print("prediction task is CONTINUOUS")
    
    all_results = []
    for iterate_idx, sparsity_param in enumerate(sparsity_params):                         
        if binary_prediction:
            if use_nonlinear_model:
                raise Exception("Not implemented.")
            linear_model = sklearn.linear_model.LogisticRegression(fit_intercept=True, 
                                                                   C=sparsity_param, 
                                                                   solver='lbfgs')
            
            linear_model.fit(train_X, train_y)
            train_yhat = linear_model.predict_proba(train_X)[:, 1]
            val_yhat = linear_model.predict_proba(val_X)[:, 1]
            test_yhat = linear_model.predict_proba(test_X)[:, 1]
        else:
            if use_nonlinear_model:
                t_fit = time.time()
                model = RandomForestRegressor(random_state=42, n_jobs=10, **sparsity_param)
            else:
                model = sklearn.linear_model.Lasso(alpha=sparsity_param, fit_intercept=True)
            model.fit(train_X, train_y)
            train_yhat = model.predict(train_X)
            val_yhat = model.predict(val_X)
            test_yhat = model.predict(test_X)
            if not use_nonlinear_model:
                if sum(np.abs(model.coef_) > 1e-6) == 0:
                    continue
            else:
                print("Using random forest model: iterate %i/%i in %2.3f seconds" % (iterate_idx + 1, len(sparsity_params), time.time() - t_fit))
        performance = {}
        performance['train'] = assess_performance(y=train_y, 
                                                 yhat=train_yhat, 
                                                 binary_prediction=binary_prediction)
        performance['val'] = assess_performance(y=val_y, 
                                                 yhat=val_yhat, 
                                                 binary_prediction=binary_prediction)
        performance['test'] = assess_performance(y=test_y, yhat=test_yhat, binary_prediction=binary_prediction) 
        performance['test']['test_yhat'] = test_yhat
        performance['sparsity_param'] = sparsity_param
        all_results.append(performance)
        
    all_results = sorted(all_results, key=lambda x:x['val'][metric_to_sort_by])[::-1] 
    #print(pd.DataFrame(all_results))
    performance = all_results[0]

    if verbose:
        print("Best performance:")
        print("Sparsity param: %s" % performance['sparsity_param'])
        for dataset in ['train', 'val', 'test']:
            if dataset == 'train':
                n_examples = len(train_y) 
            elif dataset == 'val':
                n_examples = len(val_y)
            else:
                n_examples = len(test_y)
            print("%s performance (%i examples)" % (dataset, n_examples))
            for metric in performance[dataset].keys():
                if metric == 'test_yhat':
                    continue
                print("%s: %2.3f" % (metric, performance[dataset][metric]))
    return performance['test']['test_yhat']

def load_all_results(binary, min_timestring, thing_to_filter_by=None):
    """
    Loop over all fitted models and load results + configs. 
    Checked. 
    """
    if binary:
        key_to_sort_by = 'best_val_auc' #'test_auc'
    else:
        key_to_sort_by = 'best_val_negative_rmse' # 'negative_test_rmse'
        
    results_dir = os.path.join(FITTED_MODEL_DIR, 'results')
    config_dir = os.path.join(FITTED_MODEL_DIR, 'configs')
    all_results = []
    files = os.listdir(results_dir)
    for f in sorted(files)[::-1]:
        timestring = f.replace('_results.pkl', '')
        if timestring < min_timestring:
            continue
        results = pickle.load(open(os.path.join(results_dir, f), 'rb'))
        config = pickle.load(open(os.path.join(config_dir, '%s_config.pkl' % timestring), 'rb'))
        if config['dataset_kwargs']['use_very_very_small_subset']:
            continue
        if config['model_kwargs']['binary_prediction'] != binary:
            continue
        if binary:
            best_val_loss = min([results[a]['val_loss'] for a in results if type(a) is int])
            best_val_auc = max([results[a]['val_auc'] for a in results if type(a) is int])
            best_val_auprc = max([results[a]['val_auprc'] for a in results if type(a) is int])
            results = {'timestring':timestring, 
                      'best_val_loss':best_val_loss, 
                       'best_val_auc':best_val_auc, 
                       'best_val_auprc':best_val_auprc, 
                       'test_loss':results['test_set_results']['test_loss'], 
                       'test_auc':results['test_set_results']['test_auc'], 
                       'test_auprc':results['test_set_results']['test_auprc'], 
                       'test_yhat':results['test_set_results']['test_yhat'], 
                       'minutes_to_train':results['total_seconds_to_train'] / 60.}
        else:
            best_val_loss = min([results[a]['val_loss'] for a in results if type(a) is int])
            best_val_r = max([results[a]['val_r'] for a in results if type(a) is int])
            best_val_negative_rmse = max([results[a]['val_negative_rmse'] for a in results if type(a) is int])
            binarized_aucs = [results[a]['val_binarized_auc'] for a in results if type(a) is int and results[a]['val_binarized_auc'] is not None]
            best_val_binarized_auc = max(binarized_aucs) if len(binarized_aucs) > 0 else None

            binarized_auprcs = [results[a]['val_binarized_auprc'] for a in results if type(a) is int and results[a]['val_binarized_auprc'] is not None]
            best_val_binarized_auprc = max(binarized_auprcs) if len(binarized_auprcs) > 0 else None

            test_beta_ratio_education = None
            test_beta_ratio_income = None
            test_pain_gaps_klg_geq_2 = None
            test_results_stratified_by_klg = None
            val_results_stratified_by_site = None
            #if 'test_ses_betas' in results['test_set_results']:
            #test_set_betas = results['test_set_results']['test_ses_betas']
            #test_beta_ratio_education = test_set_betas['binarized_education_graduated_college_betas']['yhat_ses_beta'] / test_set_betas['binarized_education_graduated_college_betas']['klg_beta']
            #test_beta_ratio_income = test_set_betas['binarized_income_at_least_50k_betas']['yhat_ses_beta'] / test_set_betas['binarized_income_at_least_50k_betas']['klg_beta']
            if 'test_pain_gaps_klg_geq_2' in results['test_set_results']:
                test_pain_gaps_klg_geq_2 = results['test_set_results']['test_pain_gaps_klg_geq_2']
            
            if 'stratified_by_klg' in results['test_set_results']:
                test_results_stratified_by_klg = results['test_set_results']['stratified_by_klg']
            if 'stratified_by_site' in results[0]:
                val_results_stratified_by_site = {}
                for k in results:


                    if type(k) is int:
                        val_results_stratified_by_site[k] = results[k]['stratified_by_site']
                        val_results_stratified_by_site[k]['val_negative_rmse'] = results[k]['val_negative_rmse']
   
            results = {'timestring':timestring, 
                        'highest_train_correlation':max([results[a]['train_r'] for a in results if type(a) is int]),
                        'lowest_train_loss':min([results[a]['train_loss'] for a in results if type(a) is int]),
                      'best_val_loss':best_val_loss, 
                       'best_val_r':best_val_r, 
                       'best_val_negative_rmse':best_val_negative_rmse,
                       'best_val_binarized_auc':best_val_binarized_auc,
                       'best_val_binarized_auprc':best_val_binarized_auprc,
                       'test_loss':results['test_set_results']['test_loss'], 
                       'test_r':results['test_set_results']['test_r'], 
                       'negative_test_rmse':-results['test_set_results']['test_rmse'], 
                       'test_yhat':results['test_set_results']['test_yhat'], 
                       'test_y':results['test_set_results']['test_y'],
                       'test_high_ses_negative_rmse':results['test_set_results']['high_ses_negative_rmse'], 
                       'test_low_ses_negative_rmse':results['test_set_results']['low_ses_negative_rmse'], 
                       'test_high_ses_r':results['test_set_results']['high_ses_r'], 
                       'test_low_ses_r':results['test_set_results']['low_ses_r'], 
                       'test_beta_ratio_education':test_beta_ratio_education,
                       'test_beta_ratio_income':test_beta_ratio_income,
                       'test_pain_gaps_klg_geq_2':test_pain_gaps_klg_geq_2, 
                       'test_results_stratified_by_klg':test_results_stratified_by_klg,
                       'val_results_stratified_by_site':val_results_stratified_by_site,
                       'minutes_to_train':results['total_seconds_to_train'] / 60.}


        for k in config['dataset_kwargs']:
            results[k] = config['dataset_kwargs'][k]
        for k in config['model_kwargs']:
            results[k] = config['model_kwargs'][k]
        if 'experiment_to_run' in config:
            results['experiment_to_run'] = config['experiment_to_run']
        else:
            results['experiment_to_run'] = None
        if 'weighted_ses_sampler_kwargs' in config['dataset_kwargs'] and config['dataset_kwargs']['weighted_ses_sampler_kwargs'] is not None:
            results['p_high_ses'] = config['dataset_kwargs']['weighted_ses_sampler_kwargs']['p_high_ses']
        else:
            results['p_high_ses'] = None

        all_results.append(results)
        
    all_results = pd.DataFrame(all_results).sort_values(by=key_to_sort_by)[::-1]
    if thing_to_filter_by is not None:
        assert type(thing_to_filter_by) is dict
        for col in thing_to_filter_by:
            print("Filtering by %s=%s" % (col, thing_to_filter_by[col]))
            all_results = all_results.loc[all_results[col] == thing_to_filter_by[col]]
    
        
    print("Printing parameters correlated with top results!")
    assert sorted(list(all_results[key_to_sort_by]))[::-1] == list(all_results[key_to_sort_by])
    for c in all_results.columns:
        if not any([substring in c for substring in ['train_', 'test_', 'val_', 'timestring']]):
            df = copy.deepcopy(all_results[[c, key_to_sort_by]]).fillna('None')
            df[c] = df[c].map(str)
            grouped_d = df.groupby(c).agg(['mean', 'size'])
            grouped_d.columns = [key_to_sort_by, 'n']


            print(grouped_d.sort_values(by=key_to_sort_by)[::-1])
            n_top_trials_to_take = 10
            print("Of top %i trials, the values for this parameter are" % n_top_trials_to_take)
            print(Counter(df[c].iloc[:10]))
    return all_results

