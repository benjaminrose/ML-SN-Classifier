##!/bin/env python
"""
Create RandomForest classifier to type simulated DES SNe
http://scikit-learn.org
"""
import os, sys
import numpy as np
import re
from astropy import cosmology
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, join, vstack
from astropy.io.ascii import write, read
from astropy.io.misc import fnunpickle, fnpickle
from operator import add
from time import time
import traceback
import subprocess
import matplotlib
import math as m
from matplotlib import rc
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from operator import itemgetter
from scipy.stats import randint as sp_randint
from scipy.interpolate import interp1d
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import check_array
from sklearn.externals import joblib
# from treeinterpreter import treeinterpreter as ti
import argparse
import ML_globals as g
from ML_plots import make_plots

######### SN type variables used in code ########
# allSNtypes   -list of all possible true SN types  (eg [Ia, CC, Ibc, II])
# MLtypes      -list of types used by 2-way, 3-way, .. ML classification (eg [Ia, CC], [Ia, Ibc, II] ..)
# CLFtypes     -list of types predicted by ML classifier (CLF) (eg [RFIa, RFCC], ... )
# savetypes    -list of types for special printing, pickling etc
# trueplottypes -list of true types to be plotted (default = MLtypes + [Total])
# CLFplottypes -list of CLF types to be plotted (default = CLFtypes + [CLFTotal])
# plottypes    -list of all types to be plotted (default = MLtypes + [Total] + CLFtypes + [CLFTotal])
# alltypes     -list of true types and predicted types

######### Lists of data samples to be used in code (#############
# simlist    -list of simulated samples (Training, Validation)
# datalist   -list of "data" samples (Validation, Spec etc)
# MLdatalist -list of samples typed with ML (Validation + any other)
# plotlist   -list of samples to be plotted (default = [simlist, datalist])
# CLFplotlist -list of samples for CLF-typed plots (default=[[Validation], datalist])
# realdatalist -list of non-Validation data

########## Dicts used in code ###################
# alldata    -dict with training, test and any other selected data
# masks      -dict containing masks to select sub-samples of data

# Labels used for dict keys
probability = 'probability'
MLClasses = 'MLClasses'
_save = '_save'

#classifier stages (nclass==-2)
StI = 'Stage1_'
StII = 'Stage2_'

#output formats
txt = '.txt'
hdf5 = '.hdf5'
fits = '.fits'
pkl = '.pkl'

#version
version = '3.3'

############ Parse arguments ############
def parse_args(argv):
    # check for args coming from call or from sys.argv
    args_to_parse = sys.argv[1:] if sys.argv[0] in argv else argv

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script uses a machine learning (ML) algorithm to train a photometric classifier into classes '+\
                                                 '{Ia|CC or Ia|Ibc|II or I|II->Ia|Ibc}. Select Random Forest features to use.')
    # Choose features
    parser.add_argument('--ft',
                        nargs='+', default=['fit_pr', 'x1'],
                        help='Choose SN-type features to use for classification. List them (space-separated) and select from: {%(choices)s}',
                        metavar='features',
                        choices=[f for v in g.allowed_features.values() for f in v] + \
                                [f for v in g.alternate_feature_names.values() for f in v.values()],
                       )
    # Select number of cores
    parser.add_argument('--nc', type=int, choices=range(1, 8), default=4,
                        help='Number of cores to use for parallelization', metavar='n_cores')
    # Select efficiency
    parser.add_argument('--eff', nargs='+', type=float, default=[0.95],
                        help='Efficiencies at which to evaluate purity', metavar='Fix_eff')
    # Select number of classes
    parser.add_argument('--nclass', type=int, default=3,
                        help='Number of classes used in classifier (2=Ia/CC, 3=Ia/Ibc/II,'+
                             '-2=2x2 training: 2-stage--2-way classification with I/II -> Ia/Ibc)')
    # Training, validation and data options
    parser.add_argument('--train', default='DES_training_SNR550.txt', help='Filename for training')
    parser.add_argument('--train_format', choices=[f for v in g.allowed_formats.values() for f in v],
                        default=g.allowed_formats[g.default_format][0], help='Format of training data')
    # parser.add_argument('--train',default='DES_training_fitprob.txt',help='Filename for training')
    parser.add_argument('--validation', default="''", help='Filename for validation; set to null to skip')
    parser.add_argument('--test', default='DES_test_SNR550.txt', help='Filename for test (skip if set to "")')
    parser.add_argument('--data', default=[], nargs='+',
                        help='Classify selected simulated/observed data (default=Phot); '+\
                             'choose from [Test, Spec, Spec_nofp, Phot, ...] '+\
                             'plots (if selected) will be normalized to first data sample in list; '+\
                             'customize user-data labels by adding user-supplied labels here')                    
    parser.add_argument('--spec', default='specType_forcePhoto_bazin_v3.txt', help='Filename for spectroscopic data')
    parser.add_argument('--alltypes_colname_spec', default='spec_eval',
                         help='Column name for true SN types in Spec data')
    parser.add_argument('--phot', default='host_prior_DESY1-4_bazin_v3e.txt', help='Filename for photometric data')
    parser.add_argument('--spec_nofp', default='specSN_SNANA_nofpcut.txt', 
                        help='Filename for spectroscopic data without fit probabilities')
    # User-data options
    parser.add_argument('--user_data', nargs='+', default=[],
                        help='Filenames for user-supplied data; default labels in plots etc. will be '+\
                             'User0, User1, ... if no additional arguments supplied for --data' )
    parser.add_argument('--user_alltypes_colnames', nargs='+', default=[],
                         help='Column names for types of labeled user-supplied data (unlabeled data assumed); '+\
                              'use null string for unlabeled data in list of user-data files with mixed labeling properties. ')
    # File saving options
    parser.add_argument('--filedir', default='./',
                        help='Directory name (relative or absolute) for storing output files')
    parser.add_argument('--store', action='store_true', default=False,
                        help='Save trained classifier to pklfile')
    parser.add_argument('--restore', action='store_true', default=False,
                        help='Restore trained classifier from pklfile')
    parser.add_argument('--use_filenames',  action='store_true', default=False,
                        help='Use filenames as supplied (no default additions specifying formats, etc. ')
    parser.add_argument('--pklfile', default='trained_RFclassifier',
                        help='Base filename for reading/writing trained classifier (_{format}_{nclass}way.pkl is auto-appended; ' +\
                             'pre+filename also read/written if nclass == -2)')
    parser.add_argument('--pklformat', choices=[f for v in g.allowed_formats.values() for f in v],
                        default=g.allowed_formats[g.default_format][0], 
                        help='Format of stored training data')
    parser.add_argument('--train_only', action='store_true', default=False,
                        help='Run training only, save trained classifier to pklfile, and exit')
    parser.add_argument('--done_file', default="''", help='Path to file recording SUCCESS/FAILURE of run') 
    # Choose user file-id and format
    parser.add_argument('--filestr',
                        default="''", help='Choose string to append to filenames for output files')
    parser.add_argument('--format', default=txt,
                        help='Format for output of classification data (txt, hdf5, fits)')
    # Repository option for retrieving commit hash
    parser.add_argument('--commit_hash_path', default='./ML-SN-Classifier',
                        help='Path to local github repository for retrieving commit hash ' + \
                             "(set to '' to skip)")
    # Plot options
    parser.add_argument('--plots', nargs='+', default='',
                        choices=['', g.Performance, g.SALT, g.Hubble, g.Error, g.Color, g.Magnitude, g.Bazin],
                        help='Make selected groups of plots; '+\
                             'choose from Performance, SALT, Hubble, Color, Error, Magnitude, Bazin; '+\
                             'null argument supresses all plots')
    parser.add_argument('--weights', default='',
                        help='Name of data sample to use for computing weights for plot normalizations; defaults to first data sample in data list')
    parser.add_argument('--sim', choices=[g.Training, g.Validation], default=[g.Training], nargs='+',
                        help='Plot selected simulated data')
    parser.add_argument('--totals', action='store_true', default=False,
                        help='Turn on totals in plots (default if using --data or --user-data)')
    parser.add_argument('--save', choices=[g.Ia, g.CC, g.All], default=g.Ia, nargs='+', 
                        help='Types to save for printing pickling in plotting modules')
    parser.add_argument('--user_colors', nargs='+', default=['seagreen', 'springgreen', 'lime'],
                         help='Colors for user-supplied data (3 defaults available)')
    parser.add_argument('--user_markers', nargs='+', default=['<', '>', '8'],
                         help='Markers for user-supplied data (3 defaults available)')
    parser.add_argument('--minmax', action='store_true', default=False,
                        help='Turn on minmax print-out in plots')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Turn on debug print-out in plots')
    # Classification options
    parser.add_argument('--alltypes_colname_train', default='type3', 
                        help='Column name for true SN types in validation data, '+\
                             'eg. "type3" identifies SN as Ia, Ibc, II; '+\
                             ' code checks for consistency with --nclass and creates appropriate data column(s) if needed')
    parser.add_argument('--alltypes_colname_validation', default='type3', 
                        help='Column name for true SN types in validation data, ')
    parser.add_argument('--alltypes_colname_test', default='type3', 
                        help='Column name for true SN types in test data, ')
    parser.add_argument('--type_labels', default=g.data_defaults[g.default_format]['type_labels'], nargs='+',
                        help='Labels for classes in alltypes_colname (supply one per typename); CC assumed to be ~Ia')
    parser.add_argument('--type_values', default=g.data_defaults[g.default_format]['type_values'], nargs='+', 
                        help='Corresponding class values in type_labels')
    # Cross-validation
    parser.add_argument('--cv', action='store_true', default=False, help='Turn on cross-validation')
    parser.add_argument('--sample', choices=['t', 'v', 'b'], default='t',
                        help='Sample to use for cross-validation (t=training, v=validation, b=both)')
    parser.add_argument('--niter', type=int, choices=range(1, 11), default=5,
                        help='Number of ShuffleSplit cross-validation iterations')
    parser.add_argument('--testsize', type=float, default=0.5,
                        help='Validation sample size for ShuffleSplit')
    parser.add_argument('--nfolds', type=int, default=2,
                        help='Number of folds for Kfold sample')
    # Purity tests
    parser.add_argument('--pc', action='store_true', default=False, help='Turn on purity comparison')
    parser.add_argument('--purities', nargs='+', default=[0.1, 0.25, 0.5, 0.75, 0.9],
                        help='Purities for loop over test files')

    # witholding
    parser.add_argument('--withhold',
                        choices=[t for _,v in g.allowed_templates.items() for t in v],
                        nargs='+', default='', metavar='types/templates',
                        help='Hold-out test: withhold type (20=IIP, 21=IIN, 22=IIL, 32=Ib, 33=Ic) or '+\
                             'template (206,209=IIN, 002=IIL, 201-235=IIP, 103-234=Ib, 021-218=Ic) from training sample: '+\
                             'List them (space-separated) and select from: {%(choices)s}')
    parser.add_argument('--prvar', action='store_true', default=False, help='Turn on probability-variance plots')
                        
    # template statistics
    
    # Bazin parameter options
    parser.add_argument('--Bazincuts', choices=['train', 'plots', 'off'], default='off',
                        help='Include Bazin selections in training/plots (TBD), plots only, or turn off')
    parser.add_argument('--noBazinparcuts', action='store_true', default=False,
                        help='Turn OFF Bazin parameter cuts')
    parser.add_argument('--Bazinpar_max', nargs='+', default=[800, '', 150, 't_fall', 100],
                        help='Cuts on Bazin fit parameters: [A, t0, t_fall, t_rise, C]')
    parser.add_argument('--Bazinpar_min', nargs='+', default=[-999, 1, 0, 0, -999],
                        help='Cuts on Bazin fit parameters: [A, t0, t_fall, t_rise, C]')
    parser.add_argument('--Bazinerrcuts', action='store_true', default=False,
                        help='Turn ON Bazin error cuts')
    parser.add_argument('--Bazinerr_max', nargs='+', default=[100, 50, 100, 50, 100],
                        help='Cuts on Bazin fit errors: [A_err, t0_err, t_fall_err, t_rise_err, C_err]')

    # Cosmology Planck+WMAP polar+BAO+JLA
    parser.add_argument('--H0', type=float, default=68.62, help='Value of H0')
    parser.add_argument('--OmegaM', type=float, default=.301, help='Value of OmegaM')

    args = parser.parse_args(args_to_parse)

    return args


################ FUNCTIONS ##############
# get only the selected features
def get_features(features, data):
    flist = []
    for f in features:
        if not f in data.columns:
            print('    Warning: column {} not found; replacing with dummy values {}'.format(f, g.nodata))
            data[f] = g.nodata
        flist.append(data[f])

    X = np.vstack(flist).T
    return X

def check_features(features, data, format=g.default_format):

    if not all([f in data.columns for f in features]):

        other_formats = [k for k in g.alternate_feature_names.keys() if format not in k]
        for f in features:
            if not f in data.columns:
                alt_names = [v for fmt in other_formats for k, v in g.alternate_feature_names[fmt].items() if f==k]
                alts = [a for l in alt_names for a in l if type(l)==list] + [l for l in alt_names if type(l)!=list] #flatten
                alt_features = [a for a in alts if a in data.columns]
                if len(alt_features) > 0:
                    features[features.index(f)] = alt_features[0]
                    print('    Replacing feature name {} with {} to match {} format'.format(f, alt_features[0], format))

    return features

# find index of array where value is nearest given value
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


# score function to compute purity at fixed efficiency (=TPR)
def score_func(probs, y, Fix_eff, pos_label=0):
    #Ia = (y == 0)
    #CC = (y > 0)
    pur, eff, thresh = precision_recall_curve(y, probs, pos_label=pos_label)
    purity_func = interp1d(eff[::-1], pur[::-1], kind='linear')  # reverse-order so x is monotonically increasing
    metric = purity_func(Fix_eff)  # purity at fixed efficiency
    return float(metric)


def score_func_est(estimator, X, y, Fix_eff, pos_label=0):
    probs = estimator.predict_proba(X)[:, pos_label]  # SNIa are class 0
    #correct = (y == 0)
    #wrong = (y > 0)
    pur, eff, thresh = precision_recall_curve(y, probs, pos_label=pos_label)
    purity_func = interp1d(eff[::-1], pur[::-1], kind='linear')  # reverse-order so x is monotonically increasing
    metric = purity_func(Fix_eff)  # purity at fixed efficiency
    return float(metric)


def get_purity_scores(yset, probs, effcies, pos_label=0):
    print('\n  Computing purity scores for fixed efficiencies) {}'.format(' '.join([str(f) for f in effcies])))
    pur, eff, thresh = precision_recall_curve(yset, probs, pos_label=pos_label)
    # number of unique probablility values
    print('    Length of threshold array (unique prob. values) = {}'.format(len(thresh)))
    y = thresh[::-1]  # reverse order so thresholds decrease
    x = eff[::-1][1::]  # reverse order so efficiency increasing (for interp1d), remove last zero

    P_eff_list = []
    score_list = []
    if len(x) > 1 and len(y) > 1:
        efficiency_func = interp1d(x, y, kind='linear')  # reverse-order so x is monotonically increasing

        # loop over efficiencies
        for effcy in effcies:
            P_eff = efficiency_func(effcy)  # threshold probability at efficiency=effcy
            print('\n    Threshold probability (P_thresh) (eff={:0.1f}%) = {:.3f}'.format(effcy * 100, float(P_eff)))
            print('    Purity (P_thresh=0) = {:.3f}'.format(pur[pos_label]))
            score = score_func(probs, yset, effcy)
            print('    SCORE (purity @ eff={:0.1f}%) = {:.4f}'.format(effcy * 100, score))
            P_eff_list.append(P_eff)
            score_list.append(score)
    else:
        print('    Unable to interpolate efficiency function (insufficient unique values)')
        
    return pur, eff, thresh, P_eff_list, score_list


def get_ROC(yset_ROC, probs, pos_label=0):
    ### Compute ROC curve and AUC
    if len(np.unique(yset_ROC)) > 1:
        fpr, tpr, roc_thres = roc_curve(yset_ROC, probs, pos_label=pos_label)  # restricted to binary classification
        AUC = 1 - roc_auc_score(yset_ROC, probs)  # need (1 - score) if pos_label=0
        print('\n  AUC = {:.3f}'.format(AUC))
    else:
        print('\n  Only one class present in data. ROC AUC score is not defined in that case.')
        fpr, tpr, roc_thres, AUC = None, None, None, None
    return fpr, tpr, roc_thres, AUC


def get_prob(clf, traindata, testdata, ytrain, debug=False, ntest=None):
    # computes probability from fractions of training-data classes in test-data landing leaf
    # DOES NOT account for duplicates
    classvals = list(set(y_train))
    leafs = clf.apply(traindata)  # all leaf nodes in training data
    average = np.zeros(shape=(len(testdata), len(classvals)))
    variance = np.zeros(shape=(len(testdata), len(classvals)))
    if (ntest is None):
        ntest = len(testdata)
    allnodes = clf.apply(testdata)  # landing nodes in decision trees (apply returns array of lists)
    for nt, nodes in enumerate(allnodes[0:ntest]):
        test = testdata[nt]
        if (debug):
            print("Validation data # {} = {}".format(nt, test))
            print("Leaf nodes = {}".format(nodes))
        sumpr = np.array([0.] * len(classvals))  # init sum
        sumpr2 = np.array([0.] * len(classvals))  # init sum
        for n, node in enumerate(nodes):  # loop thru nodes in trees
            match = (leafs.T[n] == node)
            classes = ytrain[match]  # classes of objects landing in same node
            if (debug):
                print('# of matching classes = {} for {}th tree'.format(len(classes), n))
                cldist = [np.sum(classes == x) for x in classvals]
                print("# of each class = {}".format(cldist))
                print("Prob from estimator = {}".format(clf.estimators_[n].predict_proba(X_test)[nt]))
            probs = [float(np.sum(classes == x)) / float(len(classes)) for x in classvals]
            sumpr = sumpr + np.asarray(probs)
            sumpr2 = sumpr2 + np.asarray(probs) * np.asarray(probs)
            if (debug):
                print("Probs = {} from #matches = {} in {}th tree".format(probs, len(classes), n))

        average[nt] = sumpr/float(clf.get_params()['n_estimators'])
        variance[nt] = np.sqrt(sumpr2/float(clf.get_params()['n_estimators']) - average[nt] * average[nt])

    # endfor
    return average, variance


def prob_errors(model, X, percentile=90, debug=False, ntest=None):
    if (ntest is None):
        ntest = len(X)
    if len(model) > 1:
        print('Prob errors: TBD')
        return
    
    model = model[0]  #one classifier
    sumpr = np.zeros_like(model.estimators_[0].predict_proba(X))
    sumpr2 = np.zeros_like(model.estimators_[0].predict_proba(X))
    err_up = np.zeros_like(model.estimators_[0].predict_proba(X))
    err_down = np.zeros_like(model.estimators_[0].predict_proba(X))
    nclass = model.n_classes_
    pdata = []
    for n, dtree in enumerate(model.estimators_):
        probs = dtree.predict_proba(X)
        if (debug):
            print("Probs = {} for {}th tree".format(probs[0:ntest], n))
        sumpr = sumpr + probs
        sumpr2 = sumpr2 + probs * probs
        for p, prob in enumerate(probs):
            if (n == 0):  # first run through pdata list
                pdata.append([[] for c in range(nclass)])
            for c in range(nclass):
                pdata[p][c].append(prob[c])  # accumulate probs for each tree

    # compute percentiles
    pdata = np.asarray(pdata)
    for p, data in enumerate(pdata):
        for c in range(nclass):
            err_down[p][c] = np.percentile(pdata[p][c], (100 - percentile) / 2.)
            err_up[p][c] = np.percentile(pdata[p][c], 100 - (100 - percentile) / 2.)

    average = sumpr / float(model.get_params()['n_estimators'])
    variance = np.sqrt(sumpr2 / float(model.get_params()['n_estimators']) - average * average)
    return average, variance, err_up, err_down, pdata


def read_data_file(key, datafile, SNRcut=5, zhi=g.zhi, SNRkey='snr1', zkey='z',
                   format=g.default_format, doBazincuts=False):

    data = Table()
    status = False
    # check for acceptable data format
    if not any([f in datafile for v in g.allowed_formats.values() for f in v]):
        print('  Unknown datafile format; expected formats are {}'.format(
                ', '.join([f for v in g.allowed_formats.values() for f in v])))
        return data, status

    # read data file 
    try:
        msg = '\nReading {} data from {}'.format(key, datafile)
        print('{}'.format(msg))
        print('{}'.format(''.join(['-' for c in msg])))
        data = read(datafile)
        print('  Initial size: {}'.format(len(data)))

        #make SNRcut
        if SNRkey in data.colnames:
            snrpass = (data[SNRkey] > SNRcut)
            nfail = len(data[SNRkey]) - np.count_nonzero(snrpass)
            if nfail > 0:
                print('  Removing {} entries failing {} > {}'.format(nfail, SNRkey, SNRcut))
                data = data[snrpass]
        else:
            print('  Data column {} not found'.format(SNRkey))

        # cut out any data with z>zhi (otherwise z-histograms get messed up)
        if zkey in data.colnames:
            zpass = (data[zkey] < zhi)
            nzfail = len(data[zkey]) - np.count_nonzero(zpass)
            if nzfail > 0:
                print('  Removing {} entries failing {} > {}'.format(nzfail, zkey, zhi))
                data = data[zpass]
        else:
            print('  Data column {} not found'.format(zkey))

        # Apply any other cuts to data eg. Bazin setup
        if (doBazincuts):
            print ('\n  Not applying Bazin cuts to {} data'.format(key))
            # would have to change get_Bazincuts to work on data_train

        print('  Total Size of {} Data (after cuts) = {}'.format(key, len(data)))
        print('  Available feature names: \n  {}'.format(', '.join(data.colnames)))
        status = True

    except BaseException as ex:
        print('  Error reading data file {}'.format(datafile))
        print('  {}'.format(str(ex)))

    return data, status


def create_type_columns(data, nclass, all_class_values, alltypes_colname='type3',
                        desired_class_values=g.desired_class_values,
                        type_colnames={}, abort=False, format=g.default_format):

    status = True
    print('\n  Checking format and values for columns with labeled data')
    alltypes_colname_default=g.data_defaults[format]['alltypes_colname_default']

    # Check for type column needed for training and create if needed
    if alltypes_colname not in data.colnames:
        print('    Warning: requested alltypes-column-name {} not found'.format(alltypes_colname))
    else:
        # Check format of alltypes_colname and convert from character types if need be
        if data[alltypes_colname].dtype != int:
            try:
                sntypes = np.unique(data[alltypes_colname]).tolist()
                print('    Found class-label values of {}'.format(' '.join(sntypes)))
                print('    Converting string class labels to integers')
                data.rename_column(alltypes_colname, alltypes_colname + _save)
                data[alltypes_colname] = g.nodata
                for t in sntypes:
                    for k, v in desired_class_values.items():
                        if k in t:    #match of class
                            tmask = data[alltypes_colname + _save] == t
                            data[alltypes_colname][tmask] = int(v)
                
                # Now overwrite target class with the newly assigned default value
                all_class_values = g.desired_class_values

            except BaseException as ex:
                print('    Unable to convert type-labels {} to standard format: skipping use of class labels'.\
                       format(all_class_values))
                print('    {}'.format(str(ex)))
                status = False

    # Check for type-columns appropriate to nclass and create them if not available
    try:
        # Copy alternate column name into alltypes_colname_default for later use (eg to generate ROC curves)
        if alltypes_colname != alltypes_colname_default and alltypes_colname_default not in data.colnames:
            data[alltypes_colname_default] = data[alltypes_colname]
            print('    Copying column {} to column {} (to be used for ROC curves etc)'.format(alltypes_colname,
                                                                                            alltypes_colname_default))
        # Check for needed column name to do n-way typing and generate if need be
        if str(nclass) in type_colnames.keys():
            colnames = type_colnames[str(nclass)] if type(type_colnames[str(nclass)])==list \
                                                  else [type_colnames[str(nclass)]]
            for cname in colnames:
                target_class_values = all_class_values[g.Ia] if type(all_class_values[g.Ia]) == list else \
                                          [all_class_values[g.Ia]]
                if cname not in data.colnames or cname == 'type2x2': #force 2x2 recompute
                    print('    Creating column "{}" used for {}-way typing'.format(cname, nclass))
                    if nclass == 2:  #binary class
                        print('    Using {} to create "{}"'.format(alltypes_colname, cname))
                        data[cname] = data[alltypes_colname] #copy column with all types
                        mask1 = np.ones(len(data), dtype=bool)
                        for t in target_class_values:
                            mask1 &= (data[cname] != int(t))

                        data[cname][mask1] =  g.desired_class_values[g.CC]  #binary data
                        data[cname][~mask1] = g.desired_class_values[g.Ia]
                    elif cname == 'type2x2':
                        data[cname] = data[alltypes_colname] # copy column with all types
                        if g.data_defaults[format]['alltypes_available']:
                            mask1 = data[cname] ==  all_class_values[g.Ibc]
                        else:
                            mask1 = get_mask_from_templates(data[g.generic_feature_names['sim_template'][format]], [g.Ib, g.Ic])
                        data[cname][mask1] = g.desired_class_values[g.Ia]  # reset label for Ibc to Ia value
                    elif nclass == 3:
                        print('    Attempting to use template information from data column {} to reconstruct SN types'.\
                                 format(g.generic_feature_names['sim_template'][format]))
                        data[cname] = g.desired_class_values[g.II]  # start with type 2
                        print('    Initialized column "{}" with values = {}'.format(cname, g.desired_class_values[g.II]))
                        for sn in [[g.Ia], [g.Ib, g.Ic]]:
                            mask = get_mask_from_templates(data[g.generic_feature_names['sim_template'][format]], sn)
                            if np.count_nonzero(mask) > 0:
                                data[cname][mask] = g.desired_class_values[sn[0]]
                    else:
                        print('  Unknown column name {}'.format(colname))
                        status=False
                    values, counts = np.unique(data[cname], return_counts=True)
                    for i, (v, c) in enumerate(zip(values, counts)):
                        print('    {} class values set to {}'.format(c, v))
                    if len(counts) != abs(nclass):    # check number of types available
                            print('    Warning: Unexpected number of types ({}) obtained in column {}'.format(len(counts), cname))
                            status = not(abort) #reset status if this would be fatal for training 
    except BaseException as ex:
        print('  Unable to read/create labeled column from available data')
        print('    {}'.format(str(ex)))
        status = False

    return data, status

def get_mask_from_templates(templates, sntypes=[g.Ia], Templates_Ia=[0]):
    
    mask = np.zeros(len(templates), dtype=bool)
    for sn in sntypes:
        if sn in g.allowed_templates or sn==g.Ia:
            allowed_templates =  Templates_Ia if sn==g.Ia else g.allowed_templates[sn]
            for t in allowed_templates:
                mask |= (templates == int(t))

    print('      Created mask for {} type {}s in data'.format(np.count_nonzero(mask), '+'.join(sntypes)))
    
    return mask


def print_heading(v):

    print('******************************************')
    print('*              {}_v{}                *'.format(g.name, v))
    print('*  {}  *'.format(g.description))
    print('* {} *'.format(', '.join(g.authors)))
    print('******************************************')    

    return


def retrieve_commit_hash(path_to_repo):
    """ Return the commit hash of the git branch currently live in the input path.
    Parameters
    ----------
    path_to_repo : string
    Returns
    -------
    commit_hash : string
    """
    cmd = 'cd {0} && git rev-parse HEAD'.format(path_to_repo)
    return subprocess.check_output(cmd, shell=True).strip()


def build_RF_classifier(data_train, nclass, features, ncores, alltypes_colname, target_class=0,
                        type_colnames=g.data_defaults[g.default_format]['type_colnames'],
                        dummy=g.nodata, start_time=-1,
                        n_estimators = 100, max_features = 'auto', min_samples_split = 5, criterion = 'entropy'):

    print
    print('\n*************************')
    print('*  BUILDING CLASSIFIER  *')
    print('*************************')
    print('Training with {} features [{}]'.format(len(features), ' '.join(features)))
        
    #min_samples_split = 5  NB: 500 works well for fit_pr alone
    #criterion = 'entropy' or 'gini'

    clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, 
                                 min_samples_split=min_samples_split, criterion=criterion, 
                                 n_jobs=ncores)

    ### fit the training data
    print('\n  Training classifier(s) across {} cores  . . . '.format(ncores))

    oldcols = data_train.columns
    if nclass == -2:  #staged classifiers in 2 steps
        # define pre-classifier to separate Iabc's from IIs
        clfpre = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                        min_samples_split=min_samples_split, criterion=criterion, 
                                        n_jobs=ncores)
        X_pretrain = get_features(features, data_train)
        colname = type_colnames[str(nclass)][0]    #type column for pre-classifier
        class_values_pretrain = data_train[colname] #Iabc's are all type 0 in this column
        print("  Pre-training stage1 classifier:\n set size = {}, \tNumber of Iabc's= {}".format(len(y_pretrain),
                                                                           np.count_nonzero(class_values_pretrain == target_class)))
        clfpre.fit(X_pretrain, class_values_pretrain)
        cut2 = (data_train[alltypes_colname] != target_class)  # identify type II
        colname = type_colnames[str(nclass)][-1]   #type  column for classifier
        data_train_abc = data_train[~cut2]
        X_train = get_features(features, data_train_abc)  # include only Ia and Ibc
        class_values_train = data_train_abc[colname]
        # train Ia/bc classifier
        print("  Training stage2 classifier:\n set size = {}, \tNumber of Ia's= {}".format(len(class_values_train),
                                                              np.count_nonzero(class_values_train == target_class)))
        clf.fit(X_train, class_values_train)

        # get probabilities        
        data_train = get_probabilities(clfpre, data_train, X_pretrain, xtralabel=StI)
        data_train_abc = get_probabilities(clf, data_train_abc, X_train)

        classifiers = [clfpre, clf]
        # save extra (truncated) columns into data_train adding dummy values for missing entries
        for cn in data_train_abc.colnames - oldcols:
            data_train[cn] = dummy
            data_train[cn][~cut2] = data_train_abc[cn]
        # create new column for stage-2 class values adding dummy values for missing entries
        data_train[colname + StII] = dummy
        data_train[colname + StII][~cut2] = data_train_abc[colname]
        class_values = [class_values_pretrain, data_train[colname + StII]]
        X_train = [X_pretrain, X_train] #NB different lengths
    else:
        X_train = get_features(features, data_train)
        colname = type_colnames[str(nclass)]   #not a list for 1-stage classifier!
        class_values = data_train[colname]
        print("  Training set size = {};   Number of Ia's = {}".format(len(class_values), 
                                         np.count_nonzero(class_values == target_class)))
        clf.fit(X_train, class_values)
        classifiers = [clf]
        data_train = get_probabilities(clf, data_train, X_train)

    print('  New columns added to table:\n    {}'.format(', '.join([c for c in data_train.columns if c not in oldcols])))

    return classifiers, data_train, class_values, X_train

def get_probabilities(clf, data, feature_values, xtralabel=''):

    allprobs = clf.predict_proba(feature_values)
    #save columns of probabilities (last column is redundant)
    for c in range(allprobs.shape[1]):
        colname = xtralabel + g.RFprobability + str(c)
        data[colname] = allprobs[:,c]

    maxprobclass = clf.predict(feature_values)
    data[xtralabel + g.MaxProbClass] = maxprobclass 
    
    return data
    

def run_RF_classifier(dkey, classifiers, data, nclass, features, 
                          method=g.MaxProbClass, target_class=0, dummy=g.nodata):

    ### fit the training data
    print('\n  Classifying {} Data'.format(dkey))

    oldcols = data.columns
    if nclass == -2:  #staged classifiers in 2 steps
        f_1 = get_features(features, data)
        clfpre = classifiers[0]
        # get probabilities        
        data = get_probabilities(clfpre, data, f_1, xtralabel=StgI)
        cut2 = data[StgI+method] != target_class  # identify type II classified by RF
        data_abc = data[~cut2]
        f_2 = get_features(features, data_abc)  # include only RF-classified Ia and Ibc
        clf = classifiers[-1]
        data_abc = get_probabilities(clf, data_abc, f_2)
        # save extra (truncated) columns into data adding dummy values for missing entries
        for cn in data_abc.colnames - oldcols:
            data[cn] = dummy
            data[cn][~cut2] = data_abc[cn]
            
        # create new column for stage-2 class values adding dummy values for missing entries
        data[colname + StII] = dummy
        data_train[colname + StII][~cut2] = data_train_abc[colname]
        feature_set = [f_1, f_2]
    else:
        feature_set = get_features(features, data)
        clf = classifiers[0]
        data = get_probabilities(clf, data, feature_set)

    print('  New columns added to table:\n    {}'.format(', '.join([c for c in data.columns if c not in oldcols])))

    return data, feature_set

def get_ML_results(dkey, dataset, classifiers, MLtypes, class_values_list, effcy, nclass,
                   masks={}, all_class_values=g.desired_class_values, target_column=0, target_class=0, 
                   P_threshold=None, CLFid=g.RF):

    performance = {}

    if g.Ia not in list(all_class_values.keys()):         # require default SN typing
        print('  Other class options/values not implemented yet')
        return performance, masks

    #find type corrresponding to target_class value
    target_type = list(all_class_values.keys())[list(all_class_values.values()).index(int(target_class))]
    
    print("\n{} DATA RESULTS\n  Target type: {}  (Class_value = {})".format(dkey.upper(),
                                                                            target_type, target_class))
    
    if not type(class_values_list) == list:
        class_values_list = [class_values_list]

    oldcols = dataset.colnames
    for n, (cl, true_classes) in enumerate(zip(classifiers, class_values_list)):
        
        # Setup keys for multi-stage classifier
        if len(classifiers) > 1:
            print('\n  Stage {} Classification'.format(n+1))
            prekey = StI if n==0 else StII
            #TBD get rid of -999s in data-columns for stage2 classification
        else:
            prekey = ''
   
        colname = prekey + g.RFprobability  #colname for probabilities in astropy table
        allprobs_colname = colname + str(target_column)
        print('  Statistics for {} Data Set'.format(dkey))
        print('  Total size = {}'.format(len(dataset[allprobs_colname])))

        # create true dict
        if true_classes is not None:
            if g.TrueType not in masks.keys():
                masks[g.TrueType] = {}
            if dkey not in masks[g.TrueType].keys():
                masks[g.TrueType][dkey] = {}
            for t, v in all_class_values.items():
                if t in MLtypes:
                    if len(classifiers) > 1 and n == 0 and t == g.II:   #2-stage
                        masks[g.TrueType][dkey][t] = true_classes != target_class # stage 0 distinguishes type II only
                    elif t not in masks.keys():             #don't overwrite existing key
                        masks[g.TrueType][dkey][t] = true_classes == int(v)   #mask for numbers in true class 
                        
            if n == len(classifiers) - 1:    #assemble final statistics and print
                if g.CC in MLtypes:
                    masks[g.TrueType][dkey][g.CC] =  true_classes != target_class
                for t in MLtypes:
                    print('  True number of {}  = {}'.format(t, np.count_nonzero(masks[g.TrueType][dkey][t])))

            #save true type id to table
            dataset[g.TrueClass_id] = str(g.nodata)
            for t in MLtypes:
                dataset[g.TrueClass_id][masks[g.TrueType][dkey][t]] = t
            
            ### Compute efficiencies, purities for theshold probability classification
            purity, efficiency, thresh, Pthresh_eff, score = get_purity_scores(true_classes, 
                                                                               dataset[allprobs_colname], effcy)
            performance[prekey + g.Purity] =  purity
            performance[prekey + g.Efficiency] =  efficiency
            performance[prekey + g.Threshold] = thresh
            for eff, pthrsh, s in zip(effcy, Pthresh_eff, score):
                eff_id = g._Eff.format(eff)
                performance[prekey + g.P_Threshold + eff_id] = pthrsh  
                performance[prekey + g.Score + eff_id] =  s
                
        print('\n  Predictions using Fixed-Efficiency Classification:')
        for j, eff in enumerate(effcy):
            # create predict dicts for storing ML results
            eff_id = g._Eff.format(eff)
            predict = g.Eff_.format(eff)
            if predict not in masks.keys():     #add sub-dict if needed
                masks[predict] = {}
            if dkey not in masks[predict].keys():
                masks[predict][dkey] = {}
            # determine P_threshold value to use
            if P_threshold is None  or len(P_threshold) ==0:
                #default to Pthresh_eff if available
                if true_classes is not None and prekey + g.P_Threshold + eff_id in performance.keys():
                    Pthresh = performance[prekey + g.P_Threshold + eff_id]
                    if dkey != g.Test:
                        print('\n    Reverting to fixed-efficiency value {:.3f} based on true types for {}'\
                                   .format(Pthresh, dkey))
                else:
                    print('\n  P_threshold value not available: skipping fixed-efficiency classification')
                    break
            else: 
                Pthresh = P_threshold[j]   #supplied value from Test data
      

            if target_type in list(all_class_values.keys()):  
                if len(classifiers) > 1 and n==0: # stage 0 distinguishes type II
                    masks[predict][dkey][CLFid + g.II] = dataset[allprobs_colname] < Pthresh
                else:
                    masks[predict][dkey][CLFid + target_type] = dataset[allprobs_colname] >= Pthresh
                    if nclass == 3:
                        masks[predict][dkey][CLFid + g.II] = (dataset[allprobs_colname] < Pthresh) & \
                                                       (dataset[colname+'2'] > dataset[colname+'1'])
                        masks[predict][dkey][CLFid + g.Ibc] = ~masks[predict][dkey][CLFid + g.Ia] & \
                                                        ~masks[predict][dkey][CLFid + g.II]
                    elif nclass == -2:   
                        masks[predict][dkey][CLFid + g.Ibc] = ~masks[predict][dkey][CLFid + g.Ia]
            else:
                print('Generic classification TBD')

            if n == len(classifiers) - 1:    #assemble final statistics
                #special case for CC (need all other classes determined first)
                if g.CC in MLtypes:
                    masks[predict][dkey][CLFid + g.CC] = ~masks[predict][dkey][CLFid + g.Ia]        

                # assemble TF classifications (TP, FP, TN, FN)
                if true_classes is not None:
                    masks[predict][dkey] = get_TF_masks(CLFid, MLtypes, target_type, masks[predict][dkey], masks[g.TrueType][dkey])
                
                # print summary
                heading = 'fixed-efficiency classification with P_thresh {:.3f}'.format(float(Pthresh))
                get_prediction_summary(CLFid, MLtypes, masks[predict][dkey], dkey, target_type, true_classes, heading)

                #save classification to table
                pthresh_id = '(PThr={:.3f})'.format(Pthresh)
                effcolname = g._FixedEffClass + eff_id + pthresh_id
                #effcolname_id = g._FixedEffClass + eff_id + pthresh_id + g._id
                dataset = save_predictions(CLFid, MLtypes, dataset, effcolname, masks[predict][dkey], 
                                           all_class_values, true_classes, target_type)
                    
        #Max-prob prediction
        if g.MaxProb not in masks.keys():
            masks[g.MaxProb] = {}
        if dkey not in masks[g.MaxProb].keys():
            masks[g.MaxProb][dkey] = {}
        maxcolname = prekey + g.MaxProbClass
        maxprobclass = dataset[maxcolname]
        for t, v in all_class_values.items():
            if t in MLtypes:
                if len(classifiers) > 1 and n==0 and t==g.II:
                    masks[g.MaxProb][dkey][CLFid + t] = maxprobclass == int(v)
                elif CLFid + t not in masks[g.MaxProb].keys():
                    masks[g.MaxProb][dkey][CLFid + t] = maxprobclass == int(v)

        if n == len(classifiers) - 1:    #assemble final statistics
            if g.CC in MLtypes:
                masks[g.MaxProb][dkey][CLFid + g.CC] = ~masks[g.MaxProb][dkey][CLFid + g.Ia]

            if true_classes is not None:
                masks[g.MaxProb][dkey] = get_TF_masks(CLFid, MLtypes, target_type, masks[g.MaxProb][dkey], masks[g.TrueType][dkey])

            # print summary
            heading = 'Maximum-Probability Classification'
            get_prediction_summary(CLFid, MLtypes, masks[g.MaxProb][dkey], dkey, target_type, true_classes, heading)

            #save classification to table
            dataset = save_predictions(CLFid, MLtypes, dataset, g._MaxProbClass, masks[g.MaxProb][dkey], 
                                       all_class_values, true_classes, target_type)
            
            if true_classes is not None:
                Eff, Pur, Rem = g.EffPur(masks[g.TrueType][dkey][g.Ia], maxprobclass == target_class)
                performance[prekey + g.Efficiency_MaxProb], performance[prekey + g.Purity_MaxProb],\
                                                  performance[prekey + g.Reject_MaxProb] = Eff, Pur, Rem
                print('\n    Performance for {} data using Max. Prob. Classification:'.format(dkey))
                print('      Purity = {:.3f}, Efficiency = {:.3f}'.format(Pur, Eff))
                print('      Number of SN rejected = {} (Fraction = {:0.2f})'.format(Rem,
                             float(Rem)/len(dataset[allprobs_colname])))

    print('\n  New columns added to data table:')
    print('    {}'.format(', '.join([c for c in dataset.columns if c not in oldcols])))

    return performance, masks

def get_TF_masks(CLFid, MLtypes, target_type, masks_predict, masks_true):

    for t in MLtypes:
        if t == target_type:
            masks_predict[CLFid + t + g.TP] = masks_predict[CLFid + t] & masks_true[t]
            masks_predict[CLFid + t + g.FP] = masks_predict[CLFid + t] & ~masks_true[t]
        else:
            masks_predict[CLFid + t + g.TN] = masks_predict[CLFid + t] & masks_true[t]
            masks_predict[CLFid + t + g.FN] = masks_predict[CLFid + t] & ~masks_true[t]
    
    return masks_predict

def get_prediction_summary(CLFid, MLtypes, masks, dkey, target_type, true_classes, heading):
    print('\n    Predictions using {}:'.format(heading))
    for t in MLtypes:
        print('\n      Predicted number of {}-Data {} = {}'.format(dkey, t,
                  np.count_nonzero(masks[CLFid + t])))
        if true_classes is not None:
            true_id = g.TP if t == target_type else g.TN
            false_id = g.FP if t == target_type else g.FN
            print('        Correct (true) number of {}-Data {} = {}'.format(dkey, 
                      t, np.count_nonzero(masks[CLFid + t + true_id])))
            print('        Incorrect (false) number of {}-Data {} = {}'.format(dkey, 
                      t, np.count_nonzero(masks[CLFid + t + false_id])))

    return

def save_predictions(CLFid, MLtypes, dataset, newcolname, masks, all_class_values, 
                     true_classes, target_type):

    newcolname_id = newcolname + g._id
    dataset[CLFid + newcolname] = g.nodata  #initial values
    dataset[CLFid + newcolname_id] = '  ' + str(g.nodata) #initial values
    if true_classes is not None:
        dataset[CLFid + g.TF + newcolname_id] = '    ' + str(g.nodata)
    for t in MLtypes:
        if t != g.CC:
            dataset[CLFid + newcolname][masks[CLFid + t]] = all_class_values[t]
        else:
            dataset[CLFid + newcolname][masks[CLFid + t]] = g.desired_class_values[g.CC]
        dataset[CLFid + newcolname_id][masks[CLFid + t]] = CLFid + t
        if true_classes is not None:
            if t == target_type:
                dataset[CLFid + g.TF + newcolname_id][masks[CLFid + t + g.TP]] = CLFid + t + g.TP
                dataset[CLFid + g.TF + newcolname_id][masks[CLFid + t + g.FP]] = CLFid + t + g.FP
            else:
                dataset[CLFid + g.TF + newcolname_id][masks[CLFid + t + g.TN]] = CLFid + t + g.TN
                dataset[CLFid + g.TF + newcolname_id][masks[CLFid + t + g.FN]] = CLFid + t + g.FN

    return dataset

def get_template_statistics(template_data, masks, MLtypes, classifications, dkey=g.Test, CLFid=g.RF):

    template_info = {}
    print("\n  Template Statistics for {} Data".format(dkey))

    for cl in classifications:
        if cl == g.MaxProb:
            print('\n    Maximum Probability Classification')
            cl_id = g.MaxProb
        else:
            print('\n    Efficiency = {}'.format(cl))
            cl_id = g.Eff_.format(float(cl))
        # check for available classifications
        if CLFid + g.Ia not in masks[cl_id][dkey].keys():
            print('    {} classification not available; skipping template statistics'.format(cl_id))
            continue
        template_info[cl_id] = {}
        predict = masks[cl_id][dkey]
        true = masks[g.TrueType][dkey]
        templates, counts = np.unique(template_data.quantity.value, return_counts=True)
        template_dict = dict(zip(templates, counts))

        # statistics for numbers of true and ML types for each template
        template_stats = {}
        for tmplt in template_dict:
            template_stats[tmplt] = {}
            for typ in MLtypes:
                template_stats[tmplt][CLFid + typ] = np.count_nonzero(predict[CLFid + typ] & (template_data == tmplt))
                template_stats[tmplt]['True' + typ] = np.count_nonzero(true[typ] & (template_data == tmplt))
                if (template_stats[tmplt]['True' + typ] > 0):
                    template_stats[tmplt]['Type'] = typ

        # count template occurrences for SN classified as Ia
        CLFIa_mask = masks[cl_id][dkey][g.RFIa]
        CLFIa_templates, CLFIa_counts = np.unique(template_data[CLFIa_mask].quantity.value, return_counts=True)
        Iatemplate_dict = dict(zip(CLFIa_templates, CLFIa_counts))

        print ('\n Templates for SN Classified as {}'.format(g.RFIa))
        print ('    Template | Value')
        # need to print in frequency order
        keys = Iatemplate_dict.keys()
        template_freq = sorted(Iatemplate_dict.values())
        template_freq.reverse()
        ordered_keys = []
        for freq in template_freq:
            # index=freqs.index(freq)
            for key in keys:
                if Iatemplate_dict[key] == freq:
                    if not (key in ordered_keys):
                        print ('    {:6} |{:6}'.format(key, freq))
                        ordered_keys.append(key)

        npop = 5
        print('   {} most popular templates and frequencies for passing {}Ia {} {}'.format(npop, CLFid, 
                                                                                    ordered_keys[0:npop], 
                                                                                    template_freq[0:npop]))
        template_info[cl_id][g.Counts] = template_dict
        template_info[cl_id][g.Stats] = template_stats
        template_info[cl_id][g.Ia] = Iatemplate_dict
        
    return template_info


def run_cross_validation(data, features, nclass, start_time=-1, nc=4, niter=5,
                         type_colnames=g.data_defaults[g.default_format]['type_colnames']):
        
    kvals = []
    avgskf = []
    stdkf = []
  
    tsvals = []
    avgss = []
    stdss = []

    if nclass == -2:
        print('TBD')
    else:
        X_data = get_features(features, data)
        y_data = data[type_colnames[str(nclass)]]

        print('\nCross-validation {}-way classification'.format(str(nway)))

        cvclf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, \
                                       min_samples_split=min_samples_split, criterion=criterion, n_jobs=nc)

        print('\n\nNow try cross-validation methods ...')

        # Stratified k-fold cross-validation and compute scoring metric each time
        print('\n----- Stratified K-fold cross-validation -----\n')


        for k in range(2, 11):
            kf = StratifiedKFold(y_data, n_folds=k, shuffle=True, random_state=42)  # define cross validator
            cv_scores_kf = cross_val_score(cvclf, X_data, y_data, scoring=score_func_est, cv=kf)
            print('k={} folds CV scores : '.format(k), cv_scores_kf)
            kvals.append(k)
            avgskf.append(np.mean(cv_scores_kf))
            stdkf.append(np.std(cv_scores_kf) / np.sqrt(float(len(cv_scores_kf))))

        # ShuffleSplit with n iterations
        print('\n\n----- ShuffleSplit iterations -----')

        test_step = 0.1
        for ts in np.arange(0.1, 1, test_step):
            print('Fractional Validation Size : {}'.format(ts))
            ss = ShuffleSplit(len(y_data), n_iter=niter, test_size=ts, random_state=42)  # BUG: don't use train_size
            for train_index, test_index in ss:
                train1as = y_data[train_index] == 0
                test1as = y_data[test_index] == 0
                print("TRAIN SNIa: {} \tTEST SNIa: {}".format(np.count_nonzero(train1as),
                                                              np.count_nonzero(test1as)))
            cv_scores_ss = cross_val_score(cvclf, X_data, y_data, scoring=score_func_est, cv=ss)  # array of score values
            print('\nCV scores (ShuffleSplit) = {}'.format(cv_scores_ss))
            # print 'Average Score = ', np.mean(cv_scores_ss)
            # print 'Score Standard Deviation = ', np.std(cv_scores_ss)
            tsvals.append(ts)
            avgss.append(np.mean(cv_scores_ss))
            stdss.append(np.std(cv_scores_ss) / np.sqrt(float(len(cv_scores_ss))))

    return kvals, avgskf, stdskf, tsvals, avgss, stdss


def get_pscores_with_purities(classifiers, features, pfilename, purities, nclass, Fix_eff, SNRcut=None):

    print('\nUsing purities {} for test files'.format(' '.join(purities)))
    pscores = []
    clf = classifiers[-1]
    blurb = ''

    for purity in purities:
        ptestfile = pfilename + str(purity) + '.txt'
        ptest = read(ptestfile)
        if SNRcut is not None:
            cutptest = (ptest['snr1'] > SNRcut)
            ptest = ptest[cutptest]
            blurb = '(after SNR cut)'
        print('Size of purity = {} test data {}: {}'.format(purity, blurb, len(ptest)))
        X_ptest = get_features(features, ptest)
        if nclass == 2:
            y_ptest = ptest['type']
        elif nclass == 3:
            y_ptest = ptest['type3']
        elif nclass == -2:
            #TODO finish this option
            #pre-classify stage 1
            #cut and classify stage2
            y_ptest = ptest['type2x2']
        pprobs = clf.predict_proba(X_ptest)[:, 0]  # SNeIa are class 0
        # print len(X_ptest),len(pprobs),len(y_ptest)
        pscore = score_func(pprobs, y_ptest, Fix_eff)
        print('Score for purity {} = {:0.4f}'.format(purity, pscore))
        pscores.append(pscore)
    
    return pscores


def get_HR(data, key, H0=68.62, Om0=.301, zkey='z', mukey='mu'):
    
    if mukey not in data.colnames:
        print('  {} not available; filling Hubble Residual column for {} data with {}'.format(mukey,
                                                                                              key, g.nodata))
        data[g.HR] = g.nodata
    else:
        if 'sim_mu' not in data.colnames:
            if 'true_mu' not in data.colnames and zkey in data.colnames:
                # compute true mu
                cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
                data['true_mu'] = cosmo.distmod(data[zkey])
            try:
                if mukey in data.colnames and 'MUMODEL' in data.colnames and 'M0DIF' in data.colnames:
                    data[g.HR] = data['MU'] - data['MUMODEL'] - data['M0DIF'] #spec fitres data
                else: 
                    data[g.HR] = data[mukey] - data['true_mu']
            except:
                print('  Unable to compute Hubble residuals using columns true_mu/{}'.format(zkey))
        else:
            data[g.HR] = data['mu'] - data['sim_mu']

    return data


def get_weights_old(masks, MLtypes, simlist, datalist, method=g.Constant, CLFid='RF', default=1.0, subclass=g.MaxProb):    
    # compute normalizations and weights for data plots 
    # scale sims/secondary data by primary observed data set
    data0 = datalist[0] if (len(datalist) > 0) else None  # primary observed dataset in datalist
    blurb = '{} {}'.format(data0, g.Data) if data0 is not None else str(default)
    print('\nEvaluating {} weights for plots; weights normalized to {}'.format(method, blurb))

    allweights = {}
    for ckey in masks.keys():  # compute normalization factors
        print('\n  Normalization factors for {} classes'.format(ckey))
        allweights[ckey] ={}
        for dkey, dmask in masks[ckey].items():
            ntotal = len(dmask[dmask.keys()[0]]) 
            print('    Computing {} weights for {} Data:'.format(ntotal, dkey))
            allweights[ckey][dkey] = {}
            norm = {}
            weights = {}
            for t, m in dmask.items():
                if data0 is not None:
                    if masks[ckey].has_key(data0) and masks[ckey][data0].has_key(t):   # t exists in data0
                        d0type = t
                        d0key = ckey
                    else:
                        btyp = t[t.find('I'):len(t)] if 'I' in t else CC  # parse t to find base SN type #TODO generalize
                        d0type = CLFid + btyp    # for unlabeled data, only CLF classes available
                        d0key = subclass
                        if masks[subclass][data0].has_key(d0type):
                            print('    Type {} not available for {} Data; using {} type {} instead'.format(t, data0, subclass, d0type))
                        else:
                            print('    No appropriate substitute weights for {} in {} and {}'.format(t, dkey, data0))
                            d0type = None
                    if (method == g.ByType):  # accumulate weights by type
                        if np.count_nonzero(m) > 0 and np.count_nonzero(masks[ckey][data0][d0type]) > 0:
                            norm[t] = float(np.count_nonzero(masks[ckey][data0][d0type]))/float(np.count_nonzero(m))
                            lengths = '({}/{})'.format(np.count_nonzero(masks[ckey][data0][d0type]), np.count_nonzero(m))
                        elif np.count_nonzero(masks[ckey][data0][d0type]) == 0:
                            print('      Zero entries for type {} in {}; defaulting to weight derived from total numbers'.format(t, data0))
                            norm[t] = float(len(masks[ckey][data0][d0type]))/float(len(m))  # default to const value
                            lengths = '({}/{})'.format(len(masks[ckey][data0][d0type]), len(m))
                        else:
                            norm[t] = 0.
                            lengths = '({}/{})'.format(np.count_nonzero(masks[ckey][data0][d0type]), np.count_nonzero(m))
                    else:   # constant weights
                        norm[t] = float(len(masks[d0key][data0][d0type]))/float(len(m))
                        lengths = '({}/{})'.format(len(masks[d0key][data0][d0type]), len(m))
                else:
                    norm[t] = 1.0
                    lengths = ''
                weights[t] = np.array([norm[t]] * len(m))
                if method == g.ByType:
                    print('      w({}) = {:0.3f} {})'.format(t, norm[t], lengths))

            # compute weights for totals; use ratio of total numbers of entries in Tables
            # ByType weights may be inconsistent if fractions in sub-classes vary 
            if data0 is not None:
                d0total = len(masks[d0key][data0][masks[d0key][data0].keys()[0]]) #length of data0 mask
                totnorm = float(d0total)/float(ntotal)
                lengths = '({}/{})'.format(d0total, ntotal)
            else:
                totnorm = 1.0
                lengths = ''
            weights[g.Total] = np.array([totnorm] * ntotal)
            blurb = '({})'.format(g.Total) if method == g.ByType else '' 
            print('      w{} = {:0.3f} {})'.format(blurb, totnorm, lengths))
                
            # save in dict
            allweights[ckey][dkey] = weights

    return allweights

def get_weights(alldata, use_data='', default=1.0):    
    # compute normalizations and weights for data plots 
    # scale sims/secondary data by primary observed data set
    data0 = use_data if len(use_data) > 0 else None  # primary observed dataset in datalist
    blurb = '{} {}'.format(data0, g.Data) if data0 is not None else str(default)
    print('\nEvaluating weights for plots; weights normalized to {}'.format(blurb))

    for dkey in alldata.keys():
        ntotal = len(alldata[dkey])
        if data0 is not None:
            d0total = len(alldata[data0])
            lengths = '({}/{})'.format(d0total, ntotal)
            weights = float(d0total)/float(ntotal)
        else:
            weights = default
            lengths = ''
        print('  w[{}] = {:0.3f} {}'.format(dkey, weights, lengths))
        # add column to all data
        alldata[dkey][g.Weights] = np.array([weights] * ntotal)
        
    return alldata


def get_MLlists(simlist, possible_datalist, data_files, alltypes_colnames, file_formats):
    datalist = []
    for k in [g.Validation, g.Test] + possible_datalist:
        if len(data_files[k]) > 0 and data_files[k] != "''":
            if file_formats[k] in g.allowed_formats:
                #if file_formats[k] == train_format:
                    if k in g.simulated_keys:
                        simlist.insert(0, k)
                    else:
                        datalist.append(k)
                    if k in g.simulated_keys:
                        if alltypes_colnames[k] != g.data_defaults[file_formats[k]]['alltypes_colname_default']:
                            print('  Warning: supplied alltypes_colname ({}) for {} data != default ({}) for file format {}'.\
                                  format(alltypes_colnames[k], k, g.data_defaults[file_formats[k]]['alltypes_colname_default'],
                                         file_formats[k]))
                    else:
                        ulabel = 'Un-labeled' if alltypes_colnames[k] == '' else 'Labeled by ' + alltypes_colnames[k]
                        print('  Including user-supplied data from {} with key/label {}; SN types: {}'.format(data_files[k], k, ulabel))
                #else:
                #    print('  Skipping {} data; file {} not in training format ({})'.format(k, data_files[k], train_format))
            else:
                print('  Skipping {} data; file {} not in allowed format'.format(k, data_files[k]))
                
    print('\nProcessing {} data sets'.format(', '.join(simlist + datalist)))

    return simlist, datalist

def copy_columns(data, features, train_format):
    #copy data columns to match defaults for training_format

    print('\n  Checking for alternate feature names to match defaults for {} training format'.format(train_format))
    for k, f in g.alternate_feature_names[train_format].items():
        fnames = f if type(f)==list else [f]
        for v in fnames:
            if k in features and v in data.columns and k not in data.columns:
                #data.rename_column(v, k)
                data[k] = data[v]
                print('  Copying column {} to {}'.format(v, k))

    return data

def check_alltypes_colname(dkey, alltypes_colname, file_format, data_file):
    # check alltypes colname to make sure it is compatible with format id supplied parameter is the default one.

    if len(alltypes_colname) > 0:
        print('\n  Checking supplied column name for labeled data')
        if alltypes_colname != g.data_defaults[file_format]['alltypes_colname'] and \
                alltypes_colname == g.data_defaults[g.default_format]['alltypes_colname']:
            alltypes_colname = g.data_defaults[file_format]['alltypes_colname']
            print('    Warning: resetting supplied alltypes_colname for {} data to "{}" (default for data file {})'.\
                      format(dkey, alltypes_colname, data_file))
        else:
            print('    Column name {} OK: compatible with {} format'.format(alltypes_colname, file_format))

    return alltypes_colname


def check_all_class_values(all_class_values, file_format):

    print('\n  Checking supplied class values for labeled data')
    if not all([v in g.data_defaults[file_format]['type_values'] for _,v in all_class_values.items()]):
        all_class_values = dict(zip(g.data_defaults[file_format]['type_labels'],
                                       g.data_defaults[file_format]['type_values']))
        print('    Warning: over-riding supplied arguments to {} based on {} data format'.format(all_class_values,
                                                                                                 file_format))
    else:
        print('    Class values {} OK: compatible with {} format'.format(all_class_values,  file_format))

    return all_class_values


def get_file_info(data_files):
    file_formats = {}
    file_types = {}
    for k, v in data_files.items():
        possible_formats = [kk for kk,vv in g.allowed_formats.items() if any([f in v for f in vv])]
        if any(possible_formats):
            file_formats[k] = possible_formats[0] # first match
        else:
            file_formats[k] = None
        possible_types = [f for vv in g.allowed_formats.values() for f in vv if f in v]
        if len(possible_types) > 0:
            file_types[k] = possible_types[0]
        else:
            file_types[k] = ''
            
    return file_formats, file_types
            

def exit_code(filename, status=g.SUCCESS, msg='', start_time=-1):
    if len(str(msg)) > 0:
        print('\nException occurred: {}'.format(str(msg)))
    if start_time > 0:
        msg = "\nEnd-to-end runtime = {0:.3f} minutes\n"
        print(msg.format((time() - start_time)/60))
    if len(filename) > 0 and filename != "''":
        with open(filename, 'w') as fh:
            fh.write(' {}'.format(g.SUCCESS if status==g.SUCCESS else g.FAILURE))
            print('Wrote {}'.format(filename))
    msg = 'Completed with status: {}'.format(status)
    print(msg)

    if status != g.EXCEPTION: # exit without throwing exception
        os._exit(1)
        

def main(args, start_time=-1):
    #############################################
    # Start of main module
    #############################################

    print_heading(version)
    Fixed_effcy = args.eff  # fix efficiency to selected values
    if len(args.commit_hash_path) > 0:
        if os.path.exists(args.commit_hash_path):
            print('\nCommit-hash for code: {}'.format(retrieve_commit_hash(args.commit_hash_path)[0:7]))
        else:
            print('\nPath {} for repository to retrieve commit hash does not exist'.format(
                   args.commit_hash_path))
 
    # Setup type lists to be used in code, printing, plotting
    MLtypes = [g.Ia, g.CC] if args.nclass == 2 else [g.Ia, g.Ibc, g.II]
    nway = '2x2' if args.nclass == -2 else args.nclass
    print('\n{}-way classification'.format(str(nway)))
    
    # setup pklfilename
    pkldir, pklname = os.path.split(args.pklfile)
    pkldir = pkldir if len(pkldir) > 0 else args.filedir
    pkl_default = '{}way.pkl'.format('_'.join([args.pklformat, 'format', str(args.nclass)]))
    pklname = pklname if args.use_filenames else '_'.join([pklname, pkl_default])
    pklfile = os.path.join(pkldir, pklname)

    # Setup dicts for non-user data files and properties
    train_file = args.train if not args.restore else pklfile
    test_file = args.test if not args.train_only else ''
    datafile_args = [train_file, args.validation, test_file, args.spec, args.spec_nofp, args.phot]
    data_files = dict(zip(g.datafile_keys, datafile_args))
    file_formats, file_types = get_file_info(data_files)
    alltypes_colname_args = [args.alltypes_colname_train, args.alltypes_colname_validation, args.alltypes_colname_test,
                             args.alltypes_colname_spec, args.alltypes_colname_spec, '']
    alltypes_colnames = dict(zip(g.datafile_keys, alltypes_colname_args))
    
    # Check format of training/pkl data
    print('\nRunning checks on supplied/default arguments')
    train_format = file_formats[g.Training]
    supplied_format = args.train_format if not args.restore else args.pklformat
    supplied_format_key = [kk for kk, vv in g.allowed_formats.items() if any([f in supplied_format for f in vv])][0]
    if train_format not in g.allowed_formats:
        print('  Training/pkl datafile ({}) may not be not in standard format: {} required'.format(data_files[g.Training], 
              ' or '.join([f for v in g.allowed_formats.values() for f in v])))
        if not args.use_filenames:
            exit_code(args.done_file, status=g.FAILURE, start_time=start_time)
        else:
            print('  Proceeding anyway: use_filenames was requested (user specified format assumed)')
            print('  Assuming training/plk format {}'.format(supplied_format))
            train_format = supplied_format_key
            file_formats[g.Training] = train_format #save

    if  supplied_format_key != file_formats[g.Training]:
        print('  Warning: over-riding supplied training/pkl file-format argument to "{}" (format of data file {})'.\
                  format(train_format, data_files[g.Training]))

    # Setup column names in training data to use for different nclass choices depending on format
    type_colnames = g.data_defaults[train_format]['type_colnames']
    if str(args.nclass) not in type_colnames:
        print('  Data for {}-way typing not available for {} training-data format'.format(args.nclass, train_format))
        exit_code(args.done_file, status=g.FAILURE, start_time=start_time)

    # Check alltypes_colname for training
    if not args.restore:
        alltypes_colnames[g.Training] = check_alltypes_colname(g.Training, alltypes_colnames[g.Training],
                                                               file_formats[ g.Training], data_files[g.Training])
        
    # Setup classification types (alternates to Ia/Ibc/II/CC typing specified here) and check 
    all_class_values = dict(zip(args.type_labels, args.type_values))
    all_class_values = check_all_class_values(all_class_values, file_formats[g.Training])

    print('\nInitial target class ({}) values: {}'.format(g.Ia, ' or '.join([str(v) for v in all_class_values[g.Ia]])))
                                                          
    # Initialize dict for storing astropy tables and feature sets
    alldata = {}
    feature_sets = {}

    SNRcut = 5
    simlist = []
    doBazincuts = args.Bazincuts != 'off' and (not args.noBazinparcuts or args.Bazinerrcuts)
    withhold_id = ''  # used in file_id later
    # Read in training and apply cuts if required by options
    if not args.restore:
        doBazincuts_train = doBazincuts and args.Bazincuts == 'train'
        # read data and apply cuts 
        data_train, status = read_data_file(g.Training, data_files[g.Training], SNRcut=SNRcut, zhi=g.zhi[g.Training], 
                                            zkey=g.generic_feature_names['z'][train_format], format=file_formats[g.Training],
                                            SNRkey=g.generic_feature_names['snrmx'][train_format], 
                                            doBazincuts=doBazincuts_train)
        if not status:
           exit_code(args.done_file, status=g.FAILURE, start_time=start_time) 
        simlist.append(g.Training) 
        
        # Check/create columns required for nclass-way training and reset target_class values as required
        data_train, status = create_type_columns(data_train, args.nclass, all_class_values,
                                                 type_colnames=type_colnames,
                                                 alltypes_colname=alltypes_colnames[g.Training],
                                                 abort=True, format=train_format)
        if not status:
            print('Aborting...')
            exit_code(args.done_file, status=g.FAILURE, start_time=start_time)

        # check for withholding and make cuts on training data
        if len(args.withhold) > 0 and not args.restore:
            print('Hold-out test: withholding {} from training sample'.format(args.withhold))
            for withhold in args.withhold:
                if len(withhold) == 2: # 2-digit argument is a type
                    typekey = g.generic_feature_names['sim_type'][train_format]
                    withcut = (data_train[typekey] == int(withhold))
                    data_train = data_train[~withcut]
                    withinfo='type'
                elif len(withhold) == 3: # 3-digit argument is a template
                    tempkey = g.generic_feature_names['sim_template'][train_format]
                    withcut = (data_train[tempkey] == int(withhold))
                    data_train = data_train[~withcut]
                    withinfo='template'
                print('Withholding {} {} from training sample'.format(withinfo, withhold))
                print('Found {} matches in training data'.format(np.count_nonzero(withcut)))
                print('Length of training data after witholding cut: {}'.format(len(data_train)))

            withhold_id = '_'.join(['withhold', '_'.join(args.withhold)])

    # Use RF for now (in future may build a loop over classifiers)
    CLFid = g.RF

    # RANDOM FOREST CLASSIFIICATION
    # --------------------------------
    if args.train_only:  # force save
        args.store = True

    if args.restore:
        try:
            clf = joblib.load(pklfile)
        except:
            print('\nUnable to load classifier from pklfile {}'.format(pklfile))
            exit_code(args.done_file, status=g.FAILURE, start_time=start_time)

        classifiers = [clf]
        if args.nclass == -2:
            try:
                clfpre = joblib.load('pre' + pklfile)  # load pre-trained classifier
            except:
                print('\nUnable to load pre-classifier from pklfile {}'.format(pklfile))
                exit_code(args.done_file, status=g.FAILURE, start_time=start_time)
            print('\nLoading pre-classifier from file pre{}'.format(pklfile))
            classifiers.insert(0, clfpre)
        print('\nLoading classifier from pklfile {}'.format(pklfile))
        # g.Training not added to simlist since feature sets etc needed for ML_stats not available
    else:
        if CLFid == g.RF:
            features = check_features(args.ft, data_train, format=train_format)
            _result = build_RF_classifier(data_train, args.nclass, args.ft, args.nc, alltypes_colnames[g.Training],
                                          type_colnames=type_colnames, start_time=start_time,
                                         ) 
            classifiers, data_train, training_class_values, feature_sets[g.Training] = _result
            if args.store:
                # setup pklfilename using file type of training data for autofil format 
                if not args.use_filenames: 
                    pklname = re.sub(args.pklformat, file_types[g.Training], pklname)
                    pklfile = os.path.join(pkldir, pklname) 

                if args.nclass == -2:
                    joblib.dump(classifiers[0], 'pre' + pklfile)
                    print('\nSaving pre-classifier to file pre{}'.format(pklfile))
                joblib.dump(classifiers[-1], pklfile)
                print('\nSaving classifier to file {}'.format(pklfile))
        else:
            print('\nUnknown classifier {}'.format(CLFid))

    # Determine feature importances for tree-based estimators (RF, AdaBoost)
    for n, cl in enumerate(classifiers):
        blurb = '' if len(classifiers) == 1 else 'Stage {}'.format(n)
        print('\nComputing feature importances for {} classifier {}'.format(CLFid, blurb))
        importances = cl.feature_importances_
        F = Table([args.ft, importances], names=('Feature', 'Importance'), dtype=['a', 'f4'])
        F.sort('Importance')
        F.reverse()
        print(F)
        print

    if args.train_only:
        exit_code(args.done_file, status=g.SUCCESS, start_time=start_time)
                             
    # DATA CLASSIFICATION WITH TRAINED CLASSIFIER
    # -------------------------------------------
    print('\n************************')
    print('*  RUNNING CLASSIFIER  *')
    print('************************')

    # Overwrite any input values to match defaults (txt format) for running classifier
    target_class = g.desired_class_values[g.Ia]  # target class values used when creating new columns
    alltypes_colnames[g.Training] = g.data_defaults[g.default_format]['type_colnames'][str(args.nclass)]

    # Check for user-supplied data and custom labels
    user_labels = [t for t in args.data if g.Spec != t and g.Phot != t and g.Spec_nofp != t]
    if len(user_labels) != len(args.user_data):
        user_labels = [g.User+str(i) if i >= len(user_labels) else user_labels[i] for i,_ in enumerate(args.user_data)]
    if len(args.user_data) > 0:
        user_files = dict(zip(user_labels, args.user_data))
        user_formats, user_types = get_file_info(user_files)
        data_files.update(user_files)
        file_formats.update(user_formats)
        file_types.update(user_types)
        user_alltypes_colnames = args.user_alltypes_colnames if len(args.user_alltypes_colnames) > 0 \
            else ['' for f in user_files]
        alltypes_colnames.update(dict(zip(user_labels, user_alltypes_colnames)))

    # Setup simlist and datalist for running classifier
    print('\nChecking requested data sets for allowed formats and compatibility')
    possible_datalist = [t for t in args.data if g.Spec == t or g.Phot == t or g.Spec_nofp == t]
    simlist, datalist = get_MLlists(simlist, possible_datalist + user_labels, data_files,
                                    alltypes_colnames, file_formats)

    # setup unique file-id for saving classifications etc
    file_id = '{}'.format(args.filestr) if args.filestr != "''" else ''
    ml_id = '_'.join([str(len(args.ft)) + 'features', str(args.nclass) + 'way-typing', withhold_id])
    file_id = re.sub('_+', '_', '_'.join([ml_id, file_id, 'v'+version]))

    # initialize dicts for ML results for the training, validation and any additional data
    performance = {}
    type_masks = {}
    P_eff_ref = None  # force evaluation of threshold probabilities for Test data
                        
    for dkey  in simlist + datalist:

        if dkey in alldata.keys():   # check if data already processed
            continue
        alltypes_colname = alltypes_colnames[dkey]
        if g.Training not in dkey:           #read in data file and apply cuts
            zkey = g.Data if dkey in datalist else dkey
            format_this = file_formats[dkey]
            if format_this not in g.allowed_formats:
                print('  Skipping data file {} in unsupported format'.format(data_files[dkey]))
                continue
            # read in data and copy column names to train_format if need be
            data_this, status = read_data_file(dkey, data_files[dkey], SNRcut, zhi=g.zhi[zkey],
                                       zkey=g.generic_feature_names['z'][format_this], format=format_this,
                                       SNRkey=g.generic_feature_names['snrmx'][format_this])
            if not status:
                print('  Skipping data file {}'.format(data_files[dkey])) 
        else:
            data_this = data_train
            format_this = train_format

        if len(data_this) == 0:
            print('  Skipping empty {} data file'.format(dkey))
            continue

        # copy feature-name columns to align with defaults for training format
        data_this = copy_columns(data_this, args.ft, train_format)

        # run classification to get classes and probabilities if required
        if dkey == g.Training and not args.restore:
            classes_this = training_class_values
        else:
            classes_this = None   #default for unlabeled data
            data_this, feature_sets[dkey] = run_RF_classifier(dkey, classifiers, data_this, args.nclass, args.ft)

            # Check alltypes_colname
            alltypes_colname = check_alltypes_colname(dkey, alltypes_colnames[dkey],
                                                      file_formats[dkey], data_files[dkey])

            if len(alltypes_colname) > 0:                 
                all_class_values_this = check_all_class_values(all_class_values, file_formats[dkey])
                #check type column name exits and try to create it if not
                data_this, status = create_type_columns(data_this, args.nclass, all_class_values_this, 
                                                        type_colnames=type_colnames,
                                                        alltypes_colname=alltypes_colname, 
                                                        abort=False, format=format_this)
                #reset alltypes_colname to unlabeled data if column for labeled data unusable/not found and save
                alltypes_colname = '' if status==False else g.data_defaults[g.default_format]['type_colnames'][str(args.nclass)]
                alltypes_colnames[dkey] = alltypes_colname

                if len(alltypes_colname) > 0:  
                    if args.nclass == -2:       #use defaults for now 
                        classes_this = [data_this[type_colnames[str(args.nclass)][0]],
                                        data_this[type_colnames[str(args.nclass)][1]]]
                    else:
                        classes_this = data_this[alltypes_colname]

        if P_eff_ref is None and dkey is not g.Test:  #use P_thresh values determined by Test data
            P_eff_ref = []
            for eff in Fixed_effcy:
                eff_id = g._Eff.format(eff)
                if g.Test in performance.keys() and g.P_Threshold + eff_id in performance[g.Test].keys():
                    P_eff_ref.append(performance[g.Test][g.P_Threshold + eff_id])

            if len(P_eff_ref) > 0:
                print('\n**Using Test-data probability threshold(s) {} for fixed-efficiency classification**'.\
                      format(', '.join(['{:.3f}'.format(p) for p in P_eff_ref])))

        # get ML results using class values obtained from created columns
        _result = get_ML_results(dkey, data_this, classifiers, MLtypes, classes_this, Fixed_effcy, args.nclass,
                                 masks=type_masks, all_class_values=g.desired_class_values, P_threshold=P_eff_ref)
        performance[dkey], type_masks = _result
        #for k, v in type_masks.items():
        #    print k, v.keys()
        #    for kk, vv in v.items():
        #        print kk, vv.keys()

        # Compute ROC curve, AUC, class probabilities for labeled types using created class values
        if len(alltypes_colname) > 0:
            ROC = {}
            print('\n Evaluating ROC curves')
            if type_colnames['2'] not in data_this.colnames: # check for binary label; use txt format class values
                data_this, status = create_type_columns(data_this, 2, all_class_values=g.desired_class_values,
                                                        type_colnames=type_colnames,
                                                        alltypes_colname=alltypes_colname,
                                                        abort=False, format=format_this)
            y_test_ROC = data_this[type_colnames['2']]  # for ROC curve, need binary labels (2-way typing)
            fpr, tpr, roc_thres, AUC = get_ROC(y_test_ROC, data_this[g.RFprobability+str(target_class)])
            ROC[g.FPR], ROC[g.TPR], ROC[g.ROC_THR], ROC[g.AUC] = fpr, tpr, roc_thres, AUC
            performance[dkey][g.ROC] = ROC
            
            # Get probability variance and percentiles from trees
            probability_errors = {}
            percentile = 68
            average, variance, err_up, err_down, pdata = prob_errors(classifiers, 
                                                                     feature_sets[dkey], percentile=percentile,
                                                                     debug=False)
            performance[dkey][g.Pr_Errors] = [average, variance, err_up, err_down, pdata]

            # Measured Classification Probability for Ias (calibration curve computes this for us)
            Nbins_RFP = 20
            true = type_masks[g.TrueType][dkey]
            print('\n  Fraction of True Positives in {} probabilty bins'.format( Nbins_RFP))
            if np.count_nonzero(true[g.Ia]) < len(true[g.Ia]):
                fraction_true, meanpr = calibration_curve(true[g.Ia],
                                                      data_this[g.RFprobability + str(target_class)],
                                                      n_bins=Nbins_RFP)
            else:  # compute manually when only Ias in sample
                N_pr, bins = np.histogram(data_this[g.RFprobability + str(target_class)],
                                    bins=np.linspace(0., 1. , Nbins_RFP+1))
                Sum_pr,_ = np.histogram(data_this[g.RFprobability + str(target_class)], bins=bins,
                                 weights=data_this[g.RFprobability + str(target_class)])
                fraction_true = np.asarray([ 1.0 if N_pr[n] > 0 else 0. for n in range(len(N_pr))])
                meanpr = np.asarray([Sum_pr[n]/N_pr[n] if N_pr[n] > 0 else 0. for n in range(len(N_pr))])
            print ('    {}'.format(' '.join(['{:.3f}'.format(f) for f in fraction_true])))
            performance[dkey][g.Fr_True] = fraction_true
            performance[dkey][g.Mean_Pr] = meanpr
                
        # save data alldata for further processing if required
        alldata[dkey] = data_this
                   
        # TEMPLATE STATISTICS for Test Data (need simulated data)
        if dkey == g.Test:
            if len(alltypes_colname) > 0 and g.generic_feature_names['sim_template'][format_this] in data_this.colnames:
                classifications = Fixed_effcy + [g.MaxProb]
                template_info = get_template_statistics(data_this[g.generic_feature_names['sim_template'][format_this]], 
                                                    type_masks, MLtypes, classifications, dkey=dkey)
            else:
                print('\n  Skipping template statistics: no labels or no template information available')

        #write out selected data to astropy table
        save_data = Table()
        _, basename = os.path.split(data_files[dkey])
        filename = os.path.splitext(basename)[0]
        filename = os.path.join(args.filedir, '_'.join([filename, file_id, MLClasses])+ args.format) # output format
        for k in data_this.colnames:
            snid = g.generic_feature_names['snid'][format_this]
            if g.RFprobability in k or g._FixedEffClass in k or snid in k or g.MaxProbClass in k or g.TrueClass_id in k or \
                                      'type' in k or (len(alltypes_colname) > 0 and alltypes_colname in k):
                save_data[k] = data_this[k]

        print('\n  Saving classified data to {}\n    Saved columns:\n    {}'.format(filename, ', '.join(save_data.colnames))) 
        save_data.write(filename, format='ascii', overwrite=True)

    # save performance dict
    filenames = '_'.join([os.path.split(data_files[k])[-1].split('.')[0] for k in simlist + datalist])
    pffile =  os.path.join(args.filedir,'_'.join([g.Performance, filenames, file_id]) + pkl)
    joblib.dump(performance, pffile)
    print('\nSaving performance to file {}'.format(pffile))

    # CROSS_VALIDATION
    # read in data
    cross_val = []
    if (args.cv):
        if (args.sample == 't'):
            data_all = data_train
        # read in validation data only
        elif (args.sample == 'v'):
            data_all = alldata[g.Validation]
        # combine both training and validation sets
        elif (args.sample == 'b'):
            data_all = vstack([ttrain, ttest])

        kvals, avgskf, stdskf, tsvals, avgss, stdss = run_cross_validation(data_all, 
                                                          args.ft, args.nclass, type_colnames=type_colnames, nc=args.nc,
                                                          niter=args.niter, start_time=start_time)
        performance[g.Cross_Validation] = [kvals, avgskf, stdskf, tsvals, avgss, stdss]
        
    if args.pc:  #TBD
        pfilename = 'DES_validation_fitprob_purity='
        pscores = get_pscores_with_purities(classifiers, args.ft, pfilename, args.purities, args.nclass, Fix_eff, SNRcut=SNRcut)
        performance[g.Purity_Scores] = [args.purities, pscores]
        
    if len(args.plots)==0 or args.plots[0] == '':
        print("Skipping plots and exiting")

    else:
        # assemble file name for plots
        fnpurity = 'purity' + re.split('purity', os.path.splitext(args.test)[0])[1] if 'purity' in args.test else ''
        SNR = 'SNR' + re.split('SNR', os.path.splitext(args.test)[0])[1] if 'SNR' in args.test else ''
        dname = '+'.join(datalist) + '_' + g.Data if len(datalist) > 0 else ''
        eff_id = 'Eff_' +  str(round(Fixed_effcy[0], 4))
        if len(Fixed_effcy) > 1:
            eff_id = eff_id + '-' + str(round(Fixed_effcy[-1], 4))
        plot_id = '_'.join(['ML', eff_id, SNR, fnpurity, dname, file_id])
        plot_id = os.path.join(args.filedir, re.sub('_+', '_', plot_id))

        # augment data columns with other variables
        for key in sorted(alldata.keys()):
            zkey = g.generic_feature_names['z'][file_formats[key]]
            mukey = g.generic_feature_names['mu'][file_formats[key]]
            alldata[key] = get_HR(alldata[key], key, H0=args.H0, Om0=args.OmegaM, 
                                  zkey=zkey, mukey=mukey) #Hubble residual
        
        # compute weights for plotting normalized simulations and data
        if len(args.weights) == 0:
            use_data = datalist[0] if len(datalist) > 0 else ''
        else:
            use_data = args.weights
        alldata =  get_weights(alldata, use_data=use_data)

        # assemble other selections
        user_prefs = {'color':dict(zip(user_labels, args.user_colors)),
                      'markers':dict(zip(user_labels, args.user_markers))}
        savetypes = g.allSNtypes if g.All in args.save else args.save #information to save in outputs

        cuts = {}
        if doBazincuts:
            if args.noBazinparcuts:
                del vars(args)[g.Bazinpar_max]
                del vars(args)[g.Bazinpar_min]
            if not args.Bazinerrcuts:
                del vars(args)[g.Bazinerr_max]
            Bazincuts = {k:v for k, v in vars(args).items() if g.Bazin in k and '_'  in k} # _par and _err cuts
            cuts = {g.Bazin: Bazincuts}
            
        make_plots(MLtypes, alldata, type_masks, Fixed_effcy, performance, alltypes_colnames,
                   template_info, user_prefs=user_prefs, CLFid=CLFid, cuts=cuts,
                   plot_groups=args.plots, plot_id=plot_id, plotlist=[simlist, datalist],
                   totals=args.totals, savetypes=savetypes, target_class=target_class,
                   minmax=args.minmax, debug=args.debug, file_formats=file_formats)

    return performance


if __name__ == '__main__':
    args = parse_args(sys.argv)
    start_time = time()
    try:
        performance = main(args, start_time=start_time)
        exit_code(args.done_file, status=g.SUCCESS, start_time=start_time)
    except BaseException as ex:
        traceback.print_exc()
        exit_code(args.done_file, status=g.EXCEPTION, msg=ex, start_time=start_time) 
