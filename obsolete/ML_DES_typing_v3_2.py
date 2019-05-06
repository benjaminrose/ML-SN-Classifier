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
# TPFPtypes    -True positive and false positive types  (TP, FP)
# allTPFPtypes -list of all combinations of allSNtypes and TPFPtypes (eg [TPIa, FPIa, TPCC, FPCC,..)
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

#default values for labeled types (CC computed to ~0)
class_values_default = {g.Ia:'0', g.Ibc:'1', g.II:'2'}

 
#default column-name for each nclass to be used for labeled data 
# no option for user to change this (code adjusts for supplied alltypes_colname)
type_colnames_default = {'2':'type', '3':'type3', '-2':['type2x2','type']}
alltypes_colname_default = 'type3'

# Labels used for dict keys
probability = 'probability'
MLClasses = 'MLClasses'
_save = '_save'

#classifier stages (nclass==-2)
StI = 'Stage1_'
StII = 'Stage2_'

CC_offset = 1 # offset from target_class to assign for 2-way typing

#filetypes
txt = '.txt'
hdf5 = '.hdf5'
fits = '.fits'

#version
version = '3.2'

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
                        metavar='features', choices=['c', 'x0', 'x1', 't0', 'z', 'chi2', 'fit_pr', 'gpeak', 'rpeak', 'ipeak', 'zpeak', 'ra',
                                     'dec', 'grpeak_m', 'ripeak_m', 'izpeak_m', 'grpeak_s', 'ripeak_s', 'izpeak_s',
                                     'Deltagr', 'Deltari', 'Deltaiz', 'snr1', 'snr2', 'snr3', 't0_err', 'x0_err', 'x1_err',
                                     'c_err', 'Bazin_g_t_rise', 'Bazin_g_t_fall', 'Bazin_r_t_rise', 'Bazin_r_t_fall',
                                     'Bazin_i_t_rise', 'Bazin_i_t_fall', 'Bazin_z_t_rise', 'Bazin_z_t_fall', 'grBazin',
                                     'riBazin', 'izBazin'])
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
    # parser.add_argument('--train',default='DES_training_fitprob.txt',help='Filename for training')
    parser.add_argument('--validation', default='', help='Filename for validation; set to null to skip')
    parser.add_argument('--test', default='DES_test_SNR550.txt', help='Filename for test')
    parser.add_argument('--data', default=[], nargs='+',
                        help='Classify selected simulated/observed data (default=Phot); '+\
                             'choose from [Test, Spec, Spec_nofp, Phot, ...] '+\
                             'plots (if selected) will be normalized to first data sample in list; '+\
                             'customize legend labels by adding user-supplied labels ')                    
    parser.add_argument('--spec', default='specType_forcePhoto_bazin_v3.txt', help='Filename for spectroscopic data')
    parser.add_argument('--alltypes_colname_spec', default='spec_eval',
                         help='Column name for true SN types in Spec data')
    parser.add_argument('--phot', default='host_prior_DESY1-4_bazin_v3e.txt', help='Filename for photometric data')
    parser.add_argument('--spec_nofp', default='specSN_SNANA_nofpcut.txt', 
                        help='Filename for spectroscopic data without fit probabilities')
    # User-data options
    parser.add_argument('--user_data', nargs='+', default=[],
                        help='Filenames for user-supplied data; default legend labels in plots will be '+\
                             'User0, User1, ... if no additional arguments supplied for --data' )
    parser.add_argument('--user_alltypes_colnames', nargs='+', default=[],
                         help='Column names for types of labeled user-supplied data (unlabeled data assumed); '+\
                              'use null string for unlabeled data in list of user-data files with mixed labeling properties. ')
    # File saving options
    parser.add_argument('--save', choices=[g.Ia, g.CC, g.All], default=g.Ia, nargs='+', 
                        help='Types to save for printing/pickling')
    parser.add_argument('--store', action='store_true', default=False,
                        help='Save trained classifier to pklfile')
    parser.add_argument('--restore', action='store_true', default=False,
                        help='Restore trained classifier from pklfile')
    parser.add_argument('--pklfile', default='trained_RFclassifier',
                        help='Base filename for reading/writing trained classifier (_n(==nclass)way.pkl is auto-appended; ' +\
                             'pre+filename also read/written if nclass == -2)')
    parser.add_argument('--train_only', action='store_true', default=False,
                        help='Run training only, save trained classifier to pklfile, and exit')
    parser.add_argument('--skip_train', action='store_true', default=False,
                        help='Skip training and reading trained classifier from pklfile')
    # Choose user file-id and format
    parser.add_argument('--filestr',
                        default='', help='Choose string to append to filenames for output')
    parser.add_argument('--format', default=txt,
                        help='Format for output of classification data (txt, hdf5, fits)')
    # Plot options
    parser.add_argument('--plots', nargs='+', default=[g.Performance, g.SALT, g.Hubble],
                        choices=['', g.Performance, g.SALT, g.Hubble, g.Error, g.Color, g.Magnitude, g.Bazin],
                        help='Make selected groups of plots; '+\
                             'choose from Performance, SALT, Hubble, Color, Error, Magnitude, Bazin; '+\
                             'null argument supresses all plots')
    parser.add_argument('--weights', default='',
                        help='Name of data sample to use for computing weights for plot normalizations; defaults to first data sample in data list')
    parser.add_argument('--filedir', default='./',
                        help='Directory name for plot pdf files')
    parser.add_argument('--sim', choices=[g.Training, g.Validation], default=[g.Training], nargs='+',
                        help='Plot selected simulated data')
    parser.add_argument('--totals', action='store_true', default=False,
                        help='Turn on totals in plots (default if using --data or --user-data)')
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
    parser.add_argument('--target_class', default=0, help='Class value of target type for computing probabilities (eg. SNIa)')
    parser.add_argument('--type_labels', choices=[g.Ia, g.II, g.Ibc, g.Ib, g.Ic], default=[g.Ia, g.II, g.Ibc, g.Ib, g.Ic], nargs='+',
                        help='Labels for classes in alltypes_colname (supply one per typename); CC assumed to be ~Ia')
    parser.add_argument('--type_values', default=['0', '2', '1', '1', '1'], nargs='+', 
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
                        choices=['20', '21', '22', '32', '33', '201', '204', '208', '210', '213', '214', '215', '216',
                                 '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230',
                                 '231', '232', '233', '235', '206', '209', '002', '103', '104', '105', '202', '203',
                                 '212', '234', '021', '022', '101', '102', '205', '207', '211', '217', '218'],
                        nargs='+', default='',
                        help='Hold-out test: withhold type (20=IIP, 21=IIN, 22=IIL, 32=Ib, 33=Ic) or '+\
                             'template (206,209=IIN, 002=IIL, 201-233=IIP, 103-234=Ib, 021-218=Ic) from training sample')
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
            print('  Warning: column {} not found; replacing with dummy values {}'.format(f, g.nodata))
            data[f] = g.nodata
        flist.append(data[f])

    X = np.vstack(flist).T
    return X


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
    efficiency_func = interp1d(x, y, kind='linear')  # reverse-order so x is monotonically increasing

    # loop over efficiencies
    P_eff_list = []
    score_list = []
    for effcy in effcies:
        P_eff = efficiency_func(effcy)  # threshold probability at efficiency=effcy
        print('\n    Threshold probability (P_thresh) (eff={:0.1f}%) = {:.3f}'.format(effcy * 100, float(P_eff)))
        print('    Purity (P_thresh=0) = {:.3f}'.format(pur[pos_label]))
        score = score_func(probs, yset, effcy)
        print('    SCORE (purity @ eff={:0.1f}%) = {:.4f}'.format(effcy * 100, score))
        P_eff_list.append(P_eff)
        score_list.append(score)
        
    return pur, eff, thresh, P_eff_list, score_list


def get_ROC(yset_ROC, probs):
    ### Compute ROC curve and AUC
    if len(np.unique(yset_ROC)) > 1:
        fpr, tpr, roc_thres = roc_curve(yset_ROC, probs, pos_label=0)  # restricted to binary classification
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
            print "Validation data #", nt, "=", test
            print "Leaf nodes =", nodes
        sumpr = np.array([0.] * len(classvals))  # init sum
        sumpr2 = np.array([0.] * len(classvals))  # init sum
        for n, node in enumerate(nodes):  # loop thru nodes in trees
            match = (leafs.T[n] == node)
            classes = ytrain[match]  # classes of objects landing in same node
            if (debug):
                print "# of matching classes=", len(classes), "for", n, "th tree"
                cldist = [np.sum(classes == x) for x in classvals]
                print "# of each class", cldist
                print "Prob from estimator=", clf.estimators_[n].predict_proba(X_test)[nt]
            probs = [float(np.sum(classes == x)) / float(len(classes)) for x in classvals]
            sumpr = sumpr + np.asarray(probs)
            sumpr2 = sumpr2 + np.asarray(probs) * np.asarray(probs)
            if (debug):
                print "Probs=", probs, "from #matches=", len(classes), "in", n, "th tree"

        average[nt] = sumpr / float(clf.get_params()['n_estimators'])
        variance[nt] = np.sqrt(sumpr2 / float(clf.get_params()['n_estimators']) - average[nt] * average[nt])

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
            print "Probs=", probs[0:ntest], "for", n, "th tree"
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


def read_data_file(key, datafile, SNRcut=5, zhi=g.zhi, SNRkey='snr1', zkey='z', doBazincuts=False):
    # read data file 
    try:
        print('\nReading {} data from {}'.format(key, datafile))
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

    except:
        print('  Error reading data file {}'.format(datafile))
        data = Table()

    return data


def create_type_columns(data, nclass, all_class_values, alltypes_colname='type3', target_class=0,  
                        type_colnames=type_colnames_default, abort=False):

    status = True
    # Check for type column needed for training and create if needed
    if alltypes_colname not in data.colnames:
        print('  Warning: requested alltypes-column-name {} not found'.format(alltypes_colname))
    else:
        # Check format of alltypes_colname and convert if need be
        if data[alltypes_colname].dtype != int:
            try:
                sntypes = np.unique(data[alltypes_colname]).tolist()
                print('  Found class-label values of {}'.format(' '.join(sntypes)))
                print('  Converting string class-labels to integers')
                data.rename_column(alltypes_colname, alltypes_colname + _save)
                data[alltypes_colname] = g.nodata
                for t in sntypes:
                    for k, v in all_class_values.items():
                        if k in t:    #match of class
                            tmask = data[alltypes_colname + _save] == t
                            data[alltypes_colname][tmask] = int(v)

            except:
                print('  Unable to convert type-labels to standard format: skipping use of labels')
                status = False

    # Check for type-columns appropriate to nclass and create them if not available
    try:
        # Copy new column into alltypes_colname_default for later use (eg to generate ROC curves)
        if alltypes_colname != alltypes_colname_default and alltypes_colname_default not in data.colnames:
            data[alltypes_colname_default] = data[alltypes_colname]
            print('  Copying column {} to column {} (for ROC curves etc)'.format(alltypes_colname, alltypes_colname_default))
        if str(nclass) in type_colnames.keys():
            colnames = type_colnames[str(nclass)] if type(type_colnames[str(nclass)])==list \
                                                  else [type_colnames[str(nclass)]]
            for cname in colnames:
                if cname not in data.colnames or cname == 'type2x2': #force 2x2 recompute
                    print('  Creating column "{}" needed for {}-way typing'.format(cname, nclass))
                    if cname == 'type':
                        data[cname] = data[alltypes_colname] #copy column with all types
                        mask1 = data[cname] != target_class 
                        data[cname][mask1] = 1   #binary data 
                    elif cname == 'type2x2':
                        data[cname] = data[alltypes_colname] #copy column with all types
                        mask1 = data[cname] ==  all_class_values[Ibc] #reset label for Ibc
                        data[cname][mask1] = 0
                    else:
                        print('  Unknown column name {}'.format(colname))
                        status=False
    except:
        print('  Unable to read/create labeled column from available data')
        status = False
        if abort:
            print('Aborting...')
            sys.exit()

    return data, status

def print_heading(v):

    print
    print('******************************************')
    print('*           {}_v{}           *'.format(g.name, v))
    print('* {} *'.format(', '.join(g.authors)))
    print('******************************************')    

    return

def build_RF_classifier(data_train, nclass, features, ncores, alltypes_colname, target_class=0,
                        type_colnames=type_colnames_default, dummy=g.nodata,
                        n_estimators = 100, max_features = 'auto', min_samples_split = 5, criterion = 'entropy'):

    print
    print('*************************')
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
    print('\nClassifying {} Data'.format(dkey))

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

    print('  New columns add to table:\n    {}'.format(', '.join([c for c in data.columns if c not in oldcols])))

    return data, feature_set

def get_ML_results(dkey, dataset, classifiers, MLtypes, class_values_list, effcy, nclass,
                   masks={}, all_class_values=class_values_default, target_column=0, target_class=0, 
                   P_threshold=None, CLFid=g.RF):

    performance = {}

    if g.Ia not in list(all_class_values.keys()):         # require default SN typing
        print('  Other class options/values not implemented yet')
        return performance, masks

    #find type corrresponding to target_class value
    target_type = all_class_values.keys()[all_class_values.values().index(str(target_class))]
    
    print "\n{} DATA RESULTS\n  Target type: {}  (Class_value = {})".format(dkey.upper(), target_type, target_class)
    
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
                if true_classes is not None: #default to Pthresh_eff if available
                    Pthresh = performance[prekey + g.P_Threshold + eff_id]
                    if dkey != g.Test:
                        print('\n    Reverting to fixed-efficiency value {:.3f} based on true types for {}'\
                                   .format(Pthresh, dkey))
                else:
                    print('\n  P_threshold value not available: skipping fixed-efficiency classification')
                    break
            else: 
                Pthresh = P_threshold[j]   #supplied value from Test data
      
            if g.Ia in list(all_class_values.keys()):  
                if len(classifiers) > 1 and n==0: # stage 0 distinguishes type II
                    masks[predict][dkey][CLFid + g.II] = dataset[allprobs_colname] < Pthresh
                else:
                    masks[predict][dkey][CLFid + g.Ia] = dataset[allprobs_colname] >= Pthresh
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

                # assemble TP/FP
                if true_classes is not None:
                    for t in MLtypes:
                        masks[predict][dkey][CLFid + g.TP + t] = masks[predict][dkey][CLFid + t] & masks[g.TrueType][dkey][t]
                        masks[predict][dkey][CLFid + g.FP + t] = masks[predict][dkey][CLFid + t] & ~masks[g.TrueType][dkey][t]

                # print summary
                print('\n    Predictions using fixed-efficiency classification with P_thresh {:.3f}:'\
                      .format(float(Pthresh)))
                for t in MLtypes:
                    print('\n      Predicted number of {}-Data {} = {}'.format(dkey, t,
                               np.count_nonzero(masks[predict][dkey][CLFid + t])))
                    if true_classes is not None:
                        print('        Correct (true positive) number of {}-Data {} = {}'.format(dkey, 
                                     t, np.count_nonzero(masks[predict][dkey][CLFid + g.TP + t])))
                        print('        Incorrect (false positive) number of {}-Data {} = {}'.format(dkey, 
                                     t, np.count_nonzero(masks[predict][dkey][CLFid + g.FP + t])))

                #save classification to table
                dataset[CLFid + g._FixedEffClass + eff_id] = g.nodata
                dataset[CLFid + g._FixedEffClass_id + eff_id] = '  ' + str(g.nodata)
                dataset[CLFid + g.TPFP + g._FixedEffClass_id + eff_id] = '    ' + str(g.nodata)
                for t in MLtypes:
                    if t != g.CC:
                        dataset[CLFid + g._FixedEffClass + eff_id][masks[predict][dkey][CLFid + t]] = \
                            all_class_values[t]
                    else:
                        dataset[CLFid + g._FixedEffClass + eff_id][masks[predict][dkey][CLFid + t]] = \
                            target_class + CC_offset
                    dataset[CLFid + g._FixedEffClass_id + eff_id][masks[predict][dkey][CLFid + t]] = CLFid + t
                    if true_classes is not None:
                        dataset[CLFid + g.TPFP + g._FixedEffClass_id + eff_id][masks[predict][dkey][CLFid + g.TP + t]] = \
                            CLFid + g.TP + t
                        dataset[CLFid + g.TPFP + g._FixedEffClass_id + eff_id][masks[predict][dkey][CLFid + g.FP + t]] = \
                            CLFid + g.FP + t
                    
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
                for t in MLtypes:
                    masks[g.MaxProb][dkey][CLFid + g.TP + t] = masks[g.MaxProb][dkey][CLFid + t] & masks[g.TrueType][dkey][t]
                    masks[g.MaxProb][dkey][CLFid + g.FP + t] = masks[g.MaxProb][dkey][CLFid + t] & ~masks[g.TrueType][dkey][t]

            print('\n  Predictions using Maximum-Probability Classification:')
            for t in MLtypes:
                print('\n    Predicted number of {}-Data {} using max prob = {}'.format(dkey,
                             t, np.count_nonzero(masks[g.MaxProb][dkey][CLFid + t])))
                if true_classes is not None:
                    print('      Correct (true positive) number of {}-Data {} = {}'.format(dkey, 
                                 t, np.count_nonzero(masks[g.MaxProb][dkey][CLFid + g.TP + t])))
                    print('      Incorrect (false positive) number of {}-Data {} = {}'.format(dkey, 
                                 t, np.count_nonzero(masks[g.MaxProb][dkey][CLFid + g.FP + t])))

            #save classification to table
            dataset[CLFid + g._MaxProbClass_id] = '  ' + str(g.nodata)
            dataset[CLFid + g.TPFP + g._MaxProbClass_id] = '    ' + str(g.nodata)
            for t in MLtypes:
                dataset[CLFid + g._MaxProbClass_id][masks[g.MaxProb][dkey][CLFid + t]] = CLFid + t
                if true_classes is not None:
                    dataset[CLFid + g.TPFP + g._MaxProbClass_id][masks[g.MaxProb][dkey][CLFid + g.TP + t]] = \
                        CLFid + g.TP + t
                    dataset[CLFid + g.TPFP + g._MaxProbClass_id][masks[g.MaxProb][dkey][CLFid + g.FP + t]] = \
                        CLFid + g.FP + t

            if true_classes is not None:
                Eff, Pur, Rem = g.EffPur(masks[g.TrueType][dkey][g.Ia], maxprobclass == target_class)
                performance[prekey + g.Efficiency_MaxProb], performance[prekey + g.Purity_MaxProb],\
                                                  performance[prekey + g.Reject_MaxProb] = Eff, Pur, Rem
                print('\n    Performance for {} data using Max. Prob. Classification:'.format(dkey))
                print('      Purity = {:.3f}, Efficiency = {:.3f}'.format(Pur, Eff))
                print('      Number of non-target SN rejected = {} (Fraction = {:0.2f})'.format(Rem,
                             float(Rem)/len(dataset[allprobs_colname])))

    print('\n  New columns add to data table:')
    print('    {}'.format(', '.join([c for c in dataset.columns if c not in oldcols])))

    return performance, masks


def get_template_statistics(template_data, masks, MLtypes, classifications, dkey=g.Test, CLFid=g.RF):

    template_info = {}
    print "\n  Template Statistics for {} Data".format(dkey)

    for cl in classifications:
        if cl == g.MaxProb:
            print('\n    Maximum Probability Classification')
            cl_id = g.MaxProb
        else:
            print('\n    Efficiency = {}'.format(cl))
            cl_id = g.Eff_.format(float(cl))
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

        print ('\n    Template | Value')
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


def run_cross_validation(data, features, nclass, type_colnames=type_colnames_default):
        
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
                                       min_samples_split=min_samples_split, criterion=criterion, n_jobs=args.nc)

        print '\n\nNow try cross-validation methods ...'

        # Stratified k-fold cross-validation and compute scoring metric each time
        print '\n----- Stratified K-fold cross-validation -----\n'


        for k in range(2, 11):
            kf = StratifiedKFold(y_data, n_folds=k, shuffle=True, random_state=42)  # define cross validator
            cv_scores_kf = cross_val_score(cvclf, X_data, y_data, scoring=score_func_est, cv=kf)
            print 'k={} folds CV scores : '.format(k), cv_scores_kf
            kvals.append(k)
            avgskf.append(np.mean(cv_scores_kf))
            stdkf.append(np.std(cv_scores_kf) / np.sqrt(float(len(cv_scores_kf))))

        # ShuffleSplit with n iterations
        print '\n\n----- ShuffleSplit iterations -----'

        test_step = 0.1
        for ts in np.arange(0.1, 1, test_step):
            print 'Fractional Validation Size : ', ts
            ss = ShuffleSplit(len(y_data), n_iter=args.niter, test_size=ts, random_state=42)  # BUG: don't use train_size
            for train_index, test_index in ss:
                train1as = y_data[train_index] == 0
                test1as = y_data[test_index] == 0
                print "TRAIN SNIa:", np.count_nonzero(train1as), "\tTEST SNIa:", np.count_nonzero(test1as)
            cv_scores_ss = cross_val_score(cvclf, X_data, y_data, scoring=score_func_est, cv=ss)  # array of score values
            print '\nCV scores (ShuffleSplit) = ', cv_scores_ss
            # print 'Average Score = ', np.mean(cv_scores_ss)
            # print 'Score Standard Deviation = ', np.std(cv_scores_ss)
            tsvals.append(ts)
            avgss.append(np.mean(cv_scores_ss))
            stdss.append(np.std(cv_scores_ss) / np.sqrt(float(len(cv_scores_ss))))

    return kvals, avgskf, stdskf, tsvals, avgss, stdss


def get_pscores_with_purities(classifiers, pfilename, purities, nclass, Fix_eff, SNRcut=None):

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
        X_ptest = get_features(args.ft, ptest)
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


def get_HR(data, key, H0=68.62, Om0=.301, zkey='z'):
    
    if 'mu' not in data.colnames:
        print('  mu not available; filling Hubble Residual column for {} data with {}'.format(key, g.nodata))
        data[g.HR] = g.nodata
    else:
        if 'sim_mu' not in data.colnames:
            if 'true_mu' not in data.colnames and zkey in data.colnames:
                # compute true mu
                cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
                data['true_mu'] = cosmo.distmod(data[zkey])
            try:
                data[g.HR] = data['mu'] - data['true_mu']
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
    

def main(args):
    #############################################
    # Start of main module
    #############################################

    print_heading(version)
    Fixed_effcy = args.eff  # fix efficiency to selected values

    # Setup type lists to be used in code, printing, plotting
    MLtypes = [g.Ia, g.CC] if args.nclass == 2 else [g.Ia, g.Ibc, g.II]
    nway = '2x2' if args.nclass == -2 else args.nclass
    print('\n{}-way classification'.format(str(nway)))

    # Setup classification types (alternates to Ia/Ibc/II/CC typing specified here) 
    all_class_values = dict(zip(args.type_labels, args.type_values))
    print('Using possible classes and class-values {}'.format(all_class_values))
    print('Target class value is {}'.format(args.target_class))

    # Setup column names in training data to use for different nclass choices (not an option for now)
    type_colnames = type_colnames_default

    # Initialize dict for storing astropy tables and feature sets
    alldata = {}
    feature_sets = {}

    # Setup locations for default data files
    data_files = {g.Training: args.train, g.Validation: args.validation, g.Test: args.test, 
                  g.Spec: args.spec, g.Spec_nofp: args.spec_nofp, g.Phot: args.phot}

    # Read in training and apply cuts if required by options
    SNRcut = 5
    simlist = []
    alltypes_colnames = []
    withhold_id = ''
    doBazincuts = args.Bazincuts != 'off' and (not args.noBazinparcuts or args.Bazinerrcuts)
    if not args.skip_train:
        doBazincuts_train = doBazincuts and args.Bazincuts == 'train'
        # read data and apply cuts 
        data_train = read_data_file(g.Training, args.train, SNRcut=SNRcut, zhi=g.zhi[g.Training], doBazincuts=doBazincuts_train)
        simlist.append(g.Training)
        alltypes_colnames.append(args.alltypes_colname_train)

        # Check/create columns required for nclass-way training
        data_train, status = create_type_columns(data_train, args.nclass, all_class_values, type_colnames=type_colnames,
                                         alltypes_colname=args.alltypes_colname_train, target_class= args.target_class,
                                         abort=True)

        # check for withholding and make cuts on training data
        if len(args.withhold) > 0 and not args.skip_train:
            print('Hold-out test: withholding {} from training sample'.format(args.withhold))
            for withhold in args.withhold:
                if len(withhold) == 2: # 2-digit argument is a type
                    withcut = (data_train['sim_type'] == int(withhold))
                    data_train = data_train[~withcut]
                elif len(withhold) == 3: # 3-digit argument is a template
                    withcut = (data_train['sim_nonIa'] == int(withhold))
                    data_train = data_train[~withcut]
                print('Withholding {} {} from training sample'.format(withinfo, withhold))
                print('Found {} matches in training data'.format(np.count_nonzero(withcut)))
                print('Length of training data after witholding cut: {}'.format(len(data_train)))

            withhold_id = '_'.join(['withhold', '_'.join(args.withhold)])

    # Use RF for now (in future may build a loop over classifiers)
    CLFid = g.RF

    # RANDOM FOREST CLASSIFIICATION
    # --------------------------------
    if args.skip_train:  # force read classifier from file
        args.restore = True

    if args.train_only:  # force save
        args.store = True
    # setup default pklfilename
    pklfile = os.path.join(args.filedir, args.pklfile + '_' + str(args.nclass) + 'way.pkl')

    if args.restore:
        try:
            clf = joblib.load(pklfile)
        except:
            print('\nUnable to load classifier from pklfile {}'.format(pklfile))
            sys.exit()

        classifiers = [clf]
        if args.nclass == -2:
            try:
                clfpre = joblib.load('pre' + pklfile)  # load pre-trained classifier
            except:
                print('\nUnable to load pre-classifier from pklfile {}'.format(pklfile))
                sys.exit()
            print('\nLoading pre-classifier from file pre{}'.format(pklfile))
            classifiers.insert(0, clfpre)
        print('\nLoading classifier from pklfile {}'.format(pklfile))
                      
    else:
        if CLFid == g.RF:
            classifiers, data_train, training_class_values, feature_sets[g.Training] = build_RF_classifier(data_train, 
                                                                         args.nclass, args.ft, args.nc,
                                                                         args.alltypes_colname_train, 
                                                                         type_colnames=type_colnames)
            if args.store:
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
        sys.exit()


    # DATA CLASSIFICATION WITH TRAINED CLASSIFIER
    # -------------------------------------------
    print('************************')
    print('*  RUNNING CLASSIFIER  *')
    print('************************')

    # Prepend Validation data if requested
    if len(args.validation) > 0: 
        simlist.insert(0, g.Validation)
        alltypes_colnames.insert(0, args.alltypes_colname_test)

    # Prepend test data to simlist if requested
    # Test data is first so that threshold probabilities for classification are setup
    if len(args.test) > 0: 
        simlist.insert(0, g.Test)
        alltypes_colnames.insert(0, args.alltypes_colname_test)

    # Create datalist and alltype_colnames and check for user-supplied data
    datalist = [t for t in args.data if g.Spec in t or g.Phot in t]
    alltypes_colnames_data = [args.alltypes_colname_spec if g.Spec in t else '' for t in datalist]
    user_data = [t for t in args.user_data]
    user_labels = [t for t in args.data if g.Spec not in t and g.Phot not in t]
    alltypes_colnames_user = [t for t in args.user_alltypes_colnames]
    print('\nProcessing {} default data sets'.format(', '.join(simlist + datalist)))

    # Check for alltype_colnames (if data is labeled) and user labels
    if len(user_data) > 0:
        for i, ds in enumerate(user_data): 
            label = user_labels[i] if len(user_labels) > i else g.User+str(i)
            #default to unlabeled data
            colname = args.user_alltypes_colnames[i] if len(args.user_alltypes_colnames) > i else '' 
            ulabel = 'Un-labeled' if colname=='' else 'Labeled by '+colname
            print('Including user-supplied data from {} with key {}; type: {}'.format(ds, label, ulabel))
            data_files[label] = ds
            datalist.append(label)
            alltypes_colnames_data.append(colname)
            user_labels.append(label)
            
    # setup unique file-id for saving classifications etc
    file_id = '{}'.format(args.filestr)
    ml_id = '_'.join([str(len(args.ft)) + 'features', str(args.nclass) + 'way-typing', withhold_id])
    file_id = re.sub('_+', '_', '_'.join([ml_id, file_id, 'v'+version]))

    # initialize dicts for ML results for the training, validation and any additional data
    performance = {}
    type_masks = {}
    P_eff_ref = None  # force evaluation of threshold probabilities for Test data
                        
    for dkey, alltypes_colname in zip(simlist + datalist, alltypes_colnames + alltypes_colnames_data):

        if dkey in alldata.keys():   # check if data already processed
            continue
        if g.Training not in dkey:           #read in data file and apply cuts
            zkey = g.Data if dkey in datalist else dkey
            data_this = read_data_file(dkey, data_files[dkey], SNRcut, zhi=g.zhi[zkey])
        else:
            data_this = data_train
        if len(data_this) == 0:
            print('Skipping {} data file'.format(dkey))
            continue
        # run classification to get classes and probabilities if required
        if dkey == g.Training and not args.restore:
            classes_this = training_class_values
        else:
            classes_this = None   #default for unlabeled data
            data_this, feature_sets[dkey] = run_RF_classifier(dkey, classifiers, data_this, args.nclass, args.ft)
            if len(alltypes_colname) > 0:                 
                #check column name exits
                data_this, status = create_type_columns(data_this, args.nclass, all_class_values, 
                                                        type_colnames=type_colnames,
                                                        alltypes_colname=alltypes_colname, target_class=args.target_class,
                                                        abort=False)
                #print(alltypes_colname, len(data_this[alltypes_colname]))
                #reset alltypes_colname to unlabeled data if column for labeled data unusable/not found
                alltypes_colname = '' if status==False else alltypes_colname
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
                if g.Test in performance.keys():
                    P_eff_ref.append(performance[g.Test][g.P_Threshold + eff_id])

            if len(P_eff_ref) > 0:
                print('\n**Using Test-data probability threshold(s) {} for fixed-efficiency classification**'.\
                      format(', '.join(['{:.3f}'.format(p) for p in P_eff_ref])))

        # get ML results
        _result = get_ML_results(dkey, data_this, classifiers, MLtypes, classes_this, Fixed_effcy, args.nclass,
                                 masks=type_masks, all_class_values=class_values_default, P_threshold=P_eff_ref)
        performance[dkey], type_masks = _result
        #for k, v in type_masks.items():
        #    print k, v.keys()
        #    for kk, vv in v.items():
        #        print kk, vv.keys()

        # Compute ROC curve, AUC, class probabilities for labeled types
        if len(alltypes_colname) > 0:
            ROC = {}
            print('\n Evaluating ROC curves')
            if type_colnames['2'] not in data_this.colnames: # check for binary label
                data_this, status = create_type_columns(data_this, 2, all_class_values,
                                                        type_colnames=type_colnames,
                                                        alltypes_colname=alltypes_colname, target_class=args.target_class,
                                                        abort=False)
            y_test_ROC = data_this[type_colnames['2']]  # for ROC curve, need binary labels (2-way typing)
            fpr, tpr, roc_thres, AUC = get_ROC(y_test_ROC, data_this[g.RFprobability+str(args.target_class)])
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
            fraction_true, meanpr = calibration_curve(true[g.Ia],
                                                      data_this[g.RFprobability + str(args.target_class)],
                                                      n_bins=Nbins_RFP)
            print('\n  Fraction of True Positives in {} probabilty bins'.format( Nbins_RFP))
            print ('    {}'.format(' '.join(['{:.3f}'.format(f) for f in fraction_true])))
            performance[dkey][g.Fr_True] = fraction_true
            performance[dkey][g.Mean_Pr] = meanpr


        # save data alldata for further processing if required
        alldata[dkey] = data_this
                   
        # TEMPLATE STATISTICS for Test Data (need simulated data)
        if dkey == g.Test:
            classifications = Fixed_effcy + [g.MaxProb]
            template_info = get_template_statistics(data_this['sim_nonIa'], 
                                                    type_masks, MLtypes, classifications, dkey=dkey)

        #write out selected data to astropy table
        save_data = Table()
        filename = os.path.splitext(data_files[dkey])[0]
        filename = os.path.join(args.filedir, '_'.join([filename, file_id, MLClasses])+ args.format)
        for k in data_this.colnames:
            if g.RFprobability in k or g._FixedEffClass in k or 'snid' in k or g.MaxProbClass in k or g.TrueClass_id in k or \
                                      'type' in k or (len(alltypes_colname) > 0 and alltypes_colname in k):
                save_data[k] = data_this[k]

        print('\n  Saving classified data to {}\n    Saved columns:\n    {}'.format(filename, ', '.join(save_data.colnames))) 
        save_data.write(filename, format='ascii', overwrite=True)


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
                                                          args.ft, args.nclass, type_colnames=type_colnames_default)
        performance[g.Cross_Validation] = [kvals, avgskf, stdskf, tsvals, avgss, stdss]
        
    if args.pc:  #TBD
        pfilename = 'DES_validation_fitprob_purity='
        pscores = get_pscores_with_purities(classifiers, pfilename, args.purities, args.nclass, Fix_eff, SNRcut=SNRcut)
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
            alldata[key] = get_HR(alldata[key], key, H0=args.H0, Om0=args.OmegaM, zkey='z') #Hubble residual
        
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
        alltypes_colnames = dict(zip(simlist + datalist, alltypes_colnames + alltypes_colnames_data))

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
                   totals=args.totals, savetypes=savetypes, target_class=args.target_class,
                   minmax=args.minmax, debug=args.debug)

    return performance


if __name__ == '__main__':
    args = parse_args(sys.argv)
    main(args)
