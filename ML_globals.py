# GLOBAL VARIABLES 

name = 'SNIRF'
description = 'SN Identification with Random Forest'
authors = ['Eve Kovacs', 'Steve Kuhlmann', 'Ravi Gupta']

# true types
SN = 'SN'
Ia = 'Ia'
CC = 'CC'
Ibc = 'Ibc'
II = 'II'
Ib = 'Ib'
Ic = 'Ic'
Total = 'Total'
SNtypes = [Ia, CC]
CCtypes = [Ibc, II]
allSNtypes = SNtypes + CCtypes

# classified types
RF = 'RF'  # Random-Forest
RFIa = RF + Ia
RFCC = RF + CC
RFIbc = RF + Ibc
RFII = RF + II
RFTotal = RF + Total

# true and false for computing PR curves
T = 'T'
F = 'F'
P = 'P'
N = 'N'
TP = T + P
FP = F + P
TN = T + N
FN = F + N
TF = T + F
TFtypes = [TP, FP, TN, FN]
RFIaTP = RFIa + TP
RFIaFP = RFIa + FP
RFCCTN = RFCC + TN
RFCCFN = RFCC + FN
RFIbcTN = RFIbc + TN
RFIbcFN = RFIbc + FN
RFIITN = RFII + TN
RFIIFN = RFII + FN
allTFtypes = [RFIaTP, RFIaFP, RFCCTN, RFCCFN, RFIbcTN, RFIbcFN, RFIITN, RFIIFN]

# data samples (read in as astropy tables)
Data = 'Data'
Training = 'Training'
Validation = 'Validation'
Test = 'Test'
Simulated = 'Simulated'
Spec = 'Spec'
Spec_nofp = 'Spec_nofp'
Phot = 'Phot'
User = 'User'

datafile_keys = ['Training', 'Validation', 'Test', 'Spec', 'Spec_nofp', 'Phot']
simulated_keys = ['Training', 'Validation', 'Test']

# Missing data in Table  
nodata = -999

# Labels used for new columns in data tables
_Eff = '_{:0.2f}'
RFprobability = RF + 'probability'
MaxProb = 'MaxProb'
_MaxProb = '_' + MaxProb
_id = '_id'
TrueClass_id = 'TrueClass' + _id
_FixedEffClass = '_FixedEffClass'
MaxProbClass = 'MaxProbClass'
_MaxProbClass = '_MaxProbClass'

# keywords/flags
Total = 'Total'
All = 'All'
Constant = "Constant"
FPR = 'FPR'
TPR  = 'TPR'
ROC_THR  = 'ROC_THR'
AUC  = 'AUC'
Purity ='Purity'
Efficiency = 'Efficiency'
Threshold = 'Threshold'
P_Threshold = 'P_' + Threshold
Score ='Score'
TrueType = 'TrueType'
Eff_ = 'Eff_{:0.2f}'
Efficiency_MaxProb = Efficiency + _MaxProb
Purity_MaxProb = Purity + _MaxProb
Reject_MaxProb = 'Reject' + _MaxProb
Fr_True = 'Fr_True'
Mean_Pr = 'Mean_Pr'
ROC = 'ROC'
Pr_Errors = 'Pr_Errors'
Cross_Validation = 'Cross_Validation'
Purity_Scores = 'Purity_Scores'
HR = 'HR'
Weights = 'Weights'
Counts = 'Counts'
Stats = 'Stats'
Bazinpar_max = 'Bazinpar_max'
Bazinpar_min = 'Bazinpar_min'
Bazinerr_max = 'Bazinerr_max'

# names for plot groups 
Performance = 'Performance'
SALT = 'SALT'
Hubble = 'Hubble'
Error = 'Error'   
Color = 'Color'
Magnitude = 'Magnitude'
Bazin = 'Bazin'

#redshift constants
zlo = 0.0
zhi = {Training: 1.4, Validation: 1.4, Test:1.4, Data: 1.1}

#status
SUCCESS = 'SUCCESS'
FAILURE = 'FAILURE'
EXCEPTION = 'EXCEPTION'

# supported input file formats and associated filetypes
allowed_formats = {'text':['txt'], 'fitres':['FITRES', 'csv']}
default_format = 'text'

# allowed feature names for each file format
allowed_features = {'text': ['c', 'x0', 'x1', 't0', 'z', 'chi2', 'fit_pr', 'gpeak', 'rpeak', 'ipeak', 'zpeak', 'ra',
                             'dec', 'grpeak_m', 'ripeak_m', 'izpeak_m', 'grpeak_s', 'ripeak_s', 'izpeak_s',
                             'Deltagr', 'Deltari', 'Deltaiz', 'snr1', 'snr2', 'snr3', 't0_err', 'x0_err', 'x1_err',
                             'c_err', 'Bazin_g_t_rise', 'Bazin_g_t_fall', 'Bazin_r_t_rise', 'Bazin_r_t_fall',
                             'Bazin_i_t_rise', 'Bazin_i_t_fall', 'Bazin_z_t_rise', 'Bazin_z_t_fall', 'grBazin',
                             'riBazin', 'izBazin'],
                    'fitres':['CID', 'IDSURVEY', 'TYPE', 'FIELD', 'CUTFLAG_SNANA', 'zHEL',
                              'zHELERR', 'zCMB', 'zCMBERR', 'zHD', 'zHDERR', 'VPEC', 'VPECERR', 'HOST_LOGMASS',
                              'HOST_LOGMASS_ERR', 'HOST_sSFR', 'HOST_sSFR_ERR', 'SNRMAX1', 'SNRMAX2', 'SNRMAX3', 'PKMJD',
                              'PKMJDERR', 'x1', 'x1ERR', 'c', 'cERR', 'mB', 'mBERR', 'x0', 'x0ERR', 'COV_x1_c', 'COV_x1_x0',
                              'COV_c_x0', 'NDOF', 'FITCHI2', 'FITPROB', 'SIM_TYPE_INDEX', 'SIM_TEMPLATE_INDEX', 'SIM_LIBID',
                              'SIM_NGEN_LIBID', 'SIM_ZCMB', 'SIM_VPEC', 'SIM_DLMAG', 'SIM_PKMJD', 'SIM_SHAPE', 'SIM_AV',
                              'SIMNULL1', 'SIM_RV', 'SIM_x0', 'SIM_mB'],
                    }

#translation dict for data formats; allows mixing of formats between training and test datasets
alternate_feature_names = {'text': {'snid':'CID', 'z':'zCMB', 't0':'PKMJD', 't0_err':'PKMJDERR', 
                                    'x0_err':'x0ERR', 'x1_err':'x1ERR', 'c_err':'cERR', 
                                    'chi2':'FITCHI2', 'dof':'NDOF', 'mu':'MU', 'muerror': 'MUERR',
                                    'fit_pr':'FITPROB', 'snr1':'SNRMAX1', 'snr2':'SNRMAX2', 'snr3':'SNRMAX3',
                                    'sim_type':'SIM_TYPE_INDEX', 'sim_nonIa':'SIM_TEMPLATE_INDEX', 'sim_mu': 'SIM_DLMAG',
                                    },
                           'fitres':{'CID':'snid','zCMB':'z', 'PKMJD':'t0', 'PKMJDERR':['PKMJDerr', 't0_err'],
                                     'FITCHI2':'chi2', 'FITPROB':'fit_pr', 'MU': 'mu', 'MUERR': 'muerror',
                                     'SNRMAX1':'snr1', 'SNRMAX2':'snr2', 'SNRMAX3':'snr3', 'SIM_TYPE_INDEX':'sim_type',
                                     'SIM_TEMPLATE_INDEX':'sim_nonIa', 'SIM_DLMAG':'sim_mu',
                                     'x0ERR':'x0_err', 'x1ERR': 'x1_err', 'cERR': 'c_err',
                                    }
                          }

# numerical values for recognized classes; code creates extra data columns if supplied labels differ 
desired_class_values = {Ia:0, Ib:1, Ic:1, Ibc:1, II:2, CC:1}

# dictionary defining default column names and contents for labeled data for allowed data formats
data_defaults = {'text':{'alltypes_colname':'type3',
                         'type_values':[[0], [2], [1], [1], [1]],
                         'type_labels':[Ia, II, Ibc, Ib, Ic],
                         'type_colnames': {'2':'type', '3':'type3', '-2':['type2x2','type']},
                         'alltypes_available':True,
                         'alltypes_colname_default':'type3',
                        },
                 'fitres':{'alltypes_colname':'TYPE',
                           'type_values':[[1, 101]],
                           'type_labels':[Ia],
                           'type_colnames': {'2':'type', '3':'type3', '-2':['type2x2','type']}, #create on the fly
                           'alltypes_available':False,
                           'alltypes_colname_default':'TYPE',  #only binary typing available
                          }
                }

# translation dictionary for generic feature names used in code  
generic_feature_names = {'z':{'text':'z', 'fitres':'zCMB'},
                         'snrmx':{'text':'snr1', 'fitres':'SNRMAX1'},
                         'sim_type':{'text':'sim_type', 'fitres':'SIM_TYPE_INDEX'},
                         'sim_template':{'text':'sim_nonIa', 'fitres':'SIM_TEMPLATE_INDEX'},
                         'snid':{'text':'snid','fitres':'CID'},
                         'mu':{'text':'mu','fitres':'MU'},
                } 

# dictionary of type <-> template number correspondence
allowed_templates = {'2P':['20'], '2N':['21'], '2L':['22'], '1b':['32'], '1bc':['33'],
                     'IIN':['206', '209'],
                     'IIL':['002'], 
                     'IIP':['201', '204', '208', '210', '213', '214', '215', '216', '219', '220', '221', '222',
                            '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '235'],
                     'IIb':['400','401', '402','403'],
                     'Ib':['103', '104', '105', '202', '203', '212', '234'],
                     'Ic':['021', '022', '101', '102', '205', '207', '211', '217', '218'],
                     'PEC1A':['502', '503', '506', '509'],
                    }


################ FUNCTIONS ##############

import numpy as np

# Compute efficiency, purity
def EffPur(true, predict):
    Tp = np.sum(true & predict)
    Fn = np.sum(true & ~predict)
    Fp = np.sum(~true & predict)
    Tn = np.sum(~true & ~predict)
    try:
        Rem = Fn + Tn  # SNe removed
        Eff = 1.0 * Tp / (Tp + Fn)
        Pur = 1.0 * Tp / (Tp + Fp)
    except:
        print('\n    **Exception in EffPur: TP+FP = {}**'.format(Tp + Fp))
        Eff = float(nodata); Pur = Eff; 

    return Eff, Pur, Rem
