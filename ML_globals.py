# GLOBAL VARIABLES 

name = 'ML_DES_typing'
authors = ['Eve Kovacs', 'Steve Kuhlmann', 'Ravi Gupta']

# true types, classified types
SN = 'SN'
Ia = 'Ia'
CC = 'CC'
Ibc = 'Ibc'
II = 'II'
Ib = 'Ib'
Ic = 'Ic'
Total = 'Total'
RF = 'RF'  # Random-Forest
RFIa = RF + Ia
RFCC = RF + CC
RFIbc = RF + Ibc
RFII = RF + II
RFTotal = RF + Total
SNtypes = [Ia, CC]
CCtypes = [Ibc, II]
allSNtypes = SNtypes + CCtypes

# true and false positives
TP = 'TP'
FP = 'FP'
TPFP = TP + FP
TPFPtypes = [TP, FP]
allTPFPtypes = [p + t for p in TPFPtypes for t in allSNtypes]
TPIa = TP + Ia
FPIa = FP + Ia
RFTPIa = RF + TP + Ia
RFFPIa = RF + FP + Ia
RFTPCC = RF + TP + CC
RFFPCC = RF + FP + CC
RFTPIbc = RF + TP + Ibc
RFFPIbc = RF + FP + Ibc
RFTPII = RF + TP + II
RFFPII = RF + FP + II

# data samples
Data = 'Data'
Training = 'Training'
Validation = 'Validation'
Test = 'Test'
Simulated = 'Simulated'
Spec = 'Spec'
Spec_nofp = 'Spec_nofp'
Phot = 'Phot'
User = 'User'

# Missing data in Table  
nodata = -999

# Labels used for new table columns
_Eff = '_{:0.2f}'
RFprobability = RF + 'probability'
MaxProb = 'MaxProb'
_MaxProb = '_' + MaxProb
_id = '_id'
TrueClass_id = 'TrueClass' + _id
_FixedEffClass = '_FixedEffClass'
_FixedEffClass_id = _FixedEffClass + _id
MaxProbClass = 'MaxProbClass'
_MaxProbClass = '_MaxProbClass'
_MaxProbClass_id = _MaxProbClass + _id

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

# plot groups
Performance = 'Performance'
SALT = 'SALT'
Hubble = 'Hubble'
Error = 'Error'   
Color = 'Color'
Magnitude = 'Magnitude'
Bazin = 'Bazin'

#constants
zlo = 0.0
zhi = {Training: 1.4, Validation: 1.4, Test:1.4, Data: 1.1}

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
        Eff = float(g.nodata); Pur = Eff; 

    return Eff, Pur, Rem
