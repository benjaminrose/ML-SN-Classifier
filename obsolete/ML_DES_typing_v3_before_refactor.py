#!/usr/bin/env python
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
from matplotlib.backends.backend_pdf import PdfPages
import warnings
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import math as m
from matplotlib import rc
from sklearn.cross_validation import cross_val_score, train_test_split, StratifiedKFold, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from operator import itemgetter
from scipy.stats import randint as sp_randint
from scipy.interpolate import interp1d
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import check_array
from sklearn.externals import joblib
#from treeinterpreter import treeinterpreter as ti
import argparse

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

######### SN type variables used in code ########
#allSNtypes   -list of all possible SN types  (eg [Ia, CC, Ibc, II])
#MLtypes      -list of types used by 2-way, 3-way, .. ML classification (eg [Ia, CC], [Ia, Ibc, II] ..)
#CLFtypes     -list of types predicted by ML classification (eg [RFIa, RFCC], ... )
#TPFPtypes    -True positive and false positive types  (TP, FP)
#allTPFPtypes -list of all combinations of allSNtypes and TPFPtypes (eg [TPIa, FPIa, TPCC, FPCC,..)
#savetypes    -list of types for special printing, pickling etc
#trueplottypes -list of true types to be plotted (default = MLtypes + [Total])
#CLFplottypes -list of CLF types to be plotted (default = CLFtypes + [CLFTotal])
#plottypes    -list of all types to be plotted (default = MLtypes + [Total] + CLFtypes + [CLFTotal])
#alltypes     -list of true types and predicted types

######### Lists of data samples to be used in code (#############
#simlist    -list of sim samples (Training, Test)
#datalist   -list of data samples (Test, Spec etc)
#MLdatalist -list of samples typed with ML (Test + any other)
#lbldMLdatalist -list of data samples typed with ML and labeled (Test, Spec, )
#plotlist   -list of samples to be plotted (default = [simlist, datalist])
#CLFplotlist -list of samples for CLF-typed plots (default=[[Test], datalist])
#realdatalist -list of non-Test data

########## Dicts used in code ###################
#alldata    -dict with training, test and any other selected data
#true       -dict of masks for true types in test data
#masks      -dict of masks to select types
#predict    -dict of masks for CLF types
#CLFstats   -dict of masks for TP and FP types

#GLOBALS
#types, typing, classification
SN='SN'
Ia='Ia'
CC='CC'
Ibc='Ibc'
II='II'
Total='Total'
CLF='CLF'  #Classifier
RF='RF'    #Random-Forest
CLFlbls={RF:' (RF)'}
RFIa=RF+Ia
RFCC=RF+CC
RFIbc=RF+Ibc
RFII=RF+II
RFTotal=RF+Total
SNtypes=[Ia,CC]
CCtypes=[Ibc,II]
allSNtypes=SNtypes+CCtypes

#true and false positives
TP='TP'
FP='FP'
TPlbl='TP '
FPlbl='FP '
TPFPtypes=[TP,FP]
allTPFPtypes=[p+t for p in TPFPtypes for t in allSNtypes]
RFTPIa=RF+TP+Ia
RFFPIa=RF+FP+Ia
RFTPCC=RF+TP+CC
RFFPCC=RF+FP+CC
RFTPIbc=RF+TP+Ibc
RFFPIbc=RF+FP+Ibc
RFTPII=RF+TP+II
RFFPII=RF+FP+II

#alt type for multiple cuts
alt='alt'
altIa=alt+Ia
altCC=alt+CC
altIbc=alt+Ibc
altII=alt+II
altTotal=alt+Total

#samples
Sim='Sim'
Data='Data'
Weights='Weights'
ConstWts='ConstWts'
Training='Training'
Test='Test'
Validation='Validation'
Simulated='Simulated'
ST=' '.join([Simulated,Training])
SV=' '.join([Simulated,Validation])
STD=' '.join([Simulated,Training,Data])
SVD=' '.join([Simulated,Validation,Data])
Spec='Spec'
Spec_nofp='Spec_nofp'
Phot='Phot'
altTest=alt+Test
altSpec=alt+Spec
altSpec_nofp=alt+Spec_nofp
altPhot=alt+Phot

#other
Constant="Constant"
ByType="ByType"
fit_bands=['g','r','i','z']
colors=['gr','ri','iz']
colorlabels=['g-r','r-i','i-z']
nodata=-999
HR='HR'
SALT='SALT '
peak_m='peak_m'
peak_s='peak_s'
peaks=[peak_m,peak_s]
peaklabels=[' (Multi-band)',' (Single-band)']
SALTfilters=[f+p for p in peaks for f in fit_bands]
SALTmfilters=[f+peak_m for f in fit_bands]
SALTsfilters=[f+peak_s for f in fit_bands]
SALTfilterlabels=['$'+f+'$'+p for p in peaklabels for f in fit_bands]
SALTmfilterlabels=['$'+f+'$'+peaklabels[0] for f in fit_bands]
SALTsfilterlabels=['$'+f+'$'+peaklabels[1] for f in fit_bands]
SALTcolors=[c+p for p in peaks for c in colors]
SALTcolorlabels=['$'+c+'$'+p for p in peaklabels for c in colorlabels]
Delta='Delta'
SALTcolordiffs=[Delta+c for c in colors]
SALTcolordifflabels=['$\Delta('+c+')$' for c in colorlabels]
diff='difference'

######## Bazin parameter names ############ 
Bazin='Bazin'
Bazin_=Bazin+'_'
err='Err'
A='A'
t0='t0'
t_fall='t_fall'
t_rise='t_rise'
C='C'
bazinpars= [A,t0,t_fall,t_rise,C]
bazinlabels={A:'$A$',t0:'$t_0$',t_fall:'$t_{fall}$',t_rise:'$t_{rise}$',C:'$C$'}
bazinerrs=[p+err for p in bazinpars]
bazinerrlabels={A+err:'$\Delta A$',t0+err:'$\Delta t_0$',t_fall+err:'$\Delta t_{fall}$',t_rise+err:'$\Delta t_{rise}$',C+err:'$\Delta C$'}
Bazincolors=[col+Bazin for col in colors]
Bazincolorlabels=[Bazin+' $'+c+'$' for c in colorlabels] 
#Bazin cuts 
All='All'
_all='_all'
_eff='_eff'
failfit=-999
Failed='Failed'
Bazin_Fail=Bazin_+Failed
Bazin_Const=Bazin_+Constant

# PLOT VARIABLES ###################
scatter='scatter'
contour='contour'
linear='linear'
log='log'
ctrlevels = [.1, .5,.9]
Title='Title'
Large='Large'
Small='Small'
Scatter='Scatter'
Ticks='Ticks'
yrescale=1.3 #scale factor for adding margins to plots
Ncols=2 #columns for scatter plots
#adjust if adding new samples
color = {Ia:'royalblue',CC:'red',Ibc:'limegreen',II:'red',
         Data:{Test:'black',Spec:'purple',Spec_nofp:'cyan',Phot:'indianred',
         altTest:'dimgrey',altSpec:'rebeccapurple',altSpec_nofp:'teal',altPhot:'lightcoral'},
         Total:'black',RFIa:'blue',RFCC:'orangered',RFIbc:'green',RFII:'orangered',RFTotal:'black',
         contour:'black',RFTPIa:'royalblue',RFFPIa:'deeppink',RFTPCC:'red',RFFPCC:'orange',RFTPII:'red',
         RFFPII:'orange',RFTPIbc:'limegreen',RFFPIbc:'magenta',altIa:'lightblue',altCC:'darkorange',
         altII:'darkorange',altIbc:'yellowgreen',altTotal:'dimgrey'}
fill = {Ia:'stepfilled',CC:'step',Ibc:'step',II:'step',Data:{Test:'o',Spec:'v',Spec_nofp:'^',Phot:'o'},
        Total:'step',altIa:'step',altCC:'step',altIbc:'step',altII:'step'}
lw = {Ia:1,CC:2,Ibc:2,II:2,Data:1,Total:1,contour:30}
alpha = {Ia:1,CC:1.0,Ibc:1,II:1,Data:1,Total:1.0,Scatter:1.0}
markers = {Ia:'.',CC:'.',Ibc:'.',II:'.',Data:{Test:'o',Spec:'v',Spec_nofp:'^',Phot:'o'}}
plotlabels = {Ia:SN+Ia,CC:SN+CC,Ibc:SN+Ibc,II:SN+II,Total:Total,Sim:{Training:STD,Test:SVD},
              Data:{Test:SVD,Spec:Spec+'. Data',Spec_nofp:re.sub('_nofp','Data (no $f_p$ cut)',
              Spec_nofp),Phot:Phot+'. Data'}}
sizes = {Title:18,Large:16,Small:12,Scatter:10,Ticks:12}

#poplulate dicts with extra keys for the classifier types (RF for now)
for key in CLFlbls.keys():  
    CLFlbl=CLFlbls[key]
    for t in allSNtypes:
        plotlabels[key+t]=SN+t+CLFlbl
        plotlabels[key+TP+t]=TPlbl+SN+t+CLFlbl
        plotlabels[key+FP+t]=FPlbl+SN+t+CLFlbl
        markers[key+t]=markers[t]
        markers[key+TP+t]=markers[t]
        markers[key+FP+t]=markers[t]
        lw[key+t]=lw[t]
        lw[key+TP+t]=lw[t]
        lw[key+FP+t]=lw[t]
        alpha[key+t]=alpha[t]
        alpha[key+TP+t]=alpha[t]
        alpha[key+FP+t]=alpha[t]
        fill[key+t]=fill[t]
        fill[key+TP+t]=fill[t]
        fill[key+FP+t]=fill[t]

    plotlabels[key+Total]=Total
    alpha[key+Total]=alpha[Total]
    fill[key+Total]=fill[Total]
    lw[key+Total]=lw[Total]

#and alt type for plots with multiple cuts
for t in allSNtypes:
    markers[alt+t]=markers[t]
    lw[alt+t]=lw[t]
    alpha[alt+t]=alpha[t]
alpha[alt+Total]=alpha[Total]
lw[alt+Total]=lw[Total]    
fill[alt+Total]=fill[Total]
lw[alt+Ia]=2 #overwrite

plotdict={'color':color,'fill':fill,'lw':lw,'alpha':alpha,'labels':plotlabels,'markers':markers,
          'levels':ctrlevels,'sizes':sizes}

#variable ranges for z and mu
zlo=0.0
zhi=1.1
Nzbins=28
mulo=36.
muhi=46.

############ Parse arguments ############
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
    description='This script uses a machine learning (ML) algorithm to train a binary or tertiary photometric classifier into classes {Ia|CC or Ia|Ibc|II or I|II->Ia|Ibc}. Select Random Forest features to use.')
# Choose features
parser.add_argument('--ft', 
    choices=['c','x0','x1','t0','z','chi2','fit_pr','gpeak','rpeak','ipeak','zpeak','ra','dec', 'grpeak_m', 'ripeak_m', 'izpeak_m','grpeak_s','ripeak_s','izpeak_s','Deltagr','Deltari','Deltaiz','snr1','snr2','snr3',
             't0_err','x0_err','x1_err','c_err','Bazin_g_t_rise','Bazin_g_t_fall','Bazin_r_t_rise', 'Bazin_r_t_fall', 'Bazin_i_t_rise', 'Bazin_i_t_fall', 'Bazin_z_t_rise', 'Bazin_z_t_fall', 'grBazin','riBazin','izBazin'],
    nargs='+', default=['fit_pr','x1'],
    help='Choose SN-type features to use for classification. List them (space-separated) and select from: {%(choices)s}',
    metavar='features')
# Select number of cores
parser.add_argument('--nc', type=int, 
    choices=range(1,8), default=4, help='Number of cores to use for parallelization', metavar='n_cores')
# Select Efficiency
parser.add_argument('--eff', type=float, 
    default=0.95, help='Efficiency to evaluate purity at', metavar='Fix_eff')
# Choose filename
parser.add_argument('--filestr', 
    default='test', help='Choose string to append to filename')
parser.add_argument('--nclass', type=int, 
    default=2, help='Number of classes used in classifier (2=Ia or CC, 3=Ia or Ibc or II, 4=2x2 training: I/II ->Ia/Ibc)')
parser.add_argument('--data',choices=[Test,Spec,Spec_nofp,Phot],default=[],nargs='+',help='Plot selected "observed" data (default=no data); plots will be normalized to first data sample in list') 
#parser.add_argument('--train',default='DES_training_fitprob.txt',help='Filename for training')
#parser.add_argument('--test',default='DES_validation_fitprob_purity=0.5.txt',help='Filename for test')
parser.add_argument('--train',default='DES_training_SNR550.txt',help='Filename for training')
parser.add_argument('--test',default='DES_validation_SNR550.txt',help='Filename for test')
parser.add_argument('--sim',choices=[Training,Test],default=[Training],nargs='+',help='Plot selected "sim" data')
parser.add_argument('--save',choices=[Ia,CC,'allSN'],default=Ia,help='Types to save for printing/pickling')
parser.add_argument('--weights',choices=[ByType,Constant],default=ByType,help='Weights for plot normalizations')
parser.add_argument('--totals', action='store_true', default=False, help='Turn off totals in plots (ignored if plotting data)')
parser.add_argument('--xplots',action='store_true', default=False, help='Turn on extra plots with error and color distributions')
parser.add_argument('--store',action='store_true', default=False,help='Save trained classifier to pklfile') 
parser.add_argument('--restore',action='store_true', default=False,help='Restore trained classifier from pklfile') 
parser.add_argument('--pklfile',default='trained_clf',help='Base filename for trained classifier (_n(==nclass)way.pkl is auto-appended; also expects pre+filename if nclass==4)')
parser.add_argument('--train_only',action='store_true', default=False,help='Run training only, save trained classifier to pklfile, and exit')
parser.add_argument('--test_only',action='store_true', default=False,help='Run test (validation) only, reading trained classifier from pklfile, and exit')
parser.add_argument('--typecolumn',default='type',help='Column name for SN type in training/test data')
parser.add_argument('--typenames',choices=[Ia,II,Ibc],default=[Ia,II],nargs='+',help='Labels for classes in typecolumn')
parser.add_argument('--typevalues',default=[0,1],nargs='+',help='Corresponding class values in typecolumn')
parser.add_argument('--plotdir',default='./',help='Directory name for plot pdf files (Plots turned off if set to "")')
parser.add_argument('--cv', action='store_true', default=False, help='Turn on cross-validation')
parser.add_argument('--sample',
    choices=['t','v','b'], default='t', help='Sample to use for cross-validation (t=training, v=validation, b=both)')
parser.add_argument('--niter', type=int,
    choices=range(1,11), default=5, help='Number of ShuffleSplit cross-validation iterations')
parser.add_argument('--testsize', type=float,
    default=0.5, help='Test sample size for ShuffleSplit')
parser.add_argument('--nfolds', type=int,
    default=2, help='Number of folds for Kfold sample')
parser.add_argument('--pc', action='store_true', default=False, help='Turn on purity comparison')
parser.add_argument('--purities',nargs='+',default=[0.1,0.25,0.5,0.75,0.9],help='Purities for loop over test files')
parser.add_argument('--withhold',choices=['20','21','22','32','33','201','204','208','210','213','214','215','216','219','220','221','222','223','224','225','226','227','228','229','230','231','232','233','235','206','209','002','103','104','105','202','203','212','234','021','022','101','102','205','207','211','217','218'], nargs='+', default='', help='Hold-out test: withhold type (20=IIP, 21=IIN, 22=IIL, 32=Ib, 33=Ic) or template (206,209=IIN, 002=IIL, 201-233=IIP, 103-234=Ib, 021-218=Ic) from training sample')
parser.add_argument('--Bazin',choices=['train','plots','off'],default='off', help='Include Bazin selections in training and plots, plots only, or turn off')
parser.add_argument('--noBazinparcuts',action='store_true', default=False, help='Turn OFF Bazin parameter cuts')
parser.add_argument('--Bazinpar_max',nargs='+',default=[800,'',150,'t_fall',100],help='Cuts on Bazin fit parameters: [A,t0,t_fall,t_rise,C]')
parser.add_argument('--Bazinpar_min',nargs='+',default=['','',-999,-999,''],help='Cuts on Bazin fit parameters: [A,t0,t_fall,t_rise,C]')
parser.add_argument('--Bazinerrcuts',action='store_true', default=False, help='Turn ON Bazin error cuts')
parser.add_argument('--Bazinerr_max',nargs='+',default=[100,50,100,50,100],help='Cuts on Bazin fit errors: [A_err,t0_err,t_fall_err,t_rise_err,C_err]')
parser.add_argument('--prvar', action='store_true', default=False, help='Turn on probability-variance plots')

args = parser.parse_args()

Fix_eff = args.eff # fix efficiency to this

################ FUNCTIONS ##############
# Read in the sample for cross-validation based on user inputs
"""
def read_sample():
    # read in training data only
    if(args.sample=='t'):
        data = read('/nova-data/DES_SN/typing/machinelearning/DES_training_fitprob.txt')
    # read in validation data only
    elif(args.sample=='v'):
        data = read('/nova-data/DES_SN/typing/machinelearning/DES_validation_fitprob.txt')
    # combine both training and validation sets
    elif(args.sample=='b'):
        t = read('/nova-data/DES_SN/typing/machinelearning/DES_training_fitprob.txt')
        v = read('/nova-data/DES_SN/typing/machinelearning/DES_validation_fitprob.txt')
        data = vstack([t, v])
    return data
"""

# Compute efficiency, purity
def EffPur(true, predict):
    Tp = np.sum(true & predict)
    Fn = np.sum(true & ~predict)
    Fp = np.sum(~true & predict)
    Tn = np.sum(~true & ~predict)
    Eff = 1.0*Tp/(Tp + Fn)
    Pur = 1.0*Tp/(Tp + Fp)
    Rem = Fn + Tn # SNe removed 
    return Eff, Pur, Rem

# get only the selected features
def get_features(features, data):
    list = [] 
    for f in features:
        list.append(data[f])
    X = np.vstack(list).T
    return X

# find index of array where value is nearest given value
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

# score function to compute purity at fixed efficiency (=TPR)
def score_func(probs, y):
    Ia = (y==0)
    CC = (y>0)
    pur, eff, thresh = precision_recall_curve(y, probs, pos_label=0)
    purity_func = interp1d(eff[::-1], pur[::-1], kind='linear') # reverse-order so x is monotonically increasing
    metric = purity_func(Fix_eff) # purity at fixed efficiency=98%
    return float(metric)

def score_func_est(estimator, X, y):                             
    probs = estimator.predict_proba(X)[:, 0] # SNIa are class 0
    correct = (y==0)                                           
    wrong = (y>0)                                              
    pur, eff, thresh = precision_recall_curve(y, probs, pos_label=0)
    purity_func = interp1d(eff[::-1], pur[::-1], kind='linear') # reverse-order so x is monotonically increasing
    metric = purity_func(Fix_eff) # purity at fixed efficiency=98%                                              
    return float(metric)              

def get_EffPur(data,cuts):
    #compute efficiency, purity for fit_pr cuts
    C_Eff = []
    C_Pur = []
    CIa = data['sim_nonIa']==0
    for D in cuts:
        cut = data['fit_pr']>D
        Eff, Pur, Rem = EffPur(CIa, cut)
        C_Eff.append(Eff)
        C_Pur.append(Pur)

    return C_Eff,C_Pur

def get_purity_scores(yset, probs, Fix_eff):
    print "\nComputing purity scores for fixed efficiency {}".format(Fix_eff)
    pur, eff, thresh = precision_recall_curve(yset, probs, pos_label=0)
    print 'Length of threshold array = ', len(thresh) #number of unique probablility values
    #y = thresh[::-1][5000:-1]
    #x = eff[::-1][5001:-1]
    y = thresh[::-1] # reverse order so thresholds decrease
    x = eff[::-1][1::] # reverse order so efficiency increasing (for interp1d), remove last zero

    efficiency_func = interp1d(x, y, kind='linear') # reverse-order so x is monotonically increasing
    P_eff = efficiency_func(Fix_eff) # threshold probability at efficiency=Fix_eff
    print '\nProb (eff={:0.1f}%) = {:.3f}'.format(Fix_eff*100, float(P_eff))
    print 'Purity (P_thresh=0) = {:.3f}'.format(pur[0])
    score = score_func(probs, yset)
    print 'SCORE (pur @ eff={:0.1f}%) = {:.3f}'.format(Fix_eff*100, score)
    return pur, eff, thresh, P_eff

def get_ROC(yset_ROC, probs):
    ### Compute ROC curve and AUC
    fpr, tpr, roc_thres = roc_curve(yset_ROC, probs, pos_label=0) # restricted to binary classification
    AUC = 1 - roc_auc_score(yset_ROC, probs) # need (1 - score) if pos_label=0
    print '\nAUC = {:.3f}\n'.format(AUC)
    return fpr, tpr, roc_thres, AUC

def bin_data(data,Nbins=None):
    Ndata=np.asarray([])
    meanX=np.asarray([])
    mask=np.isfinite(data) #remove nan's and infs
    if(np.sum(mask)<len(data)):
        print 'Warning: data truncated by',len(data)-np.sum(mask),'elements to remove nans and infs'
    data=data[mask]
    if(len(data)>0):
        if(Nbins is None):
            Ndata,bins=np.histogram(data)
            print "Warning: Nbins not supplied; using default np binning Nbins=",Nbins
        else:
            Ndata,bins=np.histogram(data, bins=Nbins)
        sumX,bins=np.histogram(data, bins=bins,weights=data)
        meanX=sumX/Ndata
    return Ndata,meanX

def get_prob(clf,traindata,testdata,ytrain,debug=False,ntest=None):
    #computes probability from fractions of training-data classes in test-data landing leaf
    #DOES NOT account for duplicates
    classvals=list(set(y_train))
    leafs=clf.apply(traindata)    #all leaf nodes in training data
    average=np.zeros(shape=(len(testdata),len(classvals)))
    variance=np.zeros(shape=(len(testdata),len(classvals)))
    if(ntest is None):
        ntest=len(testdata)
    allnodes=clf.apply(testdata)  #landing nodes in decision trees (apply returns array of lists)
    for nt,nodes in enumerate(allnodes[0:ntest]):
        test=testdata[nt]
        if(debug):
            print "Test data #",nt,"=",test
            print "Leaf nodes =",nodes
        sumpr=np.array([0.]*len(classvals))  #init sum
        sumpr2=np.array([0.]*len(classvals))  #init sum
        for n,node in enumerate(nodes):      #loop thru nodes in trees
            match=(leafs.T[n]==node)
            classes=ytrain[match]       #classes of objects landing in same node
            if(debug):
                print "# of matching classes=",len(classes),"for",n,"th tree"
                cldist=[np.sum(classes==x) for x in classvals]
                print "# of each class",cldist
                print "Prob from estimator=",clf.estimators_[n].predict_proba(X_test)[nt]
            probs=[float(np.sum(classes==x))/float(len(classes)) for x in classvals]
            sumpr=sumpr+np.asarray(probs)
            sumpr2=sumpr2+np.asarray(probs)*np.asarray(probs)
            if(debug):
                print "Probs=",probs,"from #matches=",len(classes),"in",n,"th tree"

        average[nt]=sumpr/float(clf.get_params()['n_estimators'])
        variance[nt]=np.sqrt(sumpr2/float(clf.get_params()['n_estimators'])-average[nt]*average[nt])

    #endfor
    return average,variance

def prob_errors(model, X, percentile=90,debug=False,ntest=None):
    if(ntest is None):
        ntest=len(X)    
    sumpr=np.zeros_like(model.estimators_[0].predict_proba(X))
    sumpr2=np.zeros_like(model.estimators_[0].predict_proba(X))
    err_up=np.zeros_like(model.estimators_[0].predict_proba(X))
    err_down=np.zeros_like(model.estimators_[0].predict_proba(X))
    nclass=clf.n_classes_
    pdata=[]
    for n,dtree in enumerate(model.estimators_):
        probs=dtree.predict_proba(X)
        if(debug):
            print "Probs=",probs[0:ntest],"for",n,"th tree"
        sumpr=sumpr+probs
        sumpr2=sumpr2+probs*probs
        for p,prob in enumerate(probs):
            if  (n==0):      #first run through pdata list
                pdata.append([[] for c in range(nclass)])
            for c in range(nclass):
                pdata[p][c].append(prob[c])  #accumulate probs for each tree

    #compute percentiles
    pdata=np.asarray(pdata)
    for p, data in enumerate(pdata):
        for c in range(nclass):
            err_down[p][c]=np.percentile(pdata[p][c], (100 - percentile)/2.)
            err_up[p][c]=np.percentile(pdata[p][c], 100-(100 - percentile)/2.)

    average=sumpr/float(clf.get_params()['n_estimators'])
    variance=np.sqrt(sumpr2/float(clf.get_params()['n_estimators'])-average*average)
    return average,variance,err_up,err_down,pdata

def getdatabins2d(xdata,ydata,xbinwidth,ybinwidth,xlimits=None,ylimits=None):
         H_cut=np.asarray([])
         x_cen=np.asarray([])
         y_cen=np.asarray([])
         if(len(xdata)>0 and len(ydata)>0):
             #print "2d minmax x",min(xdata),max(xdata),type(min(xdata)),type(max(xdata))
             #print "2d minmax y",min(ydata),max(ydata),type(min(ydata)),type(max(ydata))
             x_limits=[np.floor(min(xdata)),np.ceil(max(xdata))] #ranges in data
             y_limits=[np.floor(min(ydata)),np.ceil(max(ydata))]
             if(xlimits is not None):                                    #adjust plot ranges as supplied
                 xlimits=[max(xlimits[0],x_limits[0]),min(xlimits[1],x_limits[1])]
             if(ylimits is not None):
                 ylimits=[max(ylimits[0],y_limits[0]),min(ylimits[1],y_limits[1])]
             Nxbins=(x_limits[1]-x_limits[0])/xbinwidth              #use binwidth on full range of data
             Nybins=(y_limits[1]-y_limits[0])/ybinwidth
             #print Nxbins, Nybins,int(Nxbins),int(Nybins)
             H, x_bins, y_bins = np.histogram2d(xdata,ydata,bins=(int(Nxbins),int( Nybins)),range=[x_limits,y_limits])
             H_frac=H/np.max(H)
             #truncate to supplied limits (remove upper x and y edge)
             index=np.ones(len(x_bins[:-1]), dtype=bool)
             indey=np.ones(len(y_bins[:-1]), dtype=bool)
             if(xlimits is not None):
                 index=(x_bins[:-1]>=xlimits[0]) & (x_bins[:-1]<=xlimits[1])
             if(ylimits is not None):
                 indey=(y_bins[:-1]>=ylimits[0]) & (y_bins[:-1]<=ylimits[1])
             #find truncated bin centers
             x_cen, y_cen = np.meshgrid(x_bins[:-1][index], y_bins[:-1][indey])
             x_cen+=(0.5*(x_bins[1]-x_bins[0]))
             y_cen+=(0.5*(y_bins[1]-y_bins[0]))
             H_x=H_frac[index]    #select out xrows of H according to selected x_cen
             H_cut=H_x.T[indey]   #transpose H_x, and use same trick to select y_cen
         return H_cut,x_cen,y_cen

#print and write to file
def tee(line,f):
    print line
    if(f is not None):
        f.write(line+'\n')
    return

def bazin(t,A,t0,t_fall,t_rise,C):
    #print "called with",A,t0,t_fall,t_rise,C
    dt=t-t0
    if A == 0:
        bn = constantC(t,C)
    else:
        bn=A*(np.exp(-dt/t_fall)/(1+np.exp(-dt/t_rise))) + C
    return bn

def constantC(t,C):
    cc = np.zeros(len(t))
    cc[:] = C
    return cc

def plot_types(types, xvar, alldata, plotlist=[], weights=True, xlabel='', ylabel='', cuts={}, plotdict={},
        xscale=linear, yscale=linear, title='', bins=None, Nbins=10, plotid='', asize=Large, yvar='',
        minmax=False, ctrxbin=0., ctrybin=0., alt='', addlabel='', selectdata='Total', debug=False):
    #plot var for SN types
    
    color=plotdict['color']
    fill=plotdict['fill']
    lw=plotdict['lw']
    alpha=plotdict['alpha']
    sizes=plotdict['sizes']
    lgndlabels=plotdict['labels']
    plt.xlabel(xlabel,size=sizes[asize])
    plt.ylabel(ylabel,size=sizes[asize])
    if(len(yvar)>0):
        plt.tick_params(axis='both', which='major', labelsize=sizes[Ticks])

    #TODO option for uniform binning
    if (bins is None):
        bins=Nbins

    #determine what keys need to be plotted
    #allkeys=alldata.keys()       #check alldata keys 
    #checkall=[(k in allplottypes) for k in allkeys]
    datalist=[]
    simlist=[]
    if(len(plotlist)==0):  #special case; alldata has only SNtype keys
        simlist=['']
    elif(type(plotlist)==list and len(plotlist)>0):
        simlist=plotlist[0]
        if(type(simlist)!=list):
            simlist=[simlist]
        if(len(plotlist)>1):
            datalist=plotlist[1]
            if(type(datalist)!=list):
                datalist=[datalist]
    else:
        print "Invalid plotlist: supply list of dict keys to be plotted:[Sim,Data] or [[Simlist],[Datalist]]"

    if(len(datalist)>0):  #plot data
        for data in datalist:
            #print xvar,data,(xvar in alldata[data][selectdata].keys())
            if(xvar in alldata[data][selectdata].keys()):  #make sure data has required variable
                #print 'plotting',var,'in',data
                obsdatax={}
                obsdatay={}
                if (len(cuts)==0):
                    obsdatax=alldata[data][selectdata][xvar]
                    if(len(yvar)>0):
                        obsdatay=alldata[data][selectdata][yvar]
                elif (data in cuts.keys()):
                    if (selectdata in cuts[data].keys()):
                        if(np.sum(cuts[data][selectdata])>0):
                            obsdatax=alldata[data][selectdata][xvar][cuts[data][selectdata]]
                            if(len(yvar)>0):
                                obsdatay=alldata[data][selectdata][yvar][cuts[data][selectdata]]
                        else:
                            print "Skipping {} data {} {} plot: no data passing cut".format(data,plotid,xvar)
                    else:
                        print "Skipping {} data {} {} plot: no entry for {} in cuts".format(data,plotid,xvar,selectdata)
                else:
                    print "Skipping {} data {} {} plot: no entry for sample in cuts".format(data,plotid,xvar)
                if(len(obsdatax)>0):
                    if(len(yvar)==0):
                        if(minmax):
                            print 'Total {} data: {}; min= {}, max = {}'.format(data,xvar,np.min(obsdatax),np.max(obsdatax))
                        Ndata,meanX=bin_data(obsdatax,Nbins=bins)
                        #print Ndata,meanX,np.sqrt(Ndata)
                        plt.errorbar(meanX,Ndata,yerr=np.sqrt(Ndata),label=' '.join([selectdata,lgndlabels[Data][data],addlabel]),
                            color=color[Data][alt+data],fmt=fill[Data][data])
                    elif(len(obsdatay)>0):
                        if(minmax):
                            print 'Total {} data: {}; min= {}, max = {}; {}; min= {}, max = {}'.format(data,xvar,np.min(obsdatax),
                                np.max(obsdatax),yvar,np.min(obsdatay),np.max(obsdatay))
                        #print "Scattersize",sizes[Scatter]
                        plt.scatter(obsdatax,obsdatay,marker=markers[Data][data],alpha=alpha[Scatter],s=sizes[Scatter],
                            color=color[Data][alt+data], label=' '.join([selectdata,lgndlabels[Data][data],addlabel]))
                    else:
                        print "Skipping {} data {} {} scatter plot: no data passing cut".format(data,plotid,yvar)
            else:
                print "{} and/or {} not available in {} {} data".format(xvarname, yvarname, data, selectdata)

    if(len(simlist)>0):
        for sim in simlist:
            if (sim==''):
                simdict=alldata   #simple dict of types
            else:
                if(alldata.has_key(sim)):
                    simdict=alldata[sim]
                    #print sim,simdict.keys()
                else:
                    simdict={}
            if(len(yvar)>0 and weights):
                weights=False
                print 'Ignoring weights for 2-d plot'
            if(len(simdict)>0):
              if(weights and simdict.has_key(Weights)):
                sweights=simdict[Weights]
                #print sweights.keys()
              else:
                sweights={}      #set weights to None
                for t in types:
                    sweights[t]=None
              for t in types:
                  if(t in simdict.keys()):
                      simdatay={}      #initialize to empty
                      if (len(cuts)==0):
                          simdatax=simdict[t][xvar]
                          if (sweights.has_key(t)):
                              simweights=sweights[t]
                          else:       #happens if there is no data
                              simweights=np.asarray([])
                          if(len(yvar)>0):
                              simdatay=simdict[t][yvar]
                      elif (np.sum(cuts[sim][t])>0):
                          simdatax=simdict[t][xvar][cuts[sim][t]]
                          if(sweights[t] is not None):
                              simweights=sweights[t][cuts[sim][t]]
                          if(len(yvar)>0):
                              simdatay=simdict[t][yvar][cuts[sim][t]]
                      else:
                          simdatax={}
                      
                      if(len(simdatax)>0 and len(yvar)==0): #1d plot
                        if(minmax):
                            print '{} {} data: {}; min= {}, max = {}'.format(t,sim,xvar,np.min(simdatax),np.max(simdatax))
                        plt.hist(simdatax, bins=bins,label=lgndlabels[t]+addlabel,color=color[alt+t],histtype=fill[alt+t],lw=lw[alt+t],weights=simweights)
                        if(debug):
                            print 'Plotting {} {} data points for type {}'.format(len(simdatax),sim,t)
                      elif(len(simdatax)>0 and len(simdatay)>0):              #2d plot of types excluding any totals
                          if(t.find(Total)==-1):
                              if(minmax):
                                  print '{} {} data: {}; min= {}, max = {}; {}; min= {}, max = {}'.format(t,sim,xvar,np.min(simdatax),np.max(simdatax),yvar,np.min(simdatay),np.max(simdatay))
                              plt.scatter(simdatax,simdatay,marker=markers[alt+t],alpha=alpha[Scatter],s=sizes[Scatter],color=color[alt+t],label=lgndlabels[t]+addlabel)
                              if(ctrxbin>0. and ctrybin > 0. and t.find(Ia)!=-1 and t.find('FPIa')==-1):  #contours for Ia only
                                  H_cut,x_cen,y_cen=getdatabins2d(simdatax,simdatay,ctrxbin,ctrybin)
                                  c1 = plt.contour(x_cen, y_cen, H_cut, levels=plotdict['levels'], colors=color[contour],linewidth=lw[contour])
                                  plt.clabel(c1, inline=1, fontsize=8)
                                  c1.collections[0].set_label(lgndlabels[contour])                        
                      else:
                          print "Skipping type {} {} plot; {} {}: no data passing cut".format(t,plotid,xvar,yvar)
                  #else:
                  #    print "Skipping type {} {} plot: no data of this type".format(t,plotid,xvar)
            else:
                print "Skipping {} data {} plot: sample not found".format(sim,plotid)
 
    if(len(title)>0):
        #print "title:",title
        plt.title(title,size=sizes[Title])
    
    #set scales after doing plots to avoid ValueErrors for log scale
    plt.yscale(yscale)
    plt.xscale(xscale)


def get_mask(types,alldata, varlist, varlabels, SNTotals, cuts=[], ops=[], not_ops=[],good=True, mask_id='mask_id',file_id=''):
    #eg. SALTcolormask=get_mask(plottypes,alldata,SALTcolors,SALTcolorlabels,mask_id='SALT color',file_id=file_id)
    mask={}
    ff=open(mask_id+file_id+'.eff','w')
    if(good):
        cuts=[nodata]
        ops=['!=']
        not_ops=['==']
    tee('\nEfficiencies for {} cuts'.format(mask_id),ff)
    for cut,op,not_op in zip(cuts,ops,not_ops): #assume op is !=
        for var,label in zip(varlist,varlabels):
            key_pass=var+op+str(cut)
            key_fail=var+not_op+str(cut)
            mask[key_pass]={}
            mask[key_fail]={}     #need separate mask due to nans in data; can't just take ~
            txtlabel=re.sub('\$','',label)
            txtlabel=re.sub(r'\\','',txtlabel)
            tee('\n{} {}'.format(mask_id,txtlabel),ff)
            for dkey in alldata.keys():
                tee('  {} Sample'.format(dkey),ff)
                mask[key_pass][dkey]={}
                mask[key_fail][dkey]={}
                tee('       Type   NPass  Efficiency',ff)
                for t in types:
                    #setup masks for =cut and !=cut
                    if(t in alldata[dkey].keys()):
                        if(var in alldata[dkey][t].keys()):
                            if(SNTotals[dkey][t]>0):
                                mask[key_pass][dkey][t]=(alldata[dkey][t][var]!=cut) & np.isfinite(alldata[dkey][t][var])
                                mask[key_fail][dkey][t]=(alldata[dkey][t][var]==cut) & np.isfinite(alldata[dkey][t][var])
                                mask[key_pass][dkey][t+_eff]=float(np.sum(mask[key_pass][dkey][t]))/SNTotals[dkey][t]
                                mask[key_fail][dkey][t+_eff]=float(np.sum(mask[key_fail][dkey][t]))/SNTotals[dkey][t]
                                tee('        {:5} {:6} {:10.3f}'.format(t,np.sum(mask[key_pass][dkey][t]),mask[key_pass][dkey][t+_eff]),ff)
                                tee(' {:2}{:4} {:5} {:6} {:10.3f}'.format(not_op,cut,t,np.sum(mask[key_fail][dkey][t]),mask[key_fail][dkey][t+_eff]),ff)
                            #else:
                            #    tee('   No {} data for type {}'.format(dkey,t),ff)
                        else:
                            tee('{}  not available for type {}'.format(var,t),ff)
                    #else:
                    #    tee('    Type {}  not available for {} data'.format(t,dkey),ff)

    ff.close()
    return mask


def get_Bazincuts(alldata,bazinparmin,bazinparmax,bazinerrmax,SNTotals,bazinparcuts=True,bazinerrcuts=False,savetypes=[],file_id=''):
    #save efficiencies for selected types
    if(len(savetypes)>0):
        ff=open('Bazin_cut'+file_id+'.eff','w')
    else:
        ff=None

    bazinmin=[]
    bazinmax=[]
    if(bazinparcuts and bazinerrcuts):  #both par and err cuts
        bazinmax=bazinparmax+bazinerrmax
        bazinall=bazinpars+bazinpars+bazinerrs #parameters for min, max, and max error cuts
        bazinmin=bazinparmin
    elif(bazinparcuts):  #only par cuts
        bazinmax=bazinparmax
        bazinmin=bazinparmin
        bazinall=bazinpars+bazinpars
    elif(args.Bazinerrcuts):  #only err cuts
        bazinmax=bazinerrmax
        bazinall=bazinerrs        
    opmax=['<=' for b in bazinmax]
    opmin=['>' for b in bazinmin]
    bazinminmax=bazinmin+bazinmax
    opall=opmin+opmax     
    _bazinall=['_'+b for b in bazinall]
    tee("\nUsing Bazin parameter cuts:\n"+str(zip([b for b in bazinall],opall,bazinminmax)),ff)

    #Make a dict of data quality cuts
    #initialze cut dicts 
    Bazincuts={}
    for filt in fit_bands:
        Bazincuts[filt]={}
        tee("\nEfficiencies for {} band".format(filt),ff)
        Bazincuts[filt][All]={}   #setup dict for and of all cuts 
        Bazincuts[filt][Bazin_Fail]={}
        Bazincuts[filt][Bazin_Const]={}
        for dkey in alldata.keys():
            tee('  {} Sample'.format(dkey),ff)
            Bazincuts[filt][dkey]={}  
            Bazincuts[filt][All][dkey]={}
            Bazincuts[filt][Bazin_Fail][dkey]={}
            Bazincuts[filt][Bazin_Const][dkey]={}
            for t in plottypes: 
                Bazincuts[filt][dkey][t]={}
                if(t in alldata[dkey].keys()):
                  Bazincuts[filt][All][dkey][t]=np.ones(len(alldata[dkey][t]), dtype=bool) #initialze and of cuts to all true
                  allkeys=alldata[dkey][t].keys()
                  if(Bazin_+filt+'_'+A in allkeys and Bazin_+filt+'_'+t_fall in allkeys):
                    Bazincuts[filt][Bazin_Const][dkey][t]=(alldata[dkey][t][Bazin_+filt+'_'+A]==0.) & (alldata[dkey][t][Bazin_+filt+'_'+t_fall]==0.)#find constant fits
                    Bazincuts[filt][Bazin_Fail][dkey][t] =(alldata[dkey][t][Bazin_+filt+'_'+t_rise]==failfit) & (alldata[dkey][t][Bazin_+filt+'_'+t_fall]==failfit)
                    if (t in savetypes):
                        tee("    Type {}: Total # SN = {}".format(t,int(SNTotals[dkey][t])),ff)
                        tee('    # SN with constant Bazin fits = {}'.format(np.sum(Bazincuts[filt][Bazin_Const][dkey][t])),ff)
                        tee('    # SN with failed Bazin fits = {}'.format(np.sum(Bazincuts[filt][Bazin_Fail][dkey][t])),ff)
                        tee('        Parameter   Cut       NPass  Efficiency  NPass_all Cumulative Efficiency',ff) 
                    for par,_par,op,mcut in zip(bazinall,_bazinall,opall,bazinminmax):
                        varname=Bazin_+filt+_par   #table variable name
                        if (varname in allkeys):   #check if var is available
                            key=par+op+str(mcut)            #key in dict
                            if(type(mcut)!=str):   #value can be min or max
                                if(op=='<='):
                                    Bazincuts[filt][dkey][t][key]=(alldata[dkey][t][varname]<=mcut)
                                else:
                                    Bazincuts[filt][dkey][t][key]=(alldata[dkey][t][varname]>mcut)
                            elif(len(mcut)>0):
                                Bazincuts[filt][dkey][t][key]=(alldata[dkey][t][varname]<=alldata[dkey][t][Bazin_+filt+'_'+mcut]) #includes const fits
                        #cumulative efficiencies
                            if(Bazincuts[filt][dkey][t].has_key(key)):   
                                if(SNTotals[dkey][t]>0):
                                    Bazincuts[filt][All][dkey][t]= (Bazincuts[filt][All][dkey][t]) & (Bazincuts[filt][dkey][t][key])
                                    Bazincuts[filt][dkey][t][key+_eff]=float(np.sum(Bazincuts[filt][dkey][t][key]))/SNTotals[dkey][t]
                                    Bazincuts[filt][All][t+_eff]=float(np.sum(Bazincuts[filt][All][dkey][t]))/SNTotals[dkey][t]
                                    if (t in savetypes):
                                        tee('      {:11} {:2}{:6}  {:6} {:10.3f} {:8} {:10.3f}'.format(par,op,str(mcut),np.sum(Bazincuts[filt][dkey][t][key]),Bazincuts[filt][dkey][t][key+_eff],np.sum(Bazincuts[filt][All][dkey][t]),Bazincuts[filt][All][t+_eff]),ff)
                                #else:
                                #    tee('    No {} data for type {}'.format(dkey,t),ff)
                        else:
                            tee('      {:11}:No data'.format(varname))

                  else:
                    tee('    Type {}: Bazin fits not available'.format(t),ff)
                #else:
                #  tee('    Type {} not avalable for {} data'.format(t,dkey),ff)  


    #aggregate fit failures (t_rise and t_fall <= -999)
    keyAll=Bazin_Fail+'_'+All
    Bazincuts[keyAll]={}
    for dkey in alldata.keys():
        tee('\n{} Sample'.format(dkey),ff)
        Bazincuts[keyAll][dkey]={}  
        for t in plottypes:
            if(t in alldata[dkey].keys()):
                Bazincuts[keyAll][dkey][t]=np.ones(len(alldata[dkey][t]), dtype=bool) #initialze
                for filt in fit_bands:
                    if(t in Bazincuts[filt][Bazin_Fail][dkey].keys()):
                        Bazincuts[keyAll][dkey][t]= (Bazincuts[keyAll][dkey][t]) & (Bazincuts[filt][Bazin_Fail][dkey][t])

                        if (t in savetypes):
                            tee('  # SN {} with Bazin fit failures in all filters = {}'.format(t,np.sum(Bazincuts[keyAll][dkey][t])),ff)
                        #tee('some snids of failures:{}'.format(alldata[dkey][t]['snid'][Bazincuts[keyAll][dkey][t]][0:min(np.sum(Bazincuts[keyAll][dkey][t]),5)]),ff)

    if(len(savetypes)>0):
        ff.close()
    #pickle Bazincuts
    fnpickle(Bazincuts,'Bazincuts'+file_id+'.pkl')

    return Bazincuts

def get_nplot(fig,nplot,pagenum='',plotsperpage=6):
    if(nplot>=plotsperpage):
        fig.tight_layout()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            multiPdf.savefig(fig)

        print "\nStarting page {}a (for overflow data)".format(pagenum)
        fig = plt.figure(figsize=(15,7))
        nplot=1

    else:
        nplot+=1
    
    return fig,nplot

#############################################
#Start of main module
#############################################

# Planck+WMAP polar+BAO+JLA
H0=68.62
OmegaM=.301

SNRcut=5
if(not(args.test_only)):
    ttrain = read(args.train)
    print 'Training Data:', args.train
    print "Initial sizes of training data:",len(ttrain)
    cuttrain = ttrain['snr1']>SNRcut   # no cut at the moment
    data_train = ttrain[cuttrain]
if(not(args.train_only)):
    ttest =  read(args.test)
    print 'Test Data:',args.test
    print "Initial sizes of test data:",len(ttest)
    cuttest = ttest['snr1']>SNRcut   # no cut at the moment
    data_test = ttest[cuttest]
    #print ttest['sim_type']

if (args.save=='allSN'):
    savetypes=allSNtypes
else:
    savetypes=args.save

#check for withholding
if(len(args.withhold)>0):
    print "Hold-out test: withholding",args.withhold,"from training sample"
    for withhold in args.withhold:
        if(len(withhold)==2):
            withinfo='type'
            withcut=(data_train['sim_type']==int(withhold))
            data_train=data_train[~withcut]        
        elif(len(withhold)==3):
            withinfo='template'
            withcut=(data_train['sim_nonIa']==int(withhold))
            data_train=data_train[~withcut]
        else:
            print "Unknown option",args.withhold
        print "withholding",withinfo,withhold,"from training sample"
        print "Found",np.sum(withcut),"matches in training data"
        print "Length of training data after witholding cut",len(data_train)

    #endfor-withold
    withpdf='_no_'+'_'.join(args.withhold)
else:
    withpdf=''

#read data files
#TODO..shouldn't data files also have the cut applied??
if(Spec in args.data):
    #data_spec = read('specSN_SNANA.txt')
    data_spec = read('specType_forcePhoto_bazin_v3.txt')
if(Spec_nofp in args.data):
    data_spec_nofp = read('specSN_SNANA_nofpcut.txt')
if(Phot in args.data):
    #data_phot = read('DES_validation_SNR550_Ibc1mag_purity=0.25.txt')
    data_phot = read('host_prior_DESY1-4_bazin_v3.txt')

    #cut out any data with z>zhi (otherwise z-histograms get messed up)
    zpass=(data_phot['z']<zhi)
    nzfail = len(data_phot['z'])-np.sum(zpass)
    if(nzfail>0):
        print 'Removing {} entries failing z < {}'.format(nzfail,zhi)
        data_phot=data_phot[zpass]
        print 'Total Number Photometric Data = ',len(data_phot)

#Prep initial data sets 
traintype="Ia's"
blurb=''
if(not(args.test_only)):
    X_train = get_features(args.ft, data_train)
if(not(args.train_only)):
    X_test = get_features(args.ft, data_test)

if(args.nclass==2):
    print "2-way classification"
    if(not(args.test_only)):
        y_train = data_train['type'] # class 0=Ia match, 1=CC match
    if(not(args.train_only)):
        y_test = data_test['type']
elif(args.nclass==3):
    print "3-way classification"
    if(not(args.test_only)):
        y_train = data_train['type3'] #3-way typing
    if(not(args.train_only)):
        y_test = data_test['type3']
elif(args.nclass==4):                #X_train and X_test modified after classifiers are built
    print "2x2-way classification"
    traintype="Iabc's"
    blurb='Pre-'
    if(not(args.test_only)):
        X_pretrain = X_train        #setup pretrain and pretest
        y_pretrain = data_train['type2x2'] #Ia + Ibc
        cut2 = (data_train['type3']==2) #identify type II
        data_train_abc = data_train[~cut2]
        X_train_abc = get_features(args.ft, data_train_abc)  #include only Ia and Ibc
        y_train_abc = data_train_abc['type']
        y_train = y_pretrain  #assign for pre-train step
    if(not(args.train_only)):
        X_pretest = X_test
        y_pretest = data_test['type2x2']   
        y_test = y_pretest   #y_test_abc found later after pre-training 

if(not(args.train_only)):
    y_test_ROC = data_test['type'] # for ROC curve, need binary labels

nfeatures=len(args.ft)
if(not(args.test_only)):
    print 'Training with', nfeatures, 'features:',args.ft
    print blurb+'Training set size = ', len(y_train), '\tNumber of',traintype,'=', np.sum(y_train==0)
if(not(args.train_only)):
    print blurb+'Test set size = ', len(y_test), '\tNumber of',traintype,'=', np.sum(y_test==0)


#setup plotlist according to run options
simlist=args.sim
datalist=args.data                 #datalist to be included in plots (as points)
realdatalist=[t for t in datalist if not(t==Test)]
MLdatalist=[Test] + realdatalist   #list of classified data: Test data + any additional data
lbldMLdatalist= [t for t in MLdatalist if not(t==Phot)]  #list of data with known types
plotlist=[simlist,datalist]
#TODO- check this now that Training data also has RF typing
CLFplotlist=[[Test],datalist]  #plot Test data as sim, and any other data

#setup type lists to be used in code, plotting and printing
if(args.nclass==2):
    MLtypes=[Ia,CC]
else:
    MLtypes=[Ia,Ibc,II]
trueplottypes=[t for t in MLtypes]
if(args.totals or len(datalist)>0):
    trueplottypes.append(Total)

#setup TP/FP types for each MLtype for special-case plots
#setup alltypes for computing norms and weights; include CLF-predicted types
CLFtypes=[]
allTPFPtypes=[]
CLFplottypes=[]
for CLFid in CLFlbls.keys():
    for t in trueplottypes:
        CLFplottypes.append(CLFid+t)

    for t in MLtypes:
        CLFtypes.append(CLFid+t)
        for p in TPFPtypes:
            allTPFPtypes.append(CLFid+p+t)

alltypes=MLtypes +CLFtypes + allTPFPtypes #all possible variants of typing

plottypes=trueplottypes+CLFplottypes

#in future may build a loop over classifiers
#for CLFlbl in CLFlbls.keys():
#CLFid=CLFlabels[RF]
#use RF for now
CLFid=RF

#Bazin setup
Bazinparcuts=not(args.noBazinparcuts)
doBazincuts=(Bazinparcuts) or (args.Bazinerrcuts)
Bazincuts={}
if(args.Bazin=='train'):
    if(doBazincuts):
        print "\nNot applying Bazin cuts to training data yet\n"
        #need alldata defined; would have to reorganize code
        #Bazincuts=get_Bazincuts(alldata,args.Bazinpar_min,args.Bazinpar_max,args.Bazinerr_max,Bazinparcuts,args.Bazinerrcuts,savetypes=savetypes,file_id=file_id)


#RANDOM FOREST CLASSIFIICATION
#--------------------------------
if(args.test_only):   #read classifier from file
    args.restore=True
if(args.train_only):  #force save
    args.store=True
#setup default pklfilename
pklfile=args.pklfile+'_'+str(args.nclass)+'way.pkl'

if(args.restore):
    clf=joblib.load(pklfile)
    classifiers=[clf]
    classifierid=['clf']
    if(args.nclass==4):
        clfpre=joblib.load('pre'+pklfile) #load pre-trained classifier
        print "Loading pre-classifier from file",'pre'+pklfile
        classifiers.insert(0,clfpre)
        classifierid.insert('clfpre')
    print "Loading classifier from file",pklfile
else: 
    # build a classifier
    n_estimators = 100
    max_features = 'auto'
    min_samples_split = 5 # 500 works well for fit_pr alone 
    criterion = 'entropy' # or 'gini'
    clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, \
      min_samples_split=min_samples_split, criterion=criterion, n_jobs=args.nc)

    ### fit the training data
    print '\nTraining classifier(s) across %d cores  . . . ' %args.nc
    if(args.nclass==4):
        nclf=2
        #define pre-classifier to separate Iabc's from IIs
        clfpre= RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, \
                min_samples_split=min_samples_split, criterion=criterion, n_jobs=args.nc)
        clfpre.fit(X_pretrain, y_pretrain)
        #train Ia/bc classifier
        clf.fit(X_train_abc, y_train_abc)
        classifiers =[clfpre, clf]
        classifierid=['clfpre','clfabc']
        # now assemble training/test sets into lists for later use
        X_train=[X_pretrain,X_train_abc]
        y_train=[y_pretrain,y_train_abc]
    else:
        nclf=1
        clf.fit(X_train, y_train)
        classifiers =[clf]
        classifierid=['clf']
    if(args.store):
        if(args.nclass==4):
            joblib.dump(clfpre, 'pre'+pklfile)
            print "Saving pre-classifier to file",'pre'+pklfile
        joblib.dump(clf, pklfile)
        print "Saving classifier to file",pklfile

### determine feature importances for tree-based estimators (RF, AdaBoost)
for n,cl in enumerate(classifiers):
    print '\nComputing feature importances for',CLFid,'classifier',classifierid[n]
    importances = cl.feature_importances_
    F = Table([args.ft, importances], names=('Feature','Importance'), dtype=['a','f4'])
    F.sort('Importance')
    F.reverse()
    print
    print F

#setup allsimlist = [Training,Test] unless suppressed by options
if(args.train_only):
    allsimlist=[Training]
    allX=[X_train]
    ally=[y_train]
    alldat=[data_train]
elif(args.test_only):
    allsimlist=[Test]
    allX=[X_test]
    ally=[y_test]
    alldat=[data_test]
else:
    allsimlist=[Training,Test]
    allX=[X_train,X_test]
    ally=[y_train,y_test]
    alldat=[data_train,data_test]

###Type allsimlist; compute purities, efficiencies
#-------------------------------------------------------------
purity={}
efficiency={}
allprobs={}
thresh={}
P_eff={}
probs={}
true={}
for sim, xset, datset, yset in zip(allsimlist, allX, alldat, ally):
    print "\n{} DATA RESULTS".format(sim.upper())
    if(args.nclass==4):   #2x2 test
        allprobs[sim]={}
        purity[sim]={}
        efficiency[sim]={}
        thresh[sim]={}
        P_eff[sim]={}
        #loop over classifiers
        for cl, xs, ys in zip(classifiers, xset, yset): #xset=[[X_pretrain,X_train_abc],]? ie X_Train=[X_pretrain,X_train_abc]
            allprobs[sim][cl]=cl.predict_proba(xs)
        """    
        #convert this to new format and test, NOT WORKING YET
        allprobsA=clfA.predict_proba(X_testA)
        data_test['ProbA'] = allprobsA[:, 0]
        ### Compute efficiencies, purities for Stage A
        purA, effA, threshA, P_effA = get_purity_scores(y_testA, allprobsA[:,0], Fix_eff)
        fprA, tprA, roc_thresA, AUCA=get_ROC(y_testA, allprobsA[:,0])
        maxprobclass=clfA.predict(X_test)  #get predicted Ia/Ibc (use maxpr for now)
        print "Stage A classifier predicts",np.sum(maxprobclass==0),"Iabc's"
        print "Run stage B classifier on Iabc subset of test data\n"
        cutIabc=(maxprobclass==0)
        X_testB=X_testA[cutIabc]
        y_testB=data_test[cutIabc]['type']        #define y_test as type of predicted Iabc's
        y_test_ROC = data_test[cutIabc]['type']   #for ROC curve, need binary labels
        allprobs=clf.predict_proba(X_testB)
        data_test['Prob']=0.0
        data_test['Prob'][cutIabc] = allprobs[:,0]
        ### Compute efficiencies, purities for StageB
        pur, eff, thresh, P_eff = get_purity_scores(y_testB, allprobs[:,0], Fix_eff)
        """
    else:
        allprobs[sim]=clf.predict_proba(xset)
        probs[sim] = allprobs[sim][:, 0] # SNeIa are class 0
        datset['Prob'] = probs[sim]
        ### Compute efficiencies, purities
        purity[sim], efficiency[sim], thresh[sim], P_eff[sim] = get_purity_scores(yset, probs[sim], Fix_eff)
        print sim,'purity len', len(purity[sim])

    #create true & predict dicts for storing ML results
    trueIa = (yset==0)
    trueCC = (yset>0)
    trueII = (yset==2)
    trueIbc= (yset==1)
    true[sim]={Ia:trueIa, CC:trueCC, Ibc:trueIbc, II:trueII}

    print 'Statistics for',sim,'Data Set'
    for t in MLtypes:
        print 'True number of {}  = {}'.format(t,np.sum(true[sim][t]))
    print

    predictIa = datset['Prob']>P_eff[sim]
    predictCC = ~predictIa
    predict={sim:{Ia:predictIa,CC:predictCC}}
    CLFstats={sim:{TP:{},FP:{}}}
    CLFstats[sim][TP]={Ia:(predict[sim][Ia] & true[sim][Ia]),CC:(predict[sim][CC] & true[sim][CC])}
    CLFstats[sim][FP]={Ia:(predict[sim][Ia] & ~true[sim][Ia]),CC:(predict[sim][CC] & ~true[sim][CC])}
    if (args.nclass==3):
        predict[sim][II]=(allprobs[sim][:,0]<P_eff[sim]) & (allprobs[sim][:,2]>allprobs[sim][:,1])
        predict[sim][Ibc]=(~predict[sim][Ia]) & (~predict[sim][II])
        CLFstats[sim][TP][II]= predict[sim][II] & true[sim][II]
        CLFstats[sim][TP][Ibc]= predict[sim][Ibc] & true[sim][Ibc]
        CLFstats[sim][FP][II]= predict[sim][II] & ~true[sim][II]
        CLFstats[sim][FP][Ibc]= predict[sim][Ibc] & ~true[sim][Ibc]
    elif(args.nclass==4):
        predict[sim][II]=(allprobsA[:,0]<P_eff[sim])
        predict[sim][Ibc]=(~predict[sim][Ia])
    #print summary
    for t in MLtypes:
        print 'Predicted number of {}-Data {} with P_thresh {:8.3f} = {}'.format(sim, t, float(P_eff[sim]), 
                   np.sum(predict[sim][t]))
        print 'Correct (true positive) number of {}-Data {} = {}'.format(sim,t,np.sum(CLFstats[sim][TP][t]))
        print 'Incorrect (false positive) number of {}-Data {} = {}'.format(sim,t,np.sum(CLFstats[sim][FP][t]))
    print

    #end ML-peff typing

    print "Predictions using maximum probability classification"
    if(args.nclass==4):
        maxprobclass=clf.predict(xsetb)
        other='Ibc'
    else:
        maxprobclass=clf.predict(xset)
        other='CC'
    print 'Predicted number of {}-Data Ia using max prob = {}'.format(sim, np.sum(maxprobclass==0))
    print 'Predicted number of {}-Data {} using max prob = {}'.format(sim, other, np.sum(maxprobclass>0))
    if (args.nclass>=3):
        print 'Predicted number of {}-Data Ibc using max prob = {}'.format(sim, np.sum(maxprobclass==1))
        print 'Predicted number of {}-Data II using max prob = {}'.format(sim, np.sum(maxprobclass==2))
    print
    Eff, Pur, Rem =EffPur(trueIa, maxprobclass==0)
    print "Purity and efficiency for {} data using max prob = {:.3f}, {:.3f}".format(sim, Pur, Eff)
    print

#end ML-typing

### Compute ROC curve, AUC, class probabilities and template statistics for Test Data
if(not(args.train_only)):
    sim=Test
    fpr, tpr, roc_thres, AUC = get_ROC(y_test_ROC, allprobs[sim][:,0])

    #Get probability variance and percentiles from trees
    percentile=68
    average,variance,err_up,err_down,pdata=prob_errors(clf, X_test,percentile=percentile,debug=False)


    #Measured Classification Probability for Ias (calibration curve computes this for us)
    Nbins_RFP = 20
    fraction_true, meanpr = calibration_curve(true[sim][Ia], probs[sim], n_bins=Nbins_RFP)
    print "Fraction of True Positives in",Nbins_RFP,"probabilty bins"
    print '  '.join(['{:.3f}'.format(f) for f in fraction_true])

    #TEMPLATE STATISTICS
    print "\nTemplate Statistics for {} Data".format(sim)
    templates, counts = np.unique(data_test['sim_nonIa'].quantity.value, return_counts=True)
    template_dict = dict(zip(templates, counts))

    #statistics for numbers of true and ML types for each template
    template_stats={}
    for tmplt in template_dict:
        template_stats[tmplt]={}
        for typ in MLtypes:
            template_stats[tmplt][CLFid+typ]=np.sum(predict[Test][typ] & (data_test['sim_nonIa']==tmplt))
            template_stats[tmplt]['True'+typ]=np.sum(true[Test][typ] & (data_test['sim_nonIa']==tmplt))
            if(template_stats[tmplt]['True'+typ]>0):
                template_stats[tmplt]['Type']=typ

    #count template occurrences for SN classified as Ia
    CLFIa_mask = (probs[sim] > P_eff[sim])
    CLFIa_templates, CLFIa_counts = np.unique(data_test['sim_nonIa'][CLFIa_mask].quantity.value, return_counts=True)
    Iatemplate_dict = dict(zip(CLFIa_templates, CLFIa_counts))

    print "\nType\t|\tValue" 
    #need to print in frequency order
    keys=Iatemplate_dict.keys()
    template_freq = sorted(Iatemplate_dict.values())
    template_freq.reverse()
    ordered_keys=[]
    for freq in template_freq:
    #index=freqs.index(freq)
        for key in keys:
            if Iatemplate_dict[key]==freq:
                if not(key in ordered_keys):
                    print key,'\t|\t' ,freq
                    ordered_keys.append(key)

    npop=5
    print npop,"most popular templates and frequencies for passing MLIa",ordered_keys[0:npop],template_freq[0:npop]

#sys.exit()

#CROSS_VALIDATION
# read in data
if(args.cv):
    if(args.sample=='t'):
        data_all = ttrain
    # read in validation data only
    elif(args.sample=='v'):
        data_all = ttest
    # combine both training and validation sets
    elif(args.sample=='b'):
        data_all = vstack([ttrain, ttest])
    #data_all = read_sample()

    snr_cut = data_all['snr1']>SNRcut   # no cut at the moment
    data = data_all[snr_cut]

    X_data = get_features(args.ft, data)
    # Get y_data (class labels)
    if(args.nclass==2):
        print '\n\t2-WAY CLASSIFICATION'
        y_data = data['type'] # class 1=Ia match, 0=CC match
    elif(args.nclass==3):
        print '\n\t3-WAY CLASSIFICATION'
        y_data = data['type3'] #3-way typing
    #y_ROC = data['type'] # for ROC curve, need binary labels

    cvclf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, \
      min_samples_split=min_samples_split, criterion=criterion, n_jobs=args.nc)

    print '\n\nNow try cross-validation methods ...'

    # Stratified k-fold cross-validation and compute scoring metric each time
    print '\n----- Stratified K-fold cross-validation -----\n'               

    kvals = []
    avgskf = []
    stdkf = []

    for k in range(2, 11):
        kf = StratifiedKFold(y_data, n_folds=k, shuffle=True, random_state=42) # define cross validator
        cv_scores_kf = cross_val_score(cvclf, X_data, y_data, scoring=score_func_est, cv=kf)
        print 'k={} folds CV scores : '.format(k), cv_scores_kf
        kvals.append(k)
        avgskf.append(np.mean(cv_scores_kf))
        stdkf.append(np.std(cv_scores_kf)/np.sqrt(float(len(cv_scores_kf))))

    # ShuffleSplit with n iterations
    print '\n\n----- ShuffleSplit iterations -----'

    test_step = 0.1
    tsvals = []
    avgss = []
    stdss = []

    for ts in np.arange(0.1, 1, test_step):
        print 'Fractional Test Size : ', ts
        ss = ShuffleSplit(len(y_data), n_iter=args.niter, test_size=ts, random_state=42) # BUG: don't use train_size
        for train_index, test_index in ss:
            train1as = y_data[train_index]==0
            test1as = y_data[test_index]==0
            print "TRAIN SNIa:", np.sum(train1as), "\tTEST SNIa:", np.sum(test1as)
        cv_scores_ss = cross_val_score(cvclf, X_data, y_data, scoring=score_func_est, cv=ss) #array of score values
        print '\nCV scores (ShuffleSplit) = ', cv_scores_ss
        #print 'Average Score = ', np.mean(cv_scores_ss)
        #print 'Score Standard Deviation = ', np.std(cv_scores_ss)
        tsvals.append(ts)
        avgss.append(np.mean(cv_scores_ss))
        stdss.append(np.std(cv_scores_ss)/np.sqrt(float(len(cv_scores_ss))))

#endif--args.cv

if(args.pc):
    pfilename='DES_validation_fitprob_purity='
    print "\nUsing purities",args.purities,"for test files"
    pscores=[]
    for purity in args.purities:
        ptestfile=pfilename+str(purity)+'.txt'
        ptest=read(ptestfile)
        #print "Size of purity =",purity,"test data:",len(ptest)
        cutptest=(ptest['snr1']>SNRcut)  
        ptest = ptest[cutptest]  
        print "Size of purity =",purity,"test data (after SNR cut):",len(ptest)
        X_ptest=get_features(args.ft, ptest)
        if(args.nclass==2):
            y_ptest = ptest['type']
        elif(args.nclass==3):
            y_ptest = ptest['type3']
        elif(args.nclass==4):
            y_ptest = ptest['type2x2']
        pprobs=clf.predict_proba(X_ptest)[:, 0] # SNeIa are class 0
        #print len(X_ptest),len(pprobs),len(y_ptest)
        pscore=score_func(pprobs, y_ptest)
        print "Score for purity",purity,"=",pscore
        pscores.append(pscore)

    #endfor
#endif--args.pc


if(len(args.plotdir)==0):
    print "Skipping plots and exiting"
    sys.exit()
else:
    print '\n********** STARTING PLOTS **********\n'

#setup pdf file name and other filenames for saving efficiencies etc
file_id='_{}'.format(args.filestr)
fnpurity=''
SNR=''
if ('purity' in args.test):
    fnpurity='purity'+re.split('purity',os.path.splitext(args.test)[0])[1]
if ('SNR' in args.test):
    SNR='SNR'+re.split('SNR',os.path.splitext(args.test)[0])[1]
if(len(datalist)>0):
    dname='_'+'+'.join(datalist)+'Data'
else:
    dname=''
pdfname = os.path.join(args.plotdir,'_'.join(['eff_pur_DES',str(args.nclass),'way_typing',str(nfeatures)+'features',SNR,fnpurity])+withpdf+dname+file_id+'.pdf')
multiPdf = PdfPages(pdfname)

#Setup "sim" data to plot

alldata={}
mask={}
#setup cuts according to type and save in alldata dict
for sim in simlist:  
    if (sim==Training):
        simdata=data_train
        simlabel=Training
    else:
        simdata=data_test
        simlabel=Test
        simdata['true_mu']=simdata['sim_mu']

    #augment simdata with other variables
    simdata[HR]=simdata['mu']-simdata['sim_mu'] #Hubble Residual

    #add to dict
    smaskIa=simdata['sim_nonIa']==0
    smaskCC=simdata['sim_nonIa']>0
    smaskIbc=simdata['type3']==1
    smaskII=simdata['type3']==2
    mask[sim]={Ia:smaskIa,CC:smaskCC,Ibc:smaskIbc,II:smaskII}
    alldata[sim]={Ia:simdata[mask[sim][Ia]],CC:simdata[mask[sim][CC]],Ibc:simdata[mask[sim][Ibc]],II:simdata[mask[sim][II]],Total:simdata}
    #plotlabels[Sim][simlabel]=plotlabels[Sim][simlabel] + ' Data'
    #add CLFtype, TP and FP to dict for samples != Training
    if not(sim==Training):
        for t in MLtypes:
            alldata[sim][CLFid+t]=simdata[predict[sim][t]]
            alldata[sim][CLFid+TP+t]=simdata[CLFstats[sim][TP][t]]
            alldata[sim][CLFid+FP+t]=simdata[CLFstats[sim][FP][t]]

        #add CLFTotal for consistency and weights; will not need totals for TP and FP types
        alldata[sim][CLFid+Total]=simdata


for key in sorted(alldata.keys()):
    print "\nPlotting",key,"(simulated) data:"
    for t in alldata[key].keys():
        print "Number of type {} = {}".format(t,len(alldata[key][t]))

#Setup "observed" data to plot
obsdata={}
P_eff_ref = P_eff[Test]
for data in MLdatalist:   #loop over data to plot; always include Test for plots requiring CLF classified data (eg HR plots)      
    if(data==Test):       #setup entries to fill dicts
        #obslabel='Test'
        obsdata=data_test
        obsdata['true_mu']=obsdata['sim_mu']
        obsIa=obsdata['sim_nonIa']==0
        obsCC=obsdata['sim_nonIa']>0
        obsIbc=obsdata['type3']==1
        obsII=obsdata['type3']==2
    elif(data==Spec):
        #obslabel='Spec.'
        obsdata=data_spec
        obsIa=obsdata['spec_eval']=='SNIa'
        obsCC=obsdata['spec_eval']!='SNIa'
        obsII=obsdata['spec_eval']=='SNII'
        obsIbc=(obsdata['spec_eval']!='SNIa') & (obsdata['spec_eval']!='SNII')
    elif(data==Spec_nofp):
        #obsdata=data_spec_nofp
        obslabel='Spec. No $f_p$ Cut '
        obsIa=obsdata['spec_eval']=='SNIa'
        obsCC=obsdata['spec_eval']!='SNIa'
        obsII=obsdata['spec_eval']=='SNII'
        obsIbc=(obsdata['spec_eval']!='SNIa') & (obsdata['spec_eval']!='SNII')
    elif(data==Phot): 
        #obslabel='Phot.'  
        obsdata=data_phot
        #special case of phot test data
        if('sim_mu' in obsdata.keys()):
            obsdata['true_mu']=obsdata['sim_mu']
        #unknown true types; all False masks
        obsIa=np.zeros(len(obsdata['snid']),dtype=bool)
        obsCC=np.zeros(len(obsdata['snid']),dtype=bool)
        obsIbc=np.zeros(len(obsdata['snid']),dtype=bool)
        obsII=np.zeros(len(obsdata['snid']),dtype=bool)

    #save mask
    mask[data]={Ia:obsIa,CC:obsCC,Ibc:obsIbc,II:obsII}
    omask={Ia:obsIa,CC:obsCC,Ibc:obsIbc,II:obsII}

    #save label
    #plotlabels[Data][data]=plotlabels[Data][data]+' Data'

    #ML-type labeled observed data
    if(not(data in predict.keys())):   #data_test already typed and predictions printed
        predict[data]={}
        X_data = get_features(args.ft, obsdata)
        dataprobs=clf.predict_proba(X_data)
        predict[data][Ia] = dataprobs[:, 0]>P_eff_ref
        predict[data][CC] = ~predict[data][Ia]
        #compute true mu
        cosmo = FlatLambdaCDM(H0=H0, Om0=OmegaM)
        obsdata['true_mu']=cosmo.distmod(obsdata['z'])
        if (args.nclass==3):
            predict[data][II]=(dataprobs[:,0]<P_eff_ref) & (dataprobs[:,2]>dataprobs[:,1])
            predict[data][Ibc]=(~predict[data][Ia]) & (~predict[data][II])
        #print summary
        print
        for t in MLtypes:
            print 'Predicted number of {} Data {} with P_thresh {:0.3f} = {}'.format(data, 
                       t, float(P_eff_ref), np.sum(predict[data][t]))
                                       
        #fill and print some more stats
        CLFstats[data]={}
        if((np.sum(omask[Ia]) + np.sum(omask[CC])) > 0):
            CLFstats[data][TP]={Ia:(predict[data][Ia] & omask[Ia]),CC:(predict[data][CC] & omask[CC])}
            CLFstats[data][FP]={Ia:(predict[data][Ia] & ~omask[Ia]),CC:(predict[data][CC] & ~omask[CC])}
            if (args.nclass==3):
                CLFstats[data][TP][II]= predict[data][II] & omask[II]
                CLFstats[data][TP][Ibc]= predict[data][Ibc] & omask[Ibc]
                CLFstats[data][FP][II]= predict[data][II] & ~omask[II]
                CLFstats[data][FP][Ibc]= predict[data][Ibc] & ~omask[Ibc]            
            for t in MLtypes:
                print 'Correct (true positive) number of {} Data {} = {}'.format(data,t, np.sum(CLFstats[data][TP][t]))
                print 'Incorrect (false positive) number of {} Data {} = {}'.format(data,t,np.sum(CLFstats[data][FP][t]))
        else:
            print "{} data is unlabeled: true types not available".format(data)
    #augment variables in data here
    if ('true_mu' in obsdata.keys()):
        obsdata[HR]=obsdata['mu']-obsdata['true_mu']         #Hubble Residual

    #now fill dict if not already included
    if(not(data in alldata.keys())):
        alldata[data]={Ia:obsdata[obsIa],CC:obsdata[obsCC],Ibc:obsdata[obsIbc],II:obsdata[obsII],Total:obsdata}
        #add CLFtype, TP and FP types to data dict
        for t in MLtypes:
            alldata[data][CLFid+t]=obsdata[predict[data][t]]
            if(TP in CLFstats[data].keys()):
                alldata[data][CLFid+TP+t]=obsdata[CLFstats[data][TP][t]]
                alldata[data][CLFid+FP+t]=obsdata[CLFstats[data][FP][t]]

        #add CLFTotal for consistency and weights; no totals for TP and FP types
        alldata[data][CLFid+Total]=obsdata

    #summarize extra data to be plotted
    if(data in datalist):
        print "\nPlotting",plotlabels[Data][data],'as "observed" data:'
        for t in alldata[data].keys():
            print "Number of type {} = {}".format(t,len(alldata[data][t])) 

#endfor-datalist

#special printout here
if(data==Spec):
    print alldata[Spec][RFFPIa]['snid']
    print alldata[Spec][RFFPIa]['spec_eval']
    print alldata[Spec][RFFPIa]['fit_pr']
    print alldata[Spec][RFFPIa]['x1']
    print alldata[Spec][RFFPIa]['c']
    print alldata[Spec][RFFPIa]['z']
    print alldata[Spec][RFFPIa]['mB']

#compute normalizations and weights for data plots (scale sims/secondary data by primary observed data set)
weights={}
if(len(datalist)>0):
    data0=datalist[0]        #primary observed dataset in datalist
    #print alldata[data0].keys()
    #lengths=[len(alldata[data0][key]) for key in alldata[data0].keys()]
    #print lengths

for key in alldata.keys():   #compute renormalization factors
    print '\nNormalization factors for {} Data:\n'.format(key)
    weights={}
    if(args.weights==ByType): #accumulate weights by type
        norm={}
        #for t in MLtypes + CLFtypes:
        for t in MLtypes + CLFtypes + allTPFPtypes:
            if(alldata[key].has_key(t) and len(alldata[key][t])>0): #t exists and has data
                if (len(datalist)>0):
                    if(alldata[data0].has_key(t) and len(alldata[data0][t])>0): #t exists and has data
                        norm[t]=float(len(alldata[data0][t]))/float(len(alldata[key][t]))
                    else:
                        print "Type {} not available for {} Data".format(t,data0)
                        #for Phot data, only CLF weights available;
                        btyp=t[t.find('I'):len(t)] if 'I' in t else CC #parse t to find base SN type
                        subt=CLFid+btyp
                        print "Computing weights using type {} instead".format(subt)
                        if(alldata[key].has_key(subt) and len(alldata[key][subt])>0):  #Training data doesn't have CLF types
                            norm[t]=float(len(alldata[data0][subt]))/float(len(alldata[key][subt]))
                        elif(alldata[key].has_key(btyp)):
                            print "Type {} not available for {} Data".format(subt,key)
                            norm[t]=float(len(alldata[data0][subt]))/float(len(alldata[key][btyp]))
                            print "Computing weights using type {} instead".format(btyp)
                        else:
                            "No appropriate substitute weights for {} in {} and {}".format(t, key, data0) 
                else:
                    norm[t]=1.0
                weights[t]=np.array([norm[t]]*len(alldata[key][t]))
                print "Setting up {} {} weights for type {}".format(len(alldata[key][t]), args.weights, t)
            else:
                norm[t]=0.0
                print "No data: Skipping {} weights for type {}".format(args.weights, t) 
            print 'n({}) ={:0.3f}\n'.format(t, norm[t])
                
        #Total weights need different values for each type
        #Check cases for which type does not exist in data (no Ia etc. in Phot data)
        #print norm.keys()
        if (key!=Phot):
            weights[Total]=np.array([norm[Ia]]*len(alldata[key][Total]))
        if(key!=Training):
            weights[CLFid+Total]=np.array([norm[CLFid+Ia]]*len(alldata[key][Total]))
        if (key!=Phot) and (key!=Training):
            #if (norm.has_key(CLFid+TP+Ia)):
            weights[CLFid+TP+Total]=np.array([norm[CLFid+TP+Ia]]*len(alldata[key][Total]))
            #else:
            #    weights[CLFid+TP+Total]=np.array([0.]*len(alldata[key][Total]))
            #if (norm.has_key(CLFid+FP+Ia)):
            weights[CLFid+FP+Total]=np.array([norm[CLFid+FP+Ia]]*len(alldata[key][Total]))
            #else:
            #    weights[CLFid+FP+Total]=np.array([0.]*len(alldata[key][Total]))
        if (args.nclass==3):                #overwrite array to match MLtypes
            if (key!=Phot):
                weights[Total][mask[key][Ibc]]=norm[Ibc]
                weights[Total][mask[key][II]]=norm[II]
            if(key!=Training):
                weights[CLFid+Total][predict[key][Ibc]]=norm[CLFid+Ibc]
                weights[CLFid+Total][predict[key][II]]=norm[CLFid+II]
            if (key!=Phot) and (key!=Training):
                weights[CLFid+TP+Total][CLFstats[key][TP][Ibc]]=norm[CLFid+TP+Ibc]
                weights[CLFid+FP+Total][CLFstats[key][FP][Ibc]]=norm[CLFid+FP+Ibc]
                weights[CLFid+TP+Total][CLFstats[key][TP][II]]=norm[CLFid+TP+II]
                weights[CLFid+FP+Total][CLFstats[key][FP][II]]=norm[CLFid+FP+II]
        else:
            if (key!=Phot):
                weights[Total][mask[key][CC]]=norm[CC]
            if(key!=Training):
                weights[CLFid+Total][predict[key][CC]]=norm[CLFid+CC]
            if (key!=Phot) and (key!=Training):
                weights[CLFid+TP+Total][CLFstats[key][TP][CC]]=norm[CLFid+TP+CC]
                weights[CLFid+FP+Total][CLFstats[key][FP][CC]]=norm[CLFid+FP+CC]
    else:
        #constant weights (calculate for all keys even if not needed for Training etc. data)
        if (len(datalist)>0):
            weights[Total]=np.array([float(len(alldata[data0]))/float(len(alldata[key]))]*len(alldata[key][Total]))
        else:
            weights[Total]=np.array([1.0]*len(alldata[key][Total]))
        weights[CLFid+Total]=weights[Total]
        weights[CLFid+TP+Total]=weights[Total]
        weights[CLFid+FP+Total]=weights[Total]

    #save in dict
    alldata[key][Weights]=weights
    #print key,alldata[key][Weights].keys()

#old format
#Iaweights=np.array([normIa]*len(sim[Ia]))
#CCweights=np.array([normCC]*len(sim[CC]))
#Ibcweights=np.array([normIbc]*len(sim[Ibc]))
#IIweights=np.array([normII]*len(sim[II]))

#save SN Totals in dict for cuts
SNTotals={}
for key in alldata.keys():
    SNTotals[key]={}
    for t in plottypes:
        if(t in alldata[key].keys()):
            SNTotals[key][t]=float(len(alldata[key][t]))
            #print key, t, SNTotals[key][t]

print '\nDefault plottypes=',str(plottypes)

# PLOT VARIABLES ###################

npages=1


###################################################################
#page probs
print "\nStarting page",npages,"(page probs)"
#populate dict of probabilities for Test Data using array from probs dict
CLFprobs={}
data = Test
for t in MLtypes:
    CLFprobs[t]={'probs':probs[data][true[data][t]]}

fig = plt.figure(figsize=(15,7))

p_binwidth = 0.05
p_bins = np.arange(-0.1, 1.05, p_binwidth)

f = fig.add_subplot(231)
plot_types(MLtypes,'probs',CLFprobs,xlabel='Random Forest SNIa Probability',ylabel='Number',
           plotdict=plotdict,bins=p_bins,asize=Small,title=plotlabels[Sim][data])
f.set_xlim(0,1.0)
f.legend(loc='upper center', fontsize='small',numpoints=1)

f = fig.add_subplot(232)
plot_types(MLtypes,'probs',CLFprobs,xlabel='Random Forest SNIa Probability',ylabel='Number',
           plotdict=plotdict,bins=p_bins,asize=Small,yscale=log,title=plotlabels[Sim][data])
f.set_xlim(0,1.0)
#plt.yscale('log', nonposy='clip')
f.legend(loc='upper center', fontsize='small',numpoints=1)

f = fig.add_subplot(233)
f.plot(thresh[data], efficiency[data][:-1], 'b--', label='efficiency', lw=2)
f.plot(thresh[data], purity[data][:-1], c='r', label='purity', lw=2)
f.set_xlabel('Threshold Probability for Classification, $P_{t}$')
f.set_ylabel('SNIa Efficiency, Purity [$P_{Ia} \ge P_{t}$]')
f.legend(loc='lower right', fontsize='small')
f.set_ylim(0.85, 1.0)
f.set_xlim(0, 1.0)

ax1 = fig.add_subplot(234)
ax1.plot(fpr, tpr, lw=2, label='ROC curve (area = {:0.2f})'.format(AUC))
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(loc='lower right', fontsize='small')

f = fig.add_subplot(235)
f.plot(meanpr, fraction_true, label='Random Forest',marker='o',color='magenta')
f.plot(meanpr, meanpr, label='Perfectly Calibrated',linestyle='--',color='black')
plt.title(plotlabels[Sim][data])
f.set_xlabel('Random Forest SNIa Probability')
f.set_ylabel('Fraction of True SNIa')
f.legend(loc='best', fontsize='small',numpoints=1)

#get efficiency and purity for chosen data set
C_Dcut = np.arange(0.0, 1.0, 0.005)
C_Eff,C_Pur=get_EffPur(simdata,C_Dcut)
f = fig.add_subplot(236)
f.plot(C_Dcut,C_Eff, 'b--', label='efficiency', lw=2)
f.plot(C_Dcut,C_Pur, c='r', label='purity', lw=2)
f.set_xlabel('SALT fit probability')
f.set_ylabel('Efficiency, Purity [$P_{corr} \ge P_{t}$]')
f.legend(loc='best', fontsize='small',numpoints=1)
f.set_ylim(0, 1.0)
f.set_xlim(0, 1.0)

fig.tight_layout()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    multiPdf.savefig(fig)

npages+=1
#################   Page 2 Plots    #################
#page x1 c 
print "\nStarting page",npages,"(page  x1 c)"
#Select number of bins
Nbins=20
Nsbins=40
c_min=-0.6
c_max=0.6
x1_min=-5.0
x1_max=5.0
dx1=0.25 #extra xwidth for scatter plot
dcp=0.05 #extra y+width for scatter plot
dcm=0.6  #extra y-width for scatter plot
x1binwidth=(x1_max-x1_min)/Nsbins
cbinwidth=(c_max-c_min)/Nsbins
x1bins=np.linspace(x1_min,x1_max,Nbins+1)
cbins=np.linspace(c_min,c_max,Nbins+1)
#page 2 types and lists
p2list=plotlist
p2types=plottypes
#get title from simulated data list
title=' '.join([plotlabels[Sim][lbl] for lbl in p2list[0]])
fig = plt.figure(figsize=(15,7))

f = fig.add_subplot(231)
plot_types(p2types,'x1',alldata,plotlist=p2list,xlabel='SALT $x_{1}$',ylabel='Number',plotdict=plotdict,bins=x1bins,title=title)
f.set_xlim(x1_min,x1_max)
f.legend(loc='upper center', fontsize='small',numpoints=1) 

f = fig.add_subplot(232)
plot_types(p2types,'c',alldata,plotlist=p2list,xlabel='SALT $c$',ylabel='Number',plotdict=plotdict,bins=cbins,title=title)
f.set_xlim(c_min,c_max)
axes = plt.gca()
ymin, ymax = axes.get_ylim()
axes.set_ylim(ymax=ymax*yrescale)
f.legend(loc='best', fontsize='small',numpoints=1)

f = fig.add_subplot(233)
plotdict['labels'][contour]='SNIa Density'
plot_types(p2types,'x1',alldata,plotlist=[Training],yvar='c',xlabel='SALT $x_1$',ylabel='SALT $c$',plotdict=plotdict,ctrxbin=x1binwidth,ctrybin=cbinwidth,title=plotlabels[Sim][Training],weights=False)
f.set_xlim(x1_min-dx1,x1_max+dx1)
f.set_ylim(c_min-dcm/2.,c_max+dcp)
f.legend(loc='lower left', fontsize='small',scatterpoints=1,ncol=Ncols)
#ell_2a = 8.6 # x1
#ell_2b = 0.76 # color
#ellipse = Ellipse(xy=(0,0), width=ell_2a, height=ell_2b, edgecolor='red', fc='None', lw=2, ls='dashed')
#f.add_patch(ellipse)

f = fig.add_subplot(234)
plot_types(p2types,'fit_pr',alldata,plotlist=p2list,xlabel='$f_p$',ylabel='Number',plotdict=plotdict,bins=p_bins,title=title)
f.set_xlim(0,1.0)
f.legend(loc='best', fontsize='small',numpoints=1)

#x_1-c for MLtyped samples - Test + any other - will overflow to page 2a if needed
if(len(MLdatalist)>0):
    nplot=4
    for l in MLdatalist:
        fig,nplot=get_nplot(fig,nplot,pagenum=2)
        f = fig.add_subplot(230+nplot)
        plotdict['labels'][contour]='RF SNIa Density'
        #plot_types(CLFplottypes,'x1',alldata,plotlist=[l],yvar='c',xlabel='SALT $x_1$',ylabel='SALT $c$',plotdict=plotdict,ctrxbin=x1binwidth,ctrybin=cbinwidth,title=plotlabels[Data][l],weights=False)
        plot_types(CLFplottypes,'x1',alldata,plotlist=[l],yvar='c',xlabel='SALT $x_1$',ylabel='SALT $c$',plotdict=plotdict,title=plotlabels[Data][l],weights=False)   #suppress contours by omitting ctrxbin,ctrybin
        f.set_xlim(x1_min-dx1,x1_max+dx1)
        f.set_ylim(c_min-dcm/2.,c_max+dcp)
        f.legend(loc='lower left', fontsize='small',scatterpoints=1,ncol=Ncols)

        if(l in lbldMLdatalist):
            fig,nplot=get_nplot(fig,nplot,pagenum=2)
            f = fig.add_subplot(230+nplot)
            plotdict['labels'][contour]='TP SNIa Density'
            #plot_types(allTPFPtypes,'x1',alldata,plotlist=[l],yvar='c',xlabel='SALT $x_1$',ylabel='SALT $c$',plotdict=plotdict,ctrxbin=x1binwidth,ctrybin=cbinwidth,title=plotlabels[Data][l],weights=False)
            plot_types(allTPFPtypes,'x1',alldata,plotlist=[l],yvar='c',xlabel='SALT $x_1$',ylabel='SALT $c$',plotdict=plotdict,title=plotlabels[Data][l],weights=False)
            f.set_xlim(x1_min-dx1,x1_max+dx1)
            f.set_ylim(c_min-dcm,c_max+dcp)
            f.legend(loc='lower left', fontsize='small',scatterpoints=1,ncol=Ncols)


fig.tight_layout()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    multiPdf.savefig(fig)

npages+=1
##############  Page 3 Plots   ################
#page z HR HD
print "\nStarting page",npages,"(page z HR HD)"

p3types=trueplottypes
#p3list=CLFplotlist  #([Test],[Selected Data])
p3CLFtypes=[CLFid+t for t in p3types]
p3CLFtypes=[RFTPIa, RFFPIa]
Nbins=15
HRbins=bins=np.arange(-3,3,0.2)
title=' '.join([plotlabels[Sim][lbl] for lbl in CLFplotlist[0]])

fig = plt.figure(figsize=(15,7))

f = fig.add_subplot(231)
#ensure uniform binning regardless of range in data
zbins = np.linspace(zlo,zhi,Nzbins+1)
plot_types(p3CLFtypes,'z',alldata,plotlist=CLFplotlist,xlabel='Redshift',ylabel='Number',plotdict=plotdict,bins=zbins,selectdata=CLFid+Ia,title=title)
f.set_xlim(zlo,zhi)
f.legend(loc='best', fontsize='small',numpoints=1)


f = fig.add_subplot(232)
plot_types(p3CLFtypes,HR,alldata,plotlist=CLFplotlist,xlabel='Hubble Residual (RF typing)',ylabel='Number',plotdict=plotdict,bins=HRbins,title=title,selectdata=CLFid+Ia)
plt.gca().set_ylim(bottom=1)
f.legend(loc='upper left', fontsize='small',numpoints=1)

f = fig.add_subplot(233)       #HR plot for predicted type
plot_types(p3CLFtypes,HR,alldata,plotlist=CLFplotlist,xlabel='Hubble Residual (RF typing)',ylabel='Number',plotdict=plotdict,bins=HRbins,yscale=log,title=title,selectdata=CLFid+Ia)
plt.gca().set_ylim(bottom=0.1)
f.legend(loc='upper left', fontsize='small',numpoints=1)


#Side-by-side Hubble Diagrams for Data and Simulation using ML typed classifications
nplot=3
rCLFtypes= CLFtypes[::-1] #reverse order so Ia points are on top
rMLtypes = MLtypes[::-1]
#loop through plots for data ans sim
#plot MLtypes for realdata only => plotlist=[[],realdatalist[0]] (skip sim data)
#plot TP and FP for Test dat only => plotlist=[[Test],[]]  (skip real data)
#set up lists to be zipped together
typelist=[rMLtypes,[RFTPIa,RFFPIa]]
plotlst=[[[],realdatalist[0]],[[Test],[]]] #realdatalist[0]=Phot., Spec. etc
titles=[plotlabels[Data][realdatalist[0]],plotlabels[Sim][Test]]
for types,plotls,title in zip(typelist,plotlst,titles):
    nplot+=1
    f = fig.add_subplot(230+nplot)
    plot_types(types,'z',alldata,plotlist=plotls,yvar='mu',xlabel='Redshift',ylabel='$\mu$',bins=zbins,plotdict=plotdict,weights=False,selectdata=CLFid+Ia,title=title)
    f.legend(loc='lower right', fontsize='small',numpoints=1,scatterpoints=1,ncol=1)
    f.set_xlim(zlo,zhi)
    f.set_ylim(mulo,muhi)

f = fig.add_subplot(236)       #HR plot for predicted type
CLFplotlist=[[Test],datalist]  #plot Test data as sim, and any other data
oldalpha = plotdict['alpha'][RFIa]
plotdict['alpha'][RFIa] = 1.0
oldfill = plotdict['fill'][RFIa]
plotdict['fill'][RFIa] = 'step'
oldcolor = plotdict['color'][RFIa]
plotdict['color'][RFIa] = 'black'
plot_types([RFIa],HR,alldata,plotlist=CLFplotlist,xlabel='Hubble Residual (RF typing)',ylabel='Number',plotdict=plotdict,bins=HRbins,yscale=log,title=title,selectdata=CLFid+Ia)
plt.gca().set_ylim(bottom=0.1)
f.legend(loc='upper left', fontsize='small',numpoints=1)
plotdict['alpha'][RFIa] = oldalpha
plotdict['fill'][RFIa] = oldfill
plotdict['color'][RFIa] = oldcolor


fig.tight_layout()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    multiPdf.savefig(fig)

#sys.exit()

############## Old Page 3 Plots (Training Data and True/CLF Types)  ################
#page z HR HD
print "\nStarting old page",npages,"(page z HR HD)"

p3types=trueplottypes
p3list=plotlist
p3CLFtypes=[CLFid+t for t in p3types]
Nbins=15
HRbins=bins=np.arange(-4,4,0.2)
CLFplotlist=[[Test],datalist]  #plot Test data as sim, and any other data
title=' '.join([plotlabels[Sim][lbl] for lbl in p3list[0]])
CLFtitle=' '.join([plotlabels[Sim][lbl] for lbl in CLFplotlist[0]])

fig = plt.figure(figsize=(15,7))

f = fig.add_subplot(231)
#ensure uniform binning regardless of range in data
zbins = np.linspace(zlo,zhi,Nzbins+1)
plot_types(p3types,'z',alldata,plotlist=p3list,xlabel='Redshift',ylabel='Number',plotdict=plotdict,bins=zbins,title=title)
f.set_xlim(zlo,zhi)
f.legend(loc='best', fontsize='small',numpoints=1)

f = fig.add_subplot(232)
plot_types(p3types,HR,alldata,plotlist=p3list,xlabel='Hubble Residual',ylabel='Number',plotdict=plotdict,bins=HRbins,yscale=log,title=title)
f.legend(loc='upper left', fontsize='small',numpoints=1)

f = fig.add_subplot(233)       #HR plot for predicted type
plot_types(p3CLFtypes,HR,alldata,plotlist=CLFplotlist,xlabel='Hubble Residual (RF typing)',ylabel='Number',plotdict=plotdict,bins=HRbins,yscale=log,title=CLFtitle)
f.legend(loc='upper left', fontsize='small',numpoints=1)


#Hubble Diagrams for Training
nplot=3
dictkeys=[Sim,Data]
samplelist=[Training,Test]
rCLFtypes= CLFtypes[::-1]
rMLtypes = MLtypes[::-1]
typelist=[rMLtypes,[RFTPIa,RFFPIa]]
ncolumns=[1,2]
for sample,types,key,nc in zip(samplelist,typelist,dictkeys,ncolumns):
    nplot+=1
    f = fig.add_subplot(230+nplot)
    plot_types(types,'z',alldata,plotlist=[sample],yvar='mu',xlabel='Redshift',ylabel='$\mu$',plotdict=plotdict,bins=zbins,title=plotlabels[key][sample],weights=False)
    f.legend(loc='lower right', fontsize='small',numpoints=1,scatterpoints=1,ncol=nc)
    f.set_xlim(zlo,zhi)
    f.set_ylim(mulo,muhi)

f = fig.add_subplot(236)       #HR plot for predicted type
CLFplotlist=[[Test],datalist]  #plot Test data as sim, and any other data
#temporarily change fill and colors
oldalpha = plotdict['alpha'][RFIa]
plotdict['alpha'][RFIa] = 1.0
oldfill = plotdict['fill'][RFIa]
plotdict['fill'][RFIa] = 'step'
oldcolor = plotdict['color'][RFIa]
plotdict['color'][RFIa] = 'black'
plot_types([Ia,RFIa],HR,alldata,plotlist=CLFplotlist,xlabel='Hubble Residual (RF typing)',ylabel='Number',plotdict=plotdict,bins=HRbins,yscale=log,title=CLFtitle)
f.legend(loc='upper left', fontsize='small',numpoints=1)
plotdict['alpha'][RFIa] = oldalpha
plotdict['fill'][RFIa] = oldfill
plotdict['color'][RFIa] = oldcolor

fig.tight_layout()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    multiPdf.savefig(fig)


##########page 3a HD overflow #####################

if(len(realdatalist)>0):
    print "\nStarting page",npages,"a overflow (HD for data)"

    nplot=0
    fig = plt.figure(figsize=(15,7))

    for real in realdatalist:
        if real in lbldMLdatalist:
            typelist=[rMLtypes,[RFTPIa,RFFPIa]]
            ncolumns=[1,2]
        else:
            typelist=[rCLFtypes]  #plot only predicted types
            ncolumns=[1]
        for types,nc in zip(typelist,ncolumns):
            nplot+=1
            f = fig.add_subplot(230+nplot)
            plot_types(types,'z',alldata,plotlist=[real],yvar='mu',xlabel='Redshift',ylabel='$\mu$',plotdict=plotdict,title=plotlabels[Data][real],weights=False)
            f.legend(loc='lower right', fontsize='small',numpoints=1,ncol=nc)
            f.set_xlim(zlo,zhi)

    fig.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multiPdf.savefig(fig)

npages+=1

#define some constants for cuts in following plots
ne999='!=-999'
ee999='==-999'
Pass=ne999
Fail=ee999

if(args.xplots):   #check for extra plots
    #################   Page 4 Plots    ############
    #page t0_err +
    print "\nStarting page",npages,"(page t0 x1 c err)"
    p4types=plottypes
    p4list=plotlist
    Nebins=20
    x1_err_max=4.0
    c_err_max=0.3
    t0_err_max=10.0
    x1ebins=np.linspace(0.,x1_err_max,Nbins+1)
    t0ebins=np.linspace(0.,t0_err_max,Nbins+1)
    cebins=np.linspace(0.,c_err_max,Nbins+1)
    title=' '.join([plotlabels[Sim][lbl] for lbl in p4list[0]])

    fig = plt.figure(figsize=(15,7))


    f = fig.add_subplot(231)
    plot_types(p4types,'t0_err',alldata,plotlist=p4list,xlabel='$\Delta t_0$',ylabel='Number',plotdict=plotdict,bins=t0ebins,title=title)
    f.legend(loc='best', fontsize='small',numpoints=1)

    f = fig.add_subplot(232)
    plot_types(p4types,'x1_err',alldata,plotlist=p4list,xlabel='$\Delta x_1$',ylabel='Number',plotdict=plotdict,bins=x1ebins,title=title)
    f.legend(loc='best', fontsize='small',numpoints=1)

    f = fig.add_subplot(233)
    plot_types(p4types,'c_err',alldata,plotlist=p4list,xlabel='$\Delta c$',ylabel='Number',plotdict=plotdict,bins=cebins,title=title)
    f.legend(loc='best', fontsize='small',numpoints=1)

    fig.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multiPdf.savefig(fig)

    npages+=1
    #################   Page 5 Plots    ############
    #page 5 t0_err scatter
    print "\nStarting page",npages,"(page t0 err x1err cerr scatter)"
    #always plot for Test data + overflow page for other data 
    if(len(MLdatalist)>0):
        fig = plt.figure(figsize=(15,7))
        x1_err_max=2.0
        c_err_max=0.18
        t0_err_max=4.0
        margin=0.6
        marginu=.05
        dt=0.1
        x1_errbinwidth=x1_err_max/Nsbins
        c_errbinwidth=c_err_max/Nsbins
        t0_errbinwidth=t0_err_max/Nsbins
        fit_pr_max=1.0
        fit_prbinwidth=fit_pr_max/Nsbins

        yvars=['c_err','x1_err','fit_pr','x1','c']
        ylabels=['SALT $\Delta c$','SALT $\Delta x_1$','SALT $f_p$','SALT $x_1$','SALT $c$']
        binwidy=[c_errbinwidth,x1_errbinwidth,fit_prbinwidth,x1binwidth,cbinwidth]
        ymin=[0.,0.,0.,x1_min,c_min]
        ymax=[c_err_max,x1_err_max,fit_pr_max,x1_max,c_max]
        nplot=0
        plotdict['labels'][contour]='TP SNIa Density'

        for l in MLdatalist:   #make plots for Test + any other "data"; will overflow to page 5a,b if needed
        #t0_err vs yvars
        #titlel=plotlabels[Data][l]
            for yvar,dy,lbl,ymn,ymx in zip(yvars,binwidy,ylabels,ymin,ymax):
                fig,nplot=get_nplot(fig,nplot,pagenum=5)
                f = fig.add_subplot(230+nplot)
                if (l in lbldMLdatalist):
                    p5types=allTPFPtypes
                else:
                    p5types=CLFtypes
                plot_types(p5types,'t0_err',alldata,plotlist=[l],yvar=yvar,xlabel='SALT $\Delta t_0$',ylabel=lbl,plotdict=plotdict,ctrxbin=t0_errbinwidth,ctrybin=dy,title=plotlabels[Data][l],weights=False)
                f.set_xlim(-dt,t0_err_max+dt)
                f.set_ylim(ymn-margin*(ymx-ymn),ymx+marginu*(ymx-ymn))
                f.legend(loc='lower left', fontsize='small',scatterpoints=1,ncol=Ncols)

            #x1_err vs c_err
            fig,nplot=get_nplot(fig,nplot,pagenum=5)
            f = fig.add_subplot(230+nplot)
            plot_types(p5types,'x1_err',alldata,plotlist=[l],yvar='c_err',xlabel='SALT $\Delta x_1$',ylabel='SALT $\Delta c$',plotdict=plotdict,ctrxbin=x1_errbinwidth,ctrybin=c_errbinwidth,title=plotlabels[Data][l],weights=False)
            f.set_xlim(0,x1_err_max)
            f.set_ylim(0-margin*c_err_max,c_err_max)
            f.legend(loc='lower left', fontsize='small',scatterpoints=1,ncol=Ncols)

        fig.tight_layout()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            multiPdf.savefig(fig)

    #endif-MLdata)>0

    npages+=1
    ##############  Page 6 Plots   ################
    #page SALT colors                                                                       
    print "\nStarting page",npages,"(page SALT multi- and single-bandcolors)"

    #select good data for SALT colors
    SALTcolormask=get_mask(plottypes,alldata,SALTcolors,SALTcolorlabels,SNTotals,mask_id='SALT color',file_id=file_id,good=True)

    #page 6
    fig = plt.figure(figsize=(15,7))
    #not using, but leave in code for now
    binlo=[-7.,-2.,-1.,-2.,-2.,-2.]
    binhi=[7.,4.,1.,2.,2.,2.]
    nplot=0
    NCbins=40
    for col,lo,hi,label in zip(SALTcolors,binlo,binhi,SALTcolorlabels):
        ylabel='$N$'
        xlabel=SALT+ label
        nplot +=1
        f = fig.add_subplot(230+nplot)
        cbins=np.linspace(lo,hi,NCbins+1)
        key_pass=col+ne999
        pltcuts=SALTcolormask[key_pass]
        plot_types(plottypes,col,alldata,plotlist=plotlist,xlabel=xlabel,ylabel=ylabel,cuts=pltcuts,plotdict=plotdict,plotid=SALT+col,bins=cbins)
        #f.set_xlim(-5.0, 5.0)                                                                                        
        f.legend(loc='best', fontsize='small',numpoints=1)

    fig.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multiPdf.savefig(fig)

    npages+=1
    ##############  Page 7 Plots   ################
    #page SALT color diffs                                                                  
    print "\nStarting page",npages,"(page SALT color diffs)"
    #page 7
    # select good data for SALT color differences
    SALTcolordiffmask=get_mask(plottypes,alldata,SALTcolordiffs,SALTcolordifflabels,SNTotals,mask_id='SALT color difference',file_id=file_id,good=True)

    binlo=[-4.,-2.,-1.5]
    binhi=[4.,3.,2.]
    fig = plt.figure(figsize=(15,7))
    nplot=0
    for col,lo,hi,label in zip(SALTcolordiffs,binlo,binhi,SALTcolordifflabels):
        ylabel='$N$'
        xlabel=SALT+label
        nplot +=1
        f = fig.add_subplot(230+nplot)
        cbins=np.linspace(lo,hi,NCbins+1)
        key_pass=col+ne999
        pltcuts=SALTcolordiffmask[key_pass]
        plot_types(plottypes,col,alldata,plotlist=plotlist,xlabel=xlabel,ylabel=ylabel,cuts=pltcuts,plotdict=plotdict,plotid=SALT+col+'diff',bins=cbins)
        #f.set_xlim(-5.0, 5.0)                                                                                        
        f.legend(loc='best', fontsize='small',numpoints=1)

    nplot=3
    #scatter plots for color diffs; plot 3 combinations (gr vs ri, gr vs iz,ri vs iz)
    dx=0.2
    dy=0.2
    jointcolormask={}
    print "\nStarting scatter plots for SALT colors\n"
    print 'Number passing joint color cuts'
    #combos are colors in colordiffs , ie gr vs ri, gr vz iz, ri vs iz
    for combos in [(0,1),(0,2),(1,2)]:
        xvar=SALTcolordiffs[combos[0]]
        yvar=SALTcolordiffs[combos[1]]
        xlabel=SALTcolordifflabels[combos[0]]
        ylabel=SALTcolordifflabels[combos[1]]
        nplot+=1
        f = fig.add_subplot(230+nplot)
        xmin=binlo[combos[0]]
        xmax=binhi[combos[0]]
        ymin=binlo[combos[1]]
        ymax=binhi[combos[1]]
        xykey=xvar+'+'+yvar
        jointcolormask[xykey]={}
        #determine joint mask for data passing both color cuts
        rMLtypes = MLtypes[::-1]
        for dkey in alldata.keys():
            print '{} Sample'.format(dkey)
            jointcolormask[xykey][dkey]={}
            print '  Type      Colors     Number'
            for t in plottypes+CLFplottypes:
                if(t in SALTcolordiffmask[xvar+ne999][dkey].keys() and t in SALTcolordiffmask[yvar+ne999][dkey].keys()):
                    jointcolormask[xykey][dkey][t]=SALTcolordiffmask[xvar+ne999][dkey][t] & SALTcolordiffmask[yvar+ne999][dkey][t]
                    print '   {:4} {} {}'.format(t,xykey,np.sum(jointcolormask[xykey][dkey][t])) 

        pltcuts=jointcolormask[xykey]
        plotdict['labels'][contour]='SNIa Density'
        plot_types(rMLtypes,xvar,alldata,plotlist=[Training],yvar=yvar,xlabel=xlabel,ylabel=ylabel,cuts=pltcuts,plotdict=plotdict,ctrxbin=dx,ctrybin=dy,title=plotlabels[Sim][Training],weights=False)
        f.set_xlim(xmin-dx,xmax+dx)
        f.set_ylim(ymin-dy,ymax+dy)
        f.legend(loc='best', fontsize='small',scatterpoints=1,ncol=Ncols)

    fig.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multiPdf.savefig(fig)

    npages+=1
    ##############  Page 8
    print "\nStarting page",npages,"(page SALT color diff vs z)"
    #redshift distributions of pass/fail color-diff cuts
    fig = plt.figure(figsize=(15,7))
    nplot=0
    for col,lo,hi,label in zip(SALTcolordiffs,binlo,binhi,SALTcolordifflabels):
        ylabel='$N$'
        xlabel='Redshift'
        nplot +=1
        f = fig.add_subplot(230+nplot)
        zbins=np.linspace(zlo,zhi,Nzbins+1)
        key_pass=col+Pass
        key_fail=col+Fail
        pltcuts=SALTcolordiffmask[key_pass]
        plot_types(plottypes,'z',alldata,plotlist=plotlist,xlabel=xlabel,ylabel=ylabel,cuts=pltcuts,plotdict=plotdict,plotid=SALT+col+diff+Pass,bins=zbins,addlabel='Pass '+label)
        pltcuts=SALTcolordiffmask[key_fail]
        plot_types(plottypes,'z',alldata,plotlist=plotlist,xlabel=xlabel,ylabel=ylabel,cuts=pltcuts,plotdict=plotdict,plotid=SALT+col+diff+Fail,bins=zbins,alt=alt,addlabel='Fail '+label)
        #f.set_xlim(-5.0, 5.0)
        axes=plt.gca()
        f.set_ylim(0.,axes.get_ylim()[1]*1.5)
        f.legend(loc='best',numpoints=1,prop={'size':8},scatterpoints=1,ncol=Ncols)

    fig.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multiPdf.savefig(fig)

    npages+=1
    ##############  Page 9 and Page 10 Plots   ################
    #page SALT filters (single and multi), z

    SALTfiltermask=get_mask(plottypes,alldata,SALTfilters,SALTfilterlabels,SNTotals,mask_id='SALT peak magnitude',file_id=file_id,good=True)

    #6 plots per page, filters, z
    binhi=[32.,28.,26.,26.]
    binlo=[14.,14.,14.,14.]
    Nmbins=40
    nplots=[1,2,4,5]
    for filts,labels in zip([SALTmfilters,SALTsfilters],[SALTmfilterlabels,SALTsfilterlabels]):
        fig = plt.figure(figsize=(15,7))
        print "\nStarting page",npages,"(page SALT filters, z pass/fail)"
        npages+=1
        for filt,lo,hi,label,npl in zip(filts,binlo,binhi,labels,nplots):
            ylabel='$N$'
            xlabel=SALT+label
            f = fig.add_subplot(230+npl)
            mbins=np.linspace(lo,hi,Nmbins+1)
            key_pass=filt+Pass
            pltcuts=SALTfiltermask[key_pass]
            plot_types(plottypes,filt,alldata,plotlist=plotlist,xlabel=xlabel,ylabel=ylabel,cuts=pltcuts,plotdict=plotdict,plotid=SALT+filt+Pass,bins=mbins,addlabel='Pass '+label)
            f.legend(loc='upper left', fontsize='small',numpoints=1)

            #z distributions for pass i and z only
            if(filt.find('ipeak')!=-1 or filt.find('zpeak')!=-1):
                if(filt.find('ipeak')!=-1):
                    nplot=3
                else:
                    nplot=6
                f = fig.add_subplot(230+nplot)
                plot_types(plottypes,'z',alldata,plotlist=plotlist,xlabel='Redshift',ylabel=ylabel,cuts=pltcuts,plotdict=plotdict,plotid=SALT+filt+Pass,bins=zbins,addlabel=label,)
                key_fail=filt+Fail
                pltcuts=SALTfiltermask[key_fail]
                plot_types(plottypes,'z',alldata,plotlist=plotlist,xlabel='Redshift',ylabel=ylabel,cuts=pltcuts,plotdict=plotdict,plotid=SALT+filt+Fail,bins=zbins,alt=alt,addlabel='No '+label,)
                axes=plt.gca()
                f.set_ylim(0.,axes.get_ylim()[1]*1.5)
                f.legend(loc='upper left',numpoints=1,prop={'size':8},scatterpoints=1,ncol=Ncols)

        fig.tight_layout()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            multiPdf.savefig(fig)

    #end loop over page

    ###############   Page 11+ Plots  #####################
            print "\nSkipping page",npages,"(page more SALT feature plots)"

    #fig = plt.figure(figsize=(15,7))

    # c distribution for pass and fail of Delta-colors
    # scatter plots of Deltaiz, zpeak_s ipeak_s vs z for passing SN
    # x1,fit_pr, c distributions and x1 vs c  for ipeak_s </>= 24.5

#endif--xplots

npages+=1
#########    Page 12++ Bazin Histograms ##############

if(args.Bazin!='off'):
    if(doBazincuts and len(Bazincuts)==0):
        Bazincuts=get_Bazincuts(alldata,args.Bazinpar_min,args.Bazinpar_max,args.Bazinerr_max,SNTotals,bazinparcuts=Bazinparcuts,bazinerrcuts=args.Bazinerrcuts,savetypes=savetypes,file_id=file_id)
    #page 12++ Bazin vars (2 pages x 4 bands+ color)
    print "\nStarting pages",npages,"-",npages+9,"(Bazin plots)\n"
    NBbins=25
    pBtypes=plottypes
    pBlist=plotlist
    yBscale=log
    Bminmax=False

    for filt in fit_bands:
        #page a - pars
        varlist=[b for b in bazinpars if not(b==t0)]
        binlo=[0.,0.,0.,-100.]
        binhi=[600.,120.,20.,100.]
        fig = plt.figure(figsize=(15,7))
        print "\nStarting page",npages,"(",filt,"Bazin vars)\n"
        npages+=1
        nplot=0
        ylabel='$N$'
        if(doBazincuts):
            pltcuts=Bazincuts[filt][All]
        else:
            pltcuts={}
        for var,lo,hi in zip(varlist,binlo,binhi):
            nplot +=1
            f = fig.add_subplot(230+nplot)
            varname=Bazin_+filt+'_'+var
            xlabel=Bazin+ ' '+filt+' '+bazinlabels[var]
            bins=np.linspace(lo,hi,NBbins+1)
            plot_types(pBtypes,varname,alldata,plotlist=pBlist,xlabel=xlabel,ylabel=ylabel,cuts=pltcuts,yscale=yBscale,plotdict=plotdict,plotid=filt+'-band',bins=bins,minmax=Bminmax)
            #f.set_xlim(-5.0, 5.0)
            f.legend(loc='best', fontsize='small',numpoints=1)

        if(doBazincuts):
            cutlist=[Bazin_Const,Bazin_Fail]
            xlabel='Redshift'
            bins=np.linspace(zlo,zhi,Nzbins+1)

            for cut in cutlist:
                nplot +=1
                f = fig.add_subplot(230+nplot)
                varname='z'
                plot_types(pBtypes,varname,alldata,plotlist=pBlist,xlabel=xlabel,ylabel=ylabel,cuts=Bazincuts[filt][cut],yscale=yBscale,plotdict=plotdict,plotid=cut,bins=bins)
                plt.title(re.sub('_',' ',cut)+' Fits',size=sizes[Title])
                f.legend(loc='best', fontsize='small',numpoints=1)

        fig.tight_layout()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            multiPdf.savefig(fig)

        #page b - errors
        varlist=[b for b in bazinerrs]
        binlo=0.
        binhi=[50.,50.,50.,50.,50.]
        fig = plt.figure(figsize=(15,7))
        print "\nStarting page",npages,"(",filt,"Bazin errors)\n"
        npages+=1
        nplot=0
        ylabel='$N$'
        if(doBazincuts):
            pltcuts=Bazincuts[filt][All]
        else:
            pltcuts={}
        for var,hi in zip(varlist,binhi):
            nplot +=1
            f = fig.add_subplot(230+nplot)
            varname=Bazin_+filt+'_'+var
            xlabel=Bazin+ ' '+filt+' '+bazinerrlabels[var]
            bins=np.linspace(binlo,hi,NBbins+1)
            plot_types(pBtypes,varname,alldata,plotlist=pBlist,xlabel=xlabel,ylabel=ylabel,cuts=pltcuts,yscale=yBscale,plotdict=plotdict,plotid=filt+'-band',bins=bins,minmax=Bminmax)
            #f.set_xlim(-5.0, 5.0)                                                                                        
            f.legend(loc='best', fontsize='small',numpoints=1)

        fig.tight_layout()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            multiPdf.savefig(fig)

    #end-filt-bands

    #page 20 Bazin colors
    print "\nStarting page",npages,"(Bazin colors)\n"
    npages+=1

    #select good data for Bazin colors
    Bazincolormask=get_mask(plottypes,alldata,Bazincolors,colorlabels,SNTotals,mask_id='Bazin color',file_id=file_id,good=True)

    print "\nEfficiencies for combined cuts on filters (alternate goor color sample)\n"
    Bazincombomask={}
    for col,label in zip(colors,colorlabels):
        print 'Bazin Color',label
        Bazincombomask[col]={}
        for dkey in alldata.keys():
            print '   {} Sample'.format(dkey)
            Bazincombomask[col][dkey]={}
            for t in plottypes:
                if(t in Bazincuts[col[0]][All][dkey].keys() and t in Bazincuts[col[1]][All][dkey].keys()):
                    Bazincombomask[col][dkey][t]=(Bazincuts[col[0]][All][dkey][t]) & (Bazincuts[col[1]][All][dkey][t]) 
                    print '      {}  {}'.format(t,np.sum(Bazincombomask[col][dkey][t]))

    #make plots
    fig = plt.figure(figsize=(15,7))
    binlo=[-100.,-100.,-100.]
    binhi=[100.,100.,100.]
    nplot=0
    nplot2=3
    NBbins=50
    print '\n Min and Max of colors:\n'
    for col,lo,hi,label in zip(colors,binlo,binhi,Bazincolorlabels):
        varname=col+Bazin    
        ylabel='$N$'
        xlabel=label
        nplot +=1
        f = fig.add_subplot(230+nplot)
        bins=np.linspace(lo,hi,NBbins+1)
        pltcuts=Bazincolormask[varname+ne999]
        plot_types(pBtypes,varname,alldata,plotlist=pBlist,xlabel=xlabel,ylabel=ylabel,cuts=pltcuts,yscale=yBscale,plotdict=plotdict,plotid=Bazin+' '+col,bins=bins,minmax=True,addlabel='(Pass '+label+')')
        axes=plt.gca()
        f.set_ylim(1.,axes.get_ylim()[1]*10)                                
        f.legend(loc='best', fontsize='small',numpoints=1)

        nplot2 +=1
        f = fig.add_subplot(230+nplot2)
        pltcuts=Bazincombomask[col]
        plot_types(pBtypes,varname,alldata,plotlist=pBlist,xlabel=xlabel,ylabel=ylabel,cuts=pltcuts,yscale=yBscale,plotdict=plotdict,plotid=Bazin+' '+col,bins=bins,minmax=True,addlabel='(Pass All'+col+')')
        axes=plt.gca()
        f.set_ylim(1.,axes.get_ylim()[1]*10)                                
        f.legend(loc='best', fontsize='small',numpoints=1)

    fig.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multiPdf.savefig(fig)

    #page 21+ - Bazin scatter plots
    xvars=[t_rise]
    yvars=[t_fall]
    xbinlo=[0.]
    xbinhi=[20.]
    ybinlo=[0.]
    ybinhi=[120.]
    xbinwidth=[0.5]
    ybinwidth=[2]
    #add scatter plot t_rise vs t_fall;
    for l,s in zip([Training]+MLdatalist,[Sim]+[Data for m in MLdatalist]):
      print "\nStarting page",npages,"(Bazin scatterplots)\n"
      fig = plt.figure(figsize=(15,7))
      npages+=1
      nplot=0
      for filt in fit_bands:
        for xvar,yvar,xmin,xmax,ymin,ymax,dx,dy in zip(xvars,yvars,xbinlo,xbinhi,ybinlo,ybinhi,xbinwidth,ybinwidth):
            nplot+=1
            f = fig.add_subplot(230+nplot)
            xvarname=Bazin_+filt+'_'+xvar
            xlabel=Bazin+ ' '+filt+' '+bazinlabels[xvar]
            yvarname=Bazin_+filt+'_'+yvar
            ylabel=Bazin+ ' '+filt+' '+bazinlabels[yvar]
            if(doBazincuts):
                pltcuts=Bazincuts[filt][All]
            else:
                pltcuts={}
            plotdict['labels'][contour]='SNIa Density'
            if (l in lbldMLdatalist or l==Training):
                pbztypes=MLtypes
            else:
                pbztypes=CLFtypes
            plot_types(pbztypes,xvarname,alldata,plotlist=[l],yvar=yvarname,xlabel=xlabel,ylabel=ylabel,cuts=pltcuts,plotdict=plotdict,ctrxbin=dx,ctrybin=dy,title=plotlabels[s][l],weights=False)
            f.set_xlim(xmin-dx,xmax+dx)
            f.set_ylim(ymin-dy,ymax+dy)
            f.legend(loc='best', fontsize='small',scatterpoints=1,ncol=Ncols)

      fig.tight_layout()
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multiPdf.savefig(fig)

#endif--args.Bazin

##### Page 11 Cross-Validation and Purity Variation Plots ######
#
if(args.cv or args.pc):
    print "\nStarting page",npages,"(Cross-Validation and Purity)\n"
    fig = plt.figure(figsize=(15,7))
    if(args.cv):
        minval = np.fmin(np.min(avgskf), np.min(avgss))
        maxval = np.fmax(np.max(avgskf), np.max(avgss))
        ax1 = fig.add_subplot(231)
        plt.title('Stratified k-fold Scores')
        ax1.errorbar(kvals, avgskf, yerr=stdkf, fmt='o')
        ax1.scatter(kvals, avgskf,color='blue',marker=".")
        ax1.set_ylim(minval-0.02, maxval+0.02)
        ax1.set_xlabel('k (number of folds)')
        ax1.set_ylabel('Score')

        ax2 = fig.add_subplot(232)
        plt.title('ShuffleSplit Scores')
        ax2.errorbar(1.0-np.array(tsvals), avgss, yerr=stdss, fmt='o')
        ax2.scatter(1.0-np.array(tsvals), avgss, color='blue', marker='.')
        ax2.set_ylim(minval-0.02, maxval+0.02)
        ax2.set_xlabel('Training Fraction')
        ax2.set_ylabel('Score')
    #endif-cv

    if(args.pc):
        ax1 = fig.add_subplot(233)
        plt.title('Scores for Purity Variations')
        ax1.scatter(args.purities,pscores,color='blue',marker=".")
        ax1.set_xlabel('Purity of Test Data')
        ax1.set_ylabel('Score')        

    #endif-pc

    fig.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multiPdf.savefig(fig)
#endif


#Page 12 Probability Variances
if(args.prvar):
  fig = plt.figure(figsize=(15,7))
  pr_binwidth = 0.2
  pr_bins = np.arange(0.0, 1., pr_binwidth)
  pcolors=['r','g','blue','cyan','magenta','y','orange','navy','pink','purple']

  print "Decision-Tree Probabilities"
  f = fig.add_subplot(231)
  f.set_xlabel('Test-Data Decision-Tree SNIa Probability')
  f.set_ylabel('Number')
  f.set_yscale('log')
  f.tick_params(axis='both', which='major', labelsize=12)
  f.set_xlim(0, 1.0)
  for npr, pbin in enumerate(pr_bins):
    if (npr < len(pr_bins)):
        probcut=(allprobs[:,0]>=pr_bins[npr]) & (allprobs[:,0]<pr_bins[npr]+pr_binwidth) & (trueIa)
        plabel=str(pr_bins[npr])+'-'+str(pr_bins[npr]+pr_binwidth)
    else:
        probcut=(allprobs[:,0]>=pr_bins[npr]) & (allprobs[:,0]<=pr_bins[npr]+pr_binwidth) & (trueIa) #include 1.0 in last bin
        plabel=str(pr_bins[npr])+'-'+str(pr_bins[npr]+pr_binwidth)
    probdata=pdata[probcut][:,0]    #find tree probs for this pr_bin for this SN type
    print "Found",len(probdata),"SNIa matching cuts for pr_bin",npr,":",pr_bins[npr],"-",pr_bins[npr]+pr_binwidth,"(",len(probdata.flatten())," prob. values)"
    plt.hist(probdata.flatten(), bins=p_bins, color=pcolors[npr], histtype='step',alpha=totalpha, label=plabel) #,normed=True)
  f.legend(loc='upper center', scatterpoints=1,ncol=Ncols, fontsize='small')
  print ""

  if(args.nclass==2):
    f = fig.add_subplot(232)
    f.set_xlabel('Test-Data Decision-Tree SNCC Probability')
    f.set_ylabel('Number')
    f.set_yscale('log')
    f.tick_params(axis='both', which='major', labelsize=12)
    f.set_xlim(0, 1.0)
    for npr, pbin in enumerate(pr_bins):
        if (npr < len(pr_bins)):
            probcut=(allprobs[:,1]>=pr_bins[npr]) & (allprobs[:,1]<pr_bins[npr]+pr_binwidth) & (trueCC)
            plabel=str(pr_bins[npr])+'-'+str(pr_bins[npr]+pr_binwidth)
        else:
            probcut=(allprobs[:,1]>=pr_bins[npr]) & (allprobs[:,1]<=pr_bins[npr]+pr_binwidth) & (trueCC) #include 1.0 in last bin
            plabel=str(pr_bins[npr])+'-'+str(pr_bins[npr]+pr_binwidth)
        probdata=pdata[probcut][:,1]    #find tree probs for this pr_bin for this SN type
        print "Found",len(probdata),"SNCC matching cuts for pr_bin",npr,":",pr_bins[npr],"-",pr_bins[npr]+pr_binwidth,"(",len(probdata.flatten())," prob. values)"
        plt.hist(probdata.flatten(), bins=p_bins, color=pcolors[npr], histtype='step',alpha=totalpha, label=plabel) #,normed=True)
    f.legend(loc='upper center', scatterpoints=1,ncol=Ncols, fontsize='small')
    print ""
  else:
    f = fig.add_subplot(232)
    f.set_xlabel('Test-Data Decision-Tree SNIbc Probability')
    f.set_ylabel('Number')
    f.set_yscale('log')
    f.tick_params(axis='both', which='major', labelsize=12)
    f.set_xlim(0, 1.0)
    for npr, pbin in enumerate(pr_bins):
        if (npr < len(pr_bins)):
            probcut=(allprobs[:,1]>=pr_bins[npr]) & (allprobs[:,1]<pr_bins[npr]+pr_binwidth) & (trueIbc)
            plabel=str(pr_bins[npr])+'-'+str(pr_bins[npr]+pr_binwidth)
        else:
            probcut=(allprobs[:,1]>=pr_bins[npr]) & (allprobs[:,1]<=pr_bins[npr]+pr_binwidth) & (trueIbc) #include 1.0 in last bin
            plabel=str(pr_bins[npr])+'-'+str(pr_bins[npr]+pr_binwidth)
        probdata=pdata[probcut][:,1]    #find tree probs for this pr_bin for this SN type
        print "Found",len(probdata),"SNIbc matching cuts for pr_bin",npr,":",pr_bins[npr],"-",pr_bins[npr]+pr_binwidth,"(",len(probdata.flatten())," prob. values)"
        plt.hist(probdata.flatten(), bins=p_bins, color=pcolors[npr], histtype='step',alpha=totalpha, label=plabel) #,normed=True)
    f.legend(loc='upper center', scatterpoints=1,ncol=Ncols, fontsize='small')

    print ""
    f = fig.add_subplot(233)
    f.set_xlabel('Test-Data Decision-Tree SNII Probability')
    f.set_ylabel('Number')
    f.set_yscale('log')
    f.tick_params(axis='both', which='major', labelsize=12)
    f.set_xlim(0, 1.0)
    for npr, pbin in enumerate(pr_bins):
        if (npr < len(pr_bins)):
            probcut=(allprobs[:,2]>=pr_bins[npr]) & (allprobs[:,2]<pr_bins[npr]+pr_binwidth) & (trueII)
            plabel=str(pr_bins[npr])+'-'+str(pr_bins[npr]+pr_binwidth)
        else:
            probcut=(allprobs[:,2]>=pr_bins[npr]) & (allprobs[:,2]<=pr_bins[npr]+pr_binwidth) & (trueII) #include 1.0 in last bin
            plabel=str(pr_bins[npr])+'-'+str(pr_bins[npr]+pr_binwidth)
        probdata=pdata[probcut][:,2]    #find tree probs for this pr_bin for this SN type
        print "Found",len(probdata),"SNII matching cuts for pr_bin",npr,":",pr_bins[npr],"-",pr_bins[npr]+pr_binwidth,"(",len(probdata.flatten())," prob. values)"
        plt.hist(probdata.flatten(), bins=p_bins, color=pcolors[npr], histtype='step',alpha=totalpha, label=plabel) #,normed=True)
        f.legend(loc='upper center', scatterpoints=1,ncol=Ncols, fontsize='small')

  print ""
  f = fig.add_subplot(234)
  f.set_xlabel('Test-Data Random Forest SNIa Probability')
  f.set_ylabel('SNIa Probability Variance')
  f.tick_params(axis='both', which='major', labelsize=12)
  f.scatter(probs[trueIa], variance[trueIa][:,0], marker='.', alpha=scattalpha, color=Iacol, label='SNIa')
  if (args.nclass==3):
    f.scatter(probs[trueIbc],variance[trueIbc][:,0] , marker='.', alpha=scattalpha, color=Ibccol, label='SNIbc')
    f.scatter(probs[trueII], variance[trueII][:,0], marker='.', alpha=scattalpha, color=IIcol, label='SNII')
  else:
    f.scatter(probs[trueCC], variance[trueCC][:,0], marker='.', alpha=scattalpha, color=CCcol, label='SNCC')
  f.legend(loc='best', fontsize='small',scatterpoints=1)

  f = fig.add_subplot(235)
  f.set_xlabel('Test-Data Random Forest SNIa Probability')
  f.set_ylabel(str(percentile)+'th Percentile Upper Limit')
  f.tick_params(axis='both', which='major', labelsize=12)
  f.scatter(probs[trueIa], err_up[trueIa][:,0], marker='.', alpha=scattalpha, color=Iacol, label='SNIa')
  if (args.nclass==3):
    f.scatter(probs[trueIbc],err_up[trueIbc][:,0] , marker='.', alpha=scattalpha, color=Ibccol, label='SNIbc')
    f.scatter(probs[trueII], err_up[trueII][:,0], marker='.', alpha=scattalpha, color=IIcol, label='SNII')
  else:
    f.scatter(probs[trueCC], err_up[trueCC][:,0], marker='.', alpha=scattalpha, color=CCcol, label='SNCC')
  f.legend(loc='lower right', fontsize='small',scatterpoints=1)

  f = fig.add_subplot(236)
  f.set_xlabel('Test-Data Random Forest SNIa Probability')
  f.set_ylabel(str(percentile)+'th Percentile Lower Limit')
  f.tick_params(axis='both', which='major', labelsize=12)
  f.scatter(probs[trueIa], err_down[trueIa][:,0], marker='.', alpha=scattalpha, color=Iacol, label='SNIa')
  if (args.nclass==3):
    f.scatter(probs[trueIbc],err_down[trueIbc][:,0] , marker='.', alpha=scattalpha, color=Ibccol, label='SNIbc')
    f.scatter(probs[trueII], err_down[trueII][:,0], marker='.', alpha=scattalpha, color=IIcol, label='SNII')
  else:
    f.scatter(probs[trueCC], err_down[trueCC][:,0], marker='.', alpha=scattalpha, color=CCcol, label='SNCC')
  f.legend(loc='upper left', fontsize='small',scatterpoints=1)


  fig.tight_layout()
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    multiPdf.savefig(fig)
#endif--prvar


#Making the plots for the number of SN per template 

fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(111)
ax2 = ax.twinx()
templates = sorted(template_dict.keys())  #sorted list of templates
index = np.arange(len(templates))  #length of dict
TMs={}
TM0={}
tick_colors=[color[template_stats[tmpl]['Type']] for tmpl in templates]
#setup entries for bar charts for each type
#Ias are separated so different axis can be used 
for t in MLtypes:
    TM0[t]=[0]*len(templates)     #initialize all zero entries
    TM0[t][0]=template_stats[0][CLFid+t] #overwrite value for Ias
    TMs[t] = [template_stats[tmpl][CLFid+t] for tmpl in templates] #number of each type per template
    TMs[t][0] = 0   #overwrite entry for Ias to zero

rects1 = ax2.bar(index,TMs[Ia],.3,color = color[Ia], label= 'Typed as Ia')
rects3 = ax.bar(index,TM0[Ia], .3,color = color[Ia])
bottom2 = TMs[Ia]
bottom  = TM0[Ia]
nonIas = [typ for typ in MLtypes if not(typ==Ia)]
for nonIa in nonIas:
    rects2 = ax2.bar(index,TMs[nonIa],.3,bottom = bottom2,color = color[nonIa],label='Typed as '+nonIa) 
    bottom2= map(add, bottom2, TMs[nonIa])
    rects4 = ax.bar(index, TM0[nonIa], .3,bottom =bottom, color = color[nonIa])
    bottom= map(add, bottom, TM0[nonIa])

plt.xlabel('Template')
ax.set_ylabel('Number of Ia (template = 0)')
ax2.set_ylabel('Number of CC (template != 0)' )
plt.title('Number of SN classified as Ia and CC per template')
plt.xticks(index+.1,templates)
ax.set_xlim(0,max(index)+1)
for xtick, col in zip(ax.get_xticklabels(), tick_colors):
    xtick.set_color(col)
plt.legend(loc='best')
#plt.legend((rects1[0],rects2[0]),('Typed as Ia','Typed as CC'))

fig.tight_layout()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    multiPdf.savefig(fig)

fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(111)
templates=sorted(template_dict.keys())
fracIa= [float(Iatemplate_dict[tmpl])/float(template_dict[tmpl]) if (tmpl in Iatemplate_dict.keys()) else 0 for tmpl in templates ]
index = np.arange(len(templates))
rects5 = plt.bar(index,fracIa,.3,color = color[Ia],label='Typed as Ia')
plt.xlabel('Template')
plt.ylabel('Fraction of SN typed as Ia')
plt.title('Fraction of SN classified as Ia per template')
plt.xticks(index+.1,templates)
for xtick, col in zip(ax.get_xticklabels(), tick_colors):
    xtick.set_color(col)
ax.set_xlim(0,max(index)+1)
plt.legend(loc='best')

fig.tight_layout()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    multiPdf.savefig(fig)


multiPdf.close()

print "\nWrote",pdfname
print "Completed Successfully"
