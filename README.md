# ML-SN-Classifier
Machine-Learning Algorithm for Supernova Calssification

SNIRF.py performs photometric classification using a
Random-Forest machine-learing algorithm. It builds and tests the
classifier using simulated data. The user can supply additional data
files which will be classified by the code. The code saves the
classifications in auxiliary text files.

The package consists of 3 modules:
SNIRF.py (builds and runs the classifier; outputs results)
ML_globals.py (contains definitions of global constants and functions)
ML_plots.py (optional plotting code that makes a variety of plots)

Many options are available.
To get help:
python ./SNIRF.py -h  (standard argparse implementation)

This gives the following output:
usage: SNIRF.py [-h] [--ft features [features ...]] [--nc n_cores]                                                                                     
                [--eff Fix_eff [Fix_eff ...]] [--nclass NCLASS]                                                                                        
                [--train TRAIN] [--train_format {txt,FITRES,csv}]                                                                                      
                [--validation VALIDATION] [--test TEST]                                                                                                
                [--data DATA [DATA ...]] [--spec SPEC]                                                                                                 
                [--alltypes_colname_spec ALLTYPES_COLNAME_SPEC] [--phot PHOT]                                                                          
                [--spec_nofp SPEC_NOFP]                                                                                                                
                [--user_data USER_DATA [USER_DATA ...]]                                                                                                
                [--user_alltypes_colnames USER_ALLTYPES_COLNAMES [USER_ALLTYPES_COLNAMES ...]]                                                         
                [--filedir FILEDIR] [--store] [--restore] [--use_filenames]                                                                            
                [--pklfile PKLFILE] [--pklformat {txt,FITRES,csv}]                                                                                     
                [--train_only] [--done_file DONE_FILE] [--filestr FILESTR]                                                                             
                [--format FORMAT] [--commit_hash_path COMMIT_HASH_PATH]                                                                                
                [--plots {,Performance,SALT,Hubble,Error,Color,Magnitude,Bazin} [{,Performance,SALT,Hubble,Error,Color,Magnitude,Bazin} ...]]          
                [--weights WEIGHTS]                                                                                                                    
                [--sim {Training,Validation} [{Training,Validation} ...]]                                                                              
                [--totals] [--save {Ia,CC,All} [{Ia,CC,All} ...]]                                                                                      
                [--user_colors USER_COLORS [USER_COLORS ...]]                                                                                          
                [--user_markers USER_MARKERS [USER_MARKERS ...]] [--minmax]                                                                            
                [--debug] [--alltypes_colname_train ALLTYPES_COLNAME_TRAIN]                                                                            
                [--alltypes_colname_validation ALLTYPES_COLNAME_VALIDATION]                                                                            
                [--alltypes_colname_test ALLTYPES_COLNAME_TEST]                                                                                        
                [--type_labels TYPE_LABELS [TYPE_LABELS ...]]                                                                                          
                [--type_values TYPE_VALUES [TYPE_VALUES ...]] [--cv]                                                                                   
                [--sample {t,v,b}] [--niter {1,2,3,4,5,6,7,8,9,10}]                                                                                    
                [--testsize TESTSIZE] [--nfolds NFOLDS] [--pc]                                                                                         
                [--purities PURITIES [PURITIES ...]]                                                                                                   
                [--withhold types/templates [types/templates ...]] [--prvar]                                                                           
                [--Bazincuts {train,plots,off}] [--noBazinparcuts]                                                                                     
                [--Bazinpar_max BAZINPAR_MAX [BAZINPAR_MAX ...]]                                                                                       
                [--Bazinpar_min BAZINPAR_MIN [BAZINPAR_MIN ...]]                                                                                       
                [--Bazinerrcuts]                                                                                                                       
                [--Bazinerr_max BAZINERR_MAX [BAZINERR_MAX ...]] [--H0 H0]                                                                             
                [--OmegaM OMEGAM]                                                                                                                      

This script uses a machine learning (ML) algorithm to train a photometric
classifier into classes {Ia|CC or Ia|Ibc|II or I|II->Ia|Ibc}. Select Random
Forest features to use.                                                    

optional arguments:
  -h, --help            show this help message and exit
  --ft features [features ...]                         
                        Choose SN-type features to use for classification.
                        List them (space-separated) and select from: {c, x0,
                        x1, t0, z, chi2, fit_pr, gpeak, rpeak, ipeak, zpeak,
                        ra, dec, grpeak_m, ripeak_m, izpeak_m, grpeak_s,    
                        ripeak_s, izpeak_s, Deltagr, Deltari, Deltaiz, snr1,
                        snr2, snr3, t0_err, x0_err, x1_err, c_err,          
                        Bazin_g_t_rise, Bazin_g_t_fall, Bazin_r_t_rise,     
                        Bazin_r_t_fall, Bazin_i_t_rise, Bazin_i_t_fall,     
                        Bazin_z_t_rise, Bazin_z_t_fall, grBazin, riBazin,   
                        izBazin, CID, IDSURVEY, TYPE, FIELD, CUTFLAG_SNANA, 
                        zHEL, zHELERR, zCMB, zCMBERR, zHD, zHDERR, VPEC,    
                        VPECERR, HOST_LOGMASS, HOST_LOGMASS_ERR, HOST_sSFR, 
                        HOST_sSFR_ERR, SNRMAX1, SNRMAX2, SNRMAX3, PKMJD,    
                        PKMJDERR, x1, x1ERR, c, cERR, mB, mBERR, x0, x0ERR, 
                        COV_x1_c, COV_x1_x0, COV_c_x0, NDOF, FITCHI2, FITPROB,
                        SIM_TYPE_INDEX, SIM_TEMPLATE_INDEX, SIM_LIBID,        
                        SIM_NGEN_LIBID, SIM_ZCMB, SIM_VPEC, SIM_DLMAG,        
                        SIM_PKMJD, SIM_SHAPE, SIM_AV, SIMNULL1, SIM_RV,       
                        SIM_x0, SIM_mB, CID, zCMB, PKMJD, PKMJDERR, x0ERR,    
                        x1ERR, cERR, FITCHI2, NDOF, MU, MUERR, FITPROB,       
                        SNRMAX1, SNRMAX2, SNRMAX3, SIM_TYPE_INDEX,            
                        SIM_TEMPLATE_INDEX, SIM_DLMAG, snid, z, t0,           
                        ['PKMJDerr', 't0_err'], chi2, fit_pr, mu, muerror,    
                        snr1, snr2, snr3, sim_type, sim_nonIa, sim_mu, x0_err,
                        x1_err, c_err} (default: ['fit_pr', 'x1'])            
  --nc n_cores          Number of cores to use for parallelization (default:  
                        4)                                                    
  --eff Fix_eff [Fix_eff ...]                                                 
                        Efficiencies at which to evaluate purity (default:    
                        [0.95])                                               
  --nclass NCLASS       Number of classes used in classifier (2=Ia/CC,        
                        3=Ia/Ibc/II,-2=2x2 training: 2-stage--2-way           
                        classification with I/II -> Ia/Ibc) (default: 3)      
  --train TRAIN         Filename for training (default:                       
                        DES_training_SNR550.txt)                              
  --train_format {txt,FITRES,csv}                                             
                        Format of training data (default: txt)                
  --validation VALIDATION                                                     
                        Filename for validation; set to null to skip (default:
                        '')                                                   
  --test TEST           Filename for test (skip if set to "") (default:       
                        DES_test_SNR550.txt)                                  
  --data DATA [DATA ...]                                                      
                        Classify selected simulated/observed data             
                        (default=Phot); choose from [Test, Spec, Spec_nofp,   
                        Phot, ...] plots (if selected) will be normalized to  
                        first data sample in list; customize user-data labels 
                        by adding user-supplied labels here (default: [])     
  --spec SPEC           Filename for spectroscopic data (default:             
                        specType_forcePhoto_bazin_v3.txt)                     
  --alltypes_colname_spec ALLTYPES_COLNAME_SPEC                               
                        Column name for true SN types in Spec data (default:  
                        spec_eval)                                            
  --phot PHOT           Filename for photometric data (default:               
                        host_prior_DESY1-4_bazin_v3e.txt)                     
  --spec_nofp SPEC_NOFP                                                       
                        Filename for spectroscopic data without fit           
                        probabilities (default: specSN_SNANA_nofpcut.txt)     
  --user_data USER_DATA [USER_DATA ...]                                       
                        Filenames for user-supplied data; default labels in   
                        plots etc. will be User0, User1, ... if no additional 
                        arguments supplied for --data (default: [])           
  --user_alltypes_colnames USER_ALLTYPES_COLNAMES [USER_ALLTYPES_COLNAMES ...]
                        Column names for types of labeled user-supplied data  
                        (unlabeled data assumed); use null string for         
                        unlabeled data in list of user-data files with mixed  
                        labeling properties. (default: [])                    
  --filedir FILEDIR     Directory name (relative or absolute) for storing     
                        output files (default: ./)                            
  --store               Save trained classifier to pklfile (default: False)   
  --restore             Restore trained classifier from pklfile (default:     
                        False)                                                
  --use_filenames       Use filenames as supplied (no default additions       
                        specifying formats, etc. (default: False)             
  --pklfile PKLFILE     Base filename for reading/writing trained classifier  
                        (_{format}_{nclass}way.pkl is auto-appended;          
                        pre+filename also read/written if nclass == -2)       
                        (default: trained_RFclassifier)                       
  --pklformat {txt,FITRES,csv}                                                
                        Format of stored training data (default: txt)         
  --train_only          Run training only, save trained classifier to pklfile,
                        and exit (default: False)                             
  --done_file DONE_FILE                                                       
                        Path to file recording SUCCESS/FAILURE of run         
                        (default: '')                                         
  --filestr FILESTR     Choose string to append to filenames for output files 
                        (default: '')                                         
  --format FORMAT       Format for output of classification data (txt, hdf5,  
                        fits) (default: .txt)                                 
  --commit_hash_path COMMIT_HASH_PATH                                         
                        Path to local github repository for retrieving commit 
                        hash (set to '' to skip) (default: ./ML-SN-Classifier)
  --plots {,Performance,SALT,Hubble,Error,Color,Magnitude,Bazin} [{,Performance,SALT,Hubble,Error,Color,Magnitude,Bazin} ...]
                        Make selected groups of plots; choose from                                                           
                        Performance, SALT, Hubble, Color, Error, Magnitude,                                                  
                        Bazin; null argument supresses all plots (default: )                                                 
  --weights WEIGHTS     Name of data sample to use for computing weights for                                                 
                        plot normalizations; defaults to first data sample in                                                
                        data list (default: )                                                                                
  --sim {Training,Validation} [{Training,Validation} ...]                                                                    
                        Plot selected simulated data (default: ['Training'])                                                 
  --totals              Turn on totals in plots (default if using --data or                                                  
                        --user-data) (default: False)                                                                        
  --save {Ia,CC,All} [{Ia,CC,All} ...]                                                                                       
                        Types to save for printing pickling in plotting                                                      
                        modules (default: Ia)                                                                                
  --user_colors USER_COLORS [USER_COLORS ...]                                                                                
                        Colors for user-supplied data (3 defaults available)                                                 
                        (default: ['seagreen', 'springgreen', 'lime'])                                                       
  --user_markers USER_MARKERS [USER_MARKERS ...]                                                                             
                        Markers for user-supplied data (3 defaults available)                                                
                        (default: ['<', '>', '8'])                                                                           
  --minmax              Turn on minmax print-out in plots (default: False)                                                   
  --debug               Turn on debug print-out in plots (default: False)                                                    
  --alltypes_colname_train ALLTYPES_COLNAME_TRAIN                                                                            
                        Column name for true SN types in validation data, eg.                                                
                        "type3" identifies SN as Ia, Ibc, II; code checks for                                                
                        consistency with --nclass and creates appropriate data                                               
                        column(s) if needed (default: type3)                                                                 
  --alltypes_colname_validation ALLTYPES_COLNAME_VALIDATION                                                                  
                        Column name for true SN types in validation data,                                                    
                        (default: type3)                                                                                     
  --alltypes_colname_test ALLTYPES_COLNAME_TEST
                        Column name for true SN types in test data, (default:
                        type3)
  --type_labels TYPE_LABELS [TYPE_LABELS ...]
                        Labels for classes in alltypes_colname (supply one per
                        typename); CC assumed to be ~Ia (default: ['Ia', 'II',
                        'Ibc', 'Ib', 'Ic'])
  --type_values TYPE_VALUES [TYPE_VALUES ...]
                        Corresponding class values in type_labels (default:
                        [[0], [2], [1], [1], [1]])
  --cv                  Turn on cross-validation (default: False)
  --sample {t,v,b}      Sample to use for cross-validation (t=training,
                        v=validation, b=both) (default: t)
  --niter {1,2,3,4,5,6,7,8,9,10}
                        Number of ShuffleSplit cross-validation iterations
                        (default: 5)
  --testsize TESTSIZE   Validation sample size for ShuffleSplit (default: 0.5)
  --nfolds NFOLDS       Number of folds for Kfold sample (default: 2)
  --pc                  Turn on purity comparison (default: False)
  --purities PURITIES [PURITIES ...]
                        Purities for loop over test files (default: [0.1,
                        0.25, 0.5, 0.75, 0.9])
  --withhold types/templates [types/templates ...]
                        Hold-out test: withhold type (20=IIP, 21=IIN, 22=IIL,
                        32=Ib, 33=Ic) or template (206,209=IIN, 002=IIL,
                        201-235=IIP, 103-234=Ib, 021-218=Ic) from training
                        sample: List them (space-separated) and select from:
                        {20, 21, 22, 32, 33, 206, 209, 002, 201, 204, 208,
                        210, 213, 214, 215, 216, 219, 220, 221, 222, 223, 224,
                        225, 226, 227, 228, 229, 230, 231, 232, 233, 235, 400,
                        401, 402, 403, 103, 104, 105, 202, 203, 212, 234, 021,
                        022, 101, 102, 205, 207, 211, 217, 218, 502, 503, 506,
                        509} (default: )
  --prvar               Turn on probability-variance plots (default: False)
  --Bazincuts {train,plots,off}
                        Include Bazin selections in training/plots (TBD),
                        plots only, or turn off (default: off)
  --noBazinparcuts      Turn OFF Bazin parameter cuts (default: False)
  --Bazinpar_max BAZINPAR_MAX [BAZINPAR_MAX ...]
                        Cuts on Bazin fit parameters: [A, t0, t_fall, t_rise,
                        C] (default: [800, '', 150, 't_fall', 100])
  --Bazinpar_min BAZINPAR_MIN [BAZINPAR_MIN ...]
                        Cuts on Bazin fit parameters: [A, t0, t_fall, t_rise,
                        C] (default: [-999, 1, 0, 0, -999])
  --Bazinerrcuts        Turn ON Bazin error cuts (default: False)
  --Bazinerr_max BAZINERR_MAX [BAZINERR_MAX ...]
                        Cuts on Bazin fit errors: [A_err, t0_err, t_fall_err,
                        t_rise_err, C_err] (default: [100, 50, 100, 50, 100])
  --H0 H0               Value of H0 (default: 68.62)
  --OmegaM OMEGAM       Value of OmegaM (default: 0.301)


ML_globals.py contains the definitions of global constants used
throughout the code.  The code classifies SNe in either binary (Ia or CC)
or tertiary (Ia, Ibc, or II) mode, corresponding to the user choice 
--nclass 2 or --nclass 3, respectively. 

Two formats are supported for the user-supplied input data, which are read in
as astropy tables: 
1) text file with a header that labels columns of data, 
2) fitres file (typically output from SNANA).  

ML_globals.py documents the default formats for these files. User
options are available to change some of these default values (see
above).  The most important definitions are:

allowed_features: dictionary with list of allowed feature names for each allowed format
data_defaults: dictionary with information about how classes are labeled for each allowed format
allowed_templates: dictionary with list of templates used to simulate each class in the data

The contents of allowed_features and allowed_templates should be self-explanatory.
The following keys in the data_defaults dictionary describe the labels used 
for the various classes contained in user-supplied labeled data. (These defaults
can be changed on the fly).

alltypes_colname: name of data column containing the labels for the class (type) of each object
type_values: value (usually integer) given to each labeled class
type_labels: labels (Ia, Ibc, II, CC) corresponding to each value in type_values

The information in the following keys is used internally in the code.
type_colnames: name of column containing labels for different values of nclass (2, 3, -2);
	       generated on the fly (if possible) if not available in data file
alltypes_available: True (False), if labels (do not) differentiate between classes of CC SNe


In general, if your file format/variable names are different from
those listed in the above dictionaries, you will need to make changes
to ML_globals.py to get the code to run successfully.  New feature
names should be added to the list appropriate to your file format in
allowed_features.  Although it is possible to change the defaults in
data_defaults by supplying user options, it will be more convenient to
make the appropriate changes to the data_defaults dictionary by
modifying the entries for the keys alltypes_colname and type_values to
match your labeled data.

