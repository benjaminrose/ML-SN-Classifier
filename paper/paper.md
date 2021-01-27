---
title: 'SNIRF: SuperNova Identification with Random Forest'
tags:
  - Python
  - supernova
  - cosmology

authors:
  - name: Eve Kovacs
    orcid: 0000-0002-2545-1989
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Stephen Kuhlmann
    affiliation: 1
  - Ravi Gupta
    affiliation: 1
  - name: Mi Dai
    affiliation: "2, 1"
affiliations:
 - name: Argonne National Laboratory
   index: 1
 - name: Rutgers, The State University of New Jersey
   index: 2
date: 27 January 2021
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
---

# Summary

SuperNova Identification with Random Forest (SNIRF) is a stand-alone
python package for the classification of supernovae (SNe) based on their
photometric data alone. Photometric data consist of a series of measurements
of the integrated SNe light fluxes over the band passes of the filters used
for the observations. Hence, these data are missing the prominent spectral
features that can be used to classify SNe for which spectra are available.
The purpose of SNIRF is to identify, within these photometric samples, the
cosmologically useful type Ia SNe, thereby removing all of the
contamination from core-collapse (CC) SNe. In previous analyses
such as the DES-3YR analysis [@des3yr], each supernova candidate was
followed-up spectroscopically to ensure that it was indeed of type Ia.
However, this follow-up is extremely expensive and not feasible for
candidates at higher redshifts. Hence spectroscopic samples are
necessarily small and limited in redshift range.  For the DES-5YR
analysis, we wish to exploit fully the large number of supernova
candidates discovered by the survey and thus it is crucial to develop
photometric classifiers to remove contamination by non-Ia SNe.

The contamination is dominated by type Ib, Ic or II SNe, which
exhibit light curves having different shapes and evolution with time
than their type Ia counterparts. For example, CC SNe are on
average dimmer than type Ia SNe and have bluer colors, where
color is defined as the difference between the light curves at a given
epoch relative to peak brightness.  SNIRF exploits these differences
to perform the classification. It has been designed to run on
supernova candidates whose raw photometric data (typically time series
of fluxes in various broad-band filters) have been pre-processed by a
light-curve fitter such as SNANA [@snana] or sncosmo [@sncosmo]. The
light curves are characterized by parameters or "features" that
describe their properties and serve as inputs to the SNIRF
machine-learning algorithm.

SNIRF makes use of the Random Forest algorithm from the scikit-learn
[@sklearn]. Random Forest is a supervised machine-learning code that
uses labeled training data to build a set of decision trees based on
the features supplied in the training data. These features are user
determined and may be engineered to improve the accuracy of the
classifications obtained. The trained algorithm classifies objects of
unknown type by averaging the results over the ensemble of trees.  The
number of types of SN that can be classified is determined by the
number of labels available in the training data.  Training the model
on training sets of up to 50,000 SNe typically takes less than 1
minute on a 4 core processor.

The average scores output by the random forest algorithm are the
so-called "random-forest probabilities", which are provided for each
class of object in the original training set.  In general, these are
not true probabilities because the fractions of correct
classifications do not match the probability values returned. In order
to obtain a final classification, the user must select either a
threshold value of the probability above which objects will be
considered to be in the desired class, or else choose the maximum
value of each class (the maximum probability) to be the
classification.

The SNIRF code provides many options for the user to select how the
training and classification will be done. The code accepts data in
either text or so called 'FITRES' format (produced by SNANA) and can
handle both spectroscopic and photometric data sets.  Various
parameters pertaining to the random-forest algorithm can be selected.
There is an option to balance the number of type Ia and CC SNe in the
training set. The code outputs the classification of each object in
the supplied data for both user-supplied thresholds based on sample
purity and for the maximum probability method.  The format of the
output data is customizeable.  Log files of each run are automatically
generated and the level of output to the screen during interactive
running can be adjusted.  Other user-selected options include the
ability to to run on combinations of training, testing and data files,
to save trained models.

Currently, the user can choose to build a binary (distinguishes
between Type Ia and CC SNe) or tertiary (distinguishes between types
Ia, Ib/c and II SNe) classifier depending on how the training data is
labeled. In principle, the code can be expanded to classify other
types if appropriately labeled training data is available. We
typically use simulated training data based on SNANA simulations, but
it is also possible to use observational data obtained from
spectroscopic follow-up of supernova candidates. The advantages of
using simulated data are better and more uniform coverage of the
redshift and light-curve parameter ranges that characterize the
observed data. In contrast, spectroscopic data is typically selected
with a very biased selection function and does not cover the higher
redshift ranges. The disadvantage of using simulated data is that
while the light-curve model for type Ia SN is well understood, the
number and range of light-curve templates for CC SNe is known to be
incomplete. Hence simulated training data will not cover the possible
range of CC SNe that may occur in the observed data.

SNIRF uses a number of metrics to evaluate the performance of the
algorithm, including ROC curves and the so called Precision-Recall
(PR) curves that provide measures of purity and efficiency.
SNIRF is also fully integrated with the Pippin framework [@pippin] which
runs the end-to-end cosmology analysis for the DES 5YR analysis. There
are a number of additional user-supplied options that pertain to running within
the Pippin framework.

SNIRF provides an optional plotting package based on matplotlib
[@matplotlib], allowing the user to select a variety of plots that
show the performance of the algorithm and distributions of features
and other quantities that are relevant to SN Ia cosmology. SNIRF also
makes use of various numpy [@numpy], scipy [@scipy] functions.

All of the options can be seen by running the code with the -help or
-h option.  The GitHub repository [@github] contains a "how-to.txt"
file which gives numerous examples of different ways to run the
pacakage.  Any bugs or feature requests can be opened as issues on the
[development page](https://github.com/evevkovacs/ML-SN-Classifier) [@github].

# Acknowledgements
We gratefully acknowledge the contributions of Nihar Dalal, Justin
Craig, Vicki Kuhn, Camille Liotine and Alec Lancaster who participated
in summer student programs at Argonne National Laboratory and
performed a variety of verification and validation tests on the code.

# References
