import os
import numpy as np
from operator import add
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import ML_globals as g
import re
from astropy.io.misc import fnunpickle, fnpickle
from matplotlib.backends.backend_pdf import PdfPages
import warnings
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

# Labels used for cuts and plots
Weights = 'Weights'
ConstWts = 'ConstWts'
fit_bands = ['g', 'r', 'i', 'z']
fitlabels = dict(zip(fit_bands, ['$' + f + '$' for f in fit_bands]))
colors = ['gr', 'ri', 'iz']
colorlabels = ['g-r', 'r-i', 'i-z']
Ncolors= len(colors)
SALT = 'SALT '
peak_m = 'peak_m' #multi-peak fit
peak_s = 'peak_s' #single-peak fit
peaks = [peak_m, peak_s]
peaklabels = [' (Multi-band)', ' (Single-band)']
SALTfilters = [f + p for p in peaks for f in fit_bands]
SALTmfilters = [f + peak_m for f in fit_bands]
SALTsfilters = [f + peak_s for f in fit_bands]
SALTfilterlabels = ['$' + f + '$' + p for p in peaklabels for f in fit_bands]
SALTmfilterlabels = ['$' + f + '$' + peaklabels[0] for f in fit_bands]
SALTsfilterlabels = ['$' + f + '$' + peaklabels[1] for f in fit_bands]
SALTcolors = [c + p for p in peaks for c in colors]
SALTcolorlabels = ['$' + c + '$' + p for p in peaklabels for c in colorlabels]
Delta = 'Delta'
SALTcolordiffs = [Delta + c for c in colors]
SALTcolordifflabels = ['$\Delta(' + c + ')$' for c in colorlabels]
diff = 'difference'
SALTlabels = {'c_err':'SALT $\Delta c$', 'x1_err':'SALT $\Delta x_1$', 'fit_pr':'SALT $f_p$', 'x1':'SALT $x_1$', 'c':'SALT $c$',
              't0_err':'SALT $\Delta MJD$'}

# ranges and binning
Nzbins = 28
mulo = 36.
muhi = 46.

# constants for cuts
ee = '=='
ne = '!='
_in_ = '_in_'
_notin_ = '_!in_'
ne999 = '!=-999'
ee999 = '==-999'
_finite = '_finite'
Joint = 'Joint'
Max = 'Max'
Min = 'Min'
Alt = 'Alt'
AltMax = Alt + Max
AltMin = Alt + Min
_Efficiency = '_' + g.Efficiency

######## Bazin parameter names ############
Bazin_ = g.Bazin + '_'
err = 'Err'
A = 'A'
t0 = 't0'
t_fall = 't_fall'
t_rise = 't_rise'
C = 'C'
bazinpars = [A, t0, t_fall, t_rise, C]
bazinerrs = [p + err for p in bazinpars]
bazinlabels = {A: '$A$', t0: '$t_0$', t_fall: '$t_{fall}$', t_rise: '$t_{rise}$', C: '$C$',
               A + err: '$\Delta A$', t0 + err: '$\Delta t_0$', t_fall + err: '$\Delta t_{fall}$',
               t_rise + err: '$\Delta t_{rise}$', C + err: '$\Delta C$'}
Bazincolors = [col + g.Bazin for col in colors]
Bazincolorlabels = [g.Bazin + ' $' + c + '$' for c in colorlabels]

# Bazin cuts
_all = '_all'
failfit = g.nodata
Failed = 'Failed'
Bazin_Fail = Bazin_ + Failed
Bazin_Const = Bazin_ + g.Constant

# PLOT VARIABLES ###################

# plotting globals
linear = 'linear'
log = 'log'
Large = 'Large'
Small = 'Small'

scatter = 'scatter'
contour = 'contour'
ctrlevels = [.1, .5, .9]
Title = 'Title'
Scatter = 'Scatter'
Ticks = 'Ticks'
yrescale = 1.3  # scale factor for adding margins to plots
Ncols = 2  # columns for scatter plots

#figsizes and layout
figx = 15 
figy = 7
nrow6 = 2
ncol6 = 3
nrow8 = 2
ncol8 = 4
plotsperpage = 6
plot_offset6 = nrow6*100 + ncol6*10  #2x3 layout
plot_offset8 = nrow8*100 + ncol8*10 #2x4 layout

# alt type for plot of samples with 2 sets of cuts
alt = 'alt'
altIa = alt + g.Ia
altCC = alt + g.CC
altIbc = alt + g.Ibc
altII = alt + g.II
altTotal = alt + g.Total
altRFIa = alt + g.RF + g.Ia
altRFCC = alt + g.RF + g.CC
altRFIbc = alt + g.RF + g.Ibc
altRFII = alt + g.RF + g.II

altTest = alt + g.Test
altSpec = alt + g.Spec
altSpec_nofp = alt + g.Spec_nofp
altPhot = alt + g.Phot

color = {g.Ia: 'royalblue', g.CC: 'red', g.Ibc: 'limegreen', g.II: 'red',
         g.Data: {g.Training: 'magenta', g.Test: 'black', g.Validation: 'yellow', g.Spec: 'purple', g.Spec_nofp: 'cyan', g.Phot: 'indianred',
                altTest: 'dimgrey', altSpec: 'rebeccapurple', altSpec_nofp: 'teal', altPhot: 'lightcoral'},
         g.Total: 'black', g.RFIa: 'blue', g.RFCC: 'orangered', g.RFIbc: 'green', g.RFII: 'orangered', g.RFTotal: 'black',
         contour: 'black', g.RFIaTP: 'royalblue', g.RFIaFP: 'deeppink', g.RFCCTN: 'red', g.RFCCFN: 'orange', g.RFIITN: 'red',
         g.RFIIFN: 'orange', g.RFIbcTN: 'limegreen', g.RFIbcFN: 'magenta', 
         altIa: 'lightblue', altCC: 'darkorange', altII: 'darkorange', altIbc: 'yellowgreen', altTotal: 'dimgrey',
         altRFIa: 'lightskyblue', altRFCC: 'peru', altRFII: 'peru', altRFIbc: 'y'}
fill = {g.Ia: 'stepfilled', g.CC: 'step', g.Ibc: 'step', g.II: 'step', 
        g.Data: {g.Training: 'o', g.Test: 'o', g.Validation: '8', g.Spec: 'v', g.Spec_nofp: '^', g.Phot: 'o'},
        g.Total: 'step', altIa: 'step', altCC: 'step', altIbc: 'step', altII: 'step'}
lw = {g.Ia: 1, g.CC: 2, g.Ibc: 2, g.II: 2, g.Data: 1, g.Total: 1, contour: 30}
alpha = {g.Ia: 1, g.CC: 1.0, g.Ibc: 1, g.II: 1, g.Data: 1, g.Total: 1.0, Scatter: 1.0}
markers = {g.Ia: '.', g.CC: '.', g.Ibc: '.', g.II: '.', 
           g.Data: {g.Test: 'o', g.Validation: '8', g.Spec: 'v', g.Spec_nofp: '^', g.Phot: 'o'}}
STR = ' '.join([g.Simulated, g.Training])
SV = ' '.join([g.Simulated, g.Validation])
STT = ' '.join([g.Simulated, g.Test])
STRD = ' '.join([g.Simulated, g.Training, g.Data])
SVD = ' '.join([g.Simulated, g.Validation, g.Data])
STTD = ' '.join([g.Simulated, g.Test, g.Data])
plotlabels = {g.Ia: g.SN + g.Ia, g.CC: g.SN + g.CC, g.Ibc: g.SN + g.Ibc, g.II: g.SN + g.II, g.Total: g.Total, 
              g.Training: STRD, g.Test: STTD, g.Validation: SVD,
              g.Spec: g.Spec + '. ' + g.Data, g.Spec_nofp: re.sub('_nofp', '. Data (no $f_p$ cut)', g.Spec_nofp), 
              g.Phot: g.Phot + '. ' + g.Data}
sizes = {Title: 18, Large: 16, Small: 12, Scatter: 10, Ticks: 12}

CLFlbls = {g.RF: ' (RF)'}
TFlabels = {g.TP:'TP ', g.FP:'FP ', g.TN:'TN ', g.FN:'FN '} 

# poplulate dicts with extra keys for the classifier types (RF for now)
for key in CLFlbls.keys():
    CLFlbl = CLFlbls[key]
    for t in g.allSNtypes:
        plotlabels[key + t] = g.SN + t + CLFlbl
        for tf in g.TFtypes:
            if (g.Ia in t and g.P in tf) or (g.Ia not in t and g.N in tf):
                plotlabels[key + t + tf] = TFlabels[tf] + g.SN + t + CLFlbl
        for k in [''] + g.TFtypes:
            if (g.Ia in t and g.P in k) or (g.Ia not in t and g.N in k) or k == '':
                markers[key + t + k] = markers[t]
                lw[key + t + k] = lw[t]
                lw[alt + key + t + k] = lw[t]
                alpha[key + t + k] = alpha[t]
                fill[key + t + k] = fill[t]
                fill[alt + key + t + k] = fill[t]

    plotlabels[key + g.Total] = g.Total
    alpha[key + g.Total] = alpha[g.Total]
    fill[key + g.Total] = fill[g.Total]
    lw[key + g.Total] = lw[g.Total]

# and alt type for plots with multiple cuts
for t in g.allSNtypes:
    markers[alt + t] = markers[t]
    lw[alt + t] = lw[t]
    alpha[alt + t] = alpha[t]
alpha[alt + g.Total] = alpha[g.Total]
lw[alt + g.Total] = lw[g.Total]
fill[alt + g.Total] = fill[g.Total]
lw[alt + g.Ia] = 2  # overwrite

# assemble everything into giant dict
plotdict = {'color': color, 'fill': fill, 'lw': lw, 'alpha': alpha, 'labels': plotlabels, 'markers': markers,
            'levels': ctrlevels, 'sizes': sizes}

Default_markers=['o','v','s','d','H','^','D','h','<','>','.']
Default_colors=['blue','r','m','g','navy','y','purple','gray','c','orange','violet',
                'coral','gold','orchid','maroon','tomato','sienna','chartreuse','firebrick','SteelBlue']

def bin_data(data, Nbins=None):
    Ndata = np.asarray([])
    meanX = np.asarray([])
    mask = np.isfinite(data)  # remove nan's and infs
    if (np.sum(mask) < len(data)):
        print('Warning: data truncated by elements to remove nans and infs'.format(len(data) - np.sum(mask)))
    data = data[mask]
    if (len(data) > 0):
        if (Nbins is None):
            Ndata, bins = np.histogram(data)
            print('Warning: Nbins not supplied; using default np binning Nbins='.format(Nbins))
        else:
            Ndata, bins = np.histogram(data, bins=Nbins)
        sumX, bins = np.histogram(data, bins=bins, weights=data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meanX = sumX / Ndata
    return Ndata, meanX


def getdatabins2d(xdata, ydata, xbinwidth, ybinwidth, xlimits=None, ylimits=None):
    H_cut = np.asarray([])
    x_cen = np.asarray([])
    y_cen = np.asarray([])
    if (len(xdata) > 0 and len(ydata) > 0):
        # print "2d minmax x",min(xdata),max(xdata),type(min(xdata)),type(max(xdata))
        # print "2d minmax y",min(ydata),max(ydata),type(min(ydata)),type(max(ydata))
        x_limits = [np.floor(min(xdata)), np.ceil(max(xdata))]  # ranges in data
        y_limits = [np.floor(min(ydata)), np.ceil(max(ydata))]
        if (xlimits is not None):  # adjust plot ranges as supplied
            xlimits = [max(xlimits[0], x_limits[0]), min(xlimits[1], x_limits[1])]
        if (ylimits is not None):
            ylimits = [max(ylimits[0], y_limits[0]), min(ylimits[1], y_limits[1])]
        Nxbins = (x_limits[1] - x_limits[0]) / xbinwidth  # use binwidth on full range of data
        Nybins = (y_limits[1] - y_limits[0]) / ybinwidth
        # print Nxbins, Nybins,int(Nxbins),int(Nybins)
        H, x_bins, y_bins = np.histogram2d(xdata, ydata, bins=(int(Nxbins), int(Nybins)), range=[x_limits, y_limits])
        H_frac = H / np.max(H)
        # truncate to supplied limits (remove upper x and y edge)
        index = np.ones(len(x_bins[:-1]), dtype=bool)
        indey = np.ones(len(y_bins[:-1]), dtype=bool)
        if (xlimits is not None):
            index = (x_bins[:-1] >= xlimits[0]) & (x_bins[:-1] <= xlimits[1])
        if (ylimits is not None):
            indey = (y_bins[:-1] >= ylimits[0]) & (y_bins[:-1] <= ylimits[1])
        # find truncated bin centers
        x_cen, y_cen = np.meshgrid(x_bins[:-1][index], y_bins[:-1][indey])
        x_cen += (0.5 * (x_bins[1] - x_bins[0]))
        y_cen += (0.5 * (y_bins[1] - y_bins[0]))
        H_x = H_frac[index]  # select out xrows of H according to selected x_cen
        H_cut = H_x.T[indey]  # transpose H_x, and use same trick to select y_cen
    return H_cut, x_cen, y_cen


# print and write to file
def tee(line, f):
    print('{}'.format(line))
    if (f is not None):
        f.write(line + '\n')
    return


def get_mask(alldata, varlist, varlabels, cuts={}, ops=[], not_ops=[], mask_id='mask_id',
             file_id='', debug=False):
    # usage eg. SALTcolormask=get_mask(alldata, SALTcolors, SALTcolorlabels, mask_id='SALT-color', file_id=file_id)
    # default cut is to not missing data
    mask = {}
    totals = {}
    if len(cuts) > 0:
        max_values = []
        min_values = []
        if Max not in cuts and Min not in cuts:
            print('\n  Max and Min values not supplied: skipping {} mask'.format(mask_id))
            return mask
        elif len(cuts[Max]) != len(varlist) or len(cuts[Min]) != len(varlist):
            print('\n  Incorrect number of Min/Max values for {}: skipping {} mask'.format(' '.join(varlist), mask_id))
            return mask
        else:
            max_values = cuts[Max]
            min_values = cuts[Min]
            allcuts = [(min_values[i], max_values[i]) for i in range(len(max_values))] #tuple of min and max values
            alt_vars = [(cuts[AltMin][i], cuts[AltMax][i]) for i in range(len(cuts[AltMax]))] #substitution strings for str cuts
            ops = [_in_]
            not_ops = [_notin_]
    else:
        allcuts = [g.nodata for v in varlist]  # default cut is on good data 
        ops = [ne]
        not_ops = [ee]
        
    ff = open(file_id + '_' + mask_id + '.eff', 'w')
    tee('\n  Efficiencies for {} cuts (variables={})'.format(mask_id, ', '.join(varlist)), ff)

    for op, not_op in zip(ops, not_ops): #assume op is fixed
        joint_pass = Joint + op
        joint_fail = Joint + not_op
        joint_finite = Joint + _finite
        mask[joint_pass] = {}
        mask[joint_fail] = {}
        mask[joint_finite] = {}
        for i, (var, label, cut) in enumerate(zip(varlist, varlabels, allcuts)):
            if type(cut) == tuple and all(c == '' for c in cut):
                continue                  #skip empty min-max cuts
            key_pass = var + op + str(cut)
            key_fail = var + not_op + str(cut)
            mask[key_pass] = {}
            mask[key_fail] = {}  # need separate mask due to nans in data; can't just take ~
            txtlabel = re.sub('\$', '', label)
            txtlabel = re.sub(r'\\', '', txtlabel)
            qname = '{} {} ({} {} & finite)'.format(mask_id, txtlabel, op, str(cut))
            tee('\n  {}'.format(qname), ff)
            tee('    Data        Npass  Efficiency  Nfail  Nfinite/Ntotal', ff)
            for dkey in alldata.keys():
                # setup masks for =cut and !=cut
                if (var in alldata[dkey].colnames):
                    if dkey not in mask[joint_pass].keys():    # initialize
                        mask[joint_pass][dkey] = np.ones(len(alldata[dkey][var]), dtype=bool)
                        mask[joint_finite][dkey] = np.ones(len(alldata[dkey][var]), dtype=bool)
                        mask[joint_fail][dkey] = np.zeros(len(alldata[dkey][var]), dtype=bool)
                    finite = np.isfinite(alldata[dkey][var])  #remove nans and infs
                    nfinite = np.count_nonzero(finite)
                    if dkey not in totals.keys():
                        totals[dkey] = len(alldata[dkey][var])
                    #save passes and specific failures for each cut
                    if op == ne and not_op == ee: 
                        mask[key_pass][dkey] = (alldata[dkey][var] != int(cut)) & finite
                        mask[key_fail][dkey] = (alldata[dkey][var] == int(cut))
                    elif op == _in_ and not_op == _notin_:  #min and max values
                        if type(cut[0]) != str:
                            mask[key_pass][dkey] = (alldata[dkey][var] > float(cut[0])) & finite
                            mask[key_fail][dkey] = (alldata[dkey][var] <= float(cut[0])) 
                        elif len(cut[0]) > 0:      #varname for cut relative to another feature eg. t_rise < t_fall
                            var_alt = re.sub(alt_vars[i][0], cut[0], var)
                            if var_alt in alldata[dkey].colnames:
                                mask[key_pass][dkey] = (alldata[dkey][var] > alldata[dkey][var_alt]) & finite
                                mask[key_fail][dkey] = (alldata[dkey][var] <= alldata[dkey][var_alt]) 
                            else:
                                tee('    {:10} (feature-name = {}) not available for cut on {}'.format(dkey, var_alt, var), ff)
                        else:  # skipping explicit cut
                            mask[key_pass][dkey] = finite
                            mask[key_fail][dkey] = np.zeros(len(finite), dtype=bool)
                        if type(cut[1]) != str:
                            mask[key_pass][dkey] = (alldata[dkey][var] < float(cut[1])) & mask[key_pass][dkey] # and with previuos 
                            mask[key_fail][dkey] = (alldata[dkey][var] >= float(cut[1])) | mask[key_fail][dkey]  # or with previous
                        elif len(cut[1]) > 0:     #varname for cut relative to another feature eg. t_rise < t_fall
                            var_alt = re.sub(alt_vars[i][1], cut[1], var)
                            if var_alt in alldata[dkey].colnames:
                                mask[key_pass][dkey] = (alldata[dkey][var] < alldata[dkey][var_alt]) & mask[key_pass][dkey]
                                mask[key_fail][dkey] = (alldata[dkey][var] >= alldata[dkey][var_alt]) | mask[key_fail][dkey]
                            else:
                                tee('    {:10} (feature-name = {}) not available for cut on {}'.format(dkey, var_alt, var), ff)
                    if debug and dkey in mask[key_pass] and np.count_nonzero(mask[key_pass][dkey]) > 0:
                        print('    Check limits: min/max {} data = {:.4g}, {:.4g}; with cut(s) = {}: min/max = {:.4g}, {:.4g}'.format(dkey,
                                   np.min(alldata[dkey][var]), np.max(alldata[dkey][var]), cut,           
                                   np.min(alldata[dkey][var][mask[key_pass][dkey]]),
                                   np.max(alldata[dkey][var][mask[key_pass][dkey]])))
                    mask[joint_finite][dkey] = mask[joint_finite][dkey] & finite    #joint finite mask
                    if dkey in mask[key_pass] and dkey in mask[key_fail]:
                        mask[joint_pass][dkey] = mask[joint_pass][dkey] & mask[key_pass][dkey]    # joint mask passes all
                        mask[joint_fail][dkey] = mask[joint_fail][dkey] | mask[key_fail][dkey] # joint fail is or of not pass
                        # now add finite mask to fail to remove nan's and infs from count
                        mask[key_fail][dkey] = mask[key_fail][dkey] & finite
                    pass_effcy = float(np.count_nonzero(mask[key_pass][dkey]))/totals[dkey]
                    tee('    {:10} {:6} {:10.3g} {:6} {:6}/{:6}'.format(dkey, np.count_nonzero(mask[key_pass][dkey]),
                                                                        pass_effcy, np.count_nonzero(mask[key_fail][dkey]),
                                                                        nfinite, totals[dkey]), ff)
                else:
                    tee('    {:10} (feature-name = {}) not available'.format(dkey, var), ff)

        # finalize and print joint mask
        tee('\n  {} ({}) {} cuts'.format(Joint, ' & '.join(varlist), mask_id), ff)
        tee('    Data        Npass  Efficiency  Nfail  Nfinite/Ntotal', ff)
        for dkey in mask[joint_pass].keys():
            # add joint_finite mask to joint_fail to remove nan's and infs from count
            mask[joint_fail][dkey] = mask[joint_fail][dkey] & mask[joint_finite][dkey]
            pass_effcy = float(np.count_nonzero(mask[joint_pass][dkey]))/totals[dkey]
            tee('    {:10} {:6} {:10.3g} {:6} {:6}/{:6}'.format(dkey, np.count_nonzero(mask[joint_pass][dkey]),
                                                                pass_effcy, np.count_nonzero(mask[joint_fail][dkey]),
                                                                np.count_nonzero(mask[joint_finite][dkey]),
                                                                totals[dkey]), ff)
    ff.close()
    
    return mask


def bazin(t, A, t0, t_fall, t_rise, C):
    # print "called with",A,t0,t_fall,t_rise,C
    dt = t - t0
    if A == 0:
        bn = constantC(t, C)
    else:
        bn = A * (np.exp(-dt / t_fall) / (1 + np.exp(-dt / t_rise))) + C
    return bn


def constantC(t, C):
    cc = np.zeros(len(t))
    cc[:] = C
    return cc

def truncate_title(dkey, title):
    subtitle = '+ ' + plotlabels[dkey]
    if subtitle in title:
        title = re.sub('\\' + subtitle, '', title)
        print(title)
    return title

def plot_types(f, types, xvar, alldata, plotlist=[''], masks={}, weights=True, totals=False, 
               xlabel='', ylabel='', cuts={}, plotdict={}, xscale=linear, yscale=linear, 
               title='', bins=None, Nbins=10, asize=Large, yvar='', datapoints=True, type_data=g.Total, 
               minmax=False, ctrxbin=0., ctrybin=0., alt='', addlabel='', plotid='', debug=False):
    """
    Loop over selected datasets and plot xvar [yvar]
    Can plot entire data set or loop over specified types
    alldata - dict of astropy Tables (columns = features)labeled by dataset
    plotlist - [[bytypelist], [datalist]]; 
               bytypelist = list of datasets to be plotted by type
               datalist = list of datasets to be plotted for a given type selection (type_data) (default=Total)
    masks - dict of masks to select various types (Ia, RFIa, etc.)
    cuts - dict of masks to select rows of alldata (apply cuts to column data)
    """
    color = plotdict['color']
    fill = plotdict['fill']
    lw = plotdict['lw']
    alpha = plotdict['alpha']
    sizes = plotdict['sizes']
    lgndlabels = plotdict['labels']
    if len(xlabel) > 0:
        f.set_xlabel(xlabel, size=sizes[asize])
    if len(ylabel) > 0:
        f.set_ylabel(ylabel, size=sizes[asize])
    if (len(yvar) > 0):
        plt.tick_params(axis='both', which='major', labelsize=sizes[Ticks])
        yblurb = '-' + yvar
    else:
        yblurb = ''

    # TODO option for uniform binning
    if (bins is None):
        bins = Nbins

    # determine what keys need to be plotted
    datalist = []
    bytypelist = []
    if (type(plotlist) == list and len(plotlist) > 0):
        bytypelist = plotlist[0] if type(plotlist[0]) == list else [plotlist[0]]
        if (len(plotlist) > 1):
            datalist = plotlist[-1] if type(plotlist[-1]) == list else [plotlist[-1]]
    else:
        print('Invalid plotlist: supply list of dict keys to be plotted:' +\
              '[g.Test, g.Spec] or [[bytypelist], [datalist]]')

    if (len(datalist) > 0):  # plot data
        for data in datalist:
            # print xvar,data,(xvar in alldata[data][type_data].keys())
            if xvar not in alldata[data].colnames or (len(yvar) > 0 and yvar not in alldata[data].colnames):  # make sure data has required variable
                blurb = 'and/or {} '.format(yvar) if len(yvar) > 0 else ''
                print("    {} {}not available in {} data".format(xvar, blurb, data))
                title = truncate_title(data, title)
            else:
                # print 'plotting',var,'in',data
                datasetx = {}
                datasety = {}
                if type_data != g.Total and data in masks.keys() and type_data in masks[data].keys():     # plot selected subsample by type of data
                    mask_this = masks[data][type_data]
                else:
                    mask_this = np.ones(len(alldata[data][xvar]), dtype=bool)
                    if type_data != g.Total:                                     # reset if failed to find selected type
                        print('    {} type not available in {} data: defaulting to Total sample'.format(type_data, data))
                        type_data = g.Total
                if len(cuts) == 0:
                    datasetx = alldata[data][xvar][mask_this]
                    if len(yvar) > 0:
                        datasety = alldata[data][yvar][mask_this]
                elif (data in cuts.keys()):
                    if np.count_nonzero(cuts[data]) > 0:     # for 2-d plots 
                        mask_this = mask_this & cuts[data]
                        datasetx = alldata[data][xvar][mask_this]
                        if len(yvar) > 0:
                            datasety = alldata[data][yvar][mask_this]
                    else:
                        print("    Skipping {} data {} {}{} plot: no data passing cut".format(data, type_data, xvar, yblurb))
                        title = truncate_title(data, title)
                else:
                    print("    Skipping {} data {} {}{} plot: no entry in cuts dict".format(data, type_data, xvar, yblurb))
                    title = truncate_title(data, title)
                if (len(datasetx) > 0):
                    if debug:
                        print('    Plotting {} {} {} data points'.format(len(datasetx), dat, xvar))
                    if (len(yvar) == 0):
                        if (minmax):
                            print('    Total {} data: {}; min= {:.4g}, max = {:.4g}'.format(data, xvar, np.min(datasetx),
                                                                                np.max(datasetx)))
                        if not datapoints   :  #histogram option
                            f.hist(datasetx, bins=bins, color=color[g.Data][data], lw=lw[g.Data])
                        else:                  #data-point option
                            Ndata, meanX = bin_data(datasetx, Nbins=bins)
                            # print Ndata,meanX,np.sqrt(Ndata)
                            f.errorbar(meanX, Ndata, yerr=np.sqrt(Ndata),
                                     label=' '.join([type_data, lgndlabels[data], addlabel]),
                                     color=color[g.Data][alt + data], fmt=fill[g.Data][data])
                    elif (len(datasety) > 0):
                        if (minmax):
                            print('    Total {} data: {}; min= {:.4g}, max = {:.4g}; {}; min= {:.4g}, max = {:.4g}'.format(data, xvar,
                                  np.min(datasetx), np.max(datasetx), yvar, np.min(datasety), np.max(datasety)))
                        # print "Scattersize",sizes[Scatter]
                        f.scatter(datasetx, datasety, marker=markers[g.Data][data], alpha=alpha[Scatter],
                                    s=sizes[Scatter],
                                    color=color[g.Data][alt + data],
                                    label=' '.join([type_data, lgndlabels[g.Data][data], addlabel]))
                    else:
                        print("    Skipping {} data {} {}{} scatter plot: no data passing cut".format(data, type_data, xvar, yblurb))

    if len(bytypelist) > 0:  #plot by type
        for dat in bytypelist:
            datdict = alldata[dat] if dat in list(alldata.keys()) else {}
            if len(yvar) > 0 and weights:
                weights = False
                print ('    Ignoring weights for 2-d plot')
            if len(datdict) == 0:
                print('    Skipping {} data {} plot: sample not found'.format(dat, plotid))
            else:
                sweights = datdict[g.Weights] if (weights and g.Weights in datdict.colnames) else None
                for t in types:
                    if g.Total in t or (t in masks[dat].keys() and np.count_nonzero(masks[dat][t])):
                        datax = {}
                        datay = {}  # initialize to empty
                        if len(cuts) == 0:
                            mask_this = np.ones(len(datdict[xvar]), dtype=bool) if g.Total in t else masks[dat][t] 
                            datax = datdict[xvar][mask_this]
                            if len(yvar) > 0:
                                datay = datdict[yvar][mask_this]
                        elif (dat in cuts.keys()):
                            if np.count_nonzero(cuts[dat]) > 0:
                                mask_this = cuts[dat] if g.Total in t else masks[dat][t] & cuts[dat]  # mask for type and cuts
                                datax = datdict[xvar][mask_this]
                                if len(yvar) > 0:
                                    datay = datdict[yvar][mask_this]
                            else:
                                print("    Skipping {} data {} {}{} plot: no data passing cut".format(dat, t, xvar, yblurb))
                        else:
                            print("    Skipping {} data {} {}{} plot: no entry in cuts dict".format(dat, t, xvar, yblurb))

                        if len(datax) > 0 and len(yvar) == 0:  # 1d plot
                            swts = sweights[mask_this] if sweights is not None else None  #set weights for 1-d plot
                            if (minmax):
                                print('    {} {} data: {}; min= {:.4g}, max = {:.4g}'.format(t, dat, xvar, np.min(datax),
                                                                                 np.max(datax)))
                            f.hist(datax, bins=bins, label=lgndlabels[t] + addlabel, color=color[alt + t],
                                     histtype=fill[alt + t], lw=lw[alt + t], weights=swts)
                            if (debug):
                                print('    Plotting {} {} {} data points for type {}'.format(len(datax), dat, xvar, t))
                        elif (len(datax) > 0 and len(datay) > 0):  # 2d plot of types excluding any totals
                            if (minmax):
                                print('    {} {} data: {}; min= {:.4g}, max = {:.4g}; {}; min= {:.4g}, max = {:.4g}'.format(t,                                                                                            dat, xvar, np.min(datax), np.max(datax), yvar, np.min(datay), np.max(datay)))
                            f.scatter(datax, datay, marker=markers[alt + t], alpha=alpha[Scatter],
                                            s=sizes[Scatter], color=color[alt + t], label=lgndlabels[t] + addlabel)
                            # contours for Ia only
                            if ctrxbin > 0. and ctrybin > 0. and t.find(g.Ia) != -1 and t.find('FPIa') == -1:
                                H_cut, x_cen, y_cen = getdatabins2d(datax, datay, ctrxbin, ctrybin)
                                c1 = f.contour(x_cen, y_cen, H_cut, levels=plotdict['levels'],
                                                 colors=color[contour])
                                f.clabel(c1, inline=1, fontsize=8)
                                c1.collections[0].set_label(lgndlabels[contour])
                    else:
                        print('    Skipping {} type {} {} plot; {}{}: type not available'.format(dat, t, plotid, xvar, yblurb))

    if (len(title) > 0):
        # print "title:",title
        f.set_title(title, size=sizes[Title])

    # set scales after doing plots to avoid ValueErrors for log scale
    f.set_yscale(yscale)
    f.set_xscale(xscale)

    return


def get_EffPur(data, cuts, colname='fit_pr', target_class=0, alltypes_colname='type3'):
    # compute efficiency, purity for different cuts for supplied target class
    C_Eff = []
    C_Pur = []
    CIa = data[alltypes_colname] == target_class
    for D in cuts:
        cut = data[colname] > D
        Eff, Pur, Rem = g.EffPur(CIa, cut)
        C_Eff.append(Eff)
        C_Pur.append(Pur)
        
    return C_Eff, C_Pur

def plot_probabilities(fig, dkey, alldata, MLtypes, type_masks, performance, target_class=0, 
                       colname='fit_pr', debug=False, alltypes_colname='type3', y_lo=.75,
                       p_binwidth =0.05, p_min=-0.1, p_max=1.05, plot_offset=plot_offset6): 
    
    p_bins = np.arange(p_min, p_max, p_binwidth)
    RFprobability = g.RFprobability + str(target_class)

    f = fig.add_subplot(plot_offset + 1)
    plot_types(f, MLtypes, RFprobability, alldata, plotlist=[[dkey], []], masks=type_masks, 
               xlabel='Random Forest SNIa Probability', ylabel='Number',
               plotdict=plotdict, bins=p_bins, asize=Small, weights=False,
               title='{}'.format(plotlabels[dkey]))
    f.set_xlim(0, 1.0)
    f.legend(loc='upper center', fontsize='small', numpoints=1)
               
    f = fig.add_subplot(plot_offset + 2)
    plot_types(f, MLtypes, RFprobability, alldata, plotlist=[[dkey], []], masks=type_masks, 
               xlabel='Random Forest SNIa Probability', ylabel='Number',
               plotdict=plotdict, bins=p_bins, asize=Small, yscale=log, weights=False,
               title='{}'.format(plotlabels[dkey]))
    f.set_xlim(0, 1.0)
    # plt.yscale('log', nonposy='clip')
    f.legend(loc='upper center', fontsize='small', numpoints=1)
    
    thresh = performance[g.Threshold]
    purity = performance[g.Purity]
    efficiency = performance[g.Efficiency]
    f = fig.add_subplot(plot_offset + 3)
    f.plot(thresh, efficiency[:-1], 'b--', label='efficiency', lw=2)
    f.plot(thresh, purity[:-1], c='r', label='purity', lw=2)
    f.set_xlabel('Threshold Probability for Classification, $P_{t}$')
    f.set_ylabel('SNIa Efficiency, Purity [$P_{Ia} \ge P_{t}$]')
    f.legend(loc='lower right', fontsize='small')
    f.set_ylim(y_lo, 1.0)
    f.set_xlim(0, 1.0)
    f.set_title('{}'.format(plotlabels[dkey]), size=sizes[Title])
    
    ax1 = fig.add_subplot(plot_offset + 4)
    fpr = performance[g.ROC][g.FPR]
    tpr = performance[g.ROC][g.TPR]
    ax1.plot(fpr, tpr, lw=2, label='ROC curve (area = {:0.2f})'.format(performance[g.ROC][g.AUC]))
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc='lower right', fontsize='small')
    ax1.set_title('{}'.format(plotlabels[dkey]), size=sizes[Title])

    meanpr = performance.get(g.Mean_Pr)
    fraction_true = performance.get(g.Fr_True)
    if meanpr is not None:
        f = fig.add_subplot(plot_offset + 5)
        f.plot(meanpr, fraction_true, label='Random Forest', marker='o', color='magenta')
        f.plot(meanpr, meanpr, label='Perfectly Calibrated', linestyle='--', color='black')
        plt.title(plotlabels[dkey])
        f.set_xlabel('Random Forest SNIa Probability')
        f.set_ylabel('Fraction of True SNIa')
        f.legend(loc='best', fontsize='small', numpoints=1)
        f.set_title('{}'.format(plotlabels[dkey]), size=sizes[Title])

    if colname in alldata[dkey].colnames:
        # get efficiency and purity for chosen data set
        C_Dcut = np.arange(0.0, 1.0, 0.005)
        C_Eff, C_Pur = get_EffPur(alldata[dkey], C_Dcut, colname=colname, alltypes_colname=alltypes_colname)
        f = fig.add_subplot(plot_offset + 6)
        f.plot(C_Dcut, C_Eff, 'b--', label='efficiency', lw=2)
        f.plot(C_Dcut, C_Pur, c='r', label='purity', lw=2)
        f.set_xlabel('$P_{SALT}$')
        f.set_ylabel('Efficiency, Purity [$P \ge P_{SALT}$]')
        f.legend(loc='best', fontsize='small', numpoints=1)
        f.set_ylim(0, 1.0)
        f.set_xlim(0, 1.0)
        f.set_title('{}'.format(plotlabels[dkey]), size=sizes[Title])
    else:
        print('  Skipping Eff-Pur plot for {} data: {} not available'.format(dkey, colname)) 

    return

def plot_purity_vs_effcy(fig, performance, plot_offset=110, eff_lo=.8, pur_lo=.925):


    f = fig.add_subplot(plot_offset + 1)
    mark = iter(Default_markers)
    color = iter(Default_colors)
    for dkey in performance.keys():
        # plot points for classification methods
        classification_keys = [k for k in performance[dkey].keys() if g.P_Threshold in k]
        col = next(color)
        mk = next(mark)
        effs = [float(re.split(g.P_Threshold + '_', k)[-1]) for k in classification_keys]
        purs = [performance[dkey][re.sub(g.P_Threshold, g.Score, k)] for k in classification_keys]
        if len(purs) > 0 and len(effs) > 0:
            pur_lo = min(min(purs), pur_lo)
            eff_lo = min(min(effs), eff_lo)
            f.plot(effs, purs, marker=next(mark), c=col, label=dkey, ls='')
        maxprob_keys = [k for k in performance[dkey].keys() if g.MaxProb in k]
        if len(maxprob_keys) > 1:
            f.plot(performance[dkey][g.Efficiency_MaxProb], performance[dkey][g.Purity_MaxProb],
                   c=col, label=' '.join([dkey, g.MaxProb]), marker=next(mark))
            pur_lo = min(performance[dkey][g.Purity_MaxProb], pur_lo) 
            eff_lo = min(performance[dkey][g.Efficiency_MaxProb], eff_lo)

    f.set_xlabel('Efficiency')
    f.set_ylabel('Purity')
    f.legend(loc='lower left', fontsize='small', ncol=2)
    f.set_ylim(pur_lo - 0.025, 1.01)
    f.set_xlim(eff_lo - 0.025, 1.01)    
    
    return

               
def plot_template_statistics(template_info, MLtypes, npage, subpage, multiPdf, CLFid=g.RF, classification_label='',
                             plot_offset=plot_offset6):

    template_dict = template_info[g.Counts]
    template_stats = template_info[g.Stats]
    Iatemplate_dict = template_info[g.Ia]
    
    fig = plt.figure(figsize=(figx, figy))
    ax = fig.add_subplot(211)
    ax2 = ax.twinx()
    templates = sorted(template_dict.keys())  # sorted list of templates
    index = np.arange(len(templates))  # length of dict
    TMs = {}
    TM0 = {}
    tick_colors = [color[template_stats[tmpl]['Type']] for tmpl in templates]
    # setup entries for bar charts for each type
    # Ias are separated so different axis can be used
    for t in MLtypes:
        TM0[t] = [0] * len(templates)  # initialize all zero entries
        TM0[t][0] = template_stats[0][CLFid + t]  # overwrite value for Ias
        TMs[t] = [template_stats[tmpl][CLFid + t] for tmpl in templates]  # number of each type per template
        TMs[t][0] = 0  # overwrite entry for Ias to zero

    rects1 = ax2.bar(index, TMs[g.Ia], .3, color=color[g.Ia], label='Typed as Ia')
    rects3 = ax.bar(index, TM0[g.Ia], .3, color=color[g.Ia])
    bottom2 = TMs[g.Ia]
    bottom = TM0[g.Ia]
    nonIas = [typ for typ in MLtypes if not (typ == g.Ia)]
    for nonIa in nonIas:
        rects2 = ax2.bar(index, TMs[nonIa], .3, bottom=bottom2, color=color[nonIa], label='Typed as ' + nonIa)
        bottom2 = map(add, bottom2, TMs[nonIa])
        rects4 = ax.bar(index, TM0[nonIa], .3, bottom=bottom, color=color[nonIa])
        bottom = map(add, bottom, TM0[nonIa])

    plt.xlabel('Template')
    ax.set_ylabel('Number of Ia (template = 0)')
    ax2.set_ylabel('Number of CC (template != 0)')
    plt.title('Classification per template: {}'.format(classification_label), size=sizes[Title])
    plt.xticks(index + .1, templates)
    ax.set_xlim(0, max(index) + 1)
    for xtick, col in zip(ax.get_xticklabels(), tick_colors):
        xtick.set_color(col)
    plt.legend(loc='best')
    # plt.legend((rects1[0],rects2[0]),('Typed as Ia','Typed as CC'))

    ax = fig.add_subplot(212)
    templates = sorted(template_dict.keys())
    fracIa = [float(Iatemplate_dict[tmpl]) / float(template_dict[tmpl]) if (tmpl in Iatemplate_dict.keys()) else 0 for tmpl
              in templates]
    index = np.arange(len(templates))
    rects5 = plt.bar(index, fracIa, .3, color=color[g.Ia], label='Typed as Ia')
    plt.xlabel('Template')
    plt.ylabel('Ia Fraction')
    plt.title('Ia Fraction per template: {}'.format(classification_label), size=sizes[Title])
    plt.xticks(index + .1, templates)
    for xtick, col in zip(ax.get_xticklabels(), tick_colors):
        xtick.set_color(col)
    ax.set_xlim(0, max(index) + 1)
    plt.legend(loc='best')

    subpage = close_page(fig, multiPdf, npage, subpage)

    return subpage


def plot_SALT(fig, plotlist, alldata, types, type_masks, nplot=0, lgnd_title='',
                Nbins=20, x1_min=-5.0, x1_max=5.0, c_min=-0.6, c_max=0.6,
                p_binwidth =0.05, p_min=-0.1, p_max=1.05, plot_offset=plot_offset6, debug=False):
    """
    Plot simulation versus data
    """
    
    x1bins = np.linspace(x1_min, x1_max, Nbins + 1)
    cbins = np.linspace(c_min, c_max, Nbins + 1)
    # get title from plotlist
    title = get_plotlist_title(plotlist)

    nplot += 1
    f = fig.add_subplot(plot_offset + nplot)
    plot_types(f, types, 'x1', alldata, plotlist=plotlist, masks=type_masks, xlabel=SALTlabels['x1'], ylabel='Number', plotdict=plotdict,
               bins=x1bins, title=title)
    f.set_xlim(x1_min, x1_max)
    f.legend(loc='upper center', fontsize='small', numpoints=1, title=lgnd_title)
    f.set_title(title, size=sizes[Title])
    
    nplot += 1
    f = fig.add_subplot(plot_offset + nplot)
    plot_types(f, types, 'c', alldata, plotlist=plotlist, masks=type_masks, xlabel=SALTlabels['c'], ylabel='Number', plotdict=plotdict, bins=cbins,
               title=title)
    f.set_xlim(c_min, c_max)
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    axes.set_ylim(ymax=ymax * yrescale)
    f.legend(loc='best', fontsize='small', numpoints=1, title=lgnd_title)
    f.set_title(title, size=sizes[Title])
    
    nplot += 1
    f = fig.add_subplot(plot_offset + nplot)
    p_bins = np.arange(p_min, p_max, p_binwidth)
    plot_types(f, types, 'fit_pr', alldata, plotlist=plotlist, masks=type_masks, xlabel=SALTlabels['fit_pr'], ylabel='Number', plotdict=plotdict, bins=p_bins,
               title=title)
    f.set_xlim(0, 1.0)
    f.legend(loc='best', fontsize='small', numpoints=1, title=lgnd_title)
    f.set_title(title, size=sizes[Title])
    
    return nplot


def plot_scatter_bytype(fig, dkey, alldata, types, type_masks, nplot=0, lgnd_title='',
                        xvar='x1', yvar='c', contour_label='', ctrxbin=0., ctrybin=0., Nbins=40, cuts={}, plotid='x1_c_scatter',
                        x_min=-5.25, x_max=5.25, y_min=-0.9, y_max=0.65, xlabel=SALTlabels['x1'], ylabel=SALTlabels['c'],
                        ellipse=False, ell_x=8.6, ell_y=0.76, plot_offset=plot_offset6):

    if xvar in alldata[dkey].colnames and yvar in alldata[dkey].colnames:    #check for columns in alldata
        if len(contour_label) > 0:   #setup contour
            ctrxbinwidth = ctrxbin if ctrxbin > 0. else (x_max - x_min)/Nbins
            ctrybinwidth = ctrybin if ctrybin > 0. else (y_max - y_min)/Nbins
        else:
            ctrxbinwidth = 0.
            ctrybinwidth = 0.
        title = '{}'.format(plotlabels[dkey])
    
        nplot += 1
        f = fig.add_subplot(plot_offset + nplot)
        plotdict['labels'][contour] = '{} Density'.format(contour_label)
        plot_types(f, types, xvar, alldata, plotlist=[[dkey], []], masks=type_masks, yvar=yvar, xlabel=xlabel, ylabel=ylabel, cuts=cuts,
                   plotdict=plotdict, ctrxbin=ctrxbinwidth, ctrybin=ctrybinwidth, title=title, weights=False, plotid=plotid)
        f.set_xlim(x_min, x_max)
        f.set_ylim(y_min, y_max)
        f.legend(loc='lower left', fontsize='small', scatterpoints=1, ncol=Ncols, title=lgnd_title)
        f.set_title(title, size=sizes[Title])
        if ellipse:
            ellipse_this = Ellipse(xy=(0,0), width=ell_x, height=ell_y, edgecolor='red', fc='None', lw=2, ls='dashed')
            f.add_patch(ellipse_this)
                                          
    return nplot
                                          
def plot_hubble(fig, plotlist, alldata, types, type_mask, nplot=0, lgnd_title='',
                type_data=g.Total, minmax=False, debug=False,
                HR_min=-3., HR_max=3., HR_width=0.2, plot_offset=plot_offset6):
    """
    Plot simulation versus data
    """
    HRbins = bins = np.arange(HR_min, HR_max, HR_width)
    title = get_plotlist_title(plotlist)

    nplot += 1
    # use z-range from simulation (1st element of plotlist[0])
    zhi = g.zhi[plotlist[0][0]]
    zbins = np.linspace(g.zlo, zhi, Nzbins + 1)
    f = fig.add_subplot(plot_offset + nplot)
    plot_types(f, types, 'z', alldata, plotlist=plotlist, masks=type_mask, xlabel='Redshift', ylabel='Number', plotdict=plotdict,
               bins=zbins, type_data=type_data, title=title)
    f.set_xlim(g.zlo, zhi)
    f.legend(loc='best', fontsize='small', numpoints=1, title=lgnd_title)
    f.set_title(title, size=sizes[Title])
    
    nplot += 1
    f = fig.add_subplot(plot_offset + nplot) #HR linear
    plot_types(f, types, g.HR, alldata, plotlist=plotlist, masks=type_mask, xlabel='Hubble Residual', ylabel='Number',
               plotdict=plotdict, bins=HRbins, title=title, type_data=type_data)
    plt.gca().set_ylim(bottom=1)
    f.legend(loc='upper left', fontsize='small', numpoints=1, title=lgnd_title)
    f.set_title(title, size=sizes[Title])
    
    nplot += 1
    f = fig.add_subplot(plot_offset + nplot)  # HR log
    plot_types(f, types, g.HR, alldata, plotlist=plotlist, masks=type_mask, xlabel='Hubble Residual', ylabel='Number',
               plotdict=plotdict, bins=HRbins, yscale=log, title=title, type_data=type_data)
    plt.gca().set_ylim(bottom=0.1)
    f.legend(loc='upper left', fontsize='small', numpoints=1, title=lgnd_title)

    return nplot


def plot_HD(fig, dkey, alldata, types, type_mask, nplot=0, lgnd_title='', plot_offset=plot_offset6):
    
    # Scatter-plot Hubble Diagram
    rtypes = types[::-1] #reverse so Ias are in top
    title =  '{}'.format(plotlabels[dkey])

    nplot += 1
    zhi = g.zhi[dkey] if dkey == g.Training or dkey == g.Test or dkey == g.Validation else g.zhi[g.Data] 
    zbins = np.linspace(g.zlo, zhi, Nzbins + 1)
    f = fig.add_subplot(plot_offset + nplot)
    plot_types(f, types, 'z', alldata, plotlist=[[dkey], []], masks=type_mask, yvar='mu', xlabel='Redshift', ylabel='$\mu$', bins=zbins,
                   plotdict=plotdict, weights=False, title=title)
    f.legend(loc='lower right', fontsize='small', numpoints=1, scatterpoints=1, ncol=1, title=lgnd_title)
    f.set_xlim(g.zlo, zhi)
    f.set_ylim(mulo, muhi)
    
    return nplot


def plot_errors(fig, plotlist, alldata, types, type_masks, nplot=0, lgnd_title='', minmax=False, debug=False,
                x1_err_max=4.0, c_err_max=0.3, t0_err_max=10.0, Nbins=15, plot_offset=plot_offset6):

    x1ebins = np.linspace(0., x1_err_max, Nbins + 1)
    t0ebins = np.linspace(0., t0_err_max, Nbins + 1)
    cebins = np.linspace(0., c_err_max, Nbins + 1)
    title = get_plotlist_title(plotlist)
    
    nplot +=1
    f = fig.add_subplot(plot_offset + nplot)
    plot_types(f, types, 't0_err', alldata, plotlist=plotlist, masks=type_masks, xlabel='$\Delta t_0$', ylabel='Number', plotdict=plotdict,
               bins=t0ebins, title=title)
    f.legend(loc='best', fontsize='small', numpoints=1, title=lgnd_title)

    nplot +=1
    f = fig.add_subplot(plot_offset + nplot)
    plot_types(f, types, 'x1_err', alldata, plotlist=plotlist, masks=type_masks, xlabel='$\Delta x_1$', ylabel='Number', plotdict=plotdict,
               bins=x1ebins, title=title)
    f.legend(loc='best', fontsize='small', numpoints=1, title=lgnd_title)

    nplot +=1
    f = fig.add_subplot(plot_offset + nplot)
    plot_types(f, types, 'c_err', alldata, plotlist=plotlist, masks=type_masks, xlabel='$\Delta c$', ylabel='Number', plotdict=plotdict,
               bins=cebins, title=title)
    f.legend(loc='best', fontsize='small', numpoints=1, title=lgnd_title)
    
    return nplot


def plot_magnitudes(ax, plotlist, alldata, types, type_masks, nplot=0, lgnd_title='', cuts={}, Pass=ne999, minmax=False,
                    binhi=[32., 28., 26., 26.], binlo=[14., 14., 14., 14.], Nbins = 40, plot_offset=plot_offset8,
                    debug=False):
    # 8 plots per page, multi- and singlr-band filters 
    title = get_plotlist_title(plotlist)

    for i, (filts, labels) in enumerate(zip([SALTmfilters, SALTsfilters], [SALTmfilterlabels, SALTsfilterlabels])):
        for j, (filt, lo, hi, label) in enumerate(zip(filts, binlo, binhi, labels)):
            ylabel = '$N$'
            xlabel = SALT + label
            title_this = title if i==0 else ''
            nplot +=1
            mbins = np.linspace(lo, hi, Nbins + 1)
            key_pass = filt + Pass
            pltcuts = cuts[key_pass]
            plot_types(ax[i, j], types, filt, alldata, plotlist=plotlist, masks=type_masks, xlabel=xlabel, ylabel=ylabel, 
                       cuts=pltcuts, plotdict=plotdict, plotid=SALT + filt + Pass, bins=mbins, 
                       title=title_this, addlabel=' Pass ' + label, minmax=minmax, debug=debug)
            ax[i, j].legend(loc='upper left', fontsize='small', numpoints=1, title=lgnd_title)

    return nplot


def plot_SALTcolors(fig, plotlist, alldata, types, type_masks, nplot=0, lgnd_title='', cuts={}, Pass=ne999, minmax=False,
                    debug=False,
                    binlo=[-7., -2., -1., -2., -2., -2.], binhi=[7., 4., 1., 2., 2., 2.], NCbins=40, plot_offset=plot_offset6):        

    title = get_plotlist_title(plotlist)
    # 3 plots
    for col, lo, hi, label in zip(SALTcolors, binlo, binhi, SALTcolorlabels):
        ylabel = '$N$'
        xlabel = SALT + label
        nplot += 1
        f = fig.add_subplot(plot_offset + nplot)
        cbins = np.linspace(lo, hi, NCbins + 1)
        pltcuts = cuts[col + Pass] if len(cuts) > 0 else {}
        #addlabel probably not needed
        plot_types(f, types, col, alldata, plotlist=plotlist, masks=type_masks, xlabel=xlabel, ylabel=ylabel, cuts=pltcuts,
                   plotdict=plotdict, plotid=SALT + col, bins=cbins, title=title, minmax=minmax, debug=debug)
        f.legend(loc='best', fontsize='small', numpoints=1, title=lgnd_title)

    return nplot


def plot_SALTcolordiffs(fig, plotlist, alldata, types, type_masks, nplot=0, lgnd_title='', cuts={},
                        Pass=ne999, joint_Pass=ne, joint_Fail=ee, minmax=False, debug=False,
                        binlo = [-4., -2., -1.5], binhi = [4., 3., 2.], NCbins=40, plot_offset=plot_offset6):

    title = get_plotlist_title(plotlist)
    # 3 plots
    ylabel = '$N$'
    for col, lo, hi, label in zip(SALTcolordiffs, binlo, binhi, SALTcolordifflabels):
        xlabel = SALT + label
        nplot += 1
        f = fig.add_subplot(plot_offset + nplot)
        cbins = np.linspace(lo, hi, NCbins + 1)
        pltcuts = cuts[col + Pass] if len(cuts) > 0 else {}
        #addlabel probably not needed
        plot_types(f, types, col, alldata, plotlist=plotlist, masks=type_masks, xlabel=xlabel, ylabel=ylabel, cuts=pltcuts,
                   plotdict=plotdict, plotid=SALT + col + 'diff', bins=cbins, title=title, minmax=minmax, debug=debug)
        # f.set_xlim(-5.0, 5.0)
        f.legend(loc='best', fontsize='small', numpoints=1, title=lgnd_title)

    if nplot != 3:
        print('Unexpected plot count: check pagination')

    # use joint mask for other variables
    xlabel = 'Redshift'
    zhi = g.zhi[plotlist[0][0]]   #sim
    for col, lo, hi, label in zip(SALTcolordiffs, binlo, binhi, SALTcolordifflabels):
        nplot += 1
        f = fig.add_subplot(plot_offset + nplot)
        zbins = np.linspace(g.zlo, zhi, Nzbins + 1)
        pltcuts = cuts[Joint + joint_Pass] if len(cuts) > 0 else {}
        plot_types(f, types, 'z', alldata, plotlist=plotlist, masks=type_masks, xlabel=xlabel, ylabel=ylabel, cuts=pltcuts,
                   plotdict=plotdict, plotid='_'.join([SALT, col, diff, Joint, 'Pass']), bins=zbins, addlabel=' Pass ' + label, title=title)
        pltcuts = cuts[Joint + joint_Fail] if len(cuts) > 0 else {}
        plot_types(f, types, 'z', alldata, plotlist=plotlist, masks=type_masks, xlabel=xlabel, ylabel=ylabel, cuts=pltcuts,
                   plotdict=plotdict, plotid='_'.join([SALT, col, diff, Joint, 'Fail']), bins=zbins, alt=alt, addlabel=' Fail ' + label, title=title)
        axes = plt.gca()
        f.set_ylim(0., axes.get_ylim()[1] * 1.5)
        f.legend(loc='best', numpoints=1, prop={'size': 8}, scatterpoints=1, ncol=Ncols, title=lgnd_title)


    return nplot


def get_plotlist_title(plotlist):
    datsets = plotlist[0] + plotlist[1] if len(plotlist) >1 else plotlist[0]
    title = '{}'.format(' + '.join([plotlabels[lbl] for lbl in datsets]))    

    return title

def get_valid_keys(alldata, varlist):  #check that alldata has required columns
    
    valid_keys = []
    for dkey in alldata.keys():
        for var in varlist:
            if var in alldata[dkey].colnames and dkey not in valid_keys:
                valid_keys.append(dkey)
    
    if len(valid_keys) < len(alldata.keys()):
        missing_keys = [k for k in alldata.keys() if k not in valid_keys]
        print('\n  Skipping plots for {} data: columns {} not available'.format(' '.join(missing_keys), ' '.join(varlist)))

    return valid_keys

def get_Bazincuts(cuts):
    
    bazinp = bazinpars if any('par' in k for k in  cuts.keys()) else []
    bazine = bazinerrs if any('err' in k for k in  cuts.keys()) else []
    bazinvars = bazinp + bazine
    pmaxcuts = cuts.get(g.Bazinpar_max) if len(bazinp) > 0 else []
    subpmax = [bazinpars[i] if (type(pmaxcuts[i]) == str and len(pmaxcuts[i]) > 0) else '' for i in range(len(bazinpars))] \
               if len(bazinp) > 0 else []
    pmincuts = cuts.get(g.Bazinpar_min) if len(bazinp) > 0 else []
    subpmin = [bazinpars[i] if (type(pmincuts[i]) == str and len(pmincuts[i]) > 0) else '' for i in range(len(bazinpars))] \
               if len(bazinp) > 0 else []
    emaxcuts = cuts.get(g.Bazinerr_max) if len(bazine) > 0 else []   # numbers or skipped
    emincuts = [0 for p in bazine]
    sube = ['' for e in bazine]
    # make sure any character numbers are floats
    #pmaxcuts = [float(p) if type(p)==str and p.isdigit() else p for p in pmaxcuts]
    Bazincuts = {Max: pmaxcuts + emaxcuts , Min: pmincuts + emincuts, AltMax: subpmax + sube,
                 AltMin: subpmin + sube}

    return bazinvars, Bazincuts


def plot_Bazinvars(fig, plotlist, alldata, types, type_masks, filt, nplot=0, lgnd_title='',
                   cuts={}, Nbins=25, joint_Pass=_in_, joint_Fail=_notin_, errors=False,
                   binlo_par=[0., 0., 0., -100.], binhi_par=[600., 120., 20., 100.],
                   binlo_err=0., binhi_err=50., chi2_max=50., minmax=False, debug=False):

    plottitle = get_plotlist_title(plotlist)
    if not errors:
        varlist = [b for b in bazinpars if not (b == t0)]
        binlo = binlo_par
        binhi = binhi_par
        labellist = [bazinlabels[b] for b in varlist] 
    else:
        varlist = [b for b in bazinerrs]
        binlo = [binlo_err for b in bazinerrs]
        binhi = [binhi_err for b in bazinerrs]
        labellist = [bazinlabels[b] for b in bazinerrs]
        
    nplot = 0
    ylabel = '$N$'
    for var, lo, hi, label in zip(varlist, binlo, binhi, labellist):
        nplot += 1
        f = fig.add_subplot(plot_offset6 + nplot)
        varname = Bazin_ + filt + '_' + var
        xlabel = ' '.join([g.Bazin, filt, label])
        bins = np.linspace(lo, hi, Nbins + 1)
        if errors:
            cut_key = [k for k in cuts if varname in k and _in_ in k and err in k]  # will be 1 element list
        else:
            cut_key = [k for k in cuts if varname in k and _in_ in k and err not in k]
        pltcuts = cuts[cut_key[0]] if len(cuts) > 0  and len(cut_key) > 0 else {}
        addlabel = ' (Pass ' + xlabel + ')' if len(cuts) > 0  and len(cut_key) > 0 else ''
        plot_types(f, types, varname, alldata, plotlist=plotlist, masks=type_masks, xlabel=xlabel, cuts=pltcuts,
                   ylabel=ylabel,
                   yscale=log, plotdict=plotdict, plotid=Bazin_ + filt + '-band', bins=bins, minmax=minmax, title=plottitle,
                   addlabel=addlabel)
        f.legend(loc='best', fontsize='small', numpoints=1, title=lgnd_title)

    if len(cuts) > 0 and not errors:
        pltcuts = cuts[Joint + joint_Pass]   #mask for joint-pass of all fit variables
        addlabel = '(Pass Joint ' + filt +')'
        # add chisq distribution
        nplot += 1
        varname = Bazin_ + filt + '_chisq_red'
        xlabel = ' '.join([g.Bazin, filt, '$\\chi^2/dof$'])
        bins = np.linspace(0., chi2_max, Nbins + 1)
        f = fig.add_subplot(plot_offset6 + nplot)
        plot_types(f, types, varname, alldata, plotlist=plotlist, masks=type_masks, xlabel=xlabel, cuts=pltcuts,
                   ylabel=ylabel,
                   yscale=log, plotdict=plotdict, plotid=Bazin_ + filt + '-chisq', bins=bins, minmax=minmax, title=plottitle,
                   addlabel=addlabel)
        f.legend(loc='best', fontsize='small', numpoints=1, title=lgnd_title)
        
        xlabel = 'Redshift'
        zhi = g.zhi[plotlist[0][0]]   #sim
        bins = np.linspace(g.zlo, zhi, Nzbins + 1)
        nplot += 1
        f = fig.add_subplot(plot_offset6 + nplot)
        varname = 'z'
        plot_types(f, types, varname, alldata, plotlist=plotlist, masks=type_masks, xlabel=xlabel, ylabel=ylabel, title=plottitle,
                   cuts=pltcuts, yscale=log, plotdict=plotdict, plotid='_'.join([g.Bazin, varname, Joint, 'Pass']), bins=bins,
                   addlabel=addlabel)
        f.legend(loc='best', fontsize='small', numpoints=1, title=lgnd_title)

    return nplot


def get_valid_combos(combos, valid_keys):

    valid_combos = []
    for combo in combos:
        valids = [s for s in combo[0] if s in valid_keys]
        validd = [d for d in combo[1] if d in valid_keys]
        valid_combos.append([valids, validd])

    return valid_combos
        

def get_next(fig, multiPdf, nplot, npage, subpage, plotsperpage=6, new=True, force_close=False):
    if (nplot >= plotsperpage or force_close):
        subpage = close_page(fig, multiPdf, npage, subpage)
        fig = plt.figure(figsize=(figx, figy)) if new else None
        nplot = 0
        closed = True
    else:
        closed = False

    return fig, nplot, subpage, closed


def rename_features(file_formats, feature_names, data):

    for dkey in data.keys():
        if file_formats[dkey] != g.default_format: #check for file_format != 'text'
            for ft in feature_names:
                if ft not in data[dkey].colnames:
                    alt = g.alternate_feature_names[g.default_format].get(ft, None)
                    #check for columns in data
                    if len(data[dkey]) > 0 and ft is not None:
                        alts = alt if type(alt)==list else [alt]
                        for alt in alts:
                            if alt in data[dkey].colnames:
                                data[dkey].rename_column(alt, ft)
                                print('  Renaming {} to {} for {} data'.format(alt, ft, dkey))

                if ft not in data[dkey].colnames:
                    print('  No alternate feature name available for {} in {} data'.format(ft, dkey))

    return data


def close_page(fig, multiPdf, npage, subpage):
    fig.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multiPdf.savefig(fig)
        
    print('  Wrote page {}.{}'.format(npage, subpage))
    subpage += 1
    plt.close(fig)

    return subpage
    
    
def make_plots(MLtypes, alldata, type_masks, Fixed_effcy, performance, alltypes_colnames,  
               template_info, plotlist = [[], []], cuts={}, user_prefs={},
               savetypes=g.Ia, plot_groups=[g.Performance, g.SALT], plot_id='', CLFid=g.RF,
               totals=False, target_class=0, minmax=False, debug=False, file_formats={}):

    print('\n********** STARTING PLOTS **********\n')

    group_id = ''.join([gr[0:1] for gr in plot_groups])
    pdfname = plot_id + '_' + group_id + '.pdf'
    print('\nSaving plots to {}'.format(pdfname))
    multiPdf = PdfPages(pdfname)

    # setup list of types needed for making plots; include CLF-predicted types
    CLFtypes = []
    TFtypes = []
    for CLFid in CLFlbls.keys():
        for t in MLtypes:
            CLFtypes.append(CLFid + t)
            for p in g.TFtypes:
                if CLFid + t + p in g.allTFtypes:
                    TFtypes.append(CLFid + t + p)


    # include totals if requested, or if data is included
    totals = totals or len(plotlist) > 1
    MLtypes_plot = [t for t in MLtypes] + [g.Total] if totals else [t for t in MLtypes]
    CLFtypes_plot = [t for t in CLFtypes] + [g.Total] if totals else [t for t in CLFtypes]
    TFtypes_plot = [t for t in TFtypes] + [g.Total] if totals else [t for t in TFtypes]
    #alltypes = [MLtypes, CLFtypes, TFtypes]  # all possible variants of typing

    # setup combinations of simulations and data for plotting
    if len(plotlist) > 1 and len(plotlist[1]) > 0:   # data is present 
        combos = [[[s], plotlist[1]] for s in plotlist[0]] # list format for plotting with plot_types
        pairs = [[[s], [d]] for s in plotlist[0] for d in plotlist[1]]
    else:
        combos = [[[s], []] for s in plotlist[0]] # list format for plotting with plot_types
        pairs = [[[s], []] for s in plotlist[0]]
    
    # add user prefs to color and marker dicts.
    #        user_prefs = {'color':dict(zip(user_labels, args.user_colors)),
    #                      'markers':dict(zip(user_labels, args.user_markers))}
    plotdict['color'][g.Data].update(user_prefs['color'])
    plotdict['markers'][g.Data].update(user_prefs['markers'])
    
    # setup labels for classification methods (Max Prob or Fixed-efficiency)
    classification_labels = {}
    for cl_id in type_masks.keys():
        if cl_id == g.MaxProb:
            classification_labels[cl_id] = 'Max.-Prob. Classification'
        elif g.TrueType in cl_id:
            classification_labels[cl_id] = 'True Types'
        else:
            classification_labels[cl_id] = re.sub('Eff_', 'Fixed-Eff. Classification $e=$ ', cl_id)

    # pages
    npage = 1
    page_total = 0
    closed = True
    
    for group in plot_groups:

        # Performance plots
        if g.Performance in group:
            print('\nStarting pages {}.x: {} plots'.format(npage, group))
            # need labeled data for probability plots
            dkeys = [k for k in alldata.keys() if g.Training not in k and k in type_masks[g.TrueType].keys()]
            subpage = 1
            colname = 'fit_pr'
            alldata = rename_features(file_formats, [colname], alldata) #check formats and rename columns

            for dkey in dkeys:
                print('  Plotting {} Data'.format(dkey))
                fig = plt.figure(figsize=(figx, figy))
                plot_probabilities(fig, dkey, alldata, MLtypes_plot, type_masks[g.TrueType], performance[dkey],
                                   target_class=target_class, alltypes_colname=alltypes_colnames[dkey],
                                   colname=colname)
                subpage = close_page(fig, multiPdf, npage, subpage)
                page_total += 1

            #Summary plot of purity vs effcy
            fig = plt.figure(figsize=(figx, figy))
            plot_purity_vs_effcy(fig, performance)
            subpage = close_page(fig, multiPdf, npage, subpage)
            page_total += 1

            # plots for number of SN per template
            if (g.Test in alldata.keys()):
                print('  Plotting Template Statistics for:')
                for cl_id in template_info.keys():
                    cl_label = '{} for {}'.format(classification_labels[cl_id], plotlabels[g.Test]) 
                    subpage = plot_template_statistics(template_info[cl_id], MLtypes, npage, subpage, multiPdf,
                                                   CLFid=CLFid, classification_label=cl_label)
                    page_total += 1

            #TODO Cross-validation plots
            
            npage += 1
            
        # SALT plots: compare simulations and data, if any    
        if g.SALT in group:
            print('\nStarting pages {}.x: {} plots'.format(npage, group))
            subpage = 1
            nplot = 0
            fig = plt.figure(figsize=(figx, figy))
            alldata = rename_features(file_formats, ['x1', 'c', 'fit_pr'], alldata) #check formats and rename columns
            for combo in combos:                        # loop over combinations of simulation and data
                for mkey, tmask in type_masks.items():  # loop over classification options
                    types = MLtypes_plot if g.TrueType in mkey else CLFtypes_plot   # types corresponding to mask choice
                    print('  Plotting {} Data ({})'.format(' + '.join([c for sublist in combo for c in sublist]),
                                                           classification_labels[mkey]))
                    # fp, x1 and c distributions  (3 plots)
                    nplot = plot_SALT(fig, combo, alldata, types, tmask, nplot=nplot, lgnd_title=classification_labels[mkey])
                    fig, nplot, subpage, closed = get_next(fig, multiPdf, nplot, npage, subpage, plotsperpage=plotsperpage)
                    if closed:
                        page_total += 1

            if not closed:
                subpage = close_page(fig, multiPdf, npage, subpage)
                page_total += 1
                closed = True

            # x1-c scatter plots
            nplot = 0
            fig = plt.figure(figsize=(figx, figy))
            for dkey in alldata.keys():                 # loop over data sets
                for mkey, tmask in type_masks.items():  # loop over classification options
                    if dkey in tmask.keys():            # check that mask exists for dataset
                        if g.TrueType in mkey:
                            types = MLtypes
                            clabel = g.SN + g.Ia
                        else:
                            types = TFtypes if dkey in type_masks[g.TrueType].keys() else CLFtypes
                            clabel = ''  # suppress contours for CLF types
                            # code to add contours as follows
                            #clabel = '{} {}'.format(CLFid, g.SN + g.Ia)
                            #clabel = '{} {}'.format(g.TP, clabel) if g.TrueType in type_masks.keys() else clabel 
                        print('  Plotting x1-c scatter for {} Data ({})'.format(dkey, classification_labels[mkey]))
                        nplot = plot_scatter_bytype(fig, dkey, alldata, types, tmask, nplot=nplot, xvar='x1', yvar='c',
                                                    lgnd_title=classification_labels[mkey], contour_label=clabel)
                        fig, nplot, subpage, closed = get_next(fig, multiPdf, nplot, npage, subpage, 
                                                               plotsperpage=plotsperpage)
                        if closed:
                            page_total += 1
                    else:
                        print('  Skipping scatter plots for {}: not available for {} data'.format(mkey, dkey))

            if not closed:
                subpage = close_page(fig, multiPdf, npage, subpage)
                page_total += 1
                closed = True

            npage += 1

        if g.Hubble in group:
            print('\nStarting pages {}.x: {} plots'.format(npage, group))
            subpage = 1
            nplot = 0
            fig = plt.figure(figsize=(figx, figy))
            # loop over pairs of simulation and data samples (if any); plot CLFtypes and TFtypes
            for pair in pairs:
                for mkey, tmask in type_masks.items():
                    type_data = CLFid + g.Ia                                 # always available (except for TrueType)  
                    if g.TrueType in mkey:
                        types_this = [MLtypes_plot]                          # plot true types for sim
                        if len(pair[1]) > 0:
                            type_data = g.Ia if pair[1][0] in tmask.keys() else g.Total  # if true_types available in data 
                    else:
                        types_this = [CLFtypes_plot, [CLFid + g.Ia + g.TP, CLFid + g.Ia + g.FP]] # TPFP always available in sim
                    for types in types_this:
                        print('  Plotting {} Data using ({} with types {}) + ({})'.format(' + '.join([c for sublist in pair for c in sublist]),
                                                                             classification_labels[mkey],
                                                                                  ' '.join(types), type_data))
                        # z, HR linear and HR log distributions
                        nplot = plot_hubble(fig, pair, alldata, types, tmask, nplot=nplot, lgnd_title=classification_labels[mkey],
                                            type_data=type_data)
                        fig, nplot, subpage, closed = get_next(fig, multiPdf, nplot, npage, subpage, plotsperpage=plotsperpage)
                        if closed:
                            page_total += 1

            if not closed:
                subpage = close_page(fig, multiPdf, npage, subpage)
                page_total += 1
                closed = True
            
            # Hubble Diagram (scatter) plots for Ia's 
            nplot = 0
            fig = plt.figure(figsize=(figx, figy))
            for dkey in alldata.keys():                 # loop over data sets
                for mkey, tmask in type_masks.items():  # loop over classification options
                    if dkey in tmask.keys():            # check that mask exists for dataset
                        if g.TrueType in mkey:
                            types = [g.Ia]
                        else:
                            types = [g.RFIaTP, g.RFIaFP] if dkey in type_masks[g.TrueType].keys() else [g.RFIa]
                        print('  Plotting HD for {} Data ({}) using {} type(s)'.format(dkey, classification_labels[mkey], ' + '.join(types)))
                        nplot = plot_HD(fig, dkey, alldata, types, tmask, nplot=nplot, lgnd_title=classification_labels[mkey])
                        fig, nplot, subpage, closed = get_next(fig, multiPdf, nplot, npage, subpage, plotsperpage=plotsperpage)
                        if closed:
                            page_total += 1
                    else:
                        print('  Skipping scatter plots for {}: not available for {} data'.format(mkey, dkey))

            if not closed:
                subpage = close_page(fig, multiPdf, npage, subpage)
                page_total += 1
                closed = True

            npage += 1

        if g.Error in group:
            print('\nStarting pages {}.x: {} plots'.format(npage, group))
            subpage = 1
            nplot = 0
            fig = plt.figure(figsize=(figx, figy))
            alldata = rename_features(file_formats, ['t0_err', 'c_err', 'x1_err'], alldata) #check formats and rename columns
            for combo in combos:
                for mkey, tmask in type_masks.items():
                    types = MLtypes_plot if g.TrueType in mkey else CLFtypes_plot   # types corresponding to mask choice
                    print('  Plotting {} Data ({})'.format(' + '.join([c for sublist in combo for c in sublist]),
                                                           classification_labels[mkey]))
                    # t0_err, x1_err and c_err distributions
                    nplot = plot_errors(fig, combo, alldata, types, tmask, nplot=nplot, lgnd_title=classification_labels[mkey])
                    fig, nplot, subpage, closed = get_next(fig, multiPdf, nplot, npage, subpage, plotsperpage=plotsperpage)
                    if closed:
                        page_total += 1

            if not closed:
                subpage = close_page(fig, multiPdf, npage, subpage)
                page_total += 1
                closed = True

            # error scatter plots    
            xvars = ['t0_err', 't0_err', 't0_err', 'x1_err', 'c_err', 'x1_err', ]
            yvars = ['fit_pr', 'x1', 'c', 'x1', 'c', 'c_err']
            xmins = [-0.1, -0.1, -0.1, 0., 0., 0.]
            xmaxs = [4., 4., 4., 2., 0.2, 2.]
            ymins = [0., -5., -0.6, -5., -0.6, 0.]
            ymaxs = [1., 5., 0.6, 5, 0.6, 0.2]
            nplot = 0
            fig = plt.figure(figsize=(figx, figy))
            for dkey in alldata.keys():                 # loop over data sets
                for mkey, tmask in type_masks.items():  # loop over classification options
                    if dkey in tmask.keys():            # check that mask exists for dataset
                        if g.TrueType in mkey:
                            types = MLtypes
                        else:
                            types = TFtypes if dkey in type_masks[g.TrueType].keys() else CLFtypes
                        print('  Plotting error scatter for {} Data ({})'.format(dkey, classification_labels[mkey]))
                        for xvar, yvar, xmin, xmax, ymin, ymax in zip(xvars, yvars, xmins, xmaxs, ymins, ymaxs):
                            nplot = plot_scatter_bytype(fig, dkey, alldata, types, tmask, nplot=nplot, xvar=xvar, yvar=yvar,
                                                        lgnd_title=classification_labels[mkey], x_min=xmin, x_max=xmax, 
                                                        plotid='_'.join([xvar, yvar, scatter]),
                                                        y_min=ymin, y_max=ymax, xlabel=SALTlabels[xvar], ylabel=SALTlabels[yvar])
                            fig, nplot, subpage, closed = get_next(fig, multiPdf, nplot, npage, subpage, plotsperpage=plotsperpage)
                        if closed:
                            page_total += 1
                    else:
                        print('  Skipping scatter plots for {}: not available for {} data'.format(mkey, dkey))

            if not closed:
                subpage = close_page(fig, multiPdf, npage, subpage)
                page_total += 1
                closed = True

            npage += 1
            
        if g.Magnitude in group:
            print('\nStarting pages {}.x: {} plots'.format(npage, group))
            SALTmagmask = get_mask(alldata, SALTfilters, SALTfilterlabels, mask_id='SALT-peak-magnitude', file_id=plot_id, debug=debug)

            subpage = 1
            nplot = 0
            #fig = plt.figure(figsize=(figx, figy))
            valid_keys = get_valid_keys(alldata, SALTfilters)
            mag_combos = get_valid_combos(combos, valid_keys)
            for combo in mag_combos:
                for mkey, tmask in type_masks.items():
                    if closed:    #check if new fig needed
                        fig, ax = plt.subplots(nrow8, ncol8, figsize=(figx, figy), sharex='col')
                    types = MLtypes_plot if g.TrueType in mkey else CLFtypes_plot   # types corresponding to mask choice
                    print('\n  Plotting {} Data ({})'.format(' + '.join([c for sublist in combo for c in sublist]),
                                                           classification_labels[mkey]))
                    # griz distributions for multi and single band fits (8 plots per page)
                    nplot = plot_magnitudes(ax, combo, alldata, types, tmask, cuts=SALTmagmask, nplot=nplot,
                                            lgnd_title=classification_labels[mkey], plot_offset=plot_offset8, minmax=minmax, debug=debug)
                    fig.subplots_adjust(hspace=0)   #remove horizontal space
                    fig, nplot, subpage, closed = get_next(fig, multiPdf, nplot, npage, subpage, plotsperpage=8, new=False)
                    if closed:
                        page_total += 1

            if not closed:
                subpage = close_page(fig, multiPdf, npage, subpage)
                page_total += 1
                closed = True


            npage += 1

        if g.Color in group:
            print('\nStarting pages {}.x: {} plots'.format(npage, group))
            SALTcolormask = get_mask(alldata, SALTcolors, SALTcolorlabels, mask_id='SALT-color', file_id=plot_id, debug=debug)
            SALTcolordiffmask = get_mask(alldata, SALTcolordiffs, SALTcolordifflabels,
                                         mask_id='SALT-color-difference', file_id=plot_id, debug=debug)
            subpage = 1
            nplot = 0
            valid_keys = get_valid_keys(alldata, SALTcolordiffs)  #check that alldata has required columns
            color_combos = get_valid_combos(combos, valid_keys)
            fig = plt.figure(figsize=(figx, figy))
            for combo in color_combos:
                for mkey, tmask in type_masks.items():
                    types = MLtypes_plot if g.TrueType in mkey else CLFtypes_plot   # types corresponding to mask choice
                    print('\n  Plotting {} Data ({})'.format(' + '.join([c for sublist in combo for c in sublist]),
                                                           classification_labels[mkey]))
                    # gr, ri and iz distributions for multi and single band fits (6 plots)
                    nplot = plot_SALTcolors(fig, combo, alldata, types, tmask, cuts=SALTcolormask, nplot=nplot,
                                            lgnd_title=classification_labels[mkey], minmax=minmax, debug=debug)
                    fig, nplot, subpage, closed = get_next(fig, multiPdf, nplot, npage, subpage, plotsperpage=plotsperpage)
                    if closed:
                        page_total += 1

                    # color-diff and redshift distributions with cuts (6 plots) 
                    nplot = plot_SALTcolordiffs(fig, combo, alldata, types, tmask, cuts=SALTcolordiffmask, nplot=nplot,
                                                lgnd_title=classification_labels[mkey])
                    fig, nplot, subpage, closed = get_next(fig, multiPdf, nplot, npage, subpage, plotsperpage=plotsperpage)
                    if closed:
                        page_total += 1

            if not closed:  #close page
                subpage = close_page(fig, multiPdf, npage, subpage)
                page_total += 1
                closed = True

            # color-diff scatter plots
            color_selections = [('gr', 'ri'), ('gr','iz'), ('ri', 'iz')]
            indexes = [(colors.index(s[0]), colors.index(s[1])) for s in color_selections]
            collo = [-4., -2.5, -2.]
            colhi = [4., 3., 2.]
            NCbins=40
            nplot = 0
            dx = 0.2    #contour binwidths
            dy = 0.2 
            fig = plt.figure(figsize=(figx, figy))
            
            for dkey in valid_keys:                     # loop over data sets
                for mkey, tmask in type_masks.items():  # loop over classification options
                    if dkey in tmask.keys():            # check that mask exists for dataset
                        if g.TrueType in mkey:
                            types = MLtypes
                            clabel = g.SN + g.Ia
                        else:
                            types = TFtypes if dkey in type_masks[g.TrueType].keys() else CLFtypes
                            clabel = ''
                        print('  Plotting color-difference scatter for {} Data ({})'.format(dkey, classification_labels[mkey]))
                        for (c1, c2) in indexes:
                            nplot = plot_scatter_bytype(fig, dkey, alldata, types, tmask, nplot=nplot, cuts=SALTcolordiffmask[Joint+ne],
                                                        xvar=SALTcolordiffs[c1], yvar=SALTcolordiffs[c2], contour_label=clabel,
                                                        lgnd_title=classification_labels[mkey], ctrxbin=dx, ctrybin=dy, 
                                                        plotid='color_diff_scatter',
                                                        x_min=collo[c1], x_max=colhi[c1], y_min=collo[c2], y_max=colhi[c2], 
                                                        xlabel=SALTcolordifflabels[c1], ylabel=SALTcolordifflabels[c2])
                            fig, nplot, subpage, closed = get_next(fig, multiPdf, nplot, npage, subpage, plotsperpage=plotsperpage)
                        if closed:
                            page_total += 1

                    else:
                        print('  Skipping scatter plots for {}: not available for {} data'.format(mkey, dkey))

            if not closed:
                subpage = close_page(fig, multiPdf, npage, subpage)

            npage += 1

        if g.Bazin in group:
            print('\nStarting pages {}.x: {} plots'.format(npage, group))
            
            bazinvars = bazinpars + bazinerrs #default list of all Bazin features
            Bazincuts = {}
            joint_Pass = ne                   #default string for combined filter cuts
            if len(cuts) >0 and g.Bazin in cuts:  
                bazinvars, Bazincuts = get_Bazincuts(cuts[g.Bazin])
                joint_Pass = _in_
            bazinvarlabels = [bazinlabels[v] for v in bazinvars]

            Bazinfitmask = {}
            Bazin_features = []
            for filt in fit_bands:      #assemble masks for Bazin cuts; defaults to cuts on good fits (!-999) if cuts = {}
                varlist = ['_'.join([g.Bazin, filt, v]) for v in bazinvars]
                Bazin_features = Bazin_features + varlist
                varlabels = [' '.join([g.Bazin, fitlabels[filt], v]) for v in bazinvarlabels]
                Bazinfitmask[filt] = get_mask(alldata, varlist, varlabels, cuts=Bazincuts, mask_id='Bazin-fit',
                                            file_id=plot_id, debug=debug)

            # select good data for Bazin colors (NB: removing -999's alone probably not sufficient to remove flaky fits)
            Bazincolormask = get_mask(alldata, Bazincolors, colorlabels, mask_id='Bazin-color', file_id=plot_id, debug=debug)
            #print(Bazincolormask.keys()) # 'izBazin!=-999', 'Joint!=', 'riBazin!=-999'..

            Bazincombinedmask = {}
            for col in colors:  # get joint mask for SN passing filter cuts for each color
                Bazincombinedmask[col] = get_combined_fitmask(col, Bazinfitmask, mask_id='Bazin-combined', 
                                                                 joint_Pass=joint_Pass, file_id=plot_id)
            
            subpage = 1
            nplot = 0
            valid_keys = get_valid_keys(alldata, Bazin_features)  # check that data has features
            bazin_combos = get_valid_combos(combos, valid_keys)
            """
            fig = plt.figure(figsize=(figx, figy))
            #for combo in [bazin_combos[0]]:
            for combo in bazin_combos:
                for mkey, tmask in type_masks.items():
                    types = MLtypes_plot if g.TrueType in mkey else CLFtypes_plot   # types corresponding to mask choice
                    print('\n  Plotting {} Data ({})'.format(' + '.join([c for sublist in combo for c in sublist]),
                                                           classification_labels[mkey]))
                    for filt in fit_bands:
                        # 5 or 6 plots
                        nplot = plot_Bazinvars(fig, combo, alldata, types, tmask, filt, nplot=nplot, cuts=Bazinfitmask[filt],
                                               lgnd_title=classification_labels[mkey], joint_Pass=joint_Pass,
                                               minmax=minmax, debug=debug)
                        fig, nplot, subpage, closed = get_next(fig, multiPdf, nplot, npage, subpage, plotsperpage=plotsperpage,
                                                               force_close=True)
                        page_total += 1

                        nplot = plot_Bazinvars(fig, combo, alldata, types, tmask, filt, nplot=nplot, cuts=Bazinfitmask[filt],
                                               lgnd_title=classification_labels[mkey], joint_Pass=joint_Pass, errors=True,
                                               minmax=minmax, debug=debug)
                        fig, nplot, subpage, closed = get_next(fig, multiPdf, nplot, npage, subpage, plotsperpage=plotsperpage,
                                                               force_close=True)
                        page_total += 1

                        
                    # Bazin colors (3 plots each)
                    nplot = plot_Bazincolors(fig, combo, alldata, types, tmask, nplot=nplot, cuts=Bazincolormask[Joint + ne],
                                             lgnd_title=classification_labels[mkey], minmax=minmax, debug=debug)
                        
                    nplot = plot_Bazincolors(fig, combo, alldata, types, tmask, nplot=nplot, cuts=Bazincombinedmask,
                                               lgnd_title=classification_labels[mkey], cut_keys=True, minmax=minmax, debug=debug)
                    fig, nplot, subpage, closed = get_next(fig, multiPdf, nplot, npage, subpage, plotsperpage=plotsperpage)
                    if closed:
                        page_total += 1

            if not closed:
                subpage = close_page(fig, multiPdf, npage, subpage)
                page_total += 1
                closed = True
            """

            # Bazin scatter plots
            xvars = [t_rise]
            yvars = [t_fall]
            xmins = [0.]
            xmaxs = [20.]
            ymins = [0.]
            ymaxs = [120.]
            nplot = 0
            fig = plt.figure(figsize=(figx, figy))
            for dkey in alldata.keys():                 # loop over data sets                                                                                
                for mkey, tmask in type_masks.items():  # loop over classification options                                                                   
                    if dkey in tmask.keys():            # check that mask exists for dataset                                                                 
                        if g.TrueType in mkey:
                            types = MLtypes
                        else:
                            types = TFtypes if dkey in type_masks[g.TrueType].keys() else CLFtypes
                        print('  Plotting Bazin scatter for {} Data ({})'.format(dkey, classification_labels[mkey]))
                        for filt in fit_bands:
                            print(Bazinfitmask[filt].keys())
                            for xvar, yvar, xmin, xmax, ymin, ymax in zip(xvars, yvars, xmins, xmaxs, ymins, ymaxs):
                                xvarname = '_'.join([g.Bazin, filt, xvar])
                                xlabel = ' '.join([g.Bazin, filt, bazinlabels[xvar]])
                                yvarname = '_'.join([g.Bazin, filt, yvar])
                                ylabel = ' '.join([g.Bazin, filt, bazinlabels[yvar]])
                                nplot = plot_scatter_bytype(fig, dkey, alldata, types, tmask, nplot=nplot, xvar=xvarname, yvar=yvarname,
                                                            lgnd_title=classification_labels[mkey], x_min=xmin, x_max=xmax,
                                                            plotid='_'.join([xvar, yvar, scatter]), cuts={},
                                                            y_min=ymin, y_max=ymax, xlabel=xlabel, ylabel=ylabel)
                                fig, nplot, subpage, closed = get_next(fig, multiPdf, nplot, npage, subpage, plotsperpage=plotsperpage)
                        if closed:
                            page_total += 1
                    else:
                        print('  Skipping scatter plots for {}: not available for {} data'.format(mkey, dkey))

            if not closed:
                subpage = close_page(fig, multiPdf, npage, subpage)
                page_total += 1
                closed = True

            npage += 1
            
    multiPdf.close()
    plt.rcParams.update({'figure.max_open_warning': 0})

    print('\nWrote {} with {} pages'.format(pdfname, page_total))
    print('Completed Successfully')

    return

def get_combined_fitmask(col, fitmask, mask_id='', joint_Pass=_in_, file_id=''):

    #work on default
    ff = open(file_id + '_' + mask_id + '.eff', 'w')
    filters = [c for c in col]
    tee('\n  Efficiencies for {} cuts for filters {}'.format(mask_id, ' & '.join(filters)), ff)
    combined_mask = {}
    for filt in filters:
        if filt in fitmask:
            joint_pass = Joint + joint_Pass
            if joint_pass in fitmask[filt]:
                for dkey, mask in fitmask[filt][joint_pass].items():
                    if dkey in combined_mask:
                        combined_mask[dkey] = combined_mask[dkey] & mask
                    else:
                        combined_mask[dkey] = mask

    tee('    Data        Npass  Efficiency', ff)
    for dkey, cmask in combined_mask.items():
        pass_effcy = float(np.count_nonzero(cmask))/len(cmask)
        tee('    {:10} {:6} {:10.3g}'.format(dkey, np.count_nonzero(cmask), pass_effcy), ff)

    return combined_mask
    
        
def plot_var_withcuts(): #TBD

    f = fig.add_subplot(plot_offset + nplot)
    plot_types(f, plottypes, 'z', alldata, plotlist=plotlist, xlabel='Redshift', ylabel=ylabel, cuts=pltcuts,
               plotdict=plotdict, plotid=SALT + filt + Pass, bins=zbins, addlabel=label, )
    key_fail = filt + Fail
    pltcuts = SALTfiltermask[key_fail]
    plot_types(f, plottypes, 'z', alldata, plotlist=plotlist, xlabel='Redshift', ylabel=ylabel, cuts=pltcuts,
               plotdict=plotdict, plotid=SALT + filt + Fail, bins=zbins, alt=alt, addlabel=' No ' + label, )
    axes = plt.gca()
    f.set_ylim(0., axes.get_ylim()[1] * 1.5)
    f.legend(loc='upper left', numpoints=1, prop={'size': 8}, scatterpoints=1, ncol=Ncols)

    return


def plot_Bazincolors(fig, plotlist, alldata, types, type_masks, nplot=0, lgnd_title='',
                     cuts={}, Nbins=50, cut_keys=False,
                     binlo=[-100., -100., -100.], binhi=[100., 100., 100.],
                     minmax=True, debug=False, yscale='log'):
    
    plottitle = get_plotlist_title(plotlist)
    addlabel = ''
    ylabel = '$N$'

    #print '\n  Min and Max of colors:\n'
    for col, lo, hi, xlabel in zip(colors, binlo, binhi, Bazincolorlabels):
        varname = col + g.Bazin
        nplot += 1
        f = fig.add_subplot(plot_offset6 + nplot)
        bins = np.linspace(lo, hi, Nbins + 1)
        pltcuts = cuts[col] if len(cuts) > 0 and cut_keys else cuts
        if len(cuts) > 0:
            addlabel = '(Pass ' + xlabel + ')' if cut_keys else '(Pass ' + xlabel+ne999 + ')'
        plot_types(f, types, varname, alldata, plotlist=plotlist, masks=type_masks, xlabel=xlabel, ylabel=ylabel, cuts=pltcuts,
                   yscale=yscale, plotdict=plotdict, plotid=g.Bazin + ' ' + col, bins=bins, minmax=minmax,
                   addlabel=addlabel)
        axes = plt.gca()
        f.set_ylim(1., axes.get_ylim()[1] * 10)
        f.legend(loc='best', fontsize='small', numpoints=1, title=lgnd_title)

    return nplot

def plot_scatter_Bazin_color(): #TBD
        # page 21+ - Bazin scatter plots
        xvars = [t_rise]
        yvars = [t_fall]
        xbinlo = [0.]
        xbinhi = [20.]
        ybinlo = [0.]
        ybinhi = [120.]
        xbinwidth = [0.5]
        ybinwidth = [2]
        # add scatter plot t_rise vs t_fall;
        for l, s in zip([Training] + MLdatalist,  g.Simulated + [g.Data for m in MLdatalist]):
            print('\nStarting page {} (Bazin scatterplots)\n'.format(npages))
            fig = plt.figure(figsize=(figx, figy))
            npages += 1
            nplot = 0
            for filt in fit_bands:
                for xvar, yvar, xmin, xmax, ymin, ymax, dx, dy in zip(xvars, yvars, xbinlo, xbinhi, ybinlo, ybinhi,
                                                                      xbinwidth, ybinwidth):
                    nplot += 1
                    f = fig.add_subplot(plot_offset6 + nplot)
                    xvarname = Bazin_ + filt + '_' + xvar
                    xlabel = Bazin + ' ' + filt + ' ' + bazinlabels[xvar]
                    yvarname = Bazin_ + filt + '_' + yvar
                    ylabel = Bazin + ' ' + filt + ' ' + bazinlabels[yvar]
                    if (doBazincuts):
                        pltcuts = Bazincuts[filt][All]
                    else:
                        pltcuts = {}
                    plotdict['labels'][contour] = 'SNIa Density'
                    if (l in MLdatalist_labeled or l == Training):
                        pbztypes = MLtypes
                    else:
                        pbztypes = CLFtypes
                    plot_types(f, pbztypes, xvarname, alldata, plotlist=[l], yvar=yvarname, xlabel=xlabel, ylabel=ylabel,
                               cuts=pltcuts, plotdict=plotdict, ctrxbin=dx, ctrybin=dy, title=plotlabels[s][l],
                               weights=False)
                    f.set_xlim(xmin - dx, xmax + dx)
                    f.set_ylim(ymin - dy, ymax + dy)
                    f.legend(loc='best', fontsize='small', scatterpoints=1, ncol=Ncols)

            fig.tight_layout()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                multiPdf.savefig(fig)

        return

def plot_cross_validation():  #TBD
    if (args.cv or args.pc):
        print('\nStarting page {} (Cross-Validation and Purity)\n'.format(npages))
        fig = plt.figure(figsize=(figx, figy))
        if (args.cv):
            minval = np.fmin(np.min(avgskf), np.min(avgss))
            maxval = np.fmax(np.max(avgskf), np.max(avgss))
            ax1 = fig.add_subplot(plot_offset6 + nplot1)
            plt.title('Stratified k-fold Scores')
            ax1.errorbar(kvals, avgskf, yerr=stdkf, fmt='o')
            ax1.scatter(kvals, avgskf, color='blue', marker=".")
            ax1.set_ylim(minval - 0.02, maxval + 0.02)
            ax1.set_xlabel('k (number of folds)')
            ax1.set_ylabel('Score')

            ax2 = fig.add_subplot(plot_offset6 + nplot2)
            plt.title('ShuffleSplit Scores')
            ax2.errorbar(1.0 - np.array(tsvals), avgss, yerr=stdss, fmt='o')
            ax2.scatter(1.0 - np.array(tsvals), avgss, color='blue', marker='.')
            ax2.set_ylim(minval - 0.02, maxval + 0.02)
            ax2.set_xlabel('Training Fraction')
            ax2.set_ylabel('Score')
        # endif-cv

        if (args.pc):
            ax1 = fig.add_subplot(plot_offset6 + nplot3)
            plt.title('Scores for Purity Variations')
            ax1.scatter(args.purities, pscores, color='blue', marker=".")
            ax1.set_xlabel('Purity of Test Data')
            ax1.set_ylabel('Score')

            # endif-pc

        fig.tight_layout()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            multiPdf.savefig(fig)
    return

def plot_probability_variance(alldata, prvar=False): #TBD
    if prvar:
        fig = plt.figure(figsize=(figx, figy))
        pr_binwidth = 0.2
        pr_bins = np.arange(0.0, 1., pr_binwidth)
        pcolors = ['r', 'g', 'blue', 'cyan', 'magenta', 'y', 'orange', 'navy', 'pink', 'purple']

        print('Decision-Tree Probabilities')
        f = fig.add_subplot(plot_offset6 + nplot1)
        f.set_xlabel('Test-Data Decision-Tree SNIa Probability')
        f.set_ylabel('Number')
        f.set_yscale('log')
        f.tick_params(axis='both', which='major', labelsize=12)
        f.set_xlim(0, 1.0)
        for npr, pbin in enumerate(pr_bins):
            if (npr < len(pr_bins)):
                probcut = (allprobs[:, 0] >= pr_bins[npr]) & (allprobs[:, 0] < pr_bins[npr] + pr_binwidth) & (trueIa)
                plabel = str(pr_bins[npr]) + '-' + str(pr_bins[npr] + pr_binwidth)
            else:
                probcut = (allprobs[:, 0] >= pr_bins[npr]) & (allprobs[:, 0] <= pr_bins[npr] + pr_binwidth) & (
                    trueIa)  # include 1.0 in last bin
                plabel = str(pr_bins[npr]) + '-' + str(pr_bins[npr] + pr_binwidth)
            probdata = pdata[probcut][:, 0]  # find tree probs for this pr_bin for this SN type
            print('Found {} SNIa matching cuts for pr_bin {}: {}-{} ({} prob. values)'.format(len(probdata),
                                                                                              npr, pr_bins[npr], 
                                                                                              pr_bins[npr] + pr_binwidth,
                                                                                              len(probdata.flatten())))
            plt.hist(probdata.flatten(), bins=p_bins, color=pcolors[npr], histtype='step', alpha=totalpha,
                     label=plabel)  # ,normed=True)
        f.legend(loc='upper center', scatterpoints=1, ncol=Ncols, fontsize='small')
        

        if (args.nclass == 2):
            f = fig.add_subplot(plot_offset6 + nplot2)
            f.set_xlabel('Test-Data Decision-Tree SNCC Probability')
            f.set_ylabel('Number')
            f.set_yscale('log')
            f.tick_params(axis='both', which='major', labelsize=12)
            f.set_xlim(0, 1.0)
            for npr, pbin in enumerate(pr_bins):
                if (npr < len(pr_bins)):
                    probcut = (allprobs[:, 1] >= pr_bins[npr]) & (allprobs[:, 1] < pr_bins[npr] + pr_binwidth) & (trueCC)
                    plabel = str(pr_bins[npr]) + '-' + str(pr_bins[npr] + pr_binwidth)
                else:
                    probcut = (allprobs[:, 1] >= pr_bins[npr]) & (allprobs[:, 1] <= pr_bins[npr] + pr_binwidth) & (
                        trueCC)  # include 1.0 in last bin
                    plabel = str(pr_bins[npr]) + '-' + str(pr_bins[npr] + pr_binwidth)
                probdata = pdata[probcut][:, 1]  # find tree probs for this pr_bin for this SN type
                print('Found {} SNCC matching cuts for pr_bin {}: {}-{} ({} prob. values)'.format(len(probdata),
                                                                                                  npr, pr_bins[npr],
                                                                                                  pr_bins[npr] + pr_binwidth,
                                                                                                  len(probdata.flatten())))
                plt.hist(probdata.flatten(), bins=p_bins, color=pcolors[npr], histtype='step', alpha=totalpha,
                         label=plabel)  # ,normed=True)
            f.legend(loc='upper center', scatterpoints=1, ncol=Ncols, fontsize='small')
        else:
            f = fig.add_subplot(plot_offset6 + nplot2)
            f.set_xlabel('Test-Data Decision-Tree SNIbc Probability')
            f.set_ylabel('Number')
            f.set_yscale('log')
            f.tick_params(axis='both', which='major', labelsize=12)
            f.set_xlim(0, 1.0)
            for npr, pbin in enumerate(pr_bins):
                if (npr < len(pr_bins)):
                    probcut = (allprobs[:, 1] >= pr_bins[npr]) & (allprobs[:, 1] < pr_bins[npr] + pr_binwidth) & (trueIbc)
                    plabel = str(pr_bins[npr]) + '-' + str(pr_bins[npr] + pr_binwidth)
                else:
                    probcut = (allprobs[:, 1] >= pr_bins[npr]) & (allprobs[:, 1] <= pr_bins[npr] + pr_binwidth) & (
                        trueIbc)  # include 1.0 in last bin
                    plabel = str(pr_bins[npr]) + '-' + str(pr_bins[npr] + pr_binwidth)
                probdata = pdata[probcut][:, 1]  # find tree probs for this pr_bin for this SN type
                print('Found {} SNIbc matching cuts for pr_bin {}: {}-{} ({} prob. values)'.format(len(probdata),
                                                                                                  npr, pr_bins[npr],
                                                                                                  pr_bins[npr] + pr_binwidth,
                                                                                                  len(probdata.flatten())))
                plt.hist(probdata.flatten(), bins=p_bins, color=pcolors[npr], histtype='step', alpha=totalpha,
                         label=plabel)  # ,normed=True)
            f.legend(loc='upper center', scatterpoints=1, ncol=Ncols, fontsize='small')

            f = fig.add_subplot(plot_offset6 + nplot3)
            f.set_xlabel('Test-Data Decision-Tree SNII Probability')
            f.set_ylabel('Number')
            f.set_yscale('log')
            f.tick_params(axis='both', which='major', labelsize=12)
            f.set_xlim(0, 1.0)
            for npr, pbin in enumerate(pr_bins):
                if (npr < len(pr_bins)):
                    probcut = (allprobs[:, 2] >= pr_bins[npr]) & (allprobs[:, 2] < pr_bins[npr] + pr_binwidth) & (trueII)
                    plabel = str(pr_bins[npr]) + '-' + str(pr_bins[npr] + pr_binwidth)
                else:
                    probcut = (allprobs[:, 2] >= pr_bins[npr]) & (allprobs[:, 2] <= pr_bins[npr] + pr_binwidth) & (
                        trueII)  # include 1.0 in last bin
                    plabel = str(pr_bins[npr]) + '-' + str(pr_bins[npr] + pr_binwidth)
                probdata = pdata[probcut][:, 2]  # find tree probs for this pr_bin for this SN type
                print('Found {} SNII matching cuts for pr_bin {}: {}-{} ({} prob. values)'.format(len(probdata),
                                                                                                  npr, pr_bins[npr],
                                                                                                  pr_bins[npr] + pr_binwidth,
                                                                                                  len(probdata.flatten())))
                plt.hist(probdata.flatten(), bins=p_bins, color=pcolors[npr], histtype='step', alpha=totalpha,
                         label=plabel)  # ,normed=True)
                f.legend(loc='upper center', scatterpoints=1, ncol=Ncols, fontsize='small')

        f = fig.add_subplot(plot_offset6 + nplot4)
        f.set_xlabel('Test-Data Random Forest SNIa Probability')
        f.set_ylabel('SNIa Probability Variance')
        f.tick_params(axis='both', which='major', labelsize=12)
        f.scatter(probs[trueIa], variance[trueIa][:, 0], marker='.', alpha=scattalpha, color=Iacol, label='SNIa')
        if (args.nclass == 3):
            f.scatter(probs[trueIbc], variance[trueIbc][:, 0], marker='.', alpha=scattalpha, color=Ibccol, label='SNIbc')
            f.scatter(probs[trueII], variance[trueII][:, 0], marker='.', alpha=scattalpha, color=IIcol, label='SNII')
        else:
            f.scatter(probs[trueCC], variance[trueCC][:, 0], marker='.', alpha=scattalpha, color=CCcol, label='SNCC')
        f.legend(loc='best', fontsize='small', scatterpoints=1)

        f = fig.add_subplot(plot_offset6 + nplot5)
        f.set_xlabel('Test-Data Random Forest SNIa Probability')
        f.set_ylabel(str(percentile) + 'th Percentile Upper Limit')
        f.tick_params(axis='both', which='major', labelsize=12)
        f.scatter(probs[trueIa], err_up[trueIa][:, 0], marker='.', alpha=scattalpha, color=Iacol, label='SNIa')
        if (args.nclass == 3):
            f.scatter(probs[trueIbc], err_up[trueIbc][:, 0], marker='.', alpha=scattalpha, color=Ibccol, label='SNIbc')
            f.scatter(probs[trueII], err_up[trueII][:, 0], marker='.', alpha=scattalpha, color=IIcol, label='SNII')
        else:
            f.scatter(probs[trueCC], err_up[trueCC][:, 0], marker='.', alpha=scattalpha, color=CCcol, label='SNCC')
        f.legend(loc='lower right', fontsize='small', scatterpoints=1)

        f = fig.add_subplot(plot_offset6 + nplot6)
        f.set_xlabel('Test-Data Random Forest SNIa Probability')
        f.set_ylabel(str(percentile) + 'th Percentile Lower Limit')
        f.tick_params(axis='both', which='major', labelsize=12)
        f.scatter(probs[trueIa], err_down[trueIa][:, 0], marker='.', alpha=scattalpha, color=Iacol, label='SNIa')
        if (args.nclass == 3):
            f.scatter(probs[trueIbc], err_down[trueIbc][:, 0], marker='.', alpha=scattalpha, color=Ibccol, label='SNIbc')
            f.scatter(probs[trueII], err_down[trueII][:, 0], marker='.', alpha=scattalpha, color=IIcol, label='SNII')
        else:
            f.scatter(probs[trueCC], err_down[trueCC][:, 0], marker='.', alpha=scattalpha, color=CCcol, label='SNCC')
        f.legend(loc='upper left', fontsize='small', scatterpoints=1)

        fig.tight_layout()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            multiPdf.savefig(fig)
    # endif--prvar

    return
