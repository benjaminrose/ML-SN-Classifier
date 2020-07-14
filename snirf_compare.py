import sys
import os
import glob
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import joblib
from matplotlib.backends.backend_pdf import PdfPages
import warnings
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

import ML_globals as g

figx = 15
figy = 6
nrows = 1

sizes = {'Title': 20, 'Large': 16, 'Small': 12}
lsize = 16
Default_markers=['o','v','s','d','H','^','D','h','<','>','.']
Default_colors=['blue','r','m','g','navy','y','purple','gray','c','orange','violet',
'coral','gold','orchid','maroon','tomato','sienna','chartreuse','firebrick','SteelBlue']
STR = ' '.join([g.Simulated, g.Training])
STT = ' '.join([g.Simulated, g.Test])
STRD = ' '.join([g.Simulated, g.Training, g.Data])
STTD = ' '.join([g.Simulated, g.Test, g.Data])

def plot_roc(ax, performance={}, label='', color='blue', title='', finish=False):
    if not finish:
        fpr = performance[g.ROC][g.FPR]
        tpr = performance[g.ROC][g.TPR]
        ax.plot(fpr, tpr, lw=2, c=color,
                label='{} (area = {:0.3f})'.format(label, performance[g.ROC][g.AUC]))
    else:
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right', fontsize=lsize)
        ax.set_title('{}'.format(' '.join(['ROC curve', title]), size=sizes['Title'])) 

def plot_pur_vs_eff(ax, performance={}, label='', color='blue',
                     eff_lo=.8, pur_lo=.925, title='', finish=False):
    if not finish:
        ax.plot( performance[g.Purity], performance[g.Efficiency], c=color, label=label)
    else:
        ax.set_ylabel('Efficiency (Recall)')
        ax.set_xlabel('Purity (Precision)')
        ax.legend(loc='lower left', fontsize=lsize)
        ax.set_xlim(pur_lo - 0.025, 1.00)
        ax.set_ylim(eff_lo - 0.025, 1.00)
        ax.set_title('{}'.format(' '.join(['PR Curve', title]), size=sizes['Title']))
                
def plot_calibration(ax, performance={}, label='', color='blue', title='', finish=False):
    meanpr = performance.get(g.Mean_Pr)
    if not finish:
        if meanpr is not None:
            fraction_true = performance.get(g.Fr_True)
            ax.plot(meanpr, fraction_true, label=label, marker='o', color=color)
    else:
        meanpr = np.linspace(0, 1, 20)
        ax.plot(meanpr, meanpr, label='Perfectly Calibrated', linestyle='--', color='black')
        ax.set_xlabel('Random Forest SNIa Probability')
        ax.set_ylabel('Fraction of True SNIa')
        ax.legend(loc='best', fontsize=lsize, numpoints=1)
        ax.set_title('{}'.format(' '.join(['Calibration Curve', title])))#, fontsize=sizes['Title'])

    
def close_page(fig, multiPdf, npage):
    npage += 1
    fig.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multiPdf.savefig(fig)
        
    print('  Wrote page {}'.format(npage))
    plt.close(fig)
    return npage

def main(argv, title='Training Set Size', before='10x10', after='TRAIN', 
         plot_id ='pureff_ROC_prob', mode=g.Test, filedir='performance'):
    file_template = argv[0]
    if len(argv) > 1:
        title = argv[1]
    if len(argv) > 2:
        before = argv[2]
    if len(argv) > 3:
        after = argv[3] 
    if len(argv) > 4:
        plot_id = argv[4] 
    if len(argv) > 5:
        mode = argv[5] 
    files = sorted(glob.glob(file_template+'*'))
    if 'prob' in plot_id:
        plots = [plot_pur_vs_eff, plot_roc, plot_calibration]
    elif 'ROC' in plot_id:
        plots = [plot_pur_vs_eff, plot_roc]
    else:
        plots = [plot_pur_vs_eff]
    pdfname = os.path.join(filedir, '_'.join([plot_id, re.sub(' ', '_', title)]) + '.pdf')
    ncolumns = len(plots)
    if nrows == 1 and ncolumns == 1:
        fig, ax_all  = plt.subplots(nrows, ncolumns, figsize=(figx/2, figy))
        axall = [ax_all]
    else:
        fig, ax_all  = plt.subplots(nrows, ncolumns, figsize=(figx, figy))
        axall = ax_all.flat
    markers = iter(Default_markers)

    print('\nSaving plots to {}'.format(pdfname))
    multiPdf = PdfPages(pdfname)

    page_total = 0
    perf = {}  #load dicts
    for f in files:
        pff = joblib.load(f)
        label = re.sub('_', ' ', f.split(after)[0].split(before)[-1]).strip()
        #label = re.findall('\d+',  f.split(match)[0])[-1]  #get last set of digits in filename
        perf[label] = pff

    print(sorted(perf.keys()))

    #make plots
    for ax, plot in zip(axall, plots):
        colors = iter(Default_colors)
        for label, performance in perf.items():
            if mode == g.Test:
                dkeys = [g.Test]
            else:
                dkeys = [k for k in performance.keys() if g.Training not in k and g.Phot not in k]
            for dkey in dkeys:
                lgnd_label = label if mode==g.Test else ' '.join([dkey, label])
                color = next(colors)
                plot(ax, performance[dkey], label=lgnd_label, color=color)
    
        plot(ax, title=title, finish=True)

    page_total = close_page(fig, multiPdf, page_total)

    
    multiPdf.close()
    plt.rcParams.update({'figure.max_open_warning': 0})

    print('\nWrote {} with {} pages'.format(pdfname, page_total))

        
    return


if __name__ == "__main__":
    main(sys.argv[1:])
    quit()
