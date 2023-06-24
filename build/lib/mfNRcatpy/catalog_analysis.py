import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import lal

import utils0 as ut
import viz as viz
import snr as snr
from lal_wf import *
from core_wf import *
from eosdb import catalog


from scipy.integrate import simpson
from tqdm import tqdm

Y_22 = lal.SpinWeightedSphericalHarmonic(0,0,-2,2,2)

def catalogue_analysis(waveform_bank, func, sc,**func_kwargs):
    
    mode = func_kwargs['mode']
    
    R={}
    for i in range(0,len(waveform_bank['sims'])):
        print(waveform_bank['sims'][i])
        if mode=='core':
            wf = CoRe_NR_wf22(waveform_bank['db_path'], waveform_bank['sims'][i], waveform_bank['ex_param'], resol='best')
        elif mode=='lal':
            wf = lal_NR_wf1(waveform_bank['sims'][i],2,2, waveform_bank['ex_param']['dt'], waveform_bank['ex_param'])

        R[waveform_bank['sims'][i]] = func(wf, sc,**func_kwargs)
    return R



def full_analysis(waveform0, sc0, fbounds=None,Nf=None,
                  Nd=None, taufix=None,phase_opt=True,
                  mode=None, energetics=True, cut=None):
    
    if fbounds is None:
        fdounds = (500, 3500)
    if Nf is None:
        Nf=2
    if Nd is None:
        Nd = 2
    if mode is None:
        mode='core'
    
    tau1 = taufix
    pfind = cut
    
    wf_p0 = ut.find_postm(waveform0, before=0)
    templ = lambda f,d,dt,t_0=wf_p0.t_0,amp=1:ut.monochr_templ(f, d, dt, t_0,amp, phase=0)
    
    # Matched filtering
    f = np.linspace(fbounds[0],fbounds[1], Nf)
    d = np.linspace(0.1e-3, 1*wf_p0.dur, Nd) 
    
    f_mesh0, d_mesh0 = np.meshgrid(f,d)
    
    S = snr.genrlzd_SNR1(wf_p0.hp_get_tdobj(),templ,sc0,
                   f_mesh0, d_mesh0, N=2000, cut_sc=None,
                   taufix=tau1, phase_opt=True, cut_pfind=pfind)
    res = ut.optimal_template(wf_p0, sc0, S,
                 f_mesh0, d_mesh0, 3, tau1,
                 pfind, 2000, cut_sc=None,
                 plot=False, zoom=-5)
    fmax, dmax, taumax, phimax, smax, sopt = res
  
    
    if energetics==True:
        dist_m = wf_p0.mdta['dist_mpc'] * 3.086e22
        h22 = (1/Y_22)*(wf_p0.hp_get_tdobj().data+1j*wf_p0.hc_get_tdobj().data)
        _, L = ut.energylm(2,2, h22, wf_p0.td, dist_m)
        _, T = ut.angmom_Z_lm(2,2, h22, wf_p0.td,dist_m)
        e_rad, J_rad = simpson(L.data, L.td), simpson(T.data, T.td)
    
    result = {}

    result['best_f'] = (fmax, 'Hz')
    result['best_d'] = (dmax, 's')
    result['best_tau'] = (taumax, 's')
    result['best_phi'] = (phimax, 'rad')
    
    result['wf_d'] = (wf_p0.dur, 's')
    result['wf_d_wei'] = (wf_p0.amp_wei_time(),'s')
    result['wf_f_wei'] = (wf_p0.amp_wei_freq(), 'Hz')
    
    result['SNR_rec'] = (smax, 'dmless')
    result['SNR_opt'] = (sopt, 'dmless')

    
    result['mdta'] = dict(wf_p0.mdta)
    
    result['mdta']['E'] = (e_rad,'Msun')
    result['mdta']['Jz'] = (J_rad, 'Msun^2')
    
    if result['mdta']['ecc']=='':
        result['mdta']['ecc']=np.nan
    
    if mode=='core':
        if result['mdta']['eos'] in EOS_masses().keys():
            result['mdta']['maxm_TOV'] = EOS_masses()[result['mdta']['eos']]['maxm_TOV']
        else:
            result['mdta']['maxm_TOV'] = np.nan
            
    elif mode=='lal':
        cat = catalog.load_catalog()
        eos = result['mdta']['eos'].split('hybrid-pp-')[::-1][0] 
        if eos.upper()+'_LAL'  in cat.keys():
            result['mdta']['maxm_TOV'] = cat[eos.upper()+'_LAL'].tov.mg_max
        elif eos.upper()+'_EPP' in cat.keys():
            result['mdta']['maxm_TOV'] = cat[eos.upper()+'_EPP'].tov.mg_max
        else:
            result['mdta']['maxm_TOV'] = np.nan
    
    return result

#def wf_vizlysis(wf, templ_bank, path,taufix=None, sc_name='et_d',mode='core'):
#    path_code = os.path.expanduser('~/REPOS/sebas-training/thesis/Postmerger/')
#    sc = ut.ASD(sc_name, path_code)
#    templ_func = lambda fr,du,dt,t_0=0, amp=1 :ut.monochr_templ(fr,du,dt,t_0,amp)
#    freq_mesh, dur_mesh = np.meshgrid(templ_bank['frequency'], #templ_bank['duration'])
#    
#    fig = plt.figure(figsize=(20,20))
#    
#    if mode=='lal':
#        direc = path+'/'+(wf.mdta['name'].split(':R')[0])
#        if os.path.exists(direc):
#            shutil.rmtree(direc)
#        os.mkdir(direc)
#        new_path = path+'/'+(wf.mdta['name'].split(':R'))[0]
#
#        wf_p = ut.find_postm(wf).hp_get_tdobj()    
#        N = np.round(np.max(templ_bank['duration'])/wf_p.dur, decimals=0)
#        tmax=N*wf_p.dur
#
#        mask = viz.rjct_criteria_monochtempl(freq_mesh, dur_mesh, wf_p.dt)
#        S = genrlzd_SNR(wf_p,templ_func,sc,tmax,freq_mesh, dur_mesh,taufix=taufix)
#
#        fig.clear()
#        fig, ind = viz.best_SNR(S,freq_mesh, dur_mesh, 
#                          rjct_criteria=mask, fig=fig,
#                          xlabel='frequency[Hz]',ylabel='duration[s]')
#
#        fig.savefig(new_path+'/'+'surf.jpg')
#
#        fig.clear()
#        wf.show_wf(mdta=True, xlabel='time[s]', ylabel='strain', fig=fig)
#        fig.savefig(new_path+'/'+'wf.jpg')
#
#        fig.clear()
#        viz.best_monochr(ind,freq_mesh, dur_mesh, wf_p, sc, fig=fig)
#        fig.savefig(new_path+'/'+'best_match.jpg')
#        plt.close()
#    elif mode=='core':
#        direc = path+'/'+(wf.mdta['name'].split(':R')[0])
#        if os.path.exists(direc):
#            shutil.rmtree(direc)
#        os.mkdir(direc)
#        new_path = path+'/'+(wf.mdta['name'].split(':R'))[0]

#        wf_p = ut.find_postm(wf).hp_get_tdobj()    
#        N = np.round(np.max(templ_bank['duration'])/wf_p.dur, decimals=0)
#        tmax=N*wf_p.dur
#
#        mask = viz.rjct_criteria_monochtempl(freq_mesh, dur_mesh, wf_p.dt)
#        S = genrlzd_SNR(wf_p,templ_func,sc,tmax,freq_mesh, dur_mesh)
#
#        fig.clear()
#        fig, ind = viz.best_SNR(S,freq_mesh, dur_mesh, 
#                          rjct_criteria=None, fig=fig,
#                          xlabel='frequency[Hz]',ylabel='duration[s]')
#
#        fig.savefig(new_path+'/'+'surf.jpg')
#
#        fig.clear()
#        wf.show_wf(mdta=True, xlabel='time[s]', ylabel='strain', fig=fig)
#        fig.savefig(new_path+'/'+'wf.jpg')
#
#        fig.clear()
#        viz.best_monochr(ind,freq_mesh, dur_mesh, wf_p, sc, fig=fig)
#        fig.savefig(new_path+'/'+'best_match.jpg')
#        plt.close()
#    return print('the analysis of the database has been stored in '+'\n'+new_path)

#path_to = os.path.expanduser('~/analysis') #path to store plots
#snr.wf_vizlysis(wf0, template_bank, path_to)
#path_to_img =  path_to +'/'+ lal_NR_wf(lal_list[0], deltat, src).mdta['name']
#path_to_img =  path_to +'/'+ waveform_bank['core'][0]
#viz.show_results(path_to_img)



def show_results(path_to_img):
    img = [mpimg.imread(path_to_img+'/'+'surf.jpg'),
           mpimg.imread(path_to_img+'/'+'wf.jpg'),
           mpimg.imread(path_to_img+'/'+'best_match.jpg')]

    size=(20,10) 

    ax0 = plt.figure(figsize=size).add_subplot()
    ax1 = plt.figure(figsize=size).add_subplot()
    ax2 = plt.figure(figsize=(30,20)).add_subplot()

    ax0.axis('off')
    ax1.axis('off')
    ax2.axis('off')

    ax0.imshow(img[0])
    ax2.imshow(img[2])
    ax1.imshow(img[1])

    ax0.set_ylim(1000, 200)
    ax0.set_xlim(130, 1300)

    ax1.set_ylim(1000, 450)
    ax1.set_xlim(130, 1300)

    ax2.set_ylim(1300, 160)
    ax2.set_xlim(130, 1300)
    plt.show()
    
    
def catalog_plot(q, R, ax=None, mode='lal',show_ticks=True, EOS_color_patellete=None, step=4):
    
    list_ticks = list(R.keys())
    ind = (np.arange(len(q)+1, step=step))
    ind[1:] = np.arange(len(q)+1, step=step)[1:]-1
    sims1 = np.array(list_ticks)[ind]
    
    C = EOS_color_patellete
    
    if mode=='lal':
        rel = lal_relabeling(R)
        for i, sim in enumerate(sims1):
            #name = R[sim]['mdta']['name']
            #sims1[i] = rel[name]
            sims1[i] = rel[sim]

    if ax is None:
        ax = plt.axes()
    if show_ticks==False:
        ax.set_xticks(ind, sims1, rotation= 'vertical')
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    else:
        ax.set_xticks(ind, sims1, rotation= 'vertical')
    E = []
    for i, q in enumerate(q):
        eos = R[list_ticks[i]]['mdta']['eos']
        if eos not in E:
            E.append(eos)
            ax.scatter(i, q, c= C[eos], s=2, label=eos)
        else:
            ax.scatter(i, q, c= C[eos], s=2)
    ax.margins(x=0.009, y=0.5)
    ax.grid()
    ax.locator_params(axis='y', nbins=6)
    return ax

def which_sim(D, mask1, mask2):
    m = np.logical_and(mask1, mask2)    
    L = np.array(list(D.keys()))
    
    return L[m]


def lal_relabeling(A):
    j=1
    D ={}
    for i, name in enumerate(A.keys()):
        eos = A[name]['mdta']['eos']
        if eos=='hybrid-pp-APR4':
            if j<=10:
                #D[A[name]['mdta']['name']] = f'LAL:000{j}'
                D[name] = f'LAL:000{j}'
            else:
                #D[A[name]['mdta']['name']] = f'LAL:00{j}'
                D[name] = f'LAL:00{j}'
            j+=1


    for i, name in enumerate(A.keys()):
        eos = A[name]['mdta']['eos']
        if eos=='SHT':
            if j<=10:
                #D[A[name]['mdta']['name']] = f'LAL:000{j}'
                D[name] = f'LAL:000{j}'

            else:
                #D[A[name]['mdta']['name']] = f'LAL:00{j}'
                D[name] = f'LAL:00{j}'

            j+=1

    for i, name in enumerate(A.keys()):
        eos = A[name]['mdta']['eos']
        if eos=='hybrid-pp-H4':
            if j<=10:
                #D[A[name]['mdta']['name']] = f'LAL:000{j}'
                D[name] = f'LAL:000{j}'

            else:
                #D[A[name]['mdta']['name']] = f'LAL:00{j}'
                D[name] = f'LAL:00{j}'

            j+=1

    for i, name in enumerate(A.keys()):
        eos = A[name]['mdta']['eos']
        if eos=='hybrid-pp-H4':
            if j<=10:
                #D[A[name]['mdta']['name']] = f'LAL:000{j}'
                D[name] = f'LAL:000{j}'

            else:
                #D[A[name]['mdta']['name']] = f'LAL:00{j}'
                D[name] = f'LAL:00{j}'
            j+=1

    for i, name in enumerate(A.keys()):
        eos = A[name]['mdta']['eos']
        if eos=='hybrid-pp-MS1':
            if j<=10:
                #D[A[name]['mdta']['name']] = f'LAL:000{j}'
                D[name] = f'LAL:000{j}'

            else:
                #D[A[name]['mdta']['name']] = f'LAL:00{j}'
                D[name] = f'LAL:00{j}'

            j+=1

    for i, name in enumerate(A.keys()):
        eos = A[name]['mdta']['eos']
        if eos=='hybrid-pp-SLy':
            if j<=10:
                #D[A[name]['mdta']['name']] = f'LAL:000{j}'
                D[name] = f'LAL:000{j}'
            else:
                #D[A[name]['mdta']['name']] = f'LAL:00{j}'
                D[name] = f'LAL:00{j}'

            j+=1
    return D
