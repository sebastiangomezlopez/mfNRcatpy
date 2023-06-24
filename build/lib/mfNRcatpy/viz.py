import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker

import base as bs
import utils as ut
import snr as snr

def wf_regions(td_obj, ax=None,**kwargs):
    '''
    this is function that draws square contours 
    to make easier, the regions of the timeseries
    on is taking into account
    
    input:
    
    td_obj  --> time_domain objects which packs time(x axis data)
                                         and strain(y axis data)
    ax      --> pyplot axes object as a container for the plot
    
    **kwargs--> section: part pf the plot one wants to crop
                         must be a tuple and it has two cases:
                         
                * to draw just one: 
                  section=('postm'/'prem'/'merg',
                            (t_init, t_fin))
                * to draw the three of them:
                  section=('all',((t_init, t_fin)  prem
                                  (t_init, t_fin)  merg 
                                  (t_init, t_fin)) postm
                                  
    example:
                         
        fig = plt.figure(figsize=(15,5))
        ax0 = fig.add_subplot(2,1,1)
        ax1 = fig.add_subplot(2,1,2)  
        wf_regions(wf0,ax=ax0, section=('merg', merg))
        wf_regions(wf0,ax=ax1, section=('all', (prem, merg, postm)))
        
    Output:
        pyplot axes object filled with a time series and the regions 
        marked with dashed squares
    
    
    '''
    height = np.max(td_obj.data)*(1+0.1)
    if ax is None:
        ax = plt.gca()
        
    ax.plot(td_obj.td*1e3, td_obj.data)

    if 'section' in kwargs:
        key = kwargs['section'][0]
        value = kwargs['section'][1]
        
        if key=='postm' or key=='prem' or key=='merg':
            ax.add_patch( Rectangle((value[0],-height),value[1]-value[0], 2*height,
                                    fc ='none', ec ='red', lw = 1, linestyle='--'))
            print(0)
        elif key=='all':
            color = ['red', 'green','orange']
            for i in range(len(value)):
                ax.add_patch( Rectangle((value[i][0],-height),value[i][1]-value[i][0], 2*height,
                        fc ='none', ec =color[i], lw = 1, linestyle='--'))
    return ax
    
    
def rjct_criteria_monochtempl(f_mesh,d_mesh,dt,wl=1):
    '''
    This funtion creates a mask wich rejects points in the 2D parameter space for 
    monochromatic wave templates. It does it based on:
        *Nyquist thorem, templates must have a sampling rate haigh enough to avoid 
                         aliasing
        *templates with less than one full cycle are rejected
    
    Input:
        f_mesh:2D array, frequency mesh
        d_mesh:2D array, duration mesh
        dt: float, sampling rate
    Output:
        msk1, mask(2D numpy array of bools) which is False at those rejected spots
              in the param space where one gets an unaccepted template
     
    Example:
        accepted_tmpl = rjct_criteria(templ_args[params[0]],templ_args[params[1]], wf_td.dt) 
    '''
    #msk0 = np.logical_and((1/f_mesh)<d_mesh, np.ones_like(f_mesh)*dt < 1/(*f_mesh))
    msk0 = wl*(1/f_mesh)<d_mesh
    msk1 = np.logical_not(msk0)
    return msk1
    
    
def best_SNR(B, wf, sc_ps,*templ_args,params=None,
             durfix=None,rjct_criteria=None, show=True):
    '''
    This is a plotter function that maps 2D surfaces embeded en a 3D space, where
    x,y & z coordinates are frequency,duration & SNR_rec
    
    Input:
    
        B: ND array, containing recovered SNR from a recursive matched filtering algorithm
           on a bank of templates which depend on *templ_args
           
        *templ_args: ND arrays, which are n-dimensional meshs, on which the template model
                     depends
        
        params: tuple, to indicate which of the N parameters are used in the 2D surface. By 
                default one should choose (0,1)
        
        rjct_criteria: ND mask, mask for rejecting points of parameter space which for some reason
                       are not wanted in the analysis
                
    Output:
        fig: contains all the plots 
        ind: tuple associated with the spot in which the maximum recovered SNR is 
        
    '''
    
    if params is None:
        params=(0,1)
    
    if rjct_criteria is not None:
        B[rjct_criteria]=np.nan
    

    if durfix is None:
        ind = np.unravel_index(np.nanargmax(B), B.shape)
    else:
        ind1 = np.argwhere(templ_args[params[1]]>=durfix).flatten()[0]
        ind2 = np.nanargmax(B[ind1,:])
        ind = (ind1, ind2)


    # this is for plotting pursposes, setting all the invalid values to zero SNR 
    if rjct_criteria is not None:
        B[rjct_criteria]= 0.0

    S_max = B[ind]
    par0_max = templ_args[params[0]][ind]
    par1_max = templ_args[params[1]][ind]

    temp = ut.monochr_templ(par0_max,par1_max,wf.dt, t_0=wf.t_0, amp=np.mean(wf.GW_ampl))

    uf = wf.hp_get_tdobj().to_fd()
    tf = temp.to_fd()   

    if show==True:
        fig = plt.figure(figsize=(7,4.5),constrained_layout=True)
        fig.subplots_adjust(wspace=5, hspace=0.8)
        #gs = fig.add_gridspec(8,28)    

        #ax0 = fig.add_subplot(gs[1:7,17:])
        #ax0.set_xlabel('f[Hz]', fontsize=15)
        #ax0.set_ylabel(r'$\mathrm{\sqrt{f}\cdot h(f)\;\;}$'+r'[$\mathrm{\frac{1}{\sqrt{Hz}}}$]', fontsize=15)
        ax1 = fig.add_subplot()#gs[:,:10])
        ax1.set_xlabel('$\mathrm{f[Hz]}$', fontsize=30)
        ax1.set_ylabel('$\mathrm{d[ms]}$', fontsize=30)

        im1 = ax1.pcolormesh(templ_args[params[0]],templ_args[params[1]]*1e3,B,cmap='cubehelix',shading='nearest',rasterized=True,antialiased=True, linewidth=0.0)
        cbar= fig.colorbar(im1, ax=ax1)
        cbar.set_label(r'$\mathrm{\rho}$', labelpad=10,rotation=0, fontsize=30)
        cbar.ax.tick_params(labelsize=20)
        ax1.scatter(par0_max,par1_max*1e3,c='red')
        ax1.axhline(par1_max*1e3, color='red', linestyle='dashed')
        ax1.tick_params(axis='both', which='major', color='black', labelsize=20)


        #ax0.plot(uf.fd, np.sqrt(np.abs(uf.fd))*np.abs(uf.data),label=r'$\mathrm{h_+ postm}$')
        #ax0.plot(tf.fd, np.sqrt(np.abs(tf.fd))*np.abs(tf.data),label='best template')
        #ax0.plot(sc_ps.fd, sc_ps.ASD, label='ASD ET')
        #ax0.set_xscale('log')
        #ax0.set_yscale('log')

        #ax0.set_xticks([20, 200, 2000, 5000, 10000])
        #ax0.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        #ax0.tick_params(axis='x', which='major', color='black', width=2, rotation=-45, length=10)

        #ax0.set_xlim(10, 12000)
        #ax0.set_ylim(0, 1e-20)
        #fig.legend(fontsize=12)
        plt.tight_layout()
    
    return par0_max, par1_max, S_max


def best_monochr(f, d, wf_td, sc_ps, max_templ_dur=None, taufix=None, phase_opt=True, show=True):
    '''
    This function takes the output of the best_SNR function, and shows several
    plots in the time domain and frequency domain for the case of monochromatic
    template models
    
    Input:
    
        mesh_ind: tuple, contains the position of the maximum
        f_mesh: 2D array, frequency mesh
        d_mesh: 2D array, duration mesh
        
        sc_ps: Sensitivity_curve object
               which hold the following information:
               self.fd-> non-equally spaced array
               self.ASD-> non-equally spaced array
               self.PSD-> non-equally spaced array
        
        wf_td: time_domain object
               which holds the following attributes
                    dt, sampling rate
                    dur, duration
                    t_0, initial phase
                    td, time array 
                    data, data array
         
        fig: pyplot figure object, default is None, if it is not the case, the user should pass
             a custom figure, e.g. fig=plt.figure(figsize=(15,5))
    
    Output:
        fig, contains all the plots
    '''
    phi = 0
    for i in range(2):
        tmpl_td = ut.monochr_templ(f, d, wf_td.dt, t_0=wf_td.t_0, phase=-phi, amp=np.sqrt(np.mean(wf_td.data**2)))
        if i==0:
            complex_SNR = (snr.matched_filtering(wf_td,tmpl_td, sc_ps)).recovered_SNR1(max_templ_dur=max_templ_dur)
            
            complex_SNR_s = complex_SNR[0]            
            complex_SNR_ph = complex_SNR[1]
            #complex_SNR_s = complex_SNR_s.cut((0, complex_SNR_s.td[-1]))
            #complex_SNR_ph = complex_SNR_ph.cut((0, complex_SNR_ph.td[-1]))            
            
            if taufix is None and phase_opt==True:
                ind = np.argmax(np.abs(complex_SNR_s.data))
                tau = complex_SNR_s.td[ind]
                s = np.abs(complex_SNR_s.data)[ind]
                phi = complex_SNR_ph.data[ind]

            elif taufix is None and phase_opt==False:
                ind = np.argmax(complex_SNR_s.data.real)
                tau = complex_SNR_s.td[ind]
                s = np.abs(complex_SNR_s.data)[ind]
                phi = complex_SNR_ph.data[ind]

            elif taufix is not None and phase_opt==True:
                ind = np.argwhere(complex_SNR_s.td>=taufix).flatten()[0]
                tau = complex_SNR_s.td[ind]
                s = np.abs(complex_SNR_s.data)[ind]
                phi = complex_SNR_ph.data[ind]

            elif taufix is not None and phase_opt==False:
                ind = np.argwhere(complex_SNR_s.td>=taufix).flatten()[0]
                tau = complex_SNR_s.td[ind]
                s = complex_SNR_s.data.real[ind]
                phi = complex_SNR_ph.data[ind]
    
    if show==True:
        fig = plt.figure(figsize=(7,4.5))

        gs = fig.add_gridspec(8,16)

        ax0 = fig.add_subplot(gs[:,:])

        ax0.set_ylabel('Strain at 100 Mpc', fontsize=30)
        ax0.set_xlabel('$\mathrm{t-t_{merger}[ms]}$', fontsize=30)
        ax0.plot((wf_td.td-wf_td.t_0)*1e3,wf_td.data,label=r'$\mathrm{h_{+}}$ postmerger')
        ax0.plot((tmpl_td.td+tau-wf_td.t_0)*1e3,tmpl_td.data, linestyle='dashed', label='best template')
        ax0.legend(loc='upper right',fontsize=20)
        plt.tight_layout()
    return  f, d, s, tau, phi  
 
    
    
    
