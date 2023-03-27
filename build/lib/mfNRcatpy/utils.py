import numpy as np
from scipy import signal
import pathlib
import os
from scipy.integrate import cumtrapz, simpson
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import mfNRcatpy.base as bs
import mfNRcatpy.snr as snr





def ASD(filename,directory):
    '''
    This function imports sensitivity design curves from 
    current detectors ligo, virgo, kagra, Einstein telescope
    
    Input:
    
        filename: str, can be chosen from the keywords:
                  ---> 'aligo', 'advirgo', 'et_d', 'kagra'
        directory: str, path to the folder in which the files 
                   are stored
     
    Output:    
        sensitivity curve object:
               --> which contains  non-equally sampled  ASD data
               as a function of the frequency
               --> this objects have some builtin methods like interpolation and extension
                   , that come in handy when doing a matched filtering algorithm.
     
     
     Remarks: PLEASE READ!
         1. directory must end in /
         2. detector ASDs must be downloaded from ligo webpage
            https://dcc.ligo.org/LIGO-T1500293/public, or some 
            similar webpage, with updated versions of the data.
    '''
    freq, strain = np.loadtxt(str(directory)+str(filename)+'.txt').T
    
    return bs.Sensitivity_curve(freq, strain, filename)


def monochr_templ(freq, duration, dt, t_0=0, amp=1, phase=0, alpha = None):
    '''
    This is a function that computes a harmonic function of time
    
    Input:        
        freq: float
        duration: float
        t_0: float, initial phase of the wave
        dt: float, sampling rate of the wave
        
        amp: float, default-> 1e-20, similar order of magnitude as 
             waveforms of GW's from binary systems
    
    Output:
        time_domain object, which holds the following attributes
            dt, sampling rate
            dur, duration
            t_0, initial phase
            td, time array 
            data, data array

    '''
    
    if alpha is None:
        alpha = 0
        
    time = np.arange(t_0, (t_0 + duration), dt)
    tukey  = signal.windows.tukey(time.size, alpha=alpha)
    
    templ_r = amp * np.sin((2*np.pi*freq*time) + phase) * tukey
    
    return bs.time_domain(time, templ_r)
   

def rjct_criteria_monochtempl(f_mesh,d_mesh,dt, wl=2):
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
    msk0 = np.logical_and((wl/f_mesh)<d_mesh, np.ones_like(f_mesh)*dt < 1/(f_mesh))
    #msk0 = wl*(1/f_mesh)<d_mesh
    msk1 = np.logical_not(msk0)
    return msk1




def white_noise_fd(F, magnitude=1.0e-22):
    '''
    Test function which creates constant noise curve power
    in the frequency domain.
    
    Input:
        F: 1D array, frequency region in which the noise should defined
        magnitude: float, default->1.0e-22
        
    Output:
        Power_spectrum object
    '''
    ASD_noise = np.sqrt(magnitude) * np.ones_like(F)
    return bs.Sensitivity_curve(F, ASD_noise, 'test_noise')

def find_postm(wf_obj, before=None, after=None, alpha=None):
    '''
    it crops the original waveform to just the postm part
    
    it computes the first local maximum of the
    instantaneous frequency as a function of time 
    as a way to define the beginning of the postmerger phase
    
    ''' 
    if before is None:
        before = 0
    
    if after is None:
        after = 0
     
        
    if alpha is None:
        alpha = 0.2
        
    md = wf_obj.mdta.copy()
    
    N_b = int(before/wf_obj.dt)
    N_a = int(after/wf_obj.dt)
        
    ind = np.argmax(wf_obj.GW_ampl)-N_b
    ind1 = np.argwhere(wf_obj.GW_ampl[ind:]>= 0.05*wf_obj.GW_ampl[ind:].max()).flatten()[-1] + N_a
    
    tukey  = signal.windows.tukey(wf_obj.td[ind:ind+ind1].size, alpha=alpha)
    wf = bs.waveform(wf_obj.td[ind:ind+ind1], wf_obj.hp[ind:ind+ind1]*tukey,wf_obj.hc[ind:ind+ind1]*tukey, md) 
    
    return wf



def find_h5_infd(path):
    directory = os.path.expanduser(path)
    file_ext = '**/*.h5'
    a = list(pathlib.Path(directory).glob(file_ext))
    for i in range(len(a)):
        a[i]=str(a[i])
    return a




def optimal_template(wf,sc,SNR, F, D, 
                     wl, tau, pfind,N,
                     th22max, cut_sc=None, 
                     plot=True, alpha=None,
                     phase_opt=True, zoom=None):
    
    if alpha is None:
        alpha=0
        
    if zoom is None:
        zoom = 0.4
    
    SNR1 = SNR.copy()
    mask = rjct_criteria_monochtempl(F, D, wf.dt,wl=wl)
    SNR1[mask] = np.nan
    SNR1[mask] = np.nan


    indT = np.unravel_index(np.nanargmax(SNR1), SNR1.shape)
    fmaxT, dmaxT, smaxT = F[indT], D[indT], SNR1[indT]
    
    
    u = monochr_templ(fmaxT, dmaxT, wf.dt, wf.t_0,
                      amp=1e-22, phase=0, alpha=alpha)
    S = snr.matched_filtering(wf.hp_get_tdobj(), u, sc)
    s = S.recovered_SNR3(N=N, cut_sc=cut_sc)
    sopt = np.abs(S.optimal_SNR(N=N, cut_sc=cut_sc).data).max()

    if pfind is not None:
        s = s.cut((pfind, s.td[-1]))
    ph = bs.time_domain(s.td, np.unwrap(np.angle(s.data)))
    val = tau
    if val is None:
        if phase_opt==True:
            ind = np.argmax(np.abs(s.data))
        elif phase_opt==False:
            ind = np.argmax(s.data.real)
    else:
        ind = np.where(s.td>=val)[0][0]

    phimax, taumax, smax = ph.data[ind], s.td[ind], s.data[ind]
    
    if phase_opt==True:
        assert np.isclose(smaxT,np.abs(smax))
    elif phase_opt==False:
        assert np.isclose(smaxT,smax.real)
    
    if plot==True:
        fig1 = plt.figure(dpi=900)
        im = plt.pcolormesh(F, D*1e3, SNR1/sopt,
                            shading='gouraud',cmap='cubehelix',
                           antialiased=True,rasterized=True)
        im.set_clim(0,1)
        fig1.colorbar(im, label=r'$\mathrm{\alpha=\frac{\rho_{rec}}{\rho_{opt}}}$', orientation='horizontal')
        #plt.title(f'SNR={smaxT:.3f} at f={fmaxT:.2f}[Hz], d={}')
        plt.scatter(fmaxT, dmaxT*1e3, marker='x', c='black', s=7)
        
        y = D.T[0][-1]*1e3*0.3
        x = F[0][-1]
        st = 'Best match found:\n'+f'f={fmaxT:.2f}[Hz]\n d={dmaxT*1e3:.2f}[ms]\n'+r'$\mathrm{{\tau}}$'+f'={taumax*1e3:.2f}[ms]\n'+r'$\phi$'+f'={phimax:.2f}[rad]\n'+r'$\alpha$'+f'={abs(smax)/sopt:.5f}'
        plt.text(1.05*x, y, st, fontsize=10)
        plt.xlabel('f [Hz]', fontsize=10)
        plt.ylabel('d [ms]',fontsize=10)
        
        plt.tight_layout()
        
        fig2 = plt.figure(dpi=900)
        plt.subplot(1,2,1)
        plt.title(f'SNRmax={np.abs(smax):.5f}')
        plt.plot(s.td*1e3, s.data.real)
        plt.plot(s.td*1e3, s.data.imag)
        plt.plot(s.td*1e3, np.abs(s.data))
        plt.ylabel(r'$\mathrm{\alpha=\frac{SNR_{rec}}{SNR_{opt}}}$')
        plt.xlabel(r'$\tau$ [ms]')
        plt.axhline(np.abs(smax), linestyle='dashed', color='r')
        plt.axvline(taumax*1e3, linestyle='dashed', color='r')


        plt.subplot(1,2,2)
        plt.title(f'taumax={taumax*1e3:.5f} [ms]')
        plt.plot(ph.td*1e3, ph.data)
        plt.axhline(phimax, linestyle='dashed', color='r')
        plt.axvline(taumax*1e3, linestyle='dashed', color='r')
        plt.ylabel(r'$\mathrm{\phi_0}$')
        plt.xlabel(r'$\tau$ ms')
       
        plt.tight_layout()
        
        u = monochr_templ(fmaxT, dmaxT, wf.dt, 
                     wf.t_0, amp=1.2*np.mean(wf.GW_ampl), phase=phimax, alpha=alpha)
        w = wf.hp_get_tdobj()
        Mf = snr.matched_filtering(w, u, sc)
        w1, u1 = Mf.length_matching2(N=N)

        m = wf.GW_ampl.max()
        fig3  = plt.figure(dpi=900)
        ax = fig3.add_subplot()
        md = wf.mdta.copy()
        name = md['name']
        #ax.set_title(name, fontsize=10)
        ax.text(0.7*wf.dur*1e3, -1.8*m,name, fontsize=10)
        ax.plot(w1.td*1e3, w1.data)
        ax.plot((u1.td+taumax)*1e3, u1.data)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%1.0e'))

        

        m = wf.GW_ampl.max()
        ax.set_ylim(-2.*m, 2.*m)
        ax.set_xlim(-0.1*wf.dur*1e3, 1.1*wf.dur*1e3)
        ax.set_xlabel(r'$\mathrm{t-t_{merg}}$[ms]')
        ax.set_ylabel('strain @ 100.0 MPc')
        
        axins = ax.inset_axes([0.55, 0.7, 0.4, 0.2])
        axins.plot(w1.td*1e3, w1.data)
        axins.plot((u1.td+taumax)*1e3, u1.data)
        axins.set_xlim(-0.05*wf.dur*1e3, zoom*wf.dur*1e3)
        axins.set_ylim(-1.2*m, 1.2*m)

       
        
        axins.set_xticks([])
        axins.set_yticks([])
        
        ax.indicate_inset_zoom(axins, edgecolor="black")
        
        plt.tight_layout()
        
        return fig1, fig2, fig3, fmaxT, dmaxT, taumax, phimax, np.abs(smax), sopt
    else:
        return fmaxT, dmaxT, taumax, phimax, np.abs(smax), sopt
