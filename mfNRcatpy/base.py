import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal
import h5py

import mfNRcatpy.snr as snr

class power_spectrum(object):

    def __init__(self,frequencies, PSD):
        self.data = PSD
        self.fd = frequencies
        self.df = frequencies[1] - frequencies[0]
        
            
    def PSD_flip_mirror(self, F_in):
        '''
        This method takes a PSD that is only
        defined in the positive side of the time/frequency
        space.

        It symmetrizes the array by mirroring its complex
        conjugate to the left

        Used attributes:
            self.data --> 1D array, data in the frequency domain, must be equally sampled

        Input:
            F_in --> 1D array, frequency range in which one wants to extend it

                     Remark :
                     F_in should be growing ordered

        Output:
            power_spectrum object:   new extended(to the left) array, which is equally sampled
                                     by using its own complex conjugate on the left half plane 
                                     in the frequency domain
                                     
                                     power_spectrum object-> which hold the following information:
                                     self.fd-> equally spaced array
                                     self.data-> equally spaced array
        '''


        PSD_new = np.zeros_like(F_in)
        msk0 = np.logical_and(F_in>=0, F_in<=np.max(F_in)) 
        msk1 = np.logical_and(F_in>=-np.max(F_in), F_in<0) 
        PSD_new[msk0] = self.data
        PSD_new[msk1] = (self.data[1:].conjugate())[::-1]

        return power_spectrum(F_in, PSD_new)
            
        
        
class time_domain(object):
    
    def __init__(self, t, arr):
        self.dt = t[1] - t[0]
        self.dur = t[-1] - t[0]
        self.t_0 = t[0]
        self.td = t
        self.data = arr
        
    def to_fd(self):
        '''
        This is a method that takes a time_domain objects
        and tranforms it to its frequency domain form, by using
        numpy fft algorithm.
        
        Used attributes:
            self.data: 1D array
            delf.dt: float, sampling rate of the time doain data
            
        Input: None
            
        output:
            frequecy_domain object-> holds the following attributes:
                df: sampling rate in the frequency domain
                fd: frequecy array 
                data: data array       
                
        '''
        frequencies = np.fft.fftfreq(self.data.size, self.dt)
        data_fd = np.fft.fft(self.data)*self.dt
        
        frequencies = np.fft.fftshift(frequencies)
        data_fd = np.fft.fftshift(data_fd)
        
        return freq_domain(frequencies, data_fd)
    
    def spectral_phase(self, thresh=None):
        
        '''
        This function computes the spectral phase of any 
        time series. First cleaning all the noise produced 
        when computing the FFTs, using a threshold based 
        technique.
        
        tresh: float, (0, 0.7]-> recommended
        '''
        
        if thresh is None:
            thresh = 0.1
            
        y_f = self.to_fd().data.copy()
        freqs = self.to_fd().fd.copy()
        
        y_f[y_f<thresh*np.abs(y_f).max()] = 0
        
        phase = np.arctan2(y_f.imag, y_f.real)
        
        return freq_domain(freqs, phase)
        
    
    def cut(self, t_cut):
        '''
        This method crops the time_domain objects, to a region of interest
        
        Used attributes:
            self.td -> 1D array
            self.data -> 1D array
        
        Input:
            t_cut -> tuple, should be in miliseconds
                       t_cut[0]=t_lower
                       t_cut[1]=t_higher
        Output:
            time_domain object, which holds the following attributes
                dt, sampling rate
                dur, duration
                t_0, initial phase
                td, time array 
                data, data array
        '''
        msk = np.logical_and(self.td>=t_cut[0], self.td<=t_cut[1])

        new_td = self.td[msk]
        new_data = self.data[msk]
        
        return time_domain(new_td, new_data)
    
    def spectrogram_from_td(self, ax=None, axins=None):
        F = np.fft.fftfreq(self.td.size, self.dt)
        fs = F[1]-F[0]
        N = F.size
        length_pieces = N//10
        f, t, Sxx = signal.spectrogram(self.data, 
                                    fs=1/(self.dt),nperseg=length_pieces, 
                                    nfft=32*length_pieces,noverlap=N//11)

        T, F = np.meshgrid(t, f)
        ind = np.unravel_index(np.argmax(Sxx), Sxx.shape)
        Fmax = F[ind]
        
        T = (T+self.t_0)

        fig = plt.figure(dpi=900)
        
        if ax is None:
            ax = fig.add_subplot()
            
        ax.pcolormesh(T*1e3, F, np.sqrt(Sxx), 
                            shading='gouraud',cmap='cubehelix',
                           antialiased=True,rasterized=True)

        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [ms]')
        
        if axins is not None:

            axins = ax.inset_axes([0.7, 0.7, 0.2, 0.2])
            axins.pcolormesh(T*1e3, F, np.sqrt(Sxx), 
                        shading='goraud', rasterized=True,
                        cmap='YlGnBu', antialiased=True)

            axins.set_xlim(T.min()*1e3, (T.min()+self.dur/5)*1e3)
            if self.dur >7e-3:
                axins.set_ylim(Fmax-500, Fmax+500)
            else:
                axins.set_ylim(10, 5000)
            axins.tick_params(left = False, right = False , labelleft = False ,
                            labelbottom = False, bottom = False)


            ax.indicate_inset_zoom(axins, edgecolor="black");

        if self.dur <=10e-3:
            ax.set_ylim(10, 40000)
        else:
            ax.set_ylim(10, 4000)        
        return ax

        
        
        
class freq_domain(object):
    '''
    Input:
            fft ordered arrays
    '''
    def __init__(self, frequencies, arr_fd):
        self.df = frequencies[1]-frequencies[0]
        self.fd =frequencies
        self.data = arr_fd
        
    def to_td(self):
        time = np.ones_like(self.data)
        data_td = np.fft.ifft(self.data)*self.df
        return time_domain(time, data_td)

            
    
    
class waveform(object):   
    
    def __init__(self, t, hp, hc, src_mdta):
        self.t_0 = t[0]
        self.td = t
        self.dt = self.td[1]-self.td[0]
        self.dur = self.td[-1]-self.td[0]
        self.hp, self.hc = hp, hc
        self.GW_ampl = np.abs(self.hp+1j*self.hc)
        self.GW_phase = np.unwrap(-np.angle(self.hp+1j*self.hc))
        self.GW_inst_freq = np.gradient(self.GW_phase,self.dt,edge_order=2)*(1/(2*np.pi))
        self.mdta = src_mdta
        self.GW_tmax = self.td[np.argmax(self.GW_ampl)]
        self.td = self.td-self.GW_tmax
        self.GW_tmax = 0.0
        
    def sopt(self, sc, N=None):
        if N is None:
            N=5000
        ones = np.ones_like(self.td)
        ones = time_domain(ones, ones)
        
        sopt = snr.matched_filtering(self.hp_get_tdobj(), ones, sc).optimal_SNR(N=N)
        
        return np.abs(sopt.data).max()
        
    #def GW_tmax(self):
    #    ind = np.argmax(self.GW_ampl)
    #    return self.td[ind]
        
    def amp_wei_time(self):
        return 2 * np.sum(self.GW_ampl*(self.td-self.td[0]))/np.sum(self.GW_ampl)   
        
    def amp_wei_freq(self):
        return np.abs(np.sum(self.GW_ampl*self.GW_inst_freq)/np.sum(self.GW_ampl))
        
        
    def show_wf(self, mdta=False, fig=None, **kwargs):
        '''
        this method provides a plotting functionality 
        which can show metadata alonside with the wave itself
        
        Used attributes:
            self.td
            self.hp
            self.hc
            self.mdta
        
        Input:
            mdta: bool, default->False, when True the metadata will
                  occupy some space in the plot.
            fig: pyplot figure object, default->None,
                 One can pass a custom fig created beforehand, e.g.
                     fig = plt.figure(figsize=(15,5))
                     
            **kwargs: keyword args for the plotting:
                      xlabel, ylabel, xlim, ylim, ...
        
        Output:
            fig
        '''
        
        D = self.get_mdta()
        #dist = D['dist_mpc']
        if  len(D['name'])>15:
            D['name'] = D['name'][:10] +'...'
            
        table = np.array([list(D.keys()), list(D.values())]).T
        
        ind_tmerg = np.argmax(self.GW_ampl)
        
        
        if fig is None:
            fig = plt.figure(figsize=(7, 4.5), dpi=400)
            fig.subplots_adjust(wspace=0.5, hspace= 0.5)
        if mdta is True:
            gs = fig.add_gridspec(4,20)
            ax0 = fig.add_subplot(gs[:,:10])
            #ax0.set_ylabel(f'strain at {dist}Mpc ')
            ax0.set_xlabel('time [ms]')
            ax1 = fig.add_subplot(gs[:,10:])
            
            ax0.plot(self.td*1e3, self.hp, label=r'$h_+$', c='green')
            ax0.plot(self.td*1e3,self.hc, label=r'$h_x$', c='black')
            ax0.legend(loc='upper right')
            ax1.set_title('source parameters')
            table = ax1.table(table, loc = 'center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8.5)
            ax1.axis('off')
        else:
            #name, eos = D['name'], D['eos']
            ax0 = fig.add_subplot()
            #ax0.set_title(f'{name}, EOS:{eos}')
            #ax0.set_ylabel(f'strain at {dist}Mpc ')
            ax0.set_xlabel('time [ms]')
            ax0.plot(self.td*1e3, self.hp, label=r'$h_+$', c='green')
            ax0.plot(self.td*1e3,self.hc, label=r'$h_x$', c='black')
            ax0.axvline(self.GW_tmax*1e3, color='red',
                       linestyle='dashed')#, label=r'$\mathrm{Max(h_{22})}$')
            ax0.legend(loc='upper right')
            
        return ax0

            
    def h_AP(self, theta1, theta2):
        '''
        Antenna pattern using the coeficients
        depending on the angles which determine the orientation of the source
        '''
        wave = time_domain(self.td, np.ones_like(self.td))
        return wave
    
    def get_mdta(self):
        '''
        This method is used to give a nice formatted
        printed version of the metadata
        
        Used attributes:
            self.mdta: dictionary
            
        Input: None
        
        Output:
            show: str, formatted table
        '''
      
        keys = list(self.mdta.keys())
        values = list(self.mdta.values())
        A = {}
        for i in keys:
            A.update({i:self.mdta[i]})
        return A
    
    def hp_get_tdobj(self):
        '''
        This method is used to create time_domain objects
        from just one of the polarizations, hp
        
        Used attributes:
            self.hp: 1D array
            self.td: 1d array
            
        Input: None
        
        Output:
            time_domain object, which holds the following attributes
                dt, sampling rate
                dur, duration
                t_0, initial phase
                td, time array 
                data, data array
        '''
        wave = time_domain(self.td, self.hp)
        return wave
    
    def hc_get_tdobj(self):
        '''
        This method is used to create time_domain objects
        from just one of the polarizations, hc
        
        Used attributes:
            self.hc: 1D array
            self.td: 1d array
            
        Input: None
        
        Output:
            time_domain object, which holds the following attributes
                dt, sampling rate
                dur, duration
                t_0, initial phase
                td, time array 
                data, data array
        '''        
        wave = time_domain(self.td, self.hc)        
        return wave
        
    

class Sensitivity_curve(object):
    '''
    Input:
        non-equally sampled & growing ordered arrays
        [a1, a2, aj] with aj>...>a2>a1
        
        frequencies: 1D array, frequency bins
        ASD : 1D array, amplitude spectral density 
        name: str, this is used for plotting purposes later on           
            
    '''
    def __init__(self, frequencies, ASD, name):
                      
        self.fd = frequencies
        self.ASD = ASD
        self.PSD = ASD**2
        self.name = name
                              
    def cut(self, bounds):
        '''
        This method allow the matched filtering algorithm
        to just take into account certain region of the 
        frequency spectrum, to make the analysis. later on
        the other neglected regions are zeropadded, to ignore 
        weird behavior when applying fft to sharp regions on 
        some datasets
        
        Used attributes:
            self.fd: 1D array, non-equally sampled from ligo/virgo/kagra files
            sefl.ASD: 1D array, non-equally sampled from ligo/virgo/kagra files
            
        Input:
            bounds, tuple-> (f_lower,f_higher)
        
        Output:
            sensitivity_curve object, which holds the following attributes:
                self.df
                self.fd-> 1D array, non equally sampled
                self.ASD-> 1D array, non equally sampled
                self.PSD-> 1D array, non equally sampled
        '''
        msk = np.logical_and(self.fd>=bounds[0],self.fd<=bounds[1])
        new_fd = self.fd[msk]
        new_ASD = self.ASD[msk]
        
        return Sensitivity_curve(new_fd, new_ASD, self.name)
    
    def PSD_interp(self, F_in):
        '''
        This method is used to take the ligo/virgo/...
        design curves to be sampled in the same way as
        the templates, and waveforms.
        
        Used attributes:
            self.fd-> 1D array, non equally sampled
            self.ASD-> 1D array, non equally sampled
            
        Input:
            F_in, 1D array. is also growing ordered
                                a1, a2, aj 
                                with aj>...>a2>a1        
        Output:
            power_spectrum object-> which hold the following information:
                                    self.fd-> equally spaced array
                                    self.data-> equally spaced array
        
        '''
        
        msk = np.logical_and(F_in>=self.fd[0], F_in<=np.max(self.fd))
        new_F = F_in[msk]

        func = interpolate.interp1d(self.fd, self.ASD)
        new_ASD = func(new_F)
        
        return power_spectrum(new_F, new_ASD**2)   

    
    def inv_psd_extension(self,F_in):
        '''
        This method completes the spectrum for the design curves
        because usually they come defined just in the right frequency 
        plane
        
        Input:
            F_in, 1D array-> is growing ordered
                                a1, a2, aj 
                                with aj>...>a2>a1 
        
                                F_in must have a constant
                                sampling rate
        
        Output:
            power_spectrum object-> which holds the following information:
                        self.fd-> equally spaced array
                        self.data-> equally spaced array
        
        '''
        # 1. PSD needs to be equally spaced as F_in
        #    to be extended towards those regions in
        #    which it is not defined
        
        interp_data_PSD = (self.PSD_interp(F_in)).data  # this is a powerspectrum object
                                                        # we use .data to extract the PSD
        
        inv_PSD0 = 1/(interp_data_PSD)
        
        # crop the positive part of the target frequency space
        # crop the positive part of frequency space
        msk1 = np.logical_and(F_in>=0, F_in<=np.max(F_in))
        F_sc1 = F_in[msk1]

        # zero pad the inverse PSD on those regions in which it is not defined

        inv_PSD1 = np.zeros_like(F_sc1)
        msk2 = np.logical_and(F_sc1>=self.fd[0], F_sc1<=np.max(self.fd))
        inv_PSD1[msk2] = inv_PSD0
        
        PSD_extended = inv_PSD1

        return power_spectrum(F_sc1, PSD_extended)   
    
