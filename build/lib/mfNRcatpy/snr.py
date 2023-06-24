import numpy as np
from scipy import interpolate
from scipy import signal
from numba import njit, prange

import matplotlib.pyplot as plt
import os

import mfNRcatpy.base as bs
import mfNRcatpy.utils as ut
#from mfNRcatpy.core_wf import *
#from mfNRcatpy.lal_wf import *




@njit(parallel=True, fastmath=True)
def integral(df, tf, new_inv_PSD, F, delta_f, tau):
    '''
    to use this you better install numba and SVML

    conda install numba
    conda install -c numba icc_rt

    '''

    templ_dot_templ = 4 * (np.sum(tf * tf.conjugate() * new_inv_PSD)) * delta_f 
    N = np.sqrt(templ_dot_templ)

    # shifting inner product
    L = []
    for i in prange(tau.size):
    #for t in tau:
        #S = 4*(df * tf.conjugate()) * new_inv_PSD * np.exp(1j*2*np.pi*F*t) * delta_f
        S = 4*(df * tf.conjugate()) * new_inv_PSD * np.exp(1j*2*np.pi*F*tau[i]) * delta_f

        rho = S/N
        L.append(np.sum(rho))

    L = np.array(L)
    return L

def extend_t(da,N, method=None):
    '''
    this function extends an array of floats
    symmetrically to left and right with N new 
    values based on the sampling of the original
    array da e.g.
    
    a = np.array([1,2,3,4,5])
    extend_t(a,4)
    '''
    delta = abs(da[1]-da[0])
    new_da = da
    if method=='loop':
        for i in range(1,N+1):
            # insert the left value
            left = (-delta) + new_da[0]
            new_da = np.flip(new_da)
            new_da = np.append(new_da, left)
            new_da = np.flip(new_da)
            # insert the right value
            right = (delta) + new_da[-1]
            new_da = np.append(new_da, right)
    elif method is None or method=='vector':
        arr = np.linspace(1,N, N)*delta
        left = da[0]-np.flip(arr)
        right = da[-1]+arr
        new_da = np.concatenate((left,da,right))
        
    return new_da


def tau_shift_gen(A, dt):
    '''
    REMARK: You might want to multiply the output
    by some dt, to scale correctly the time shift 
    domain.
    
    
    This function generates the timeshift 
    domain for convolutions or cross correlations
    
    Since this depends on if the array's lenght is
    odd or even, this uses the convention in which,
    
    if the input array has even lenght (it has 2 center values),
    then it picks the one which is on the right, in order 
    to get more negative values before the zero value
    
    Arrays with odd leghts do not represent 
    any confusion since they have just 1 center
    
    Input
    A : cross correlation or convolution array
    
    Output:
    B : time shift domain for a cross correlation
    
    example:
    
    a = np.arange(8)
    b = np.arange(9)

    print('Array with even lenght {}'.format(tau_shift_gen(a)))
    print('')
    print('Array with odd lenght {}'.format(tau_shift_gen(b)))
    
    '''
    N = A.size
    if N%2==0: #par
        B = np.arange(-(N//2),(N//2))
    else:      #impar
        B = np.arange(-(N//2),(N//2)+1)
    return B * dt



class matched_filtering(object):
    '''
    Here I'm supposed to have already decided 
    if i want the full strain or just hp, hc
    
    template  and waveform need to be converted
    to time domain objects beforehand
    
    '''
    
    def __init__(self, td_wf, td_templ, ASD):
        
        self.wf = td_wf
        self.templ = td_templ
        self.ASD = ASD
        
    def cross_correlation(self,max_templ_dur=None):        
        data_td, template_td = self.length_matching(max_templ_dur)
        key = 'same'
        dt = data_td.dt
        c =  signal.correlate(data_td.data, template_td.data, mode=key)*data_td.dt
        
        tau0 = np.arange(c.size)
        tau = tau_shift_gen(c, dt)
        
        return bs.time_domain(tau, c)
        
        
        
    
    def optimal_SNR(self, N=None, cut_sc=None):
        a = self.templ
        self.templ = self.wf
        s_opt = self.recovered_SNR3(N=N, cut_sc=cut_sc)
        self.templ = a
        return s_opt
    


    
    def recovered_SNR3(self, N=None, cut_sc=None):
        
        w = self.wf
        t = self.templ
        sc = self.ASD
        
        if N is None:
            N = 5

        if cut_sc is None:
            sc = sc.cut([50, 4500])
        else:
            sc = sc.cut(cut_sc)



        Mf = matched_filtering(w, t, sc)
        # extend the arrays and iterpolate to evaluate the same samples
        data, templ = Mf.length_matching2(N=N)
        delta_t = data.dt

        # to frequency domain using ffts    
        # but reorder them to get zero freq in the center
        # in this way is easier to debug
        F = np.fft.fftshift(np.fft.fftfreq(data.data.size, delta_t))
        delta_f = F[1]-F[0]

        df, tf = [np.fft.fftshift(np.fft.fft(data.data)*delta_t),
                  np.fft.fftshift(np.fft.fft(templ.data)*delta_t)]

        # define the inverse power spectrum of the detector    
        new_inv_PSD = sc.inv_psd_extension(F).PSD_flip_mirror(F).data    
        data_dot_templ = (df * tf.conjugate()) * new_inv_PSD    
        mask = F<0
        data_dot_templ[mask]=0

        # pull the zero frequency back to the first array position
        data_dot_templ = np.fft.ifftshift(data_dot_templ)
        # use ifft
        data_dot_templ = 4* (np.fft.ifft(data_dot_templ)*data_dot_templ.size) * delta_f
        S = data_dot_templ

        # 2. denominator --> Calculate the norm of the template

        templ_dot_templ = tf* tf.conjugate() * new_inv_PSD
        templ_dot_templ[mask] = 0
        templ_dot_templ =  4*(np.sum(templ_dot_templ) * delta_f)

        N = np.sqrt(templ_dot_templ)
        rho = np.fft.fftshift(S/N)
        tau = tau_shift_gen(data.data, delta_t)
        return bs.time_domain(tau, rho)
        

    def length_matching2(self, N=None):

        A = (self.wf).data
        da = (self.wf).td

        B = (self.templ).data
        db = (self.templ).td

        if N is None:
            N=50

        dur_a = da[-1]-da[0]
        dur_b = db[-1]-db[0]

        delta_t = da[1]-da[0]

        ti, tf =  np.take(np.sort([da[0], da[-1], db[0], db[-1]]),[0,3])
        ti, tf = ti-N*delta_t, tf+N*delta_t

        new_da = np.arange(ti, tf, delta_t)
        new_A = np.zeros_like(new_da)

        ind0 = np.where(new_da>=da[0])[0][0]
        ind1 = np.where(new_da>=da[-1])[0][0]
        x0 = new_da[ind0:ind1]

        f = interpolate.interp1d(da, A)
        f_p = f(x0)
        new_A[ind0:ind1] = f_p

        new_db = new_da
        new_B = np.zeros_like(new_da)
        ind2 = np.where(new_da>=db[0])[0][0]
        ind3 = np.where(new_da>=db[-1])[0][0]
        x1 = new_da[ind2:ind3]

        f = interpolate.interp1d(db, B)
        f_p = f(x1)

        new_B[ind2:ind3] = f_p


        return bs.time_domain(new_da, new_A), bs.time_domain(new_db, new_B)
    





      
def genrlzd_SNR1(wf_td,templ_gen, sc_ps,*templ_args,
                 N=None, cut_sc=None,taufix=None,
                 phase_opt=None, cut_pfind=None):    
    '''
    This is a generalized version of the SNR caluclator, built in 
    the matched filtering object as a method.
    
    Input:
        wf_td: time_domain object, real data from NR simualtions or
               from the GW detectors.
               
        templ_gen: function, this has to be a function that generates 
                   time_domain object. e.g.
                   
                   f = lambda f,d,t0,dt :monochr_templ(f,d,t0,dt, 
                                           amp=0.5*np.max(wf2.data))
                   
        sc_ps: Sensitivity_curve object
        
        *templ_args: ND arrays, can be scalar, 1D, 2D, ...
                     are those parameters on which the template generator depends
                     
        N: int, number of points to extend(zeropad) on the array domains
        
        cut_sc: tuple, (finit, fend) window where the sensitivity curve is 
                considered. outside of those limits it is zero padded.
        
        taufix: float, fixed timeshift at which the algorithm takes the SNR value
        
        phase_opt: bool, 
                   True-> computes everything based on the magnitude of the complex
                          SNR time series
                   False-> computes everything based on the real part of the complex
                           SNR time series
        cut_pfind: float, parameter that tells the algorithm to find the maximum ignoring
                   every timeshift value behind  it. WARNING! try not mixing it with taufix.
                   

    
    Output:
        SNR_rec_max: ND array, maximum recovered SNR calulated from the waveform, sensitivity
                     curve, and template by using the matched filtering algorithm.
    '''
    if phase_opt is None:
        phase_opt = True
    
    def SNR(*templ_args):
        tmpl_td = templ_gen(*templ_args, wf_td.dt)
        complex_SNR = matched_filtering(wf_td, tmpl_td , sc_ps)
        complex_SNR = complex_SNR.recovered_SNR3(N=N, cut_sc = cut_sc)
        
        if cut_pfind is None:
            pass
        else:
            complex_SNR = complex_SNR.cut((cut_pfind, complex_SNR.td[-1]))
        
        if taufix is None and phase_opt==True:
            SNR_rec_max = np.max(np.abs(complex_SNR.data))
        elif taufix is None and phase_opt==False:
            SNR_rec_max = np.max(complex_SNR.data.real)
        elif taufix is not None and phase_opt==True:
            ind = np.argwhere(complex_SNR.td>=taufix).flatten()[0]
            SNR_rec_max = np.abs(complex_SNR.data)[ind]
        elif taufix is not None and phase_opt==False:
            ind = np.argwhere(complex_SNR.td>=taufix).flatten()[0]
            SNR_rec_max = (complex_SNR.data.real)[ind]            
        
        if phase_opt==True:
            assert SNR_rec_max>0, 'warning! you are not actually using phase optimization'
        
        return SNR_rec_max
    
    SNR_vfunc = np.vectorize(SNR)

    return np.array(SNR_vfunc(*templ_args))



def genrlzd_SNR2(wf_td,templ_gen, sc_ps,max_templ_dur,*templ_args, taufix=None, phase_opt=True):    
    '''
    This is a generalized version of the SNR caluclator, built in 
    the matched filtering object as a method.
    
    Input:
        wf_td: time_domain object, real data from NR simualtions or
               from the GW detectors.
               
        templ_gen: function, this has to be a function that generates 
                   time_domain object. e.g.
                   
                   f = lambda f,d,t0,dt :monochr_templ(f,d,t0,dt, amp=0.5*np.max(wf2.data))
                   
        sc_ps: Sensitivity_curve object
        
        max_templ_dur: float, it has to be a multiple of the duration of the 
                       waveform or the template, which is needed to fix the padding
                       when doing the matched filtering algorithm.
        *templ_args: ND arrays, can be scalar, 1D, 2D, ...
                     are those parameters on which the template generator will depend
    
    Output:
        SNR_rec_max: ND array, maximum recovered SNR calulated from the waveform, sensitivity
                     curve, and template by using the matched filtering algorithm.
    '''

    def SNR(*templ_args):
        crop0, crop1 = [10, 3000], None
        tmpl_td = templ_gen(*templ_args, wf_td.dt,t_0=wf_td.t_0, amp= 1)
        complex_SNR = (matched_filtering(wf_td, tmpl_td , sc_ps).recovered_SNR1(max_templ_dur=max_templ_dur))[0]
        #complex_SNR = complex_SNR.cut((0, complex_SNR.td[-1]))
        return complex_SNR

    
    SNR_vfunc = np.vectorize(SNR)

    return np.array(SNR_vfunc(*templ_args))

def SNR_2D(wf, tp_func, var1, var2, sc, max_templ_dur=None):
    results = np.zeros_like(np.outer(var1, var2))
    for i, param0 in enumerate(var1):
        for j, param1 in enumerate(var2):
            if 1/param0 < param1 and 1/(8*param1) > wf.dt:
                template = tp_func(param0, param1, wf.dt)
                #wf_pm = ut.find_postm(wf)
                Mf0 = matched_filtering(wf.hp_get_tdobj(), template , sc).recovered_SNR1(max_templ_dur=max_templ_dur)
                ind = np.nanargmax(np.abs(Mf0[0].data))
                overlap = Mf0[0].data[ind]
                tau = Mf0[0].td[ind]
                phase = Mf0[1].data[ind]
                results[i,j] = overlap
            else:
                pass
    return results.T

            
    
