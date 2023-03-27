import numpy as np
import base as bs
import utils_wf as utw 
from waverep import save_gw_ligo as gwlvc
#import lalsimulation as ls
import lal

def get_lalwfattr(wf_gwlvc):
    '''
    Helper function for getting the metadata from 
    the lal waveforms
    '''
    m2, m1 = np.sort(np.array([wf_gwlvc.grav_mass1,
              wf_gwlvc.grav_mass2]))
    spin_1, spin_2 = [np.linalg.norm(np.array(wf_gwlvc.spin1)),
                    np.linalg.norm(np.array(wf_gwlvc.spin2))]
    
    ecc = wf_gwlvc.eccentricity
    if ecc=='':
        ecc=np.nan
    
    attr ={'name': wf_gwlvc.name, 
           'eos':wf_gwlvc.eos_name,
           'grav_mass1':m1,
           'grav_mass2':m2,
           'lambda_tilde': np.nan,
           'Xeff': utw.chi_eff(m1,m2, spin_1,spin_2),
           'spin1':spin_1,
           'spin2':spin_2,
           'ecc':ecc
           }
    return attr 


def lal_NR_wf1(path, l, m, dt, src_mdata):
        
    '''
    This is a function that creates a waveform object from lal
    
    Input:
         path: str,
                  path to the folder in which the database is located
         dt : float,
              sampling rate of the waveform usually one should pick
              something like 1/40000
               
         src_mdta: dictionary,
                   contains information about the sky
                   positioning of the source:(typical params)
                       *src_mdta['inclination']
                       *src_mdta['dist_mpc']
                  
    Output: 
         waveform object that holds hp, hc, time array and metadata 
               
    '''
    
    MSUN_METER = lal.MSUN_SI * lal.G_SI/(lal.C_SI**2)
    MSUN_SECOND = MSUN_METER / lal.C_SI
    
    inclination = src_mdata['inclination']
    dist_mpc = src_mdata['dist_mpc']
    distance = dist_mpc * lal.PC_SI * 1.0e6
    
    wf_file = path
    wfrm = gwlvc.WaveFormFile(wf_file)
    
    ct,ch = wfrm.strain_units_msol(l,m, samp_rate=MSUN_SECOND/dt)
    
    fac = lal.SpinWeightedSphericalHarmonic(inclination,0,-2,l,m)

    hp = fac * ch.real * MSUN_METER / distance
    hc = fac * ch.imag * MSUN_METER / distance
    
    t =  ct * MSUN_SECOND
    
    wf_data = get_lalwfattr(wfrm)
    src_mdata.update(wf_data)
    
    return bs.waveform(t, hp, hc, src_mdata)
