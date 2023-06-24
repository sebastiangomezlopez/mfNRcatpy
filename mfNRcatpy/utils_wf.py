import numpy as np
from scipy.integrate import cumtrapz

import mfNRcatpy.base as bs

def EOS_masses():
    D = {'BHBlp':{'maxm_TOV':2.100904, 'maxm_rigrot':2.549901, 'maxm_bar_TOV':2.462751,'maxm_bar_rigrot':2.961749,'ref':'Banik et al., Astrophys.J.Suppl. 214 (2014) no.2, 22'},
         'DD2':{'maxm_TOV':2.422658, 'maxm_rigrot':2.955318, 'maxm_bar_TOV':2.926458,'maxm_bar_rigrot':3.521266,'ref':'Typel et al., Phys.Rev. C81 (2010) 015803'},
         'LS220':{'maxm_TOV':2.043796, 'maxm_rigrot':2.451828, 'maxm_bar_TOV':2.429123,'maxm_bar_rigrot':2.881334,'ref':'Lattimer and Swesty, Nucl.Phys. A535 (1991) 331-376'},
         'SFHo':{'maxm_TOV':2.058882, 'maxm_rigrot':2.473178, 'maxm_bar_TOV':2.457619,'maxm_bar_rigrot':2.912048,'ref':'Steiner and Hempel and Fischer, Astrophys.J. 774 (2013) 17'},
         '2B':{'maxm_TOV':1.783269, 'maxm_rigrot':2.162003, 'maxm_bar_TOV':2.136538,'maxm_bar_rigrot':2.580824,'ref':'Read et al., Phys.Rev. D79 (2009) 124032'},
         '2H':{'maxm_TOV':2.834886, 'maxm_rigrot':3.461828, 'maxm_bar_TOV':3.409756,'maxm_bar_rigrot':4.145367,'ref':'Read et al., Phys.Rev. D79 (2009) 124032'},
         'ALF2':{'maxm_TOV':1.99092, 'maxm_rigrot':2.510254, 'maxm_bar_TOV':2.319515,'maxm_bar_rigrot':2.928461,'ref':'Read et al., Phys. Rev. D79 (2009) 124032'},
         'ENG':{'maxm_TOV':2.25052, 'maxm_rigrot':2.759074, 'maxm_bar_TOV':2.734237,'maxm_bar_rigrot':3.33214,'ref':'Read et al., Phys. Rev. D79 (2009) 124032'},
         'H4':{'maxm_TOV':2.028068, 'maxm_rigrot':2.476984, 'maxm_bar_TOV':2.332485,'maxm_bar_rigrot':2.855665,'ref':'Read et al., Phys. Rev. D79 (2009) 124032'},
         'MPA1':{'maxm_TOV':2.470327, 'maxm_rigrot':3.065678, 'maxm_bar_TOV':3.035605,'maxm_bar_rigrot':3.74128,'ref':'Read et al., Phys.Rev. D79 (2009) 124032'},
         'MS1':{'maxm_TOV':2.769299, 'maxm_rigrot':3.427183, 'maxm_bar_TOV':3.348443,'maxm_bar_rigrot':4.127858,'ref':'Read et al., Phys.Rev. D79 (2009) 124032'},
         'MS1b':{'maxm_TOV':2.76283, 'maxm_rigrot':3.439019, 'maxm_bar_TOV':3.353737,'maxm_bar_rigrot':4.155828,'ref':'Read et al., Phys.Rev. D79 (2009) 124032'},
         'SLy':{'maxm_TOV':2.060435, 'maxm_rigrot':2.507342, 'maxm_bar_TOV':2.46085,'maxm_bar_rigrot':2.985524,'ref':'Read et al., Phys.Rev. D79 (2009) 124032'}}
    return D

def chi_eff(m1,m2, x1, x2):
    '''
    equation 4 in core paper
    https://arxiv.org/abs/1806.01625
    
    m1 & m2 masses in solar masses
    x1 & x2 spin magnitudes
    
    The label 1 is reserved for the most massive object
    
    '''
    M = m1+m2
    chi = (m1*x1)/M + (m2*x2)/M - (38/113)*(m1*m2/(M**2))*(x1+x2)
    return chi

def lambda_tilde(m1, m2, l1, l2):
    '''
    eq 5 core paper
    https://arxiv.org/abs/1806.01625
    
    m1 & m2 masses in solar masses
    l1 & l2 spin magnitudes
    
    The label 1 is reserved for the most massive object
    '''
    L = (16/13) *( ((m1+(12*m2))*m1*l1)/((m1+m2)**5) + ((m2+(12*m1))*m2*l2)/((m2+m1)**5) )
    
    return L
def lambda_from_compactness(m, R):
    '''
    universal relation taken from 
    https://journals.aps.org/prd/pdf/10.1103/PhysRevD.102.043023
    '''
    
    C = m/R
    pass
    
    
    

def energylm(l,m , hlm, t, r):
    '''
    uses equation(10) from https://arxiv.org/abs/1311.4443
    
    input:    
        l,m: int,int. Indices from spin weighted spherical harmonics expansion
        hlm: complex array. Spherical harmonics terms coming from integrating twice the weyl scalar ones.
        t:  time array. Must be in seconds
        r:  float. Distance to the source, must be in meters
        
    output:
        e_MSUN: time domain object.
            e_MSUN.td: time array. given in seconds
            e_MSUN.data: data array. energy radiated given in solar masses
    
        luminosity: time domain object.
            luminosity.td: time array. given in seconds
            luminosity.data: data array. dimensionless
    
    '''
    MSUN_SEC = 4.92e-6
    meter_in_sec = (1/3e8)
    r=r*meter_in_sec
    
    integrand = np.abs(r*np.gradient(hlm,t,edge_order=2))**2 
    
    e_MSUN =  bs.time_domain(t,(1/(16*np.pi))*cumtrapz(integrand, t, initial=0)/MSUN_SEC)
    luminosity = bs.time_domain(t, integrand)
    
    return e_MSUN, luminosity

def angmom_Z_lm(l,m, hlm, t,r):
    '''
    uses equation(11) from https://arxiv.org/abs/1311.4443 with a minor correction
    i.e. here i take the imaginary part of the integrand as suggested by
    https://arxiv.org/abs/0707.4654 in equation (49)
    input:    
        l,m: int,int. Indices from spin weighted spherical harmonics expansion
        hlm: complex array. Spherical harmonics terms coming from integrating twice the weyl scalar ones.
        t:  time array. Must be in seconds
        r:  float. Distance to the source, must be in meters
        
    output:
        e_MSUN: time domain object.
            e_MSUN.td: time array. given in seconds
            e_MSUN.data: data array. energy radiated given in solar masses
    
        luminosity: time domain object.
            luminosity.td: time array. given in seconds
            luminosity.data: data array. dimensionless
    
    '''

    MSUN_SEC = 4.92e-6
    meter_in_sec = (1/3e8)
    r=r*meter_in_sec
    
    integrand = -(m*(r**2)* hlm * np.gradient(np.conj(hlm), t,edge_order=2)).imag
    
    Jz_MSUNsq = bs.time_domain(t, (1/(16*np.pi))*cumtrapz(integrand, t, initial=0)/MSUN_SEC**2)
    torquez_msun = bs.time_domain(t, integrand/MSUN_SEC)
    
    return Jz_MSUNsq, torquez_msun
