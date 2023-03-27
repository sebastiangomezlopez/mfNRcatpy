import numpy as np
from watpy.coredb.coredb import *
import lal
import mfNRcatpy.base as bs
import mfNRcatpy.utils as ut
import mfNRcatpy.utils_wf as utw 
from IPython.utils import io



def get_CoRewfattr(wf_CoRedb):
    '''
    Helper function for getting the metadata from 
    the CoRe waveforms
    '''
    spin1 =np.array(wf_CoRedb.md.data['id_spin_starA'].split(',')).astype('float')
    spin2 =np.array(wf_CoRedb.md.data['id_spin_starB'].split(',')).astype('float')
    
    spin_norm1, spin_norm2 = np.linalg.norm(spin1),np.linalg.norm(spin2)
    spin_z1, spin_z2 = spin1[2],spin2[2]
    
    m2, m1 = np.sort(np.array([np.float64(wf_CoRedb.md.data['id_rest_mass_starA']),
             np.float64(wf_CoRedb.md.data['id_rest_mass_starB'])]))
    

    attr ={'name':wf_CoRedb.md.data['database_key'],
           'eos':wf_CoRedb.md.data['id_eos'],
           'grav_mass1':m1,
           'grav_mass2':m2,
           'lambda_tilde': np.float64(wf_CoRedb.md.data['id_Lambda']),
           'Xeff': utw.chi_eff(m1,m2, spin_norm1,spin_norm2),
           'spin1':spin_norm1,
           'spin2':spin_norm2,
           'ecc': wf_CoRedb.md.data['id_eccentricity']
          }
    
    if attr['ecc'] is None or attr['ecc']=='':
        attr['ecc']=0.0
    else:
        attr['ecc']=np.float64(attr['ecc'])
    
    return attr

def resolution_selector(cdb, sim,resol):
    if resol is None:
        f = [[res,cdb.sim[sim].run[res].md.data['grid_spacing_min']] for res in cdb.sim[sim].run.keys()]
        f = np.array(f)
        run = input(f'simulation {sim} has several available resolutions(grid_spacing_min) \n {f} \n where the smaller the better, please choose one \n')
        min_gridres = cdb.sim[sim].run[run].md.data['grid_spacing_min']
        
    elif resol=='best':
        f = [[res,cdb.sim[sim].run[res].md.data['grid_spacing_min']] for res in cdb.sim[sim].run.keys()]
        f = np.array(f)
        ind = np.argmin(f[:,1].astype('float'))
        
        run = f[ind, 0]
        min_gridres = cdb.sim[sim].run[run].md.data['grid_spacing_min']

    else:
        run = resol
        min_gridres = cdb.sim[sim].run[run].md.data['grid_spacing_min']
    
    return run, min_gridres 



def CoRe_NR_wf22(db_path, sim, src_mdta,resol='best',sync_db=False, energetics=False):
    
    '''
    This is a function that creates a waveform object from the 
    CoRe colaboration catalogue
    
    This waveforms come as r*h22 as a function of a timelike coordinate U/M, 
    where instead of taking M is the total mass of the system
    in solar masses.
    
    Where [r]=[Msun], h22 is dimensionless, [U/M]=[Msun/Msun]
    
    
    Input:
         db_path: str,
                  path to the folder in which the database is located
         sim : str,
               name of the simulation, more info can be found in the 
               webpage: http://www.computational-relativity.org/gwdb/
               
         src_mdta: dictionary,
                   contains information about the sky
                   positioning of the source:(typical params)
                       *src_mdta['inclination'] in radians
                       *src_mdta['phiref'] in radians
                       *src_mdta['dist_mpc'] in megaparsec
                       
         sync_db: bool, 
                  set False by default. When True, updates the database
                  to the latest version from its original repository
                  
    Output: 
         waveform object that holds hp, hc, time array and metadata 
               
    '''
    with io.capture_output() as captured: ## avoiding nasty printing
        cdb = CoRe_db(db_path)
        if sync_db is True:
            idb = CoRe_index(db_path, lfs=False, verbose=False)
            idb.sync_database(lfs=True, verbose=False)
            
            
    run, min_gridres = resolution_selector(cdb, sim, resol)

    try:
        with io.capture_output() as captured:  
            wf = CoRe_db(db_path).sim[sim].run[run]
    except ValueError:
        print(f' there is a problem with simulation {sim} run {run}, min_grid_res={min_gridres}')
            
    i = src_mdta['inclination']

    Y_22 = lal.SpinWeightedSphericalHarmonic(i,0,-2,2,2)  

    rh22 = wf.data.read('rh_22')
    
    U_M, rh = rh22[:,0], (rh22[:,1] + 1j*rh22[:,2])*Y_22
    
    attr = {}
    attr = get_CoRewfattr(wf)
    keys = list(attr.keys())
    
    for i in range(len(keys)):
        src_mdta[keys[i]] = attr[keys[i]]
                
    m_tot = float(attr['grav_mass1'])+float(attr['grav_mass2'])
    dist_msun = (float(src_mdta['dist_mpc'])*3.0856776e+22) * (1/(1476.62*m_tot))
    
    m_tot_sec = m_tot * 4.925e-6
            
    return bs.waveform(U_M*m_tot_sec, rh.real/dist_msun, rh.imag/dist_msun, src_mdta)
