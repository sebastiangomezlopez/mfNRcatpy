import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
import ray

def catalog_analysis(waveform_bank, func, **func_kwargs):
    res = [func(wf_id, **func_kwargs) for wf_id in waveform_bank]
    res = {list(x.keys())[0]:x[list(x.keys())[0]] for x in res}
    return res

def par_catalog_analysis(waveform_bank, func, **func_kwargs):
    res = ray.get([func.remote(wf_id, **func_kwargs) for wf_id in waveform_bank])
    res = {list(x.keys())[0]:x[list(x.keys())[0]] for x in res}
    return res


def relabler_cat(A, input_string, maxdigs=None):
    '''
    relabeling function, 
    For those cases where the names of simulations
    on catalogs are ridiculously long
    
    Input: 
        A, dict, original dictionary
        input_tring: first characters of the key, e.g
                     lal ---> lal001
                         ---> lal002
                         ---> ...
        maxdigs:int, number of digits after the
                input string. e.g.
                3--> lal001
                6--> lal000001
    Output:
    
        C, dict. New relabled dictionary
        D, dict, small dictionary mapping old and new keys
                
    '''
    
    
    B = A.copy()
    C = {}
    D = {}
    if maxdigs is None:
        maxdigs = 4
    
    for i, key in enumerate(B.keys()):
        i+=1
        st = input_string + str(i).rjust(maxdigs, '0')
        D[st] = key
        C[st] = B[key]
    return C, D

def scatter_cat(A, func1, func2, var1, var2,ax=None,
                bounds=None, boundsv=None, mdta=None):
    '''
    scatter plot with highlighted regions 
    from results dictionary.
    
    Input:
        A: dict. results from the catalog analysis
        func1&2: ufunc, defines the values of each axis 
        var1&2: str, keys to values in the dictionary
        
        bounds: list of lists, defines the boundaries 
                each region will be highlighted with
                different colors.e.g.
                
                bounds=[[x1,x2, y1,y2],   -> bound region 1
                        [x1,x2, y1,y2],   -> bound region 2
                        ...,
                        ]
                        
        boundsv: list, highligh according to metadata.e.g.
                boundsv= ['mass-ratio', 1.1,  2.1] 
        
    Output:
        ax: pyplot axes
        R: list of waveforms lying on the highlighted regions.
    
    '''
    
    if mdta is None:
        mdta = False
    

    paramsx0 = lambda key: (A[key]['mdta'][param][0] for param in var1)
    paramsx1 = lambda key: (A[key][param][0] for param in var1)

        

    paramsy0 = lambda key: (A[key]['mdta'][param][0] for param in var2)
    paramsy1 = lambda key: (A[key][param][0] for param in var2)    
     
    
    if ax is None:
        fig = plt.figure(dpi=1200)
        ax = fig.add_subplot()
    
    for key in A.keys():
        
        try:
            x = func1(*paramsx0(key))
        except:
            x = func1(*paramsx1(key))
            
        try:
            y = func2(*paramsy0(key))
        except:
            y = func2(*paramsy1(key))
            
        ax.scatter(x, y, color='blue', s=1)
        
        
    if bounds is not None:
        c = ['r', 'darkgreen', 'black', 'orange']
        R = {'r':[], 'darkgreen':[], 'black':[], 'orange':[]}
        for key in A.keys():
            try:
                x = func1(*paramsx0(key))
            except:
                x = func1(*paramsx1(key))

            try:
                y = func2(*paramsy0(key))
            except:
                y = func2(*paramsy1(key))
                
            for i in range(len(bounds)):
                if x>bounds[i][0] and x<bounds[i][1] and y>bounds[i][2] and y<bounds[i][3]:
                    ax.scatter(x, y, marker='x', color=c[i], s=15)
                    R[c[i]].append(key)
        return ax, R           
    
    if boundsv is not None:
        R = {'r':[]}
        for key in A.keys():
            var = boundsv[0]
            
            try:
                x = func1(*paramsx0(key))
            except:
                x = func1(*paramsx1(key))

            try:
                y = func2(*paramsy0(key))
            except:
                y = func2(*paramsy1(key))
            
            
            z = A[key]['mdta'][var][0]
            
            if z>boundsv[1] and z<=boundsv[2] :
                ax.scatter(x, y, marker='x', color='r', s=15)
                R['r'].append(key)
        return ax, R
    
    return ax, {'blue':list(A.keys())}
      

def hist_cat(A, L, g, ax=None,var=None, varmdta=None, bins=None):
    
    if ax is None:
        fig = plt.figure(dpi=1200)
        ax = fig.add_subplot()
    
    
    
    def extract(A, key, var=None, varmdta=None):
        if varmdta is None:
            gen = (A[key][param][0] for param in var)
            res = g(*gen)
        else:
            gen = (A[key]['mdta'][param][0] for param in varmdta)
            res = g(*gen)
        return res
    
 
    f = lambda key: extract(A, key, var=var, varmdta=varmdta)
    
    
    res = [[*map(f, row)] for row in list(L.values())]
    
    #print(res)

    
    ax.hist(res, color=list(L.keys()) , stacked=True, rwidth=0.9, bins=bins)
    
    return ax
    



def layer_cat(layers, A, mdta = False, 
              show_eos=True ,eos_palette=None, step=4):
    

    
    eos = set( [A[name]['mdta']['eos'][0]  for name in A] )
    eos = {name:eos_palette[i] for i, name in enumerate(eos)}
    
    fig = plt.figure(figsize=(30,20))#, dpi=1200)
    n = len(layers)
    gs = GridSpec(n,9, figure=fig)
    if show_eos==True:
        axes = [ fig.add_subplot(gs[i-1, :8]) if i<n+1 else fig.add_subplot(gs[:,8: ]) for i in range(1,n+2)]


    else:
        axes = [ fig.add_subplot(gs[i-1, :]) for i in range(1,n+1)]
        [ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) for i,ax in enumerate(axes) if i<n-1]

    
    for j, layer in enumerate(layers):
        L, V, C = ['name' for x in A.keys()], [0 for x in A.keys()], ['color' for x in A.keys()]
        for i, name in enumerate(A.keys()):
            eos_name = A[name]['mdta']['eos'][0]
            #order = int(name.split(':')[-1])
            
            if mdta==False:
                value = A[name][layer][0]
            else:
                value = A[name]['mdta'][layer][0]
                
            c = eos[eos_name]
            #print(order-1)
            
            L[i] = name
            V[i] = value
            C[i] = c
            
        bot, top = np.nanmin(V), np.nanmax(V)
        epsilon =0.2*(0.5 * (bot+top))
        V = [top*10 if np.isnan(x) else x for x in V]
        #print(len(C), len(V))
        axes[j].scatter(L, V, c=C, marker='x', s=40)
        axes[j].set_xticks(np.arange(len(L)), L, rotation=90)
        axes[j].set_yticks(np.linspace(bot-epsilon, top+epsilon, 5))
        
        axes[j].tick_params(axis='x',labelsize=40)
        axes[j].tick_params(axis='y',labelsize=30)

        

        axes[j].xaxis.set_major_locator(ticker.MultipleLocator(step))
        axes[j].yaxis.set_major_formatter(FormatStrFormatter('%1.1e'))
        axes[j].grid()
        
        axes[j].set_ylim(bot-epsilon, top+epsilon)

        
    axes[-1].set_xticks([])
    axes[-1].set_yticks([])

    [ax.set_xticklabels([]) for i,ax in enumerate(axes) if i<n-1]

    [axes[-1].scatter(0,0, color=eos[name], marker='x', label=name, s=100) for name in eos.keys()]
    axes[-1].set_xlim(12, 100)
    axes[-1].set_ylim(12, 100)


    axes[-1].legend(loc='center', fontsize=5, prop={'size': 35})
    axes[-1].axis('off')
    
    plt.tight_layout()
        
            

    return axes