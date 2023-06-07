"""Climate Security Index Source Code - Helper Functions

Authors: Omar A. Guerrero & Daniele Guariso
Written in Python 3.7


This are the main functions needed to construct the Climate Security Index (CSI)
developed by CGIAR and The Alan Turing Institute. The CSI consists of three
components that aim at quantifying three crucial elements of complex adaptive
systems:
    
    - The likelihood of extreme events such as natural disasters
    - The connectivity structure enabling indirect impacts from extreme events
    - The overall synchronicity of the system, which is often related to
    tipping points and phase transitions
      
There is one function for each of these components. Note that the CSI methodology
has been designed to work with small datasets, so each component has been
carefully chosen to (1) capture the corresponding feature of a complex socio-
economic system, and (2) work well with real-world datasets on development indicators.


Required external libraries
--------------------------
- Numpy
- Scipy
- NetworkX


"""

import numpy as np
import networkx as nx
import scipy.stats as st
import os, warnings
warnings.simplefilter("ignore")


home =  os.getcwd()[:-4]
os.chdir(home)






## CONNECTIVITY
def get_connectivity(nodes_list, edges_list):
    
    """Function to compute the connectivity component of the CSI. The method 
    consists of using a, previously estimated, network of conditional dependencies 
    and constructing a new network of compounded conditional dependencies (CCD). 
    The initial network can be estimated using the time series of the indicators 
    in the sample data. Since this type of data tends to be small (short series),
    a method that is adequate for this task is the method of sparse Bayesian 
    networks developed by:
        
        - Aragam, B., Gu, J., & Zhou, Q. Learning large-scale bayesian networks
        with the sparsebn package. to appear. Journal of Statistical Software.

    Provided such network, this function constructs a denser graph by compounding
    the conditional dependencies between every par of nodes (when paths exist).
    Then, the function uses the labels of the indicators (the development dimensions) 
    to determine the modularity of the CCD network. The intuition is that a more
    modular structure implies that shocks in a specific dimension will tend to
    stay contained within that dimension. On the contrary, low modularity means
    that shocks could propagate between different development dimensions more
    easily. The modularity score is the one for weighted directed networks, 
    provided by:
        
        - Leicht, E. A., & Newman, M. E. (2008). Community structure in directed 
        networks. Physical review letters, 100(11), 118703.
        

    Parameters
    ----------
        nodes_list: numpy array 
            An array with two columns where each row corresponds to an indicator.
            The first column should contain the indices of the indicators, 
            ranging from 0 to N-1. The second column should indicate the 
            development dimension to which the indicator belongs.
            
        edges_list: numpy array 
            An array with three columns where each row corresponds to a directed
            link from a, previously estimated, network of conditional dependencies.
            The first and second column should contain the source and target nodes
            respectively. The third column should indicate the weight of the edge.
        
    Returns
    -------
        Q: float
            The modularity score of the CCD network. The score ranges between
            -1 and 1. In the CSI, the modularity score has the opposite sign as
            in the network literature (for consistency with the other components).
            Negative values indicate a more modular structure while positive 
            ones suggest more connectivity between development dimensions.
        
        E: 2D numpy array
            The CCD network on which the modularity score is calculated.
    """
    
    # prepare nodes and edges
    n = len(nodes_list)
    nodes = nodes_list[:,0]
    labels = nodes_list[:,-1]
    
    # create graph objects to obtain all paths between node pairs
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges_list)

    # construct adjacency matrix of compounded conditional dependencies (CCDs)
    E = np.zeros((n,n))
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if node1 != node2:
                paths = list(nx.all_simple_paths(G, node1, node2))[::-1]
                # compute the mean CCD across all possible paths between this pair of nodes
                ccd = np.mean([np.prod([G[path[k]][path[k+1]]['weight'] for k in range(len(path)-1)]) for path in paths])
                # we focus on the positive CCDs
                if ccd > 0:
                    E[i,j] = ccd

    # compute the modularity of the CCD network
    mp = np.sum(E)
    qs = []
    for i, j in np.column_stack(np.where(E!=0)):
        wij = E[i,j]
        si_out = E[i,:].sum()
        sj_in = E[:,j].sum()
        delta = 0
        if labels[i] == labels[j]:
            delta = 1
        qij = (wij - (si_out*sj_in)/mp) * delta
        qs.append( -qij )
    Q = np.sum(qs)/mp
    
    # return both the modularity score and the adjacency matrix of the CCD network
    return Q, E





## SYNCHRONIZATION
def get_synchronization(X):
    
    """Function to compute the synchronisation component of the CSI. In complex
    systems, the synchronised behaviour of the different components often lead
    to qualitative changes in the properties of the system. In physics, such 
    changes are known as transition phases. In socioeconomic systems, such 
    qualitative changes could be an economic crisis or a civil conflict, for 
    example. An example of such phenomena can be found in civil violence models
    such as:
        
        - Fonoberova, M., Mezić, I., Mezić, J., Hogg, J., & Gravel, J. (2019). 
        Small-world networks and synchronisation in an agent-based model of 
        civil violence. Global Crime, 20(3-4), 161-195.
    
    The CSI considers synchronicity as a main feature as it can set the conditions
    for major socioeconomic changes, given a such such as an extreme environmental
    event. The literature on quantifying synchronisation in a system of time
    series is large. However, most methods requite large-scale data to work
    properly. The CSI uses a well-established approach called phase locking
    value (PLV) that originates in the intersection of signal processing and
    neuroscience. PLV measures the degree to which two time series synchronise
    their phases. The CSI employs the index developed by:
    
        - Lachaux, J. P., Rodriguez, E., Martinerie, J., & Varela, F. J. (1999). 
        Measuring phase synchrony in brain signals. Human brain mapping, 8(4), 194-208.
       
    PLV is a pairwise measure. Therefore, for the entire system, the CSI uses
    the average PLV across all pairs of indicators. The PLV is originally bound
    between 0 and 1. However, the CSI uses a rescaled version that stretches this
    feature to be between -1 and 1. The reason for this rescaling is simply for
    and easy interpretation of the CSI as the other two components also tend to 
    be between -1 and 1.
    
    If the PLV is close to 1, it means that the system exhibits a high degree
    of syntonisation (being 1 full synchronicity). On the other hand, if the 
    PLV is close to -1, is means that synchronisation is low. Arguably, 
    socioeconomic systems with higher synchronisation may be more likely to 
    experience qualitative changes due to exogenous shocks. Note that, the data
    on development indicators should have certain balance across the different
    development dimensions as, often, indicators belonging to the same dimension
    (e.g., economic) may be correlated by construction. Thus, the data preparation
    process should make sure to avoid redundancies in indicators that may come
    from the same source.
    

    Parameters
    ----------
        X: 2D numpy array 
            A matrix with the time series of the indicators. Each row corresponds
            to an indicator and each column to a time period. There cannot be
            missing values, so prior work on data imputation should be conducted.
            

    Returns
    -------
        PLV: float
            The system-wide phase locking value (between -1 and 1).
        
        S: 2D numpy array
            The matrix with the phase locking values between every pair of
            indicators.
    """
    
    n = X.shape[0]
    
    # normalize the time series
    deltaX = X[:,1::] - X[:,0:-1]
    means = np.tile(np.mean(deltaX, axis=1), (deltaX.shape[1], 1)).T
    stds = np.tile(np.std(deltaX, axis=1), (deltaX.shape[1], 1)).T
    X_normed = (deltaX - means) / stds

    # Compute the fast Fourier transform of the normalized time series
    X_freq_domain = np.fft.fft(X_normed, axis=1)

    # Compute the phase of the FFT components
    X_phase = np.angle(X_freq_domain)

    # Construct matrix with pairwise phase locking values (PLVs)
    S = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            phase_diff = X_phase[i] - X_phase[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            # we rescale the PLV to match the units of the other two components, so instead of being in (0,1), it is in (-1,1)
            S[i,j] = (2 * plv) - 1

    PLV = np.mean(S) # this is the average phase locking value for the entire system
    
    # return both the average phase locking value of the system as well as the matrix with the pairwise values
    return PLV, S






## EXTREME EVENTS
def get_heavy_tailness(X, percentile, n_resamples):
    
    """Function to compute the extreme events component of the CSI. This component
    aims at quantifying how likely it is to experience extreme events in a given
    set of observations. In the extreme-value-theory literature, there exist numerous
    methods and indices to achieve this. Since the CSI aims at (1) providing
    intuitive measures and (2) operating with small data, the method of choice
    for this component is the one proposed by:
        
        - Dekkers, A. L., Einmahl, J. H., & De Haan, L. (1989). A moment estimator 
        for the index of an extreme-value distribution. The Annals of Statistics, 1833-1855.
    
    The moment estimator is not strictly bound, but it generally tends to lie
    between -1 and 1. Negative values indicate that the distribution has slim
    tails, so extreme events are unlikely. On the other hand, when the estimator
    is positive, it suggests heavy tails. The larger the estimator, the more
    likely it is to experience extreme events.
    
    To overcome small sample issues, this function bootstraps the observations
    of the observations and returns the average moment estimator calculated
    across all the resampled datasets.
    

    Parameters
    ----------
        X: numpy array 
            A vector with the observations to be used. They should be positive values.
            
        percentile: int
            Since indices such as the moment estimator are calculated for the tail
            of the distribution, it is necessary to define the order statistic
            from which the tail begins. The CSI translates this query into the
            percentile of interest to the user (between 0 and 100).
            
        n_resamples: int
            The number of resamples to be performed in the bootstrapping procedure.


    Returns
    -------
        mean_dehaan_estimate: float
            The moment estimator.

    """
    
    def compute_dehaan_estimate(X):
        n = len(X)
        X = np.sort(X)
        x = np.percentile(X, percentile)
        k = np.argmin(np.abs(X - x))+1
        hill_estimate = (1/k) * np.sum(np.log(X[n-k::]) - np.log(X[n-k-1]))
        M2 = (1/k) * np.sum( (np.log(X[n-k::]) - np.log(X[n-k-1]))**2 )
        dehaan_estimate = hill_estimate + 1 - 0.5 * (1 - hill_estimate**2/M2)**(-1)
        return dehaan_estimate
    
    dehaan_estimates = st.bootstrap((X,), compute_dehaan_estimate, n_resamples=n_resamples)
    mean_dehaan_estimate = dehaan_estimates.bootstrap_distribution.mean()
    return mean_dehaan_estimate
































