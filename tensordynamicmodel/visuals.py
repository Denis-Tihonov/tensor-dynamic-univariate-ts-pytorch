import numpy as np
from numpy import pi
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.linalg import hankel

############################################################################################################
############################################################################################################
def delay_embedding_matrix(s, nlags):
    """Make a matrix with delay embeddings.

    Parameters
    ----------
    s : np.array
        The time series data.

    nlags : int
        Size of time lags.

    Returns
    -------
    delay_embedding_matrix : np.array of shape  (len(s) - lags + 1 , lags)
        Matrix with lags.
    """ 
    N = len(s)
    delay_embedding_matrix = hankel(s[ : N - nlags + 1], s[N - nlags : N])
    return delay_embedding_matrix
############################################################################################################
############################################################################################################
def plot_short_timeseries(s, frequency, path = None):
    """Plot and save example of time series.

    Parameters
    ----------
    s : np.array
        Training time series.

    frequency : int
        Frequency, the number of repetitions per second.

    path : str or path-like
        See for fname in  matplotlib.pyplot.savefig.
    """ 
    plt.plot(np.arange(0,len(s))/frequency, s)
    plt.xlabel('$t,c$', size=20)
    plt.ylabel('$s(t)$', size=20)
    plt.tight_layout()
    
    if path is not None:
        plt.savefig(path, format='png', dpi=200)
        
    plt.show()
############################################################################################################
############################################################################################################
def plot_phase_trajectory(
    phase_trajectory,
    path = None,
    rotation = (0,0,0),
    title=None
):
    """Plot and save example of phase trajectory.

    Parameters
    ----------
    phase_trajectory : np.array with shape (n_samples, 3)
        Training time series.

    path : str or path-like
        See for fname in  matplotlib.pyplot.savefig

    rotation : np.array with shape (3)
        Degree to rotate over axis.
    """ 
    ax = rotation[0]/180 * np.pi
    ay = rotation[1]/180 * np.pi
    az = rotation[2]/180 * np.pi
    
    T_X = np.array([[1,0,0],
                    [0,np.cos(ax),-np.sin(ax)],
                    [0,np.sin(ax), np.cos(ax)]])
    
    T_Y = np.array([[np.cos(ay),-np.sin(ay),0],
                    [np.sin(ay), np.cos(ay),0],
                    [0,0,1]])
    
    T_Z = np.array([[ np.cos(az),0,np.sin(az)],
                    [ 0,1,0],
                    [-np.sin(az),0,np.cos(az)]])
    
    phase_trajectory = phase_trajectory@T_Z@T_Y@T_X
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(phase_trajectory[:,0],phase_trajectory[:,1],phase_trajectory[:,2],lw = 1)
    
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.view_init(elev=20, azim=135)
    
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.set_title(title)
    if path is not None:
        fig.tight_layout()
        fig.savefig(path, format='png', dpi=200, bbox_inches='tight')
        
    return ax

############################################################################################################
############################################################################################################
def plot_phase_trajectory_and_phase(
    phase_trajectory,
    expextation_values,
    phase_history,
    path = None,
    rotation = (0,0,0)
):
    """Plot and save example of phase trajectory with phase.

    Parameters
    ----------
    phase_trajectory : np.array with shape (n_samples, 3)
        Training time series.

    expextation_values : np.array with shape (n_samples, 3)
        Expextation model for phase trajectory of initial time series.

    phase_history : np.array with shape (n_samples)
        Phase of initial time series.

    path : str or path-like
        See for fname in  matplotlib.pyplot.savefig

    rotation : np.array with shape (3)
        Degree to rotate over axis.
    """ 
    ax = rotation[0]/180 * np.pi
    ay = rotation[1]/180 * np.pi
    az = rotation[2]/180 * np.pi
    
    T_X = np.array([[1,0,0],
                    [0,np.cos(ax),-np.sin(ax)],
                    [0,np.sin(ax), np.cos(ax)]])
    
    T_Y = np.array([[np.cos(ay),-np.sin(ay),0],
                    [np.sin(ay), np.cos(ay),0],
                    [0,0,1]])
    
    T_Z = np.array([[ np.cos(az),0,np.sin(az)],
                    [ 0,1,0],
                    [-np.sin(az),0,np.cos(az)]])
    
    phase_trajectory = phase_trajectory@T_Z@T_Y@T_X
    expextation_values = expextation_values@T_Z@T_Y@T_X
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
  
    img =ax.scatter(
        phase_trajectory[:,0],
        phase_trajectory[:,1],
        phase_trajectory[:,2],
        c=cm.rainbow(phase_history[:]/(2*np.pi)),
        alpha = 1,
        s=5
    )
    
    m = cm.ScalarMappable(cmap=cm.rainbow)
    m.set_array(phase_history)
    
    cbar = fig.colorbar(m,shrink=0.5)
    
    cbar.ax.get_yaxis().set_ticks([])
    for i, lab in enumerate(['$0$','$\pi$/2','$\pi$','3$\pi$/2','2$\pi$']):
        cbar.ax.text(1, (3 * np.pi * i / 6), lab, size=16)
    
    ax.plot(
        expextation_values[:,0],
        expextation_values[:,1],
        expextation_values[:,2],
        color = 'orange',
        lw = 7
    )
    
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.view_init(elev=20, azim=135)
    
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    if path is not None:
        fig.savefig(path, format='png', dpi=200, bbox_inches='tight')
    
    fig.show()
    return fig, ax
############################################################################################################
############################################################################################################
class SSA(object):
    
    __supported_types = (pd.Series, np.ndarray, list)
    
    def __init__(self, tseries, L, save_mem=True):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.
        
        Parameters
        ----------
        tseries : The original time series, in the form of a Pandas Series, NumPy array or list. 
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.
        
        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """
        
        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")
        
        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N/2:
            raise ValueError("The window length must be in the interval [2, N/2].")
        
        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1
        
        # Embed the time series in a trajectory matrix
        self.X = np.array([self.orig_TS.values[i:L+i] for i in range(0, self.K)]).T
        
        # Decompose the trajectory matrix
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.d = np.linalg.matrix_rank(self.X)
        
        self.TS_comps = np.zeros((self.N, self.d))
        
        if not save_mem:
            # Construct and save all the elementary matrices
            self.X_elem = np.array([ self.Sigma[i]*np.outer(self.U[:,i], VT[i,:]) for i in range(self.d) ])

            # Diagonally average the elementary matrices, store them as columns in array.           
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.V = VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(self.d):
                X_elem = self.Sigma[i]*np.outer(self.U[:,i], VT[i,:])
                X_rev = X_elem[::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."
            
            # The V array may also be very large under these circumstances, so we won't keep it.
            self.V = "Re-run with save_mem=False to retain the V matrix."
        
        # Calculate the w-correlation matrix.
        self.calc_wcorr()
            
    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d
        
        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)
            
    
    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.
        
        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
        """
        if isinstance(indices, int): indices = [indices]
        
        ts_vals = self.TS_comps[:,indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)
    
    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """
             
        # Calculate the weights
        w = np.array(list(np.arange(self.L)+1) + [self.L]*(self.K-self.L-1) + list(np.arange(self.L)+1)[::-1])
        
        def w_inner(F_i, F_j):
            return w.dot(F_i*F_j)
        
        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.TS_comps[:,i], self.TS_comps[:,i]) for i in range(self.d)])
        F_wnorms = F_wnorms**-0.5
        
        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i+1,self.d):
                self.Wcorr[i,j] = abs(w_inner(self.TS_comps[:,i], self.TS_comps[:,j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j,i] = self.Wcorr[i,j]
    
    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d
        
        if self.Wcorr is None:
            self.calc_wcorr()
        
        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0,1)
        
        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d-1
        else:
            max_rnge = max
        
        plt.xlim(min-0.5, max_rnge+0.5)
        plt.ylim(max_rnge+0.5, min-0.5)