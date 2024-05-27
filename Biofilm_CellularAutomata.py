# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:20:53 2024

@author: Raphaël RUBRICE - AgroParisTech

This is my project for the 2nd year 'Engineering by numerical simulation' 
project course at AgroParisTech.

My goals were :
    1) To siimulate bacterial biofilm growth using the Cellular Automaton approach
    2) To simulate bacterial interactions between strains within the biofilm

To do so I mainly based my work on the following scientific papers :
    [1] Sarukhanian, S.; Maslovskaya, A.; Kuttler, C. (2023) - 
    Three-Dimensional Cellular Automaton for Modeling of Self-Similar 
    Evolution in Biofilm-Forming Bacterial Populations - 
    Mathematics 2023, 11, 3346. https://doi.org/10.3390/ math11153346
    
    [2] Irene Guzman-Soto,1 Christopher McTiernan,1 Mayte Gonzalez-Gomez,1 
    Alex Ross,1,2 Keshav Gupta,1 Erik J. Suuronen,1 Thien-Fah Mah,2 
    May Griffith,3,4 and Emilio I. Alarcon1,2, (2021) - 
    Mimicking biofilm formation and development: Recent progress in in vitro 
    and in vivo biofilm models - iScience 24, 102443, May 21, 2021

For further bibliography, read the project report accessible on github.

You have the possibility to make simulations in both 2D and 3D. 
However, note that simulations in 3D takes a lot of time to complete on regular 
computers so you should use simpler simulations and small size spaces when 
trying 3D mode.

Enjoy :D ! 

~ Raphaël RUBRICE.

PS : As you may notice, the following code is not at all perfect, so feel free
to make improvements and share them ! :D
"""

# =============================================================================
#                                 IMPORTS
# =============================================================================
# To manipulate files and folders
import shutil
import os
# Set working directory
wd = 'C:/Coding Projects/UP_IngSum'
os.chdir(wd)

# To access time infos
import datetime
import time

# To use deepcopy
from copy import deepcopy

# To save any kind of python object
import pickle as pkl

# To make the simulation
import numpy as np
import random

# For multiprocessing 
from joblib import Parallel, delayed

# To make animations and plots
import matplotlib.pyplot as plt
import cellpylib3d as cpl3d
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
import fitz
import io
from PIL import Image

# =============================================================================
#                               HELPER FUNCTIONS
# =============================================================================
def generate_n_colors(n, mode='strain'):
    """
    Generates a list of n colors using matplotlib color maps viridis if n <= 5, 
    Dark2_r otherwise.

    Parameters
    ----------
    n : int
        Number of colors to generate.
    mode : str, optional
        The default is 'strain'.

    Returns
    -------
    colors : list
        list of n colors.

    """
    if n<=5 and mode == 'strain':
        colormap = plt.cm.viridis
    else:
        colormap = plt.cm.ocean_r
    colors = [colormap(i) for i in np.linspace(0, 1, n)]
    if mode=='strain':
        colors[0] = '#ff0000' # dead cells in red
        colors[1] = '#F5F5F5' # medium in white     
    else:
        colors[0] = '#F5F5F5' 
    return colors


def format_dict(d, indent=0):
    """
    Function to format the dictionary into a readable string

    Parameters
    ----------
    d : dictionary.
    indent : int, optional
            The default is 0.

    Returns
    -------
    s : string
        String corresponding to the given dictionary.

    """
    s = ""
    for key, value in d.items():
        s += ' ' * indent + f"{key}: "
        # if the value is a dictionary
        if isinstance(value, dict):
            s += "\n" + format_dict(value, indent+4)
        else:
            s += f"{value}\n"
    return s

# =============================================================================
#                            CELLULAR AUTOMATON
# =============================================================================

# Define Cellular Automaton (Grid class)
class Grid:
    def __init__(self, size):
        """
        Parameters
        ----------
        size : tuple (height,length) if 2D, else (height,length,depth)
        """
        self.height, self.length = size[0], size[1]
        if len(size) == 3:
            self.dimensionality = 3
            self.depth = size[2]
        else:
            self.dimensionality = 2
                    
        self.grid = np.zeros(size)  #Biomass Matrix
        self.dead = -1              # Value to represent dead cells
        self.dead_pos = []          # List of dead cells positions 
        self.medium = 0             # Value to represent the medium
        
        
    # =========================================================================
    #                  INITIALIZATION METHODS AND FUNCTIONS
    # =========================================================================
    
    def possibles_idx(self, height, length, depth=None, d_min=0, d_max=None,
                      h_min=0, h_max=None, l_min=0, l_max=None, memory=None):
        """
        Generates all possible positions between a set of boundaries.
        
        Parameters
        ----------
        height : int
            Maximum height value.
        length : int
            Maximum length value.
        depth : int, optional
            Maximum depth value. The default is None.
        d_min : int, optional
            Minimum depth value. The default is 0.
        d_max : int, optional
            Maximum depth value. The default is None.
        h_min : int, optional
            Minimum depth value. The default is 0.
        h_max : int, optional
            Maximum height value. The default is None.
        l_min : int, optional
            Minimum length value. The default is 0.
        l_max : int, optional
            Maxium length value. The default is None.
        memory : list, optional
            List of indices already used. The default is None.

        Returns
        -------
        L : List
            List of all potential indices.

        """
        if self.dimensionality == 2: # 2D
            L = []
            if memory:
                if h_max == None and l_max == None:
                    for i in range(h_min, height):
                        for j in range(l_min, length):
                            if (i,j) not in memory:
                                L.append((i,j))
                    return L
                else :
                    assert [h_max, l_max]!=[None, None], 'You need to specify both h_max and l_max.'
                    for i in range(h_min, h_max):
                        for j in range(l_min, l_max):
                            if (i,j) not in memory:
                                L.append((i,j))
            else:
                if h_max == None and l_max == None:
                    for i in range(h_min, height):
                        for j in range(l_min, length):
                            L.append((i,j))
                    return L
                else :
                    assert [h_max, l_max]!=[None, None], 'You need to specify both h_max and l_max.'
                    for i in range(h_min, h_max):
                        for j in range(l_min, l_max):
                            L.append((i,j))
        else: # 3D
            L = []
            if memory:
                if h_max == None and l_max == None and d_max == None and depth!=None:
                    for i in range(h_min, height):
                        for j in range(l_min, length):
                            for k in range(d_min, depth):
                                if (i,j,k) not in memory:
                                    L.append((i,j,k))
                    return L
                else :
                    assert [h_max, l_max, d_max]!=[None, None, None], 'You need to specify both h_max and l_max.'
                    for i in range(h_min, h_max):
                        for j in range(l_min, l_max):
                            for k in range(d_min, d_max):
                                if (i,j,k) not in memory:
                                    L.append((i,j,k))
            else:
                if h_max == None and l_max == None and d_max == None and depth!=None:
                    for i in range(h_min, height):
                        for j in range(l_min, length):
                            for k in range(d_min, depth):
                                L.append((i,j,k))
                    return L
                else :
                    assert [h_max, l_max, d_max]!=[None, None, None], 'You need to specify both h_max and l_max.'
                    for i in range(h_min, h_max):
                        for j in range(l_min, l_max):
                            for k in range(d_min, d_max):
                                L.append((i,j,k))
        return L
        
    def unique_random(self, size, height, length, depth=None, d_min=0, d_max=None, 
                      h_min=0, h_max=None, l_min=0, l_max=None, memory=None):
        """
        Generates the list of indices used for inoculation.

        Parameters
        ----------
        size : int
            Number of indices to choose, this is the length of the output.
        height : int
            Maximum height value.
        length : int
            Maximum length value.
        depth : int, optional
            Maximum depth value. The default is None.
        d_min : int, optional
            Minimum depth value. The default is 0.
        d_max : int, optional
            Maximum depth value. The default is None.
        h_min : int, optional
            Minimum depth value. The default is 0.
        h_max : int, optional
            Maximum height value. The default is None.
        l_min : int, optional
            Minimum length value. The default is 0.
        l_max : int, optional
            Maxium length value. The default is None.
        memory : list, optional
            List of indices already used. The default is None.

        Returns
        -------
        indices : List
            List of indices used to inoculate the grid.

        """
        indices = []
        possible_idx = self.possibles_idx(height=height, length=length, depth=depth, d_min=d_min, d_max=d_max,
                                          h_min=h_min, h_max=h_max, l_min=l_min, l_max=l_max,
                                          memory=memory)
        # Among posisble indices, randomly choose 'size' indices
        for i in range(size):
            cell_pos = possible_idx[random.randint(0,len(possible_idx)-1)]
            indices.append(cell_pos)
            possible_idx.remove(cell_pos)
        return indices
                    
    def get_grids(self, TOX_dictionary,
                  O_init=7e-3, O_init_grad=True, scale_O='linear',
                  N_init=0.7e-3, N_init_grad=True, scale_N='linear',
                  C_init=25e-3, C_init_grad=True, scale_C='linear',
                  TOX_init=0, TOX_init_grad=True, scale_TOX='linear',
                  nportions=5):
        """
        Creates a dictionary of nutrients and Toxin matrices. Those are 
        used in several aspects of the simulation such as molecule diffusion, 
        cell reproduction, cellular death and cellular movement.

        Parameters
        ----------
        TOX_dictionary : dicitonary
                {'TOX0':{'+':[0,2], '-':[1], 'DTOX':0.7e-9},
                'TOX1':{'+':[1], '-':[2], 'DTOX':0.7e-9},
                'TOX2':{'+':[3], '-':[0], 'DTOX':0.7e-9}}.
        O_init : int, optional
            Maximum amount of Oxygen in space. The default is 1e9.
        O_init_grad : boolean, optional
            Whether to make a gradient or not (if True, highest values 
            are at the surface). The default is True.
        scale_O : string, optional
            The scale to use if init_grad set to True ('log', 'log10', 
            'log2', 'linear'). The default is 'linear'.
        N_init : int, optional
            Maximum amount of Nitrogen in space. The default is 1e9.
        N_init_grad : boolean, optional
            Whether to make a gradient or not (if True, highest values 
            are at the bottom). The default is True.
        scale_N : string, optional
            The scale to use if init_grad set to True ('log', 'log10', 
            'log2', 'linear'). The default is 'linear'.
        C_init : int, optional
            Maximum amount of Carbon in space. The default is 1e9.
        C_init_grad : boolean, optional
            Whether to make a gradient or not (if True, highest values 
            are at the bottom). The default is True.
        scale_C : string, optional
            The scale to use if init_grad set to True ('log', 'log10', 
            'log2', 'linear'). The default is 'linear'.
        TOX_init : int, optional
            Maximum amount of Toxins in space. The default is 0.
        TOX_init_grad : boolean, optional
            Whether to make a gradient or not (if True, highest values 
            are at the surface). The default is True.
        scale_TOX : string, optional
            The scale to use if init_grad set to True ('log', 'log10', 
            'log2', 'linear'). The default is 'linear'.
        nportions : int, optional
            How many parts for the gradient to be made of. The default is 5.

        Returns
        -------
        A dictionary of matrices :
            {'Biomass_Matrix':self.grid, 
            'O_Matrix': self.O_Matrix, 
            'N_Matrix': self.N_Matrix,
            'C_Matrix': self.C_Matrix,
            'TOX_Matrix': self.TOX_Matrix}

        """
        # Helper function to apply gradient scaling
        def apply_gradient(matrix, portion, init_value, scale_type, reverse=True):
            if scale_type == 'linear':
                assert nportions <= self.height, 'When using linear gradient, nportions must be smaller than grid height.'
                for k in range(nportions):
                    factor = (nportions - k) if reverse else (k + 1)
                    matrix[k * portion:(k + 1) * portion] /= factor
            elif scale_type.startswith('log'):
                base = {'log': np.e, 'log2': 2, 'log10': 10}[scale_type]
                for k in range(nportions):
                    factor = base**(nportions-k-1) if reverse else base**k
                    matrix[k * portion:(k + 1) * portion] = init_value * (1 / factor)
            return matrix
    
        portion = int(np.ceil(self.height / nportions))  # Used if .._init_grad is set to true
    
        # Oxygen Matrix
        self.O_Matrix = self.grid + O_init
        if O_init_grad:
            self.scale_O = scale_O
            self.O_Matrix = apply_gradient(self.O_Matrix, portion, O_init, scale_O, reverse=False)
    
        # Nitrogen Matrix
        self.N_Matrix = self.grid + N_init
        if N_init_grad:
            self.scale_N = scale_N
            self.N_Matrix = apply_gradient(self.N_Matrix, portion, N_init, scale_N)
    
        # Carbon Matrix
        self.C_Matrix = self.grid + C_init
        if C_init_grad:
            self.scale_C = scale_C
            self.C_Matrix = apply_gradient(self.C_Matrix, portion, C_init, scale_C)
    
        # Toxins Matrix
        self.TOX_Matrices = []
        for tox in range(len(TOX_dictionary)):
            M = self.grid + TOX_init
            if TOX_init_grad:
                self.scale_TOX = scale_TOX
                M = apply_gradient(M, portion, TOX_init, scale_TOX, reverse=True)
            self.TOX_Matrices.append(M)
        return {'Biomass_Matrix': self.grid,
                'O_Matrix': self.O_Matrix,
                'N_Matrix': self.N_Matrix,
                'C_Matrix': self.C_Matrix,
                'TOX_Matrices': self.TOX_Matrices}

    def place_cells(self, indices, encoding, ncells, nstrains, biomass_init=10, 
                    place_only=False, strainID=None, start_lists=False):
        """
        Inoculates the space by placing cells. Creates the Strain Matrix which 
        represents where are each cell of each strain, where are empty spaces 
        and dead cells. Updates the biomass matrix (self.grid).
        

        Parameters
        ----------
        indices : list of tuples
            Coordonates of each cell to place.
        encoding : list of integers
            List containing the values that represent each strain in 
            the Strains Matrix.
        ncells : int
            number of cells to place.
        nstrains : int
            number of strains in the simulation.
        biomass_init : int, optional
            Biomass quantity for a single cell. The default is 10.
        place_only : boolean, optional
            Whether to only place one strain or not. The default is False.
        strainID : int, optional
            ID of the strain to place with place_only. The default is None.
        start_lists : boolean, optional
            Whether to create self.cells_pos, self.Index_by_strain, 
            self.INDEX_HISTORY or not. The default is False.

        Returns
        -------
        Strains_Matrix : np.array, dtype = int
            Strains Matrix (Matrix of main interest to visualize the simulation).
        Index : List of tuples
            List of positions of every cell placed.

        """
    
        if place_only:
            if start_lists:
                Index = [] # To know each living cell's position
                Index_by_strain = list([])
                INDEX_HISTORY = list([])
                self.cells_pos = Index
                self.Index_by_strain = Index_by_strain
                
                # To be able to know the strainID of both living and dead cells
                self.INDEX_HISTORY = INDEX_HISTORY 
                
                if self.dimensionality == 2:
                    Strains_Matrix = np.zeros((self.height, self.length), dtype=int)
                else:
                    Strains_Matrix = np.zeros((self.height, self.length, self.depth), dtype=int)
            else:
                Index = deepcopy(self.cells_pos)
                Index_by_strain = deepcopy(self.Index_by_strain)
                INDEX_HISTORY = deepcopy(self.INDEX_HISTORY)
                Strains_Matrix = self.Strains_Matrix.copy()
            assert strainID!=None, 'No strainID specified but place_only was set to True.'
            
            Index += indices
            Index_by_strain.append(indices)
            INDEX_HISTORY.append(indices)
            
            if self.dimensionality == 2:
                i_strain, j_strain = zip(*indices)
                
                # Place cell in Strains Matrix
                Strains_Matrix[i_strain, j_strain] = strainID 
                
                # Place cell in Biomass Matrix
                self.grid[i_strain, j_strain] = biomass_init
            else: #3D
                i_strain, j_strain, k_strain = zip(*indices)
                
                # Place cell in Strains Matrix
                Strains_Matrix[i_strain, j_strain, k_strain] = strainID 
                
                # Place cell in Biomass Matrix
                self.grid[i_strain, j_strain, k_strain] = biomass_init
                
            self.cells_pos = Index #Index of live cells
            self.Index_by_strain = Index_by_strain
            self.INDEX_HISTORY = INDEX_HISTORY
        else:
            Index = [] # To know each living cell's position
            Index_by_strain = []
            
            # To be able to know the strainID of both living and dead cells
            INDEX_HISTORY = [] 
            
            if self.dimensionality == 2:
                Strains_Matrix = np.zeros((self.height, self.length), dtype=int)
            else:
                Strains_Matrix = np.zeros((self.height, self.length, self.depth), dtype=int)

            for strain in encoding:
                # Chose a random set of indices for this strain
                idx_strain = random.sample(indices, ncells)
                
                # Remove those used 
                for idx in idx_strain:
                    indices.remove(idx)
                
                Index += idx_strain
                Index_by_strain.append(idx_strain)
                INDEX_HISTORY.append(idx_strain)
                
                if self.dimensionality == 2:
                    i_strain, j_strain = zip(*idx_strain)
                    
                    # Place cell in Strains Matrix
                    Strains_Matrix[i_strain, j_strain] = strain 
                    
                    # Place cell in Biomass Matrix
                    self.grid[i_strain, j_strain] = biomass_init
                else: #3D
                    i_strain, j_strain, k_strain = zip(*idx_strain)
                    
                    # Place cell in Strains Matrix
                    Strains_Matrix[i_strain, j_strain, k_strain] = strain 
                    
                    # Place cell in Biomass Matrix
                    self.grid[i_strain, j_strain, k_strain] = biomass_init
                
            self.cells_pos = Index 
            self.Index_by_strain = Index_by_strain
            self.INDEX_HISTORY = INDEX_HISTORY
        self.nstrains = nstrains 
        self.strain_popsize = ncells 
        self.strainID = encoding 
        self.Strains_Matrix = Strains_Matrix
        return Strains_Matrix, Index
    
    def on_cluster_inoculation(self, cluster_strain, cluster_height, cluster_width,
                                max_depth_other, ncells, nstrains, cluster_depth=None, biomass_init=10):
        """
        Initialization method that creates a cluster of one strain only and 
        inoculates the remaining strains above it.

        Parameters
        ----------
        cluster_strain : int
            Which strain to place (e.g for 3 strains total, put 0 for Strain0).
        cluster_height : int
            Cluster height.
        cluster_width : int
            Cluster width.
        max_depth_other : int
            Maximum height value for the remaining strains (in our simulation, 
            height 0 is the surface).
        ncells : int
            Number of cells to place for each strain (all strains start with 
            the same population size).                                          
        nstrains : int
            Number of strains in the simulation.
        cluster_depth : int, optional
            Cluster depth. The default is None.
        biomass_init : int, optional
            Biomass quantity for a single cell. The default is 10.

        Returns
        -------
        Strains_M : np.array, dtype = int
            Strains Matrix (Matrix of main interest to visualize the simulation).
        Indx : List of tuples
            List of positions of every cell placed.

        """
        if self.dimensionality == 2:
            assert ncells <= cluster_height*cluster_width, 'Too much cells to place for the indicated cluster size.'
            assert ncells <= max_depth_other*self.length, 'Too much cells to place for the indicated inoculation depth.'
            assert max_depth_other <= self.height-cluster_height, 'max_depth_other is too deep.'
        else:
            assert cluster_depth != None, 'You must specify cluster_depth.'
            assert ncells <= cluster_height*cluster_width*cluster_depth, 'Too much cells to place for the indicated cluster size.'
            assert ncells <= max_depth_other*self.length*self.depth, 'Too much cells to place for the indicated inoculation depth.'
            assert max_depth_other <= self.height-cluster_height, 'max_depth_other is too deep.'

        encoding = [i*2+3 for i in range((nstrains))] # Strain IDs
        self.Biomass_init = biomass_init
        
        memory = []
        for i, strain in enumerate(encoding):
            if i==0:
                start_L = True
            else:
                start_L = False
            if i == cluster_strain:
                # Cluster inoculation
                if self.dimensionality == 2: # 2D
                    center = int(self.length/2)
                    L_min = center-int(cluster_width/2)
                    cluster = self.unique_random(ncells, height=self.height, length=center+int(cluster_width/2), 
                                                 h_min=self.height-cluster_height, l_min=L_min)
                    Strains_M, Indx = self.place_cells(cluster, encoding, ncells, nstrains, biomass_init=biomass_init,
                                                   place_only=True, strainID=strain, start_lists=start_L)
                else: # 3D
                    l_center = int(self.length/2)
                    d_center = int(self.depth/2)
                    L_min = l_center-int(cluster_width/2)
                    D_min = d_center - int(cluster_depth/2)
                    cluster = self.unique_random(ncells, height=self.height, length=l_center+int(cluster_width/2), 
                                                 depth=d_center+int(cluster_depth/2), d_min=D_min,
                                                 h_min=self.height-cluster_height, l_min=L_min)
                    Strains_M, Indx = self.place_cells(cluster, encoding, ncells, nstrains, biomass_init=biomass_init,
                                                   place_only=True, strainID=strain, start_lists=start_L)

            else:
                #Other strains inoculation
                if self.dimensionality == 2: # 2D
                    colonie = self.unique_random(ncells, height=max_depth_other, length=self.length, memory=memory)
                    memory.extend(colonie)
                    Strains_M, Indx = self.place_cells(colonie, encoding, ncells, nstrains, biomass_init=biomass_init,
                                                   place_only=True, strainID=strain, start_lists=start_L)
                else: # 3D
                    colonie = self.unique_random(ncells, height=max_depth_other, length=self.length, 
                                                 depth=self.depth,
                                                 memory=memory)
                    memory.extend(colonie)
                    Strains_M, Indx = self.place_cells(colonie, encoding, ncells, nstrains, biomass_init=biomass_init,
                                                   place_only=True, strainID=strain, start_lists=start_L)

        return Strains_M, Indx
        
    
    def separate_colonies_inoculation(self, col_height, col_width, ncells, nstrains, col_depth=None, biomass_init=10):
        """
        Initialization method that creates a cluster for each strain.

        Parameters
        ----------
        col_height : int
            Height of the colony.
        col_width : int
            Width of the colony.
        ncells : int
            Number of cells to place for each strain (all strains start with 
            the same population size).                                          
        nstrains : int
            Number of strains in the simulation.
        col_depth : int, optional
            Depth of the colony. The default is None.
        biomass_init : int, optional
            Biomass quantity for a single cell. The default is 10.

        Returns
        -------
        Strains_M : np.array, dtype = int
            Strains Matrix (Matrix of main interest to visualize the simulation).
        Indx : List of tuples
            List of positions of every cell placed.

        """
        assert col_height <= self.height, 'Colony height is too high to be placed.'
        
        if self.dimensionality == 2:
            assert ncells <= col_height*col_width, 'Too much cells to place for the indicated colony size.'
        else:
            assert col_depth != None, 'You must specify col_depth.'
            assert ncells <= col_height*col_width*col_depth, 'Too much cells to place for the indicated colony size.'
            
        encoding = [i*2+3 for i in range((nstrains))] # Strain IDs
        self.Biomass_init = biomass_init
        
        H_min = self.height-col_height      # Because 0 is the surface
        if self.dimensionality == 3:
            D_min = int(self.depth/2 - col_depth/2)
            assert D_min > 0, 'col_depth is too big to be placed.'
            
        inter_col = int((self.length - nstrains*col_width)/(nstrains + 1))
        assert  inter_col > 0, 'Colony width is too big for all colonies to be placed.'
        start_L =True
        end_of_col = 0
        
        for i,strain in enumerate(encoding):
            if i>0:
                start_L =False
            if i==0:
                L_min = inter_col
            else:
                L_min = end_of_col+ inter_col
            end_of_col = col_width + L_min
            
            if self.dimensionality == 2: # 2D
                if end_of_col < self.length:
                    colonie = self.unique_random(ncells, height=self.height, length=end_of_col, 
                                             h_min=H_min, l_min=L_min)
                else:
                    colonie = self.unique_random(ncells, height=self.height, length=self.length, 
                                             h_min=H_min, l_min=L_min)
                Strains_M, Indx = self.place_cells(colonie, encoding, ncells, nstrains, biomass_init=biomass_init,
                                               place_only=True, strainID=strain, start_lists=start_L)
            else: # 3D
                if end_of_col < self.length:
                    colonie = self.unique_random(ncells, height=self.height, length=end_of_col,
                                                 depth=D_min+col_depth,
                                             h_min=H_min, l_min=L_min, d_min=D_min)
                else: 
                    colonie = self.unique_random(ncells, height=self.height, length=self.length, 
                                             depth=D_min+col_depth,
                                         h_min=H_min, l_min=L_min, d_min=D_min)
                Strains_M, Indx = self.place_cells(colonie, encoding, ncells, nstrains, biomass_init=biomass_init,
                                               place_only=True, strainID=strain, start_lists=start_L)

        return Strains_M, Indx
    
    def floor_inoculation(self, thickness, ncells, nstrains, biomass_init=10):
        """
        Initialization method that creates a cell floor containing all strains.

        Parameters
        ----------
        thickness : int
            thickness of the floor.
        ncells : int
            Number of cells to place for each strain (all strains start with 
            the same population size).                                          
        nstrains : int
            Number of strains in the simulation.
        biomass_init : int, optional
            Biomass quantity for a single cell. The default is 10.

        Returns
        -------
        Strains_M : np.array, dtype = int
            Strains Matrix (Matrix of main interest to visualize the simulation).
        Indx : List of tuples
            List of positions of every cell placed.

        """
        assert ncells*nstrains <= thickness*self.length, 'Too much cells to place for the indicated thickness.'
        encoding = [i*2+3 for i in range((nstrains))] # Strain IDs
        self.Biomass_init = biomass_init
        
        if self.dimensionality == 2: # 2D
            indices = self.unique_random(ncells*nstrains, height=self.height, length=self.length, 
                                         h_min=self.height-thickness)
        else: # 3D
            indices = self.unique_random(ncells*nstrains, height=self.height, length=self.length, 
                                         depth=self.depth,
                                         h_min=self.height-thickness)
        Strains_M, Indx = self.place_cells(indices, encoding, ncells, nstrains, biomass_init=biomass_init)
        return Strains_M, Indx
    
    def surface_inoculation(self, thickness, ncells, nstrains, biomass_init=10):
        """
        Initialization method that places all cells close to the surface. 

        Parameters
        ----------
        thickness : int
            thickness of the floor.
        ncells : int
            Number of cells to place for each strain (all strains start with 
            the same population size).                                          
        nstrains : int
            Number of strains in the simulation.
        biomass_init : int, optional
            Biomass quantity for a single cell. The default is 10.

        Returns
        -------
        Strains_M : np.array, dtype = int
            Strains Matrix (Matrix of main interest to visualize the simulation).
        Indx : List of tuples
            List of positions of every cell placed.

        """
        assert ncells*nstrains <= thickness*self.length, 'Too much cells to place for the indicated thickness.'
        encoding = [i*2+3 for i in range((nstrains))] # Strain IDs
        self.Biomass_init = biomass_init
        
        if self.dimensionality == 2: # 2D
            indices = self.unique_random(ncells*nstrains, height=thickness, length=self.length)
        else: # 3D
            indices = self.unique_random(ncells*nstrains, height=thickness, length=self.length, 
                                         depth=self.depth)
        
        Strains_M, Indx = self.place_cells(indices, encoding, ncells, nstrains, biomass_init=biomass_init)
        return Strains_M, Indx
    
    def random_inoculate(self, ncells, nstrains, biomass_init=10):
        """
        Initialization method that randomly places all cells across 
        the entire space. 

        Parameters
        ----------
        thickness : int
            thickness of the floor.
        ncells : int
            Number of cells to place for each strain (all strains start with 
            the same population size).                                          
        nstrains : int
            Number of strains in the simulation.
        biomass_init : int, optional
            Biomass quantity for a single cell. The default is 10.

        Returns
        -------
        Strains_M : np.array, dtype = int
            Strains Matrix (Matrix of main interest to visualize the simulation).
        Indx : List of tuples
            List of positions of every cell placed.

        """
        encoding = [i*2+3 for i in range((nstrains))] # Strain IDs
        self.Biomass_init = biomass_init
        
        
        if self.dimensionality == 2: # 2D
            indices = self.unique_random(ncells*nstrains, height=self.height, length=self.length)
        else: # 3D
            indices = self.unique_random(ncells*nstrains, height=self.height, length=self.length, depth=self.depth)
        
        Strains_M, Indx = self.place_cells(indices, encoding, ncells, nstrains, biomass_init=biomass_init)
        return Strains_M, Indx
    
    # =========================================================================
    #                             NEIGHBORHOOD
    # =========================================================================

    def neighborhood(self, neighborhood='VonNeumman', radius=1, dead_mode=False, all_mode=False):
        """
        Generates the list of neighbors positions for each cell.

        Parameters
        ----------
        neighborhood : str, optional
            Type of neighborhood to use. The default is 'VonNeumman'.
        radius : int, optional
            Radius of the neighborhood. The default is 1.
        dead_mode : boolean, optional
            Whether to compute the neighborhood of dead or living cells. 
            The default is False.
        all_mode : boolean, optional
            Whether to compute the neighborhood for all cells. 
            The default is False.    
        Returns
        -------
        Near : List of lists
            List containing a list of all neighbors positions for each cell.

        """
        assert 'cells_pos' in self.__dict__, "Medium not yet inoculated, check the orthograph of 'init_strategy' parameter"
        Near = []
        if dead_mode:
            positions_to_look = self.dead_pos
        elif all_mode:
            positions_to_look = []
            if self.dimensionality == 3:
                for i in range(self.height):
                    for j in range(self.length):
                        for k in range(self.depth):
                            positions_to_look.append((i,j,k))
            else:
                for i in range(self.height):
                    for j in range(self.length):
                        positions_to_look.append((i,j))
        else:
            positions_to_look = self.cells_pos
        
        if neighborhood=='VonNeumman':
            for cell in positions_to_look:
                if self.dimensionality == 2:
                    i, j = cell
                    L = []
                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                                if abs(di) + abs(dj) <= radius:
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < self.height and 0 <= nj < self.length:
                                        L.append((ni, nj))
                else:
                    i, j, k = cell
                    L = []
                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                            for dk in range(-radius, radius + 1):
                                if abs(di) + abs(dj) + abs(dk) <= radius:
                                    ni, nj, nk = i + di, j + dj, k + dk
                                    if 0 <= ni < self.height and 0 <= nj < self.length and 0 <= nk < self.depth:
                                        L.append((ni, nj, nk))
                Near.append(L)
                    
        if neighborhood=='Moore':
            for cell in positions_to_look:
                if self.dimensionality == 2:
                    i, j = cell
                    L = []
                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.height and 0 <= nj < self.length:
                                L.append((ni, nj))
                else:
                    i, j, k = cell
                    L = []
                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                            for dk in range(-radius, radius + 1):
                                ni, nj, nk = i + di, j + dj, k + dk
                                if 0 <= ni < self.height and 0 <= nj < self.length and 0 <= nk < self.depth:
                                    L.append((ni, nj, nk))
                Near.append(L)
        if neighborhood=='halfVN':
            for cell in positions_to_look:
                if self.dimensionality == 2:
                    i, j = cell
                    L = []
                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                            if abs(di) + abs(dj) <= radius:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < self.height and 0 <= nj < self.length:
                                    L.append((ni, nj))
                else:
                    i, j, k = cell
                    L = []
                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                            for dk in range(-radius, radius + 1):
                                if abs(di) + abs(dj) + abs(dk) <= radius:
                                    ni, nj, nk = i + di, j + dj, k + dk
                                    if 0 <= ni < self.height and 0 <= nj < self.length and 0 <= nk < self.depth:
                                        L.append((ni, nj, nk))
                for i in range(int(len(L)/2)):
                    element = random.sample(L, 1)[0]
                    L.remove(element)
                Near.append(L)
        if dead_mode:
            return Near
        elif all_mode:
            return Near, positions_to_look
        else:
            self.neighbors = deepcopy(Near)
            return Near
    
    # =========================================================================
    #                   NUTRIENT and ToxinS EVOLUTION
    # =========================================================================
    def calculate_diffusion(self, Doxy, Dnitro, Dcar):
        """
        Computes the diffusion of nutrients and Toxins.
        Based on the equations found in reference [1] 
        (Approximation of the spatial-temporal distribution of nutrient 
         concentrations based on Fick's Law).

        Parameters
        ----------
        Doxy : float
            Diffusion constant for Oxygen.
        Dnitro : float
            Diffusion constant for Nitrogen.
        Dcar : float
            Diffusion constant for Carbon.
        Returns
        -------
        diff_O : np.array, dtype = float
            New state of the diffusion Matrix.
        diff_N : np.array, dtype = float
            New state of the diffusion Matrix.
        diff_C : np.array, dtype = float
            New state of the diffusion Matrix.
        diff_TOX : np.array, dtype = float
            New state of the diffusion Matrix.

        """
        # For the chosen diffusion estimation equation, neighborhood is fixed
        All_neighbors, all_pos = self.neighborhood(neighborhood='VonNeumman', radius=1, all_mode=True)
        if self.dimensionality == 2:
            i, j = zip(*all_pos)
        else:
            i, j, k = zip(*all_pos)
        
        n_tox = len(self.TOX_Matrices)
        diff_O = self.O_Matrix.copy()
        diff_N = self.N_Matrix.copy()
        diff_C = self.C_Matrix.copy()
        diff_TOX = [self.TOX_Matrices[i].copy() for i in range(n_tox)]
        
        # To have a random order of computation we shuffle cell's positions
        # Shuffle cells_pos and neighbors in unison
        combined = list(zip(all_pos, All_neighbors))
        random.shuffle(combined)
        
        # Unpack shuffled tuples
        all_pos, All_neighbors = zip(*combined)
        
        # Convert back to list
        all_pos = list(self.cells_pos)
        All_neighbors = list(self.neighbors)
        
        # Vectorized calculation of neighbors' sums
        neighbors_sums_O = np.zeros_like(self.O_Matrix)
        neighbors_sums_N = np.zeros_like(self.N_Matrix)
        neighbors_sums_C = np.zeros_like(self.C_Matrix)
        
        neighbors_sums_TOX = [np.zeros_like(self.TOX_Matrices[0])  for i in range(n_tox)]
        
        for n, cell in enumerate(all_pos):
            neighbors_list = All_neighbors[n] # Neighbors of this cell
            if self.dimensionality == 2: # 2D
                i_near, j_near = zip(*neighbors_list)
                
                neighbors_sums_O[cell] += np.sum(self.O_Matrix[i_near, j_near])
                neighbors_sums_N[cell] += np.sum(self.N_Matrix[i_near, j_near])
                neighbors_sums_C[cell] += np.sum(self.C_Matrix[i_near, j_near])
                for tox in range(n_tox):
                    neighbors_sums_TOX[tox][cell] += np.sum(self.TOX_Matrices[tox][i_near, j_near])
                
            else: # 3D
                i_near, j_near, k_near = zip(*neighbors_list)
                
                neighbors_sums_O[cell] += np.sum(self.O_Matrix[i_near, j_near, k_near])
                neighbors_sums_N[cell] += np.sum(self.N_Matrix[i_near, j_near, k_near])
                neighbors_sums_C[cell] += np.sum(self.C_Matrix[i_near, j_near, k_near])
                for tox in range(n_tox):
                    neighbors_sums_TOX[tox][cell] += np.sum(self.TOX_Matrices[tox][i_near, j_near, k_near])
                
            # Diffusion calculation using pre-calculated sums
            n_neigh = len(neighbors_list)
            
            # Diffusion coeffictient are in m2.s-1 but biological processes are
            # in h-1 so we multiply by 3600
            diff_O[cell] += Doxy * (neighbors_sums_O[cell] - n_neigh * self.O_Matrix[cell]) * 3600
            diff_N[cell] += Dnitro * (neighbors_sums_N[cell] - n_neigh * self.N_Matrix[cell]) * 3600
            diff_C[cell] += Dcar * (neighbors_sums_C[cell] - n_neigh * self.C_Matrix[cell]) * 3600
            for i in range(n_tox):
                diff_TOX[i][cell] += TOX_dictionary['TOX'+str(i)]['DTOX'] * (neighbors_sums_TOX[i][cell] - n_neigh * self.TOX_Matrices[i][cell]) * 3600
            
        return diff_O, diff_N, diff_C, diff_TOX
    
    def calculate_TOX_production(self, strains_dictionary):
        """
        Computes the Antiobiotic production of each cell.
        
        !! The equation used is not from any of the two papers cited though 
        they inspired me. I built it after thinking about a way to simulate 
        Toxin production. !! The idea is that a cell will produce an 
        Toxin if there are cells of other strains around it. Production is 
        proportionate to the number of other cells and inversely to the number 
        of cells of the same strain to represent the cooperative effort of same
        strain cells when facing another strain. (since producing Toxins is 
        costly, if cells are in clusters they can mitigate individual 
        production because thanks to quorum sensing each cell is 'aware' that 
        other cells will also participate in the production effort)
        
        This is very simple but enough for the purpose of this project.
        

        Parameters
        ----------
        strains_dictionary : dictionary
            Dictionary containing all strain specific values for all 
            strain parameters\n 
            (e.g {'Strain0':{'mu_O':0.3,'mu_N':1.5,
            'mu_C':0.3, 'mu_max':4.2e-9,
            'mobility':0.5,
            'O_status':0,
            'O_threshold':2e9,
            'N_threshold':5e-3, 
            'C_threshold':5e-3,
            'alpha_TOX':1.8e-6,
            'TOX_threshold':1e-1, 
            'Cdiv':2.7e-6, 
            'Shared_portion':0.4}, ...}
            mu_nutrient is the maximum specific consumption).

        Returns
        -------
        TOX_prod : np.array, dtype = float
            New state of Toxin production matrix.

        """
        self.TOX_prod = np.zeros_like(self.TOX_Matrices)
        alpha_TOX_distrib =[]
        for idx, list_strain in enumerate(self.Index_by_strain):
            # Mean production rate for this strain
            alpha = strains_dictionary['Strain'+str(idx)]['alpha_TOX']
            
            # Define standard deviation 
            sigma = alpha*100
            
            # Gaussian distribution with mean alpha and standard deviation sigma
            distrib = sigma*np.random.randn(1,len(list_strain)) + alpha
            
            # Making sure that minimum is 0
            alpha_TOX_distrib.append(np.where(distrib<0,0,distrib))
        
        if len(self.cells_pos) != 0:
            # To have a random order of computation we shuffle cell's positions
            # This prevents patterns due to order of computation
            # Shuffle cells_pos and neighbors in unison
            combined = list(zip(self.cells_pos, self.neighbors))
            random.shuffle(combined)
            
            # Unpack shuffled tuples
            self.cells_pos, self.neighbors = zip(*combined)
            
            # Convert back to list
            self.cells_pos = list(self.cells_pos)
            self.neighbors = list(self.neighbors)
        if len(self.TOX_dictionary) == 1 and len(self.strains_dictionary) == 1: # Si une seule souche et une seule Toxine, la production est constitutive et indépendante des voisins
            for n, cell in enumerate(self.cells_pos):
                strain = self.Strains_Matrix[cell] # Strain ID for this cell
                neighbors_list = self.neighbors[n] 
                strain_idx = self.strainID.index(strain)
                tox_id = strains_dictionary['Strain'+str(strain_idx)]['tox_+_id']
                

                distrib_idx = int(self.Index_by_strain[strain_idx].index(cell))
                
                self.TOX_prod[tox_id][cell] = alpha_TOX_distrib[strain_idx].tolist()[0][distrib_idx]
        else:
            for n, cell in enumerate(self.cells_pos):
                strain = self.Strains_Matrix[cell] # Strain ID for this cell
                neighbors_list = self.neighbors[n] 
                strain_idx = self.strainID.index(strain)
                tox_id = strains_dictionary['Strain'+str(strain_idx)]['tox_+_id']
                if self.dimensionality == 2: # 2D
                    i_near, j_near = zip(*neighbors_list)
                    
                    # Vectorized counting using NumPy
                    neighbor_strains = self.Strains_Matrix[i_near, j_near]
                    
                    #Count how many cells have the same ID in neighborhood
                    n_same = np.sum(neighbor_strains == strain)
                    
                    #Count how many grid cells represent medium in neighborhood
                    n_medium = np.sum(neighbor_strains == 0)
                    
                    #Count dead neighbors 
                    n_dead = np.sum(neighbor_strains == -1)
                    
                    #Count how many cells of other strain ID in neighborhood
                    n_other = len(neighbors_list) - n_medium - n_same - n_dead
                    
                else: # 3D
                    i_near, j_near, k_near = zip(*neighbors_list)
                    
                    # Vectorized counting using NumPy
                    neighbor_strains = self.Strains_Matrix[i_near, j_near, k_near]
                    
                    #Count how many cells have the same ID in neighborhood
                    n_same = np.sum(neighbor_strains == strain)
                    
                    #Count how many grid cells represent medium in neighborhood
                    n_medium = np.sum(neighbor_strains == 0)
                    
                    #Count dead neighbors 
                    n_dead = np.sum(neighbor_strains == -1)
                    
                    #Count how many cells of other strain ID in neighborhood
                    n_other = len(neighbors_list) - n_medium - n_same - n_dead
    
                distrib_idx = int(self.Index_by_strain[strain_idx].index(cell))
                
                self.TOX_prod[tox_id][cell] = (n_other**2)*alpha_TOX_distrib[strain_idx].tolist()[0][distrib_idx]/np.exp(n_same+1)

        return self.TOX_prod.copy()
    
    def calculate_uptakes(self, strains_dictionary,epsilon=1e-13, KO2=3.1e-3, KN=10.3e-3, KC=0.2e-3):
        """
        Computes the uptake of nutrients by each cell as well as 
        the growthrate of each cell.
        Based on the equations found in reference [1] 
        (Using the Monod equation).

        Parameters
        ----------
        strains_dictionary : dictionary
            Dictionary containing all strain specific values for all 
            strain parameters\n 
            (e.g {'Strain0':{'mu_O':0.3,'mu_N':1.5,
            'mu_C':0.3, 'mu_max':4.2e-9,
            'mobility':0.5,
            'O_status':0,
            'O_threshold':2e9,
            'N_threshold':5e-3, 
            'C_threshold':5e-3,
            'alpha_TOX':1.8e-6,
            'TOX_threshold':1e-1, 
            'Cdiv':2.7e-6, 
            'Shared_portion':0.4}, ...}
            mu_nutrient is the maximum specific consumption).
        KO2 : float, optional
            Saturation constant. The default is 3.1e-3 based on [1].
        KN : float, optional
            Saturation constant. The default is 10.3e-3 based on [1].
        KC : float, optional
            Saturation constant. The default is 0.2e-3 based on [1].

        Returns
        -------
        cons_O : np.array, dtype = float
            Cell consumption Matrix.
        cons_N : np.array, dtype = float
            Cell consumption Matrix.
        cons_C : np.array, dtype = float
            Cell consumption Matrix.
        growthrate : np.array, dtype = float
            Cell growthrate Matrix.

        """
        cons_O = np.zeros_like(self.O_Matrix)
        cons_N = np.zeros_like(self.N_Matrix)
        cons_C = np.zeros_like(self.C_Matrix)
        growthrate = np.zeros_like(self.C_Matrix)
                
        #Contains distributions of maximum specific consumptions coefficients 
        # for each strain
        mu_O_distrib = [] 
        mu_N_distrib = []
        mu_C_distrib = []
        
        for idx, list_strain in enumerate(self.Index_by_strain):
            tox_id = strains_dictionary['Strain'+str(idx)]['tox_+_id']
            if len(list_strain) >0: 
                mu_O = strains_dictionary['Strain'+str(idx)]['mu_O']
                mu_N = strains_dictionary['Strain'+str(idx)]['mu_N']
                mu_C = strains_dictionary['Strain'+str(idx)]['mu_C']
                
                std_O, std_N, std_C = mu_O*10, mu_N*20, mu_C*20
                
                distrib_O = std_O*np.random.randn(1,len(list_strain)) + mu_O
                distrib_C = std_N*np.random.randn(1,len(list_strain)) + mu_N
                distrib_N = std_C*np.random.randn(1,len(list_strain)) + mu_C
                mu_O_distrib.append(list(np.where(distrib_O<0,distrib_O/10,distrib_O)))
                mu_N_distrib.append(list(np.where(distrib_N<0,distrib_N/20,distrib_N)))
                mu_C_distrib.append(list(np.where(distrib_C<0,distrib_C/20,distrib_C)))
                        
                #Bacteria uptake and growthrate
                if self.dimensionality == 2: # 2D
                    i, j = zip(*list_strain)
                    monod_O = self.O_Matrix[i,j]/(KO2+self.O_Matrix[i,j])
                    monod_N = self.N_Matrix[i,j]/(KN+self.N_Matrix[i,j])
                    monod_C = self.C_Matrix[i,j]/(KC+self.C_Matrix[i,j])
                    
                    cons_O[i,j] = mu_O_distrib[-1]*monod_O
                    cons_N[i,j] = mu_N_distrib[-1]*monod_N #*monod_O
                    cons_C[i,j] = mu_C_distrib[-1]*monod_C #*monod_O
                    growthrate[i,j] = strains_dictionary['Strain'+str(idx)]['mu_max']*monod_O*monod_N*monod_C
                    
                    # Producing Toxins is biologically expensive so it slightly reduces growthrate
                    growthrate[i,j] *= np.exp(-strains_dictionary['Strain'+str(idx)]['tox_prod_penalty'] * self.TOX_prod[tox_id][i,j])
                    # print(growthrate[i,j])
                else: # 3D
                    i, j, k = zip(*list_strain)
                    monod_O = self.O_Matrix[i,j,k]/(KO2+self.O_Matrix[i,j,k])
                    monod_N = self.N_Matrix[i,j,k]/(KN+self.N_Matrix[i,j,k])
                    monod_C = self.C_Matrix[i,j,k]/(KC+self.C_Matrix[i,j,k])
                    
                    cons_O[i,j,k] = mu_O_distrib[-1]*monod_O
                    cons_N[i,j,k] = mu_N_distrib[-1]*monod_N #*monod_O
                    cons_C[i,j,k] = mu_C_distrib[-1]*monod_C #*monod_O                    
                    growthrate[i,j,k] = strains_dictionary['Strain'+str(idx)]['mu_max']*monod_O*monod_N*monod_C
                    
                    # Producing Toxins is biologically expensive so it reduces growthrate
                    growthrate[i,j,k] *= np.exp(-strains_dictionary['Strain'+str(idx)]['tox_prod_penalty'] * self.TOX_prod[tox_id][i,j,k])

        return cons_O, cons_N, cons_C, growthrate
    
    def update_abiotic_environment(self, dico_diff, dico_cons, TOX_prod):
        """
        Updates nutrients and Toxins matrices.

        Parameters
        ----------
        dico_diff : dictionary
            dictionary of diffusion states.
        dico_cons : dictionary
            dictionary of consumption states.
        TOX_prod : np.array, dtype = float
            Production matrix returned by self.calculate_TOX_production.

        Returns
        -------
        None.

        """
        diff_O, diff_N, diff_C, diff_TOX = dico_diff.values()
        cons_O, cons_N, cons_C, growthrate = dico_cons.values()
        
        self.O_Matrix = diff_O - cons_O * diff_O #multiplication par diff_O car cons_O est un taux de consommation
        self.O_Matrix = np.where(self.O_Matrix<0, 0, self.O_Matrix)
        self.N_Matrix = diff_N - cons_N * diff_N
        self.N_Matrix = np.where(self.N_Matrix<0, 0, self.N_Matrix)
        self.C_Matrix = diff_C - cons_C * diff_C
        self.C_Matrix = np.where(self.C_Matrix<0, 0, self.C_Matrix)
        self.TOX_Matrices = [diff_TOX[i] + TOX_prod[i] for i in range(len(self.TOX_dictionary))]
        self.grid = self.grid + growthrate*(cons_O * diff_O + cons_N * diff_N + cons_C * diff_C + self.grid) #self.grid*(1 + growthrate)
        # print(self.grid[self.grid!=0][0])
    # =========================================================================
    #                   REPRODUCTION, MOVEMENT and DEATHS 
    # =========================================================================
    def reproduction(self, strains_dictionary, look_TOX=False, move_by_nutri=False):
        """
        Compute cell division and cell movement. The division mechanism is 
        based on [1].

        Parameters
        ----------
        strains_dictionary : dictionary
            Dictionary containing all strain specific values for all 
            strain parameters\n 
            (e.g {'Strain0':{'mu_O':0.3,'mu_N':1.5,
            'mu_C':0.3, 'mu_max':4.2e-9,
            'mobility':0.5,
            'O_status':0,
            'O_threshold':2e9,
            'N_threshold':5e-3, 
            'C_threshold':5e-3,
            'alpha_TOX':1.8e-6,
            'TOX_threshold':1e-1, 
            'Cdiv':2.7e-6, 
            'Shared_portion':0.4}, ...}
            mu_nutrient is the maximum specific consumption).
        look_TOX : boolean, optional
            Whether to put an Toxin constraint upon division. 
            The default is False.
        move_by_nutri : boolean, optional
            Whether to also take into account nutrient quantities when moving.
            The default is False.

        Returns
        -------
        new_cells : List of tuples
            List of new born cells positions.
        new_cells_by_strain : List of lists
            List of new born cells positions grouped by strain.
        moving : int
            Number of moving cells.

        """
        new_cells = []
        new_cells_by_strain = [[] for i in self.strainID]
        positions = deepcopy(self.cells_pos)
        moving = 0
        INDEX_HISTORY = deepcopy(self.INDEX_HISTORY)
        Index_by_strain = deepcopy(self.Index_by_strain)
        
        if len(self.cells_pos) != 0:
            # To have a random order of computation we shuffle cell's positions
            # This prevents patterns due to order of computation
            # Shuffle cells_pos and neighbors in unison
            combined = list(zip(self.cells_pos, self.neighbors))
            random.shuffle(combined)
            
            # Unpack shuffled tuples
            self.cells_pos, self.neighbors = zip(*combined)
            
            # Convert back to list
            self.cells_pos = list(self.cells_pos)
            self.neighbors = list(self.neighbors)
        
        # Coefficient that describes percentage of mobile cells for this strain
        mobility_coeff = [strains_dictionary['Strain'+str(i)]['mobility'] for i in range(len(self.strainID))]
        for n, cell in enumerate(self.cells_pos):
            strain = int(self.Strains_Matrix[cell]) # Strain ID for this cell
            strain_idx = self.strainID.index(strain)
            neighbors_list = self.neighbors[n] 
            tox_id = strains_dictionary['Strain'+str(strain_idx)]['tox_-_id'] # Toxin that affects the strain
            if self.dimensionality == 2:
                i_near, j_near = zip(*neighbors_list)
            else:
                i_near, j_near, k_near = zip(*neighbors_list)
                
            O_status = strains_dictionary['Strain'+str(strain_idx)]['O_status']
            empty_space_near = [neighbors_list[i] for i in range(len(neighbors_list)) if self.Strains_Matrix[neighbors_list[i]]==0]

            if look_TOX:
                TOX_threshold = strains_dictionary['Strain'+str(strain_idx)]['TOX_threshold']
            
            # Can the cell divide ?
            if self.grid[cell] >= strains_dictionary['Strain'+str(strain_idx)]['Cdiv']: 
                
                if look_TOX:
                    potential_daughter_cells = [neighbors_list[i] for i in range(len(neighbors_list)) if self.Strains_Matrix[neighbors_list[i]]==0 and self.TOX_Matrices[tox_id][neighbors_list[i]] < TOX_threshold] 
                else:
                    potential_daughter_cells = [neighbors_list[i] for i in range(len(neighbors_list)) if self.Strains_Matrix[neighbors_list[i]]==0] 
                if potential_daughter_cells != []:
                    daughter_cell = random.choice(potential_daughter_cells)
                    
                    # Division of the cell in 2
                    new_biomass = self.grid[cell].copy()/2
                    self.grid[daughter_cell], self.grid[cell] = new_biomass, new_biomass
                    
                    # Update the Strains Matrix
                    self.Strains_Matrix[daughter_cell] = strain
                    
                    # Update lists of indices
                    Index_by_strain[strain_idx].append(daughter_cell)
                    INDEX_HISTORY[strain_idx].append(daughter_cell)
                    
                    new_cells.append(daughter_cell)
                    new_cells_by_strain[strain_idx].append(daughter_cell)
            # If it can't divide, can it move ?
            elif empty_space_near and random.random() < mobility_coeff[strain_idx]:
                if self.dimensionality == 2: # 2D
                    i_empty, j_empty = zip(*empty_space_near)
                else: # 3D
                    i_empty, j_empty, k_empty = zip(*empty_space_near)
                    
                # (-) to keep argmax synthax
                tox_thresh = strains_dictionary['Strain'+str(strain_idx)]['TOX_threshold']
                coeff_nutri = -self.TOX_Matrices[tox_id].copy()/tox_thresh
                
                if move_by_nutri:
                    c_thresh = strains_dictionary['Strain'+str(strain_idx)]['C_threshold']
                    n_thresh = strains_dictionary['Strain'+str(strain_idx)]['N_threshold']
                    o_thresh = strains_dictionary['Strain'+str(strain_idx)]['O_threshold']
                    if self.dimensionality == 2: # 2D
                        coeff_nutri[i_empty, j_empty] += self.C_Matrix[i_empty, j_empty]/c_thresh + self.N_Matrix[i_empty, j_empty]/n_thresh
                        if O_status ==1:
                            coeff_nutri[i_empty, j_empty] += self.O_Matrix[i_empty, j_empty]/o_thresh
                        else:
                            coeff_nutri[i_empty, j_empty] -= self.O_Matrix[i_empty, j_empty]/o_thresh
                    else: # 3D
                        coeff_nutri[i_empty, j_empty, k_empty] += self.C_Matrix[i_empty, j_empty, k_empty]/c_thresh + self.N_Matrix[i_empty, j_empty, k_empty]/n_thresh
                        if O_status ==1:
                            coeff_nutri[i_empty, j_empty, k_empty] += self.O_Matrix[i_empty, j_empty, k_empty]/o_thresh
                        else:
                            coeff_nutri[i_empty, j_empty, k_empty] -= self.O_Matrix[i_empty, j_empty, k_empty]/o_thresh
                
                if self.dimensionality == 2: # 2D
                    M = list(coeff_nutri[i_empty, j_empty].reshape(-1,1))
                else: # 3D
                    M = list(coeff_nutri[i_empty, j_empty, k_empty].reshape(-1,1))
                if np.sum((M==0)) != len(M):
                    # Choose best near cell to move to
                    M_pos = np.argmax(M)
                    best_pos = empty_space_near[M_pos]
                else: # If only zeros, choose randomly
                    M_pos = M.index(random.sample(M))
                    best_pos = empty_space_near[M_pos]

                # Update biomass matrix
                self.grid[best_pos], self.grid[cell] = self.grid[cell], 0
                
                # update strains matrix
                self.Strains_Matrix[best_pos], self.Strains_Matrix[cell] = strain, 0
                
                # Update lists of position
                Index_by_strain[strain_idx].remove(cell)
                Index_by_strain[strain_idx].append(best_pos)
                
                INDEX_HISTORY[strain_idx].remove(cell)
                INDEX_HISTORY[strain_idx].append(best_pos)
                
                positions.remove(cell)
                positions.append(best_pos)
                moving+=1
            # If it can't move either, share biomass with neighborhood
            else:
                share_coeff = strains_dictionary['Strain'+str(strain_idx)]['Shared_portion']
                total_biomass_to_share = share_coeff*self.grid[cell]
                nshare = len(neighbors_list)
                
                if self.dimensionality == 2: # 2D
                    self.grid[i_near, j_near] = total_biomass_to_share/nshare
                else: # 3D
                    self.grid[i_near, j_near, k_near] = total_biomass_to_share/nshare
                    
                # Update initial cell to respect mass conservation
                self.grid[cell] = self.grid[cell] - total_biomass_to_share 
                
        print(f'\n{len(new_cells)} new cells born.')
        print(f'\n{moving} cells moved.')
        self.cells_pos = positions + new_cells
        self.INDEX_HISTORY = INDEX_HISTORY
        self.Index_by_strain = Index_by_strain
        return new_cells, new_cells_by_strain, moving
        
    def update_deaths(self, strains_dictionary, i, complexity=1):
        """
        Compute deaths.

        Parameters
        ----------
        strains_dictionary : dictionary
            Dictionary containing all strain specific values for all 
            strain parameters\n 
            (e.g {'Strain0':{'mu_O':0.3,'mu_N':1.5,
            'mu_C':0.3, 'mu_max':4.2e-9,
            'mobility':0.5,
            'O_status':0,
            'O_threshold':2e9,
            'N_threshold':5e-3, 
            'C_threshold':5e-3,
            'alpha_TOX':1.8e-6,
            'TOX_threshold':1e-1, 
            'Cdiv':2.7e-6, 
            'Shared_portion':0.4}, ...}
            mu_nutrient is the maximum specific consumption).
        i : int
            current iteration.
        complexity : int, optional
            Death rules complexity, 1 = death by TOX only, 2 = death by TOX 
            and Oxygen, 3 = death by all nutrients and TOX. The default is 1.

        Returns
        -------
        dead_cells : list of tuples
            List of  dead cells positions.
        D_byO : int
            Number of deaths by poor Oxygen conditions.
        D_byN : int
            Number of deaths by poor Nitrogen conditions.
        D_byC : int
            Number of deaths by poor Carbon conditions.
        D_byTOX : List
            Number of deaths by poor Toxins conditions.

        """
        dead_cells = []
        live_cells = deepcopy(self.cells_pos)
        if self.dimensionality == 2:
            i, j = zip(*live_cells)
        else:
            i, j, k = zip(*live_cells)
        idx_by_strain = deepcopy(self.Index_by_strain)
        
        D_byO = 0
        D_byN = 0
        D_byC = 0
        D_byTOX = [0 for i in range(len(self.TOX_dictionary))]
        
        if len(self.cells_pos) != 0:
            # To have a random order of computation we shuffle cell's positions
            # This prevents patterns due to order of computation
            # Shuffle cells_pos and neighbors in unison
            combined = list(zip(self.cells_pos, self.neighbors))
            random.shuffle(combined)
            
            # Unpack shuffled tuples
            self.cells_pos, self.neighbors = zip(*combined)
            
            # Convert back to list
            self.cells_pos = list(self.cells_pos)
            self.neighbors = list(self.neighbors)
        
        for n, cell in enumerate(live_cells):
            c = 0 # Used to know if we already have found a death for this cell
            strain = self.Strains_Matrix[cell] # Strain ID for this cell
            strain_idx = self.strainID.index(strain)
            neighbors_list = self.neighbors[n] 
            coeff_neighbors = 1/(len(neighbors_list)+1) # To compute the mean 
            tox_id = strains_dictionary['Strain'+str(strain_idx)]['tox_-_id'] # Toxin that affects the strain

            if self.dimensionality == 2: # 2D
                i_near, j_near = zip(*neighbors_list)
            else: # 3D
                i_near, j_near, k_near = zip(*neighbors_list)
            
            # Retrieve parameters from dictionary
            strain_data = strains_dictionary['Strain' + str(strain_idx)]
            O_status, O_threshold = strain_data['O_status'], strain_data['O_threshold']
            N_threshold, C_threshold, TOX_threshold = strain_data['N_threshold'], strain_data['C_threshold'], strain_data['TOX_threshold']
            # Mean over neighbors 
            if self.dimensionality == 2:
                i_near, j_near = zip(*neighbors_list)
                s_O = coeff_neighbors * (self.O_Matrix[i_near, j_near].sum() + self.O_Matrix[cell])
                s_N = coeff_neighbors * (self.N_Matrix[i_near, j_near].sum() + self.N_Matrix[cell])
                s_C = coeff_neighbors * (self.C_Matrix[i_near, j_near].sum() + self.C_Matrix[cell])
                s_TOX = coeff_neighbors * (self.TOX_Matrices[tox_id][i_near, j_near].sum() + self.TOX_Matrices[tox_id][cell])
            else:
                i_near, j_near, k_near = zip(*neighbors_list)
                s_O = coeff_neighbors * (self.O_Matrix[i_near, j_near, k_near].sum() + self.O_Matrix[cell])
                s_N = coeff_neighbors * (self.N_Matrix[i_near, j_near, k_near].sum() + self.N_Matrix[cell])
                s_C = coeff_neighbors * (self.C_Matrix[i_near, j_near, k_near].sum() + self.C_Matrix[cell])
                s_TOX = coeff_neighbors * (self.TOX_Matrices[tox_id][i_near, j_near, k_near].sum() + self.TOX_Matrices[tox_id][cell])
    
            # Death rules based on complexity
            if s_TOX > TOX_threshold:
                self.Strains_Matrix[cell] = self.dead
                self.Index_by_strain[strain_idx].remove(cell)
                dead_cells.append(cell)
                live_cells.remove(cell)
                D_byTOX[tox_id] += 1
                c = 1
            if complexity >= 2 and c == 0:
                if (O_status == 1 and s_O < O_threshold) or (O_status == 0 and s_O > O_threshold):
                    self.Strains_Matrix[cell] = self.dead
                    self.Index_by_strain[strain_idx].remove(cell)
                    dead_cells.append(cell)
                    live_cells.remove(cell)
                    D_byO += 1
                    c = 1
            if complexity == 3 and c == 0:
                if s_N < N_threshold:
                    self.Strains_Matrix[cell] = self.dead
                    self.Index_by_strain[strain_idx].remove(cell)
                    dead_cells.append(cell)
                    live_cells.remove(cell)
                    D_byN += 1
                    c = 1
                if c == 0 and s_C < C_threshold:
                    self.Strains_Matrix[cell] = self.dead
                    self.Index_by_strain[strain_idx].remove(cell)
                    dead_cells.append(cell)
                    live_cells.remove(cell)
                    D_byC += 1
                    c = 1                
        
        self.dead_pos.extend(dead_cells)
        self.Index_by_strain = idx_by_strain
        
        self.cells_pos = live_cells
        if complexity == 3:
            print(f'\n {np.sum(D_byTOX)} death(s) due to poor TOX conditions.')
            print(f'\n {D_byO} death(s) due to poor Oxygen conditions.')
            print(f'\n {D_byN} death(s) due to poor Nitrogen conditions.')
            print(f'\n {D_byC} death(s) due to poor Carbon conditions.')
        if complexity == 2:
            print(f'\n {np.sum(D_byTOX)} death(s) due to poor TOX conditions.')
            print(f'\n {D_byO} death(s) due to poor Oxygen conditions.')
        if complexity == 1:
            print(f'\n {np.sum(D_byTOX)} death(s) due to poor TOX conditions.')
        return dead_cells, D_byO, D_byN, D_byC, D_byTOX
    
    def share_death(self, neighborhood='VonNeumman', radius=1, min_Biom=1e-17, renew_deads=False):
        """
        Compute dead cells decay.

        Parameters
        ----------
        neighborhood : str, optional
            Type of neighborhood. The default is 'VonNeumman'.
        radius : int, optional
            Neighborhood radius. The default is 1.
        min_Biom : float, optional
            Minimum biomass before complete decay. The default is 1e-17.
        renew_deads : boolean, optional
            Whether to replace dead cells by empty space when 
            biomass < min_Biom. The default is False.

        Returns
        -------
        None.

        """
        if renew_deads:
            count = 0
        # Neighbors of dead cells
        dead_neighbors = self.neighborhood(neighborhood='VonNeumman', radius=1, dead_mode=True)
        
        if len(self.dead_pos) != 0:
            # To have a random order of computation we shuffle cell's positions
            # Shuffle cells_pos and neighbors in unison
            combined = list(zip(self.dead_pos, dead_neighbors))
            random.shuffle(combined)
            
            # Unpack shuffled tuples
            self.dead_pos, dead_neighbors = zip(*combined)
            
            # Convert back to list
            self.dead_pos = list(self.dead_pos)
            dead_neighbors = list(dead_neighbors)
        
        for n, dead_cell in enumerate(self.dead_pos):
            neighbors_list = dead_neighbors[n]
            
            biomass_to_share = self.decaying_matter_coeff*self.grid[dead_cell]
            if biomass_to_share > min_Biom:
                nshare = len(neighbors_list)
                
                if self.grid[dead_cell] - biomass_to_share > min_Biom:
                    if self.dimensionality == 2: # 2D
                        i_near, j_near = zip(*neighbors_list)
                        # Distribute biomass
                        self.grid[i_near, j_near] = biomass_to_share/nshare
                    else: # 3D
                        i_near, j_near, k_near = zip(*neighbors_list)
                        # Distribute biomass
                        self.grid[i_near, j_near, k_near] = biomass_to_share/nshare
                    
                    # Update initial cell to respect mass conservation
                    self.grid[dead_cell] = self.grid[dead_cell] - biomass_to_share
            if renew_deads:
                # If very low biomass
                if self.grid[dead_cell] <= min_Biom*1e-3: 
                    # Converts to empty space
                    self.dead_pos.remove(dead_cell)
                    find_strain = [dead_cell in self.INDEX_HISTORY[i] for i in range(len(self.strainID))]
                    if np.sum(find_strain)!=0:
                        strain_idx = np.argmax(find_strain)
                        self.INDEX_HISTORY[strain_idx].remove(dead_cell)
                        self.Strains_Matrix[dead_cell] = 0
                        count+=1
        if renew_deads:
            print(f'{count} dead cells renewed.')
            
    # =========================================================================
    #                          PLOTS and ANIMATIONS
    # =========================================================================
    def make_legend_patches(self, mode='strain', val_medium=None, return_bounds=False):
        """
        Make legend patches for future plots. To do so it creates a list 
        of colors and a list of labels before creating a list of boundaries 
        and converting the colors list to a matplotlib colormap. 
        It finally uses those to build the legend patch.

        Parameters
        ----------
        mode : string, optional
            'strain' mode or 'biomass' mode (changes labels). The default is 'strain'.
        val_medium : int, optional
            value that represents the medium. The default is None.
        return_bounds : boolean, optional
            . The default is False.

        Returns
        -------
        patches, cmap, norm, colors and bounds if return_bounds

        """
        if mode=='strain':
            values = [self.dead, self.medium]+self.strainID  # Values
            colors = generate_n_colors(len(values), mode=mode)  # List of colors
            #Labels
            labels = ['Dead', 'Medium']+['Strain'+str(i) for i in range(len(self.strainID))]
        elif mode=='biomass':
            values = [val_medium] + [i*2+3 for i in range(4)] 
            colors = generate_n_colors(len(values), mode=mode)  
            labels = ['Background', 'Very Low', 'Low', 'High', 'Very High']
        
        cmap = ListedColormap(colors)
        bounds = [values[i-1]-0.5 for i in range(1,len(values)+1)] + [values[-1]+0.5]
        norm = BoundaryNorm(bounds, cmap.N)
        
        patches = [plt.plot([],[], marker="s", ms=10, ls="", mec=None, color=colors[i], 
                    label="{:s}".format(labels[i]))[0]  for i in range(len(values))]
        if return_bounds:
            return patches, cmap, norm, colors, bounds
        return patches, cmap, norm, colors
    
    def plot_Matrices(self, i, show=False, save=False, min_Biom=1e-17, file_type='png', save_TOX=False):
        """
        Plots every Matrix of the simulation.

        Parameters
        ----------
        i : int
            current iteration.
        show : boolean, optional
            Whether to show the plots. The default is False.
        save : boolean, optional
            Whether to save the plots. The default is False.
        min_Biom : float, optional
            Minimum biomass value. The default is 1e-17.

        Returns
        -------
        None.

        """
        
        plt.style.use('default')
        plt.grid(False)
        plt.figure(figsize=(17, 8))
        
        # Strains Matrix
        plt.subplot(3, 2, 1)
        patches, cmap, norm, colors = self.make_legend_patches()
        plt.imshow(self.Strains_Matrix, cmap=cmap, norm=norm)
        plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.25 )
        plt.title(f'Space after {i} iterations')
        
        # Biomass Matrix
        plt.subplot(3, 2, 2)
        patches, cmap, norm, colors, bounds = self.make_legend_patches(mode='biomass', 
                                                   val_medium=0, return_bounds=True)
        # Due to the very large disparities of values in biomass Matrix,
        # we need to create a new matrix to be plot
        Mcopy = self.grid.copy()
        maxi = np.max(Mcopy)
        scale = np.linspace(np.log10(min_Biom), np.log10(maxi), num=4)
        for k in range(self.Strains_Matrix.shape[0]):
            for j in range(self.Strains_Matrix.shape[1]):
                if self.Strains_Matrix[k,j] != 0:
                    if np.isfinite(np.log10(Mcopy[k,j])):
                        if (np.log10(Mcopy[k,j])>=scale[0]) & (np.log10(Mcopy[k,j])< scale[1]):
                            Mcopy[k,j] = bounds[2]-0.5
                        elif (np.log10(Mcopy[k,j])>=scale[1]) & (np.log10(Mcopy[k,j])< scale[2]):
                            Mcopy[k,j] = bounds[3]-0.5
                        elif (np.log10(Mcopy[k,j])>=scale[2]) & (np.log10(Mcopy[k,j])< scale[3]):
                            Mcopy[k,j] = bounds[4]-0.5
                        else:
                            Mcopy[k,j] = bounds[2]-0.5
                    else:
                        Mcopy[k,j] = bounds[2]-0.5
        
        plt.imshow(Mcopy, norm=norm, cmap=cmap)
        plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.25 )
        plt.title(f'Biomass after {i} iterations')
        
        # Oxygen Matrix
        plt.subplot(3, 2, 3)
        # if 'scale_O' in self.__dict__.keys() and self.scale_O.startswith('log'):
        norm = LogNorm(vmax=np.max(self.O_Matrix))
        # else:
        #     norm = None
        plt.imshow(self.O_Matrix, norm=norm, cmap='viridis')
        plt.colorbar()
        plt.title(f'Oxygen after {i} iterations')
        
        # Nitrogen Matrix
        plt.subplot(3, 2, 4)
        # if 'scale_N' in self.__dict__.keys() and self.scale_N.startswith('log'):
        norm = LogNorm(vmax=np.max(self.N_Matrix))
        # else:
        #     norm = None
        plt.imshow(self.N_Matrix, norm=norm, cmap='viridis')
        plt.colorbar()
        plt.title(f'Nitrogen after {i} iterations')
        
        # Carbon Matrix
        plt.subplot(3, 2, 5)
        # if 'scale_C' in self.__dict__.keys() and self.scale_C.startswith('log'):
        norm = LogNorm(vmax=np.max(self.C_Matrix))
        # else:
        #     norm = None
        plt.imshow(self.C_Matrix, norm=norm, cmap='viridis')
        plt.colorbar()
        plt.title(f'Carbon after {i} iterations')
        
        if save_TOX:
            # Toxin Matrix
        
            if save:
                plt.savefig(fname=f'./Animation/simulation_{i}.{file_type}', format=file_type)
            if show:
                plt.show()
                
            # Toxin Matrix
            plt.figure(figsize=(17, 8))
            for tox in range(len(TOX_dictionary)):
                plt.subplot(1, len(TOX_dictionary), tox+1)
                # if 'scale_TOX' in self.__dict__.keys() and self.scale_TOX.startswith('log') and np.sum(self.TOX_Matrices[tox]) > 0:
                if np.max(self.TOX_Matrices[tox]) != 0:
                    norm = LogNorm(vmax=np.max(self.TOX_Matrices[tox]))
                else:
                    norm = None
                plt.imshow(self.TOX_Matrices[tox], norm=norm, cmap='viridis')
                plt.colorbar()
                plt.title(f'Toxin{tox} after {i} iterations')
        if save:
            plt.savefig(fname=f'./Animation/simulation_Tox_{i}.{file_type}', format=file_type)
        if show:
            plt.show()
    def plot_OneMatrix(self, i, M, matrix_name, save=False, show=False, min_Biom=1e-17, file_type='png'):
        """
        Plots either Strains or Biomass Matrix.

        Parameters
        ----------
        i : int
            current iteration.
        M : np.array, dtype = float
            self.Strains_Matrix or self.grid (Biomass Matrix).
        matrix_name : string
            'Strains'. or 'Biomass'.
        save : boolean, optional
            Whether to save the plot. The default is False.
        show : boolean, optional
            Whether to save the plot. The default is False.
        min_Biom : float, optional
           Minimum biomass value. The default is 1e-17.

        Returns
        -------
        None.

        """
        plt.ioff()
        plt.style.use('default')
        plt.figure(figsize=(17, 8))
        plt.grid(False)
        
        if matrix_name == 'Strains':
            patches, cmap, norm, colors = self.make_legend_patches()
            plt.imshow(M, cmap=cmap, norm=norm)
            plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.25 )
            plt.title(f'{matrix_name} after {i} iterations')
            
        elif matrix_name== 'Biomass': 
            patches, cmap, norm, colors, bounds = self.make_legend_patches(mode='biomass', 
                                                                   val_medium=0, return_bounds=True)
            # Due to the very large disparities of values in biomass Matrix,
            # we need to create a new matrix to be plot
            Mcopy = M.copy()
            maxi = np.max(Mcopy)
            scale = np.linspace(np.log10(min_Biom), np.log10(maxi), num=4)
            for k in range(self.Strains_Matrix.shape[0]):
                for j in range(self.Strains_Matrix.shape[1]):
                    if self.Strains_Matrix[k,j] != 0:
                        if np.isfinite(np.log10(Mcopy[k,j])):
                            if (np.log10(Mcopy[k,j])>=scale[0]) & (np.log10(Mcopy[k,j])< scale[1]):
                                Mcopy[k,j] = bounds[2]-0.5
                            elif (np.log10(Mcopy[k,j])>=scale[1]) & (np.log10(Mcopy[k,j])< scale[2]):
                                Mcopy[k,j] = bounds[3]-0.5
                            elif (np.log10(Mcopy[k,j])>=scale[2]) & (np.log10(Mcopy[k,j])< scale[3]):
                                Mcopy[k,j] = bounds[4]-0.5
                            else:
                                Mcopy[k,j] = bounds[2]-0.5
                        else:
                            Mcopy[k,j] = bounds[2]-0.5
            plt.imshow(Mcopy, cmap=cmap, norm=norm)
            plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.25 )
            plt.title(f'{matrix_name} after {i} iterations')
        elif matrix_name.startswith('TOX'):
            # Toxin Matrix
            for tox in range(len(TOX_dictionary)):
                plt.subplot(1, len(TOX_dictionary), tox+1)
                # if 'scale_TOX' in self.__dict__.keys() and self.scale_TOX.startswith('log') and np.sum(self.TOX_Matrices[tox]) > 0:
                maxi = np.max(self.TOX_Matrices[tox])
                mini = np.min(self.TOX_Matrices[tox])
                if maxi != 0 :
                    norm = LogNorm(vmax=maxi)
                else:
                    norm = None
                plt.imshow(self.TOX_Matrices[tox], norm=norm, cmap='viridis')
                plt.colorbar()
                plt.title(f'Toxin{tox} after {i} iterations')
        else:
            plt.imshow(M, cmap='viridis')
            plt.colorbar()
            plt.title(f'{matrix_name} after {i} iterations')
        if save:
            plt.savefig(fname=f'./Animation/{matrix_name}_images/{matrix_name}_{i}.{file_type}', format=file_type)
            self.image_files[matrix_name].append(f'./Animation/{matrix_name}_images/{matrix_name}_{i}.{file_type}')
        if show:
            plt.show()
        plt.close()
    
    def color_by_condition(self, array_list):
        # Use dtype=object for an array of color names
        COLORS = np.empty(array_list[0].shape, dtype=object)
        patches, cmap, norm, colors = self.make_legend_patches()
        colors = np.array(colors, dtype=object)
        Lref = [self.dead, self.medium] + self.strainID
        L_index_map = {key: index for index, key in enumerate(Lref)}
        
        for arr in array_list:
            L = list(np.sort(np.unique(arr)))
            if len(L) != len(Lref):
                L_index_map = {key: Lref.index(key) for i, key in enumerate(L)}
            # Create an index array from arr using L_index_map
            index_array = np.vectorize(L_index_map.get)(arr)
            # Use advanced indexing to set colors
            COLORS = colors[index_array]
        
        return COLORS

    def make3D_plots(self, space_matrix, file_type='png'):
        h, l, d = space_matrix[0].shape
        # Initialize an empty 3D grid
        grid = cpl3d.init_simple3d(h, l, d, val=0)
        
        grid = space_matrix
        FILES = []
        for i,arr in enumerate(grid):
            L = [arr]
            cube_colors = self.color_by_condition(L)
            cpl3d.plot3d(L, shade=True, title=f'Strains Matrix at iteration {i}',face_color = cube_colors, show=False, show_grid= True, show_axis=True)
            plt.savefig(f'./Animation/Strains_images/Strains_{i}.{file_type}', format=file_type)
            FILES.append(f'./Animation/Strains_images/Strains_{i}.{file_type}')
            plt.close()
        return FILES

    def make3D_gif(self, files, output, delay):
        images = []
        for filename in files:
            images.append(Image.open(filename))
        images[0].save(output, save_all=True, append_images=images[1:], duration=delay*1000, loop=0)

    
    def make_GIF(self, matrix_name, output, delay):
        images = []
        for filename in self.image_files[matrix_name]:
            if filename.lower().endswith('.pdf'):
                # Open the PDF file
                pdf = fitz.open(filename)
                for page_num in range(len(pdf)):
                    # Get the page
                    page = pdf[page_num]
                    # Render the page to an image
                    pix = page.get_pixmap()
                    # Store the image as a PNG in memory
                    img_data = pix.tobytes("png")
                    # Open the image with PIL
                    img = Image.open(io.BytesIO(img_data))
                    images.append(img)
                pdf.close()
            else:
                images.append(Image.open(filename))
        images[0].save(output, save_all=True, append_images=images[1:], duration=delay*1000, loop=0)

    def make_DynamicCurves(self, new_by_strainEVOL, pop_size_by_strainEVOL, 
                           death_complexity,
                           death_by_strainEVOL,
                           live_cellsEVOL,
                           dead_cellsEVOL,
                           new_cellsEVOL,
                           moving_cellsEVOL,
                           death_byO, death_byC, death_byN, death_byTOX,
                           growthrate_by_strainEVOL, 
                           TOX_by_strainEVOL, 
                           each_TOX_tot_EVOL,
                           each_TOX_prod_EVOL,
                           file_type='png', show=False):
        
        FILES = []
        patches, cmap, norm, colors = self.make_legend_patches()
        if self.dimensionality == 2 :
            # Strain Matrix without dead cells
            plt.figure(figsize=(17, 8))
            M = self.Strains_Matrix.copy()
            for i,strain in enumerate(self.strainID):
                if self.INDEX_HISTORY[i]:
                    i_strain, j_strain = zip(*self.INDEX_HISTORY[i])
                    M[i_strain, j_strain] = strain
            plt.imshow(M, cmap=cmap, norm=norm)
            plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.25 )
            plt.title('Strains positions (dead or alive) after simulation.', fontsize=16)
            plt.savefig(f'StrainMatrix_deadORalive.{file_type}', format=file_type)
            FILES.append(f'StrainMatrix_deadORalive.{file_type}')
            if show:
                plt.show()
            
        n_iter = len(pop_size_by_strainEVOL)
        plt.style.use('ggplot')  # Use 'ggplot' style for nicer plots
        
        # %Population by strain
        plt.figure(figsize=(17, 8))
        for i,strain in enumerate(self.strainID):
            strain_evol = [pop_size_by_strainEVOL[k][i] for k in range(n_iter)]
            plt.plot(strain_evol, label='Strain'+str(i), linewidth=2, color=colors[i+2])
        plt.title('Structure of total Population over time.', fontsize=16)
        plt.legend()
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('% of Total population', fontsize=14)
        plt.savefig(f'Percent_Population.{file_type}', format=file_type)
        FILES.append(f'Percent_Population.{file_type}')
        if show:
            plt.show()
        
        if death_complexity != None:
            # Deaths by nutrients
            plt.figure(figsize=(17, 8))
            plt.plot(death_byO, label='Oxygen related deaths', linewidth=2)
            plt.plot(death_byC, label='Carbon related deaths', linewidth=2)
            plt.plot(death_byN, label='Nitrogen related deaths', linewidth=2)
            for tox in range(len(self.TOX_Matrices)):
                tox_list = [death_byTOX[k][tox] for k in range(n_iter)]
                plt.plot(tox_list, label=f'Toxins{tox} related deaths', linewidth=2)
            plt.title('Death causes for each iteration.', fontsize=16)
            plt.legend()
            plt.xlabel('Iterations', fontsize=14)
            plt.ylabel('Number of deaths', fontsize=14)
            plt.savefig(f'Deaths_by_nutrients.{file_type}', format=file_type)
            FILES.append(f'Deaths_by_nutrients.{file_type}')
            if show:
                plt.show()
            
            # Death over time
            plt.figure(figsize=(17, 8))
            plt.plot(dead_cellsEVOL, label='Total dead cells', linewidth=2)
            plt.title('Dead cells over time', fontsize=16)
            plt.legend()
            plt.xlabel('Iterations', fontsize=14)
            plt.ylabel('Number of cells', fontsize=14)
            plt.savefig(f'Dead_cells.{file_type}', format=file_type)
            FILES.append(f'Dead_cells.{file_type}')
            if show:
                plt.show()
                
            # Death by strain over time
            plt.figure(figsize=(17, 8))
            for i,strain in enumerate(self.strainID):
                strain_evol = [death_by_strainEVOL[k][i] for k in range(n_iter)]
                plt.plot(strain_evol, label='Deaths for Strain'+str(i), linewidth=2, color=colors[i+2])
            plt.title('Dead cells over time', fontsize=16)
            plt.legend()
            plt.xlabel('Iterations', fontsize=14)
            plt.ylabel('Number of cells', fontsize=14)
            plt.savefig(f'Dead_cells_by_strain.{file_type}', format=file_type)
            FILES.append(f'Dead_cells_by_strain.{file_type}')
            if show:
                plt.show()
        
        # TOX production by strain
        plt.figure(figsize=(17, 8))
        for i,strain in enumerate(self.strainID):
            strain_evol = -np.log10([TOX_by_strainEVOL[k][i] for k in range(n_iter)])
            plt.plot(strain_evol, label='Strain'+str(i), linewidth=2, color=colors[i+2])
        plt.title('Mean Toxins production by strain (log10 scale)', fontsize=16)
        plt.legend()
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('-log10 production of Toxins [a.u]', fontsize=14)
        plt.savefig(f'TOX_by_strain.{file_type}', format=file_type)
        FILES.append(f'TOX_by_strain.{file_type}')
        if show:
            plt.show()
        
        # Total Toxin quantities over time
        plt.figure(figsize=(17, 8))
        for i,tox in enumerate(self.TOX_Matrices):
            tox_evol = np.log10([each_TOX_tot_EVOL[k][i] for k in range(n_iter)])
            plt.plot(tox_evol, label='Toxin'+str(i), linewidth=2, color=colors[i+2])
        plt.title('Total Toxin quantities over time. (log10 scale)', fontsize=16)
        plt.legend()
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('log10 production of Toxins [a.u]', fontsize=14)
        plt.savefig(f'each_TOX_tot.{file_type}', format=file_type)
        FILES.append(f'each_TOX_tot.{file_type}')
        if show:
            plt.show()
        
        # Toxin quantities produced at each iteration
        plt.figure(figsize=(17, 8))
        for i,tox in enumerate(self.TOX_Matrices):
            tox_evol = -np.log10([each_TOX_prod_EVOL[k][i] for k in range(n_iter)])
            plt.plot(tox_evol, label='Toxin'+str(i), linewidth=2, color=colors[i+2])
        plt.title('Toxin quantities produced at each iteration. (log10 scale)', fontsize=16)
        plt.legend()
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('-log10 production of Toxins [a.u]', fontsize=14)
        plt.savefig(f'each_TOX_prod.{file_type}', format=file_type)
        FILES.append(f'each_TOX_prod.{file_type}')
        if show:
            plt.show()
        
        # Living and Moving cells over time
        plt.figure(figsize=(17, 8))
        plt.plot(live_cellsEVOL, label='Living cells', linewidth=2)
        plt.plot(moving_cellsEVOL, label='Moving cells', linewidth=2)
        plt.title('Total number of living and moving cells over time', fontsize=16)
        plt.legend()
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Number of cells', fontsize=14)
        plt.savefig(f'Live_Moving_Total.{file_type}', format=file_type)
        FILES.append(f'Live_Moving_Total.{file_type}')
        if show:
            plt.show()
            
        # Birth over time
        plt.figure(figsize=(17, 8))
        plt.plot(new_cellsEVOL, label='Total new cells', linewidth=2)
        plt.title('New cells over time', fontsize=16)
        plt.legend()
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Number of cells', fontsize=14)
        plt.savefig(f'New_cells.{file_type}', format=file_type)
        FILES.append(f'New_cells.{file_type}')
        if show:
            plt.show()
            
        # Birth Structure over time
        plt.figure(figsize=(17, 8))
        for i,strain in enumerate(self.strainID):
            strain_evol = [new_by_strainEVOL[k][i] for k in range(n_iter)]
            plt.plot(strain_evol, label='Births for Strain'+str(i), linewidth=2, color=colors[i+2])
        plt.title('New cells by strain over time', fontsize=16)
        plt.legend()
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Number of cells', fontsize=14)
        plt.savefig(f'New_cells_by_strain.{file_type}', format=file_type)
        FILES.append(f'New_cells_by_strain.{file_type}')
        if show:
            plt.show()
        
        # Growthrate by strain
        plt.figure(figsize=(17, 8))
        for i,strain in enumerate(self.strainID):
            strain_evol = np.log10([growthrate_by_strainEVOL[k][i] for k in range(n_iter)])
            plt.plot(strain_evol, label='Strain'+str(i), linewidth=2, color=colors[i+2])
        plt.title('Mean growthrate by strain over time (log10 scale)', fontsize=16)
        plt.legend()
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('log10(Mean growthrate)', fontsize=14)
        plt.savefig(f'growthrate_by_strain.{file_type}', format=file_type)
        FILES.append(f'growthrate_by_strain.{file_type}')
        if show:
            plt.show()
        return FILES
    
    # =========================================================================
    #                       OTHER SIMULATION FUNCTIONS
    # =========================================================================
    def check_Equilibrium(self, previous_matrices):
        """
        Function used to check if equilibrium has been reached. Relevant only
        if movement and reproduction are no longer possible and renew_deads 
        is turned off.

        Parameters
        ----------
        previous_matrices : List of array

        Returns
        -------
        check : boolean
            True if Equilibrium has been reached, False otherwise.

        """
        current_matrices = [self.Strains_Matrix]
        check = np.allclose(previous_matrices, current_matrices)
        return check
    
    
    def write_Log(self, strain_params, TOX_dictionary, sim_params, filepath, finaliter):
        """
        Log containing all informations about the current biofilm simulation.
        Parameters
        ----------
        strain_params : {'Strain0':{'mu_O':0.3,'mu_N':1.5, 'mu_C':0.13, 
                                    'mu_max':4.2e-9,'O_status':0,
                                    'O_threshold':0.001, 'N_threshold':5e-3, 
                                    'C_threshold':5e-3, 'alpha_TOX':1.8e-9, 
                                    'TOX_threshold':1, 'Cdiv':1.9e-10}, 
                         'Strain1':{...}, ...}
        
        sim_params : {'Dimensionality':self.dimensionality,
                      'Height':self.height, 'Length': self.length, 'Depth':depth,
                      'Death Complexity':death_complexity, 
                      'TOX reproduction rule':look_TOX, 'Max #iterations':niter, 
                      'save_nutrients':save_nutrients, 
                      'save_TOX':save_TOX,
                      'move_by_nutri':move_by_nutri,
                      'min_Biom':min_Biom, 
                      'decaying_coeff':decaying_coeff,
                      'renew_deads':renew_deads,
                      'init' : init, 'init_strategy':init_strategy,
                      'biomass_init':biomass_init,
                      'O_init':O_init, 'O_init_grad':O_init_grad,
                      'N_init':N_init, 'N_init_grad':N_init_grad, 
                      'C_init':C_init, 'C_init_grad':C_init_grad, 
                      'TOX_init':TOX_init, 'TOX_init_grad':TOX_init_grad,
                      'KO2':KO2, 'KN':KN, 'KC':KC,
                      'Doxy':Doxy, 'Dnitro':Dnitro, 'Dcar':Dcar, 'DTOX':DTOX,
                      'thickness':thickness,
                      'col_height':col_height, 'col_width':col_width, 'col_depth':col_depth,
                      'cluster_strain':cluster_strain, 'cluster_height':cluster_height, 'cluster_width':cluster_width, 'cluster_depth':cluster_depth,
                      'max_depth_other':max_depth_other, 
                      'ncells':ncells, 'nstrains':nstrains, 
                      'neighborhood':neighborhood, 'radius':radius, 
                      'file_type_dynamic':file_type, 'show_dynamic':show_dynamic,
                      'animation_duration(s)':duration}
        filepath : string.
        finaliter : {'Final Iteration': i, 
                                 'Equilibrium':Equi, 
                                 'Initialization time': t_init, 
                                 'Simulation time': t_iter}

        Returns
        ----------
        None.
        """
        # Format the parameters
        strain_params_str = format_dict(strain_params, 4)
        tox_params_str = format_dict(TOX_dictionary, 4)
        sim_params_str = format_dict(sim_params, 4)
        finaliter_str = format_dict(finaliter, 4)
        
        # Combine the strings into one log entry
        log_entry = f"""Strain Parameters:\n{strain_params_str}\n
                    Toxins Parameters:\n{tox_params_str}\n
                    Simulation Parameters:\n{sim_params_str}\n
                    Ending Informations:\n{finaliter_str}"""
    
        # Write the log entry to the file
        with open(filepath, 'w') as f:
            f.write(log_entry)
            
    # =============================================================================
    #              MAIN METHOD : RUNNING THE CELLULAR AUTOMATON
    # =============================================================================
    def run_Simulation(self, strains_dictionary, TOX_dictioanry,
                       death_complexity=1, look_TOX=False, niter=50, 
                       random_state=None,
                       save_nutrients=False, 
                       save_TOX=False,
                       init = True, init_strategy='random',
                       move_by_nutri=True,
                       min_Biom=1e-17, 
                       decaying_coeff=0.5,
                       renew_deads=False,
                       biomass_init=10,
                       O_init=1e9, O_init_grad=True, scale_O='log2',
                       N_init=1e9, N_init_grad=True, scale_N='log2',
                       C_init=1e9, C_init_grad=True, scale_C='log2',
                       TOX_init=0, TOX_init_grad=True, scale_TOX='log2',
                       nportions=5,
                       KO2=3.1e-3, KN=10.3e-3, KC=0.2e-3,
                       Doxy=2.1e-9, Dnitro=1.9e-9, Dcar=2.4e-9, DTOX=0.7e-9,
                       thickness=None,
                       col_height=None, col_width=None, col_depth=None,
                       cluster_strain=None, cluster_height=None, cluster_width=None, cluster_depth=None,
                       max_depth_other=None, 
                       ncells=100, nstrains=2, 
                       neighborhood='VonNeumman', radius=1,
                       file_type='png', show_dynamic=False,
                       duration=60):
        """
        Main function of this project, runs the cellular automata.
        ----------
        strain_dictionary : dictionary
                {'Strain0':{'mu_O':0.3,'mu_N':1.5, 'mu_C':0.13, 
                        'mu_max':4.2e-9,'O_status':0, 'mobility':0.5,
                        'O_threshold':0.001, 'N_threshold':5e-3, 
                        'C_threshold':5e-3, 'alpha_TOX':1.8e-9, 
                        'TOX_threshold':1, 'Cdiv':1.9e-10}, 
                         'Strain1':{...}, ...}.
        TOX_dictionary : dicitonary
                {'TOX0':{'+':[0,2], '-':[1], 'DTOX':0.7e-9},
                'TOX1':{'+':[1], '-':[2], 'DTOX':0.7e-9},
                'TOX2':{'+':[3], '-':[0], 'DTOX':0.7e-9}}.
        decaying_coeff : float, optional
            Dead matter decaying coefficient. The default is 0.5.
        look_TOX : boolean, optional
            The default is False.
        niter : int, optional,
            Number of iterations. The default is 50.
        random_state : int, optional
            Which seed to use for random processes. Useful for reproducible 
            initialization. The default is None,
        save_nutrients : boolean, optional
            Whether to save nutrients plots. The default is False.
        save_TOX : boolean, optional
            Whether to save Toxin plot. The default is False.
        init : boolean, optional
            Whether to initialize grids. The default is True. 
        init_strategy : string, optional
            Initialization method, either 'random', 'surface', 'floor', 
            'separate' or 'on_cluster'. The default is 'random'.
        renew_deads : boolean, optional
            The default is False.
        file_type : string, optional
            Either 'pdf' or 'png'. The default is 'png'.
        show_dynamic : boolean, optional
            Whether to show the dynamic curves. The default is False.
        duration : int, optional,
            The number of seconds the animation should last. The default is 60.

        Returns
        ----------
        dictionary
        {'O': self.O_Matrix, 'N': self.N_Matrix, 'C':self.C_Matrix, 
                'TOX':self.TOX_Matrix, 'Biomass': self.grid, 
                'Strains':self.Strains_Matrix}
        """
        plt.close('all')
        self.strains_dictionary = strains_dictionary
        self.TOX_dictionary = TOX_dictionary
        self.neighborhood_type = neighborhood
        self.radius = radius
        if random_state:
            random.seed(random_state)
            np.random.seed(random_state)
        self.decaying_matter_coeff = decaying_coeff
        if self.dimensionality == 2: # 2D
            depth = None
        else: # 3D
            depth = self.depth
        n_tox = len(TOX_dictionary)
        images = ['Strains', 'Biomass', 'Oxygen', 'Nitrogen', 'Carbon', 'TOX']
        self.image_files = {key: [] for key in images}
        
        # Retrieve time
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create simulation folder
        simulation_folder = f"./Simulation_{init_strategy}_{timestamp}"
        os.makedirs(simulation_folder, exist_ok=True)
        
        # Create animation folders
        animation_folder = "./Simulations/Animation"
        os.makedirs(animation_folder, exist_ok=True)
        os.makedirs("./Simulations/Animation/Strains_images", exist_ok=True)
        os.makedirs("./Simulations/Animation/Biomass_images", exist_ok=True)
        
        if save_nutrients:
            L = ['Oxygen', 'Carbon', 'Nitrogen']
            for nutrient in L:
                os.makedirs(f"./Simulations/Animation/{nutrient}_images", exist_ok=True)
        if save_TOX:
            os.makedirs(f"./Simulations/Animation/TOX_images", exist_ok=True)
        os.chdir(wd+'/Simulations')
        
        # MATRIX INITIALIZATION
        if init:
            t0_init = time.time()
            # Initialize useful Matrices
            Biomass_Matrix, O_Matrix, N_Matrix, C_Matrix, TOX_Matrices = self.get_grids(TOX_dictionary, O_init=O_init, O_init_grad=O_init_grad, scale_O=scale_O,
            N_init=N_init, N_init_grad=N_init_grad, scale_N=scale_N,
            C_init=C_init, C_init_grad=C_init_grad, scale_C=scale_C,
            TOX_init=TOX_init, TOX_init_grad=TOX_init_grad, scale_TOX=scale_TOX,
            nportions=nportions).values()
            
            # Initialize cell positions
            if init_strategy == 'random':
                S_M, indices = self.random_inoculate(ncells, nstrains, biomass_init=biomass_init)
                
            elif init_strategy == 'surface':
                S_M, indices = self.surface_inoculation(thickness, ncells, nstrains, biomass_init=biomass_init)
            
            elif init_strategy == 'floor':
                S_M, indices = self.floor_inoculation(thickness, ncells, nstrains, biomass_init=biomass_init)
            
            elif init_strategy == 'separate':
                S_M, indices = self.separate_colonies_inoculation(col_height, col_width, ncells, nstrains, col_depth=col_depth, biomass_init=biomass_init)
            
            elif init_strategy == 'on_cluster':
                S_M, indices = self.on_cluster_inoculation(cluster_strain, cluster_height, cluster_width, 
                                                           max_depth_other, ncells, nstrains, cluster_depth=cluster_depth, biomass_init=biomass_init)

            # Initialize Neighborhood
            Near = self.neighborhood(neighborhood=neighborhood, radius=radius)
            t_init = time.time() - t0_init
            
        previous_matrices = [S_M.copy()]
        Equi = False # To identify if the simulation stopped due to equilibrium
        
        # List of [[new cells Strain0_iter0, new cells Strain1_iter0, ..], [...]]
        new_by_strainEVOL = [[ncells for strain in range(nstrains)]] 
        
        # List of [%cellStrain0_iter0, %cellStrain1_iter0, ..]
        pop_size_by_strainEVOL = [] 
        
        # List of [ncelldeadStrain0_iter0, ncelldeadStrain1_iter0, ...]
        death_by_strainEVOL = [] 
        
        # List of Total number of living cells at each iteration
        live_cellsEVOL = [] 
        
        # List of Total number of dead cells at each iteration
        dead_cellsEVOL = [] 
        
        # List of [N new cells at iter0, at iter1, ...]
        new_cellsEVOL = [ncells*nstrains] 
        
        # List of total deaths by ...
        death_byO = []
        death_byC = []
        death_byN = []
        death_byTOX = []
        
        # List of mean growthrate for each strain at each iteration
        growthrate_by_strainEVOL = []
        
        # List of mean Toxin production by strain at each iteration
        TOX_by_strainEVOL = []
        
        # List of each total Toxin quantity
        each_TOX_tot_EVOL = []
        
        # List of each Toxin PRODUCTION 
        each_TOX_prod_EVOL = []
        
        # List of number of moving cells at each iteration
        moving_cellsEVOL = []
        
        if depth:
            # List of Strains Matrix at each iteration
            space_matrix = [S_M.copy()] 
            
        t0_iter = time.time()
        for i in range(niter):
            if i % 100 == 0 and self.dimensionality == 2 :
                self.plot_Matrices(i, show=True, min_Biom=min_Biom, save=True, file_type=file_type, save_TOX=save_TOX)
            print(f'\nIteration {i+1} started\n')
            # DIFFUSION
            diff_O, diff_N, diff_C, diff_TOX = self.calculate_diffusion(Doxy=Doxy, Dnitro=Dnitro, Dcar=Dcar)

            if len(self.cells_pos) > 0:
                n = np.sum((self.Strains_Matrix!=0) & (self.Strains_Matrix!=-1))
                live_cellsEVOL.append(n)
                if n > 0:
                    pop_size_by_strainEVOL.append([np.sum((self.Strains_Matrix==strain))/n for strain in self.strainID])
                else:
                    pop_size_by_strainEVOL.append([0 for strain in self.strainID])
                
                ndead = np.sum((self.Strains_Matrix==-1))
                dead_cellsEVOL.append(ndead)
            
                # CONSUMPTION and TOXIN PRODUCTION
                TOX_prod = self.calculate_TOX_production(strains_dictionary)
                cons_O, cons_N, cons_C, growthrate = self.calculate_uptakes(strains_dictionary)
                
                L_mu = []
                L_TOX = []
                for k in range(len(self.strainID)):
                    tox_id = strains_dictionary['Strain'+str(k)]['tox_+_id']
                    if self.Index_by_strain[k]:
                        if depth==None:
                            i_growth, j_growth = zip(*self.Index_by_strain[k])
                            L_mu.append(np.mean(growthrate[i_growth, j_growth]))
                            L_TOX.append(np.mean(TOX_prod[tox_id][i_growth, j_growth]))
                        else:
                            i_growth, j_growth, k_growth = zip(*self.Index_by_strain[k])
                            L_mu.append(np.mean(growthrate[i_growth, j_growth, k_growth]))
                            L_TOX.append(np.mean(TOX_prod[tox_id][i_growth, j_growth, k_growth]))
                    else:
                        L_mu.append(0)
                        L_TOX.append(0)
                growthrate_by_strainEVOL.append(L_mu)
                TOX_by_strainEVOL.append(L_TOX)
                each_TOX_tot_EVOL.append([self.TOX_Matrices[i].sum() for i in range(n_tox)])
                each_TOX_prod_EVOL.append([TOX_prod[i].sum() for i in range(n_tox)])

                # UPDATE ENVIRONMENT
                dico_diff = {'diff_O': diff_O, 'dff_N': diff_N, 'diff_C': diff_C, 'diff_TOX':diff_TOX}
                dico_cons = {'cons_O': cons_O, 'cons_N': cons_N, 'cons_C': cons_C, 'growthrate': growthrate}
                self.update_abiotic_environment(dico_diff, dico_cons, TOX_prod)
                
                if death_complexity != None:
                    # DEATHS
                    new_dead_cells_pos, D_byO, D_byN, D_byC, D_byTOX = self.update_deaths(strains_dictionary, i, complexity=death_complexity) 
                    
                    if new_dead_cells_pos:
                        L = []
                        for strain in self.strainID:
                            if depth == None:
                                i_dead, j_dead = zip(*new_dead_cells_pos)
                                L.append(np.sum(previous_matrices[0][i_dead, j_dead]==strain))
                            else:
                                i_dead, j_dead, k_dead = zip(*new_dead_cells_pos)
                                L.append(np.sum(previous_matrices[0][i_dead, j_dead, k_dead]==strain))
                        death_by_strainEVOL.append(L)
                    else:
                        death_by_strainEVOL.append([0 for strain in self.strainID])
                    
                    death_byTOX.append(D_byTOX)
                    if death_complexity==2:
                        death_byO.append(D_byO)
                    if death_complexity==3:
                        death_byC.append(D_byC)
                        death_byN.append(D_byN)
                        
                    # update neighborhood
                    self.neighbors = self.neighborhood(neighborhood=neighborhood, radius=radius)
                    # On mélange l'ordre des voisins pour chaque cellule
                    for liste in self.neighbors:
                        random.shuffle(liste)
                # REPRODUCTION, MOVEMENT AND RESOURCE SHARING
                new_cells_pos, new_cell_by_strain, moving = self.reproduction(strains_dictionary, look_TOX=look_TOX, move_by_nutri=move_by_nutri)
                new_cellsEVOL.append(len(new_cells_pos))
                moving_cellsEVOL.append(moving)
                new_by_strain = [len(new_cell_by_strain[i]) for i in range(len(new_cell_by_strain))]
                new_by_strainEVOL.append(new_by_strain) 
                
                # update neighborhood
                self.neighbors = self.neighborhood(neighborhood=neighborhood, radius=radius)
                for liste in self.neighbors:
                    random.shuffle(liste)
                if death_complexity != None:
                    # DEAD CELLS RESOURCE SHARING
                    self.share_death(neighborhood=neighborhood, radius=radius, min_Biom=min_Biom, renew_deads=renew_deads)
                print(f'\nIteration {i+1} ended\n')
                
                # PLOTS
                if depth==None:
                    self.plot_OneMatrix(i, self.Strains_Matrix, 'Strains', save=True, file_type=file_type)
                    self.plot_OneMatrix(i, self.grid, 'Biomass', save=True, min_Biom=min_Biom, file_type=file_type)
                    if save_nutrients:
                        self.plot_OneMatrix(i, self.O_Matrix, 'Oxygen', save=True, file_type=file_type)
                        self.plot_OneMatrix(i, self.C_Matrix, 'Carbon', save=True, file_type=file_type)
                        self.plot_OneMatrix(i, self.N_Matrix, 'Nitrogen', save=True, file_type=file_type)
                    if save_TOX:
                        self.plot_OneMatrix(i, self.TOX_Matrices, f'TOX', save=True, file_type=file_type)
                else:
                    # when in 3D, store the strain matrix for later plotting
                    space_matrix.append(self.Strains_Matrix.copy())
                # if self.check_Equilibrium(previous_matrices):
                #     Equi = True
                #     print(f'Equilibrium has been reached after iteration {i+1}')
                #     break
                # else:
                previous_matrices =[self.Strains_Matrix.copy()]
            else : 
                Equi = 'No More living cells'
                print(f'No more living cells after iteration {i+1}')
                break
        if self.dimensionality == 2 :
            self.plot_Matrices(i, show=True, min_Biom=min_Biom, save=True, file_type=file_type, save_TOX=save_TOX)
        t_iter = time.time() - t0_iter
        if i >= 60:
            time_step = duration/i
        else:
            time_step = 0.8
        print('\nMaking Animations...')
        
        if depth: # 3D Animation making
            with open(f'./Animation/Strains_{init_strategy}_{timestamp}.pkl', 'wb') as f:
                pkl.dump(space_matrix, f)
            FILES = self.make3D_plots(space_matrix, file_type=file_type)
            self.make3D_gif(FILES, './Animation/Strains_Evolution.gif', time_step)
            
        else: # 2D Animation making
            self.make_GIF('Strains', './Animation/Strains_Evolution.gif',time_step)
            self.make_GIF('Biomass', './Animation/Biomass_Evolution.gif',time_step)
            if save_nutrients:
                self.make_GIF('Oxygen', './Animation/Oxygen_Evolution.gif',time_step)
                self.make_GIF('Nitrogen', './Animation/Nitrogen_Evolution.gif',time_step)
                self.make_GIF('Carbon', './Animation/Carbon_Evolution.gif',time_step)
            if save_TOX:
                self.make_GIF(f'TOX', f'./Animation/TOX_Evolution.gif',time_step)
        print('\nAnimations ready.')
        
        # PLOT DYNAMIC CURVES
        DYNAMICS = self.make_DynamicCurves(new_by_strainEVOL, pop_size_by_strainEVOL, 
                               death_complexity,
                               death_by_strainEVOL,
                               live_cellsEVOL,
                               dead_cellsEVOL,
                               new_cellsEVOL,
                               moving_cellsEVOL,
                               death_byO, death_byC, death_byN, death_byTOX,
                               growthrate_by_strainEVOL, 
                               TOX_by_strainEVOL, 
                               each_TOX_tot_EVOL,
                               each_TOX_prod_EVOL,
                               file_type=file_type, show=show_dynamic)
        # LOG WRITING
        print('\nWriting log..')
        sim_params = {'Dimensionality':self.dimensionality,
                      'Height':self.height, 'Length': self.length, 'Depth':depth,
                        'Death Complexity':death_complexity, 
                        'TOX reproduction rule':look_TOX, 'Max #iterations':niter,
                        'Random seed': random_state,
                        'save_nutrients':save_nutrients, 
                        'save_TOX':save_TOX,
                        'move_by_nutri':move_by_nutri,
                        'min_Biom':min_Biom, 
                        'decaying_coeff':decaying_coeff,
                        'renew_deads':renew_deads,
                        'init' : init, 'init_strategy':init_strategy,
                        'biomass_init':biomass_init,
                        'O_init':O_init, 'O_init_grad':O_init_grad,
                        'N_init':N_init, 'N_init_grad':N_init_grad, 
                        'C_init':C_init, 'C_init_grad':C_init_grad, 
                        'TOX_init':TOX_init, 'TOX_init_grad':TOX_init_grad,
                        'nportions gradient':nportions,
                        'KO2':KO2, 'KN':KN, 'KC':KC,
                        'Doxy':Doxy, 'Dnitro':Dnitro, 'Dcar':Dcar, 'DTOX':DTOX,
                        'thickness':thickness,
                        'col_height':col_height, 'col_width':col_width, 'col_depth':col_depth,
                        'cluster_strain':cluster_strain, 'cluster_height':cluster_height, 'cluster_width':cluster_width, 'cluster_depth':cluster_depth,
                        'max_depth_other':max_depth_other, 
                        'ncells':ncells, 'nstrains':nstrains, 
                        'neighborhood':neighborhood, 'radius':radius, 
                        'file_type_dynamic':file_type, 'show_dynamic':show_dynamic,
                        'animation_duration(s)':duration}
        finaliter = {'Final Iteration': i, 'Equilibrium':Equi, 'Initialization time': t_init, 'Simulation time': t_iter}
        log_filepath = f'log_{init_strategy}_{timestamp}.txt'
        self.write_Log(strains_dictionary, TOX_dictionary, sim_params, log_filepath, finaliter)
        print('\nLog written.')
        # Move all generated files into the simulation folder
        folders_to_save = ['./Animation/Biomass_images', './Animation/Strains_images']
        if depth:
            files_to_save = DYNAMICS + [log_filepath, './Animation/Strains_Evolution.gif', f'./Animation/Strains_{init_strategy}_{timestamp}.pkl']
        else:
            all_matrices = [f'./Animation/simulation_0.{file_type}', f'./Animation/simulation_{i}.{file_type}']
            files_to_save = all_matrices + DYNAMICS + [log_filepath,'./Animation/Biomass_Evolution.gif', './Animation/Strains_Evolution.gif']
        if save_nutrients and depth==None:
            folders_to_save.extend(['./Animation/Carbon_images', 
                                    './Animation/Nitrogen_images', 
                                    './Animation/Oxygen_images'])
            files_to_save.extend(['./Animation/Carbon_Evolution.gif', 
                                  './Animation/Nitrogen_Evolution.gif',
                                  './Animation/Oxygen_Evolution.gif'])
        if save_TOX and depth==None:
            folders_to_save.extend([f'./Animation/TOX_images'])
            files_to_save.extend([f'./Animation/TOX_Evolution.gif'])
        
        for folder in folders_to_save:
            shutil.move(folder, os.path.join(simulation_folder, folder))
        
        for file in files_to_save:
            shutil.move(file, os.path.join(simulation_folder, file))
        print(f"All files and folders have been moved to {simulation_folder}.")
        
        print(f'\nInitialization took ~ {t_init} seconds.\nSimulation took ~ {int(t_iter)} seconds.')
        print(f'For a total of ~ {int(t_init+t_iter)} seconds.')
        for i in range(len(self.strainID)):
            print(f'\nStrain{i} is {self.strainID[i]}.')
        return {'O': self.O_Matrix, 'N': self.N_Matrix, 'C':self.C_Matrix, 
                'TOX':self.TOX_Matrices, 'Biomass': self.grid, 
                'Strains':self.Strains_Matrix}
    
# =============================================================================
#                               PROGRAM
# =============================================================================
    
# Instantiate the cellular automaton object
GRID = Grid((160, 160)) # (height, length)

# Define strains characteristics

# O_status == 1 means the strains needs minimum levels of Oxygen
# O_status == 0 means the strains can't survive a maximum level of Oxygen
# Deaths thresholds are defined by thresholds parameters
# Cdiv is the minimum biomass needed for the strain to divide
# Shared_portion sets how much of its own biomass this strain can share with its neighbors

# Main impact parameters are {Cdiv, mobility, alpha_TOX and thresholds 
# according to the deaths complexity you chose}
strains_dictionary = {'Strain0':{'mu_O':0.3,'mu_N':0.075, 'mu_C':2, 'mu_max':0.85, #Potential antagonist-2
                                'mobility':0.3, 
                                'O_status':1,
                                'O_threshold':2e-10, 'N_threshold':5e-7, 'C_threshold':0.2e-7,
                                'alpha_TOX':0.4e-15, 'TOX_threshold':7.1e-6, 
                                'tox_prod_penalty':1, 'tox_+_id':0,'tox_-_id':1,
                                'Cdiv':8.1e-13, 'Shared_portion':0.5},
                      'Strain1':{'mu_O':0.3,'mu_N':0.15, 'mu_C':1.5, 'mu_max':0.95, #Potential antagonist-1
                                  'mobility':0.095, 
                                  'O_status':1,
                                  'O_threshold':2e-10, 'N_threshold':5e-7, 'C_threshold':0.2e-7,
                                  'alpha_TOX':0.8e-15, 'TOX_threshold':7.1e-6, 
                                  'tox_prod_penalty':1, 'tox_+_id':0,'tox_-_id':1,
                                  'Cdiv':8.1e-13, 'Shared_portion':0.5},
                        'Strain2':{'mu_O':0.3,'mu_N':0.15, 'mu_C':1.5, 'mu_max':3, #E.COLI
                                'mobility':1, 
                                'O_status':1,
                                'O_threshold':2e-10, 'N_threshold':5e-7, 'C_threshold':0.2e-7,
                                'alpha_TOX':0, 'TOX_threshold':0.1e-13, 
                                'tox_prod_penalty':1, 'tox_+_id':1,'tox_-_id':0,
                                'Cdiv':8.1e-12, 'Shared_portion':0.5}}

TOX_dictionary = {'TOX0':{'+':[0], '-':[1], 'DTOX':4.5e-6},
                    'TOX1':{'+':[1], '-':[0], 'DTOX':4.5e-6}}
nstrains = len(strains_dictionary)

# Run Biofilm Simulation 
resultats = GRID.run_Simulation(strains_dictionary,
                                TOX_dictionary,
                                death_complexity=3,
                                random_state=25,
                                niter=24,      
                                ncells=32, 
                                save_nutrients=False, 
                                save_TOX=True,
                                init_strategy='random',
                                move_by_nutri=False,
                                min_Biom=1e-50, 
                                decaying_coeff=0.05,
                                renew_deads=False,
                                biomass_init=1e-12,
                                O_init=7e-3, O_init_grad=True, scale_O='linear',
                                N_init=0.7e-3, N_init_grad=True, scale_N='linear',
                                C_init=25e-3, C_init_grad=True, scale_C='linear',
                                TOX_init=0, TOX_init_grad=True, scale_TOX='linear',
                                nportions=100,
                                KO2=3.1e-3, KN=10.3e-3, KC=0.2e-3,
                                Doxy=2.1e-9, Dnitro=1.9e-9, Dcar=2.4e-9, DTOX=0.7e-9,
                                look_TOX=True, # reproduction depends on [TOX]
                                thickness = 1,
                                col_height=8,col_width=5,
                                cluster_strain=0, cluster_height=10, cluster_width=20,
                                max_depth_other=90,
                                nstrains=nstrains,
                                neighborhood='Moore', #'Moore'or 'VonNeumman'
                                radius=1, 
                                file_type='png', show_dynamic=False, 
                                duration=30) 
