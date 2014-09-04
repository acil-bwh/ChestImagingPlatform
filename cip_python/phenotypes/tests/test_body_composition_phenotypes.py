import os.path
import pandas as pd
import nrrd
from cip_python.ChestConventions import ChestConventions
from cip_python.phenotypes.body_composition_phenotypes import *
import pdb

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

this_dir = os.path.dirname(os.path.realpath(__file__))
lm_name = this_dir + '/../../../Testing/Data/Input/simple_lm.nrrd'
lm, lm_header = nrrd.read(lm_name)
ct_name = this_dir + '/../../../Testing/Data/Input/simple_ct.nrrd'
ct, ct_header = nrrd.read(ct_name)

def test_execute():
    c = ChestConventions()
    wc = c.GetChestWildCardName()
    spacing = np.array([0.5, 0.4, 0.3])
    
    bc_pheno = BodyCompositionPhenotypes()    
    df = bc_pheno.execute(ct, lm, 'simple', spacing)

    for i in xrange(0, 14):
        r = df['Region'].iloc[i]
        t = df['Type'].iloc[i]

        if (r == 'LeftLung' and t == wc):
            assert np.isclose(df['HUMean'].iloc[i], -773.333333), \
                'Phenotype not as expected'
            assert np.isclose(df['AxialCSA'].iloc[i], \
                              9*spacing[0]*spacing[1]), \
                'Phenotype not as expected'
            assert np.isclose(df['CoronalCSA'].iloc[i], \
                              9*spacing[0]*spacing[2]), \
                'Phenotype not as expected'
            assert np.isclose(df['SagittalCSA'].iloc[i], \
                              9*spacing[1]*spacing[2]), \
                'Phenotype not as expected'            
        if (r == 'WholeLung' and t == wc):
            assert df['HUMedian'].iloc[i] == -825, 'Phenotype not as expected'            
            assert np.isclose(df['HUStd'].iloc[i], 256.9695), \
                'Phenotype not as expected'
            assert np.isclose(df['AxialCSA'].iloc[i], \
                              18*spacing[0]*spacing[1]), \
                'Phenotype not as expected'
            assert np.isclose(df['CoronalCSA'].iloc[i], \
                              18*spacing[0]*spacing[2]), \
                'Phenotype not as expected'
            assert np.isclose(df['SagittalCSA'].iloc[i], \
                              18*spacing[1]*spacing[2]), \
                'Phenotype not as expected'                        
        if (r == 'WholeLung' and t == 'UndefinedType'):
            assert np.isclose(df['HUKurtosis'].iloc[i], -2.363636), \
                'Phenotype not as expected'
            assert np.isclose(df['AxialCSA'].iloc[i], \
                              14*spacing[0]*spacing[1]), \
                'Phenotype not as expected'
            assert np.isclose(df['CoronalCSA'].iloc[i], \
                              14*spacing[0]*spacing[2]), \
                'Phenotype not as expected'
            assert np.isclose(df['SagittalCSA'].iloc[i], \
                              14*spacing[1]*spacing[2]), \
                'Phenotype not as expected'                        
        if (r == 'WholeLung' and t == 'Vessel'):
            assert df['HUSkewness'].iloc[i] == 0.0, 'Phenotype not as expected'
            assert np.isclose(df['AxialCSA'].iloc[i], \
                              2*spacing[0]*spacing[1]), \
                'Phenotype not as expected'
            assert np.isclose(df['CoronalCSA'].iloc[i], \
                              2*spacing[0]*spacing[2]), \
                'Phenotype not as expected'
            assert np.isclose(df['SagittalCSA'].iloc[i], \
                              2*spacing[1]*spacing[2]), \
                'Phenotype not as expected'                        
        if (r == 'RightLung' and t == wc):
            assert df['HUMode'].iloc[i] == -800, 'Phenotype not as expected'
            assert np.isclose(df['AxialCSA'].iloc[i], \
                              9*spacing[0]*spacing[1]), \
                'Phenotype not as expected'
            assert np.isclose(df['CoronalCSA'].iloc[i], \
                              9*spacing[0]*spacing[2]), \
                'Phenotype not as expected'
            assert np.isclose(df['SagittalCSA'].iloc[i], \
                              9*spacing[1]*spacing[2]), \
                'Phenotype not as expected'                        
        if (r == 'WholeLung' and t == 'Airway'):
            assert df['HUMin'].iloc[i] == -980, 'Phenotype not as expected'
            assert df['HUMax'].iloc[i] == -950, 'Phenotype not as expected'
