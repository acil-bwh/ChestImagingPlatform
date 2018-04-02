from cip_python.phenotypes import BodyCompositionPhenotypes
from cip_python.input_output import ImageReaderWriter
import cip_python.common as common
import numpy as np
np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

image_io = ImageReaderWriter()
lm_name = ct_name = common.Paths.testing_file_path('simple_lm.nrrd')
lm, lm_header = image_io.read_in_numpy(lm_name)
ct_name = common.Paths.testing_file_path('simple_ct.nrrd')
ct, ct_header=image_io.read_in_numpy(ct_name)

def test_execute():
    c = common.ChestConventions()
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
        if (r == 'WholeLung' and t == wc):
            assert df['leanHUMin'].iloc[i] == -50, 'Phenotype not as expected'
            assert df['leanHUMax'].iloc[i] == -30, 'Phenotype not as expected'
            assert np.isclose(df['leanAxialCSA'].iloc[i],
                2*spacing[0]*spacing[1]), 'Phenotype not as expected'
            assert np.isclose(df['leanCoronalCSA'].iloc[i],
                2*spacing[0]*spacing[2]), 'Phenotype not as expected'
            assert np.isclose(df['leanSagittalCSA'].iloc[i],
                2*spacing[2]*spacing[1]), 'Phenotype not as expected'
            assert df['leanHUMedian'].iloc[i] == -40, \
                'Phenotype not as expected'
            assert df['leanHUStd'].iloc[i] == 10, \
                'Phenotype not as expected'
            assert df['leanHUMode'].iloc[i] == -50, \
                'Phenotype not as expected'                        
            assert df['leanHUKurtosis'].iloc[i] == -2.0, \
                'Phenotype not as expected'
            assert df['leanHUSkewness'].iloc[i] == 0.0, \
                'Phenotype not as expected'                            


