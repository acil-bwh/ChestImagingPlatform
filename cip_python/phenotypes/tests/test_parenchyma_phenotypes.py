import os.path
import numpy as np

from cip_python.input_output import ImageReaderWriter
from cip_python.common import ChestConventions, Paths
from cip_python.phenotypes import ParenchymaPhenotypes

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

image_io = ImageReaderWriter()

ct_name = Paths.testing_file_path('simple_ct.nrrd')
lm_name = Paths.testing_file_path('simple_lm.nrrd')

lm, lm_header = image_io.read_in_numpy(lm_name)
ct, ct_header=image_io.read_in_numpy(ct_name)

def test_execute():
    c = ChestConventions()
    wc = c.GetChestWildCardName()
    
    paren_pheno = ParenchymaPhenotypes()    
    df = paren_pheno.execute(lm, 'simple', np.array([1., 1., 1.]), ct=ct)

    for i in xrange(0, 14):
        r = df['Region'].iloc[i]
        t = df['Type'].iloc[i]
        val_950 = df['LAA950'].iloc[i]
        val_910 = df['LAA910'].iloc[i]
        val_856 = df['LAA856'].iloc[i]
        if (r == 'UndefinedRegion' and t == wc) or \
            (r == wc and t == 'UndefinedRegion') or \
            (r == 'UndefinedRegion' and t == 'UndefinedType'):
            assert val_950 == wc, 'Phenotype not as expected'
            assert val_910 == wc, 'Phenotype not as expected'
            assert val_856 == wc, 'Phenotype not as expected'
            assert val_perc10 == wc, 'Phenotype not as expected'
            assert val_perc15 == wc, 'Phenotype not as expected'
        elif (r == 'WholeLung' and t == wc) or \
            (r == 'RightLung' and t == wc) or \
            (r == 'LeftLung' and t == wc):
            test_val_950=dict()
            test_val_950['WholeLung']=0.0555555
            test_val_950['RightLung']=0
            test_val_950['LeftLung']=0.1111111
            assert np.isclose(val_950, test_val_950[r]), 'Phenotype not as expected'
            assert np.isclose(val_910, 0.1111111), 'Phenotype not as expected'
            assert np.isclose(val_856, 0.1111111), 'Phenotype not as expected'
        elif (r == wc and t == 'Airway') or \
            (r == 'UndefinedRegion' and t == 'Airway') or \
            (r == 'WholeLung' and t == 'Airway') or \
            (r == 'RightLung' and t == 'Airway') or \
            (r == 'LeftLung' and t == 'Airway'):
            test_val_950=dict()
            test_val_950[wc,'Airway']=0.75
            test_val_950['UndefinedRegion','Airway']=1.
            test_val_950['WholeLung','Airway']=0.5
            test_val_950['RightLung','Airway']=0
            test_val_950['LeftLung','Airway']=1.
            assert np.isclose(val_950, test_val_950[r,t]), 'Phenotype not as expected'
            assert np.isclose(val_910, 1.0), 'Phenotype not as expected'
            assert np.isclose(val_856, 1.0), 'Phenotype not as expected'
        elif (r == 'WholeLung' and t == 'UndefinedType') or \
            (r == 'RightLung' and t == 'UndefinedType') or \
            (r == 'LeftLung' and t == 'UndefinedType') or \
            (r == 'WholeLung' and t == 'Vessel') or \
            (r == 'RightLung' and t == 'Vessel') or \
            (r == 'LeftLung' and t == 'Vessel') or \
            (r == wc and t == 'Vessel'):
            assert np.isclose(val_950, 0.0), 'Phenotype not as expected'
            assert np.isclose(val_910, 0.0), 'Phenotype not as expected'
            assert np.isclose(val_856, 0.0), 'Phenotype not as expected'

        if (r == 'WholeLung' and t == wc):
            assert np.isclose(df['HAA700'].iloc[i], 0.1111111), \
                'Phenotype not as expected'
        if (r == wc and t == 'Vessel'):
            assert df['HAA700'].iloc[i] == 1., 'Phenotype not as expected'
        if (r == 'UndefinedRegion' and t == 'Airway'):
            assert df['HAA600'].iloc[i] == 0., 'Phenotype not as expected'
        if (r == 'WholeLung' and t == 'Vessel'):
            assert df['HAA500'].iloc[i] == 1., 'Phenotype not as expected'            
        if (r == 'WildCard' and t == 'Vessel'):
            assert df['HAA250'].iloc[i] == 1., 'Phenotype not as expected'
        if (r == 'WholeLung' and t == 'Airway'):
            assert df['Perc10'].iloc[i] == -977, 'Phenotype not as expected'
        if (r == 'WholeLung' and t == 'Vessel'):
            assert df['Perc15'].iloc[i] == -47, 'Phenotype not as expected'
        if (r == 'LeftLung' and t == wc):
            assert np.isclose(df['HUMean'].iloc[i], -773.333333), \
                'Phenotype not as expected'
        if (r == 'WholeLung' and t == wc):
            assert np.isclose(df['HUStd'].iloc[i], 256.9695), \
                'Phenotype not as expected'            
        if (r == 'WholeLung' and t == 'UndefinedType'):
            assert np.isclose(df['HUKurtosis'].iloc[i], -2.363636), \
                'Phenotype not as expected'           
        if (r == 'WholeLung' and t == 'Vessel'):
            assert df['HUSkewness'].iloc[i] == 0.0, 'Phenotype not as expected'
        if (r == 'RightLung' and t == wc):
            assert df['HUMode'].iloc[i] == -800, 'Phenotype not as expected'
        if (r == 'WholeLung' and t == wc):
            assert df['HUMedian'].iloc[i] == -825, 'Phenotype not as expected'
        if (r == 'WholeLung' and t == 'Airway'):
            assert df['HUMin'].iloc[i] == -980, 'Phenotype not as expected'
        if (r == 'WholeLung' and t == 'Airway'):
            assert df['HUMax'].iloc[i] == -950, 'Phenotype not as expected'
        if (r == 'WholeLung' and t == wc):
            assert df['HUMean500'].iloc[i] == -842.5, 'Phenotype not as expected'
        if (r == 'WholeLung' and t == wc):
            assert np.isclose(df['HUStd500'].iloc[i], 52.1416340), \
                'Phenotype not as expected'
        if (r == 'WholeLung' and t == wc):
            assert np.isclose(df['HUKurtosis500'].iloc[i], 2.37885301), \
                'Phenotype not as expected'
        if (r == 'WholeLung' and t == wc):
            assert np.isclose(df['HUSkewness500'].iloc[i], -1.613628), \
                'Phenotype not as expected'
        if (r == 'WholeLung' and t == wc):
            assert df['HUMode500'].iloc[i] == -850, 'Phenotype not as expected'
        if (r == 'WholeLung' and t == wc):
            assert df['HUMedian500'].iloc[i] == -850, 'Phenotype not as expected'
        if (r == 'WholeLung' and t == wc):
            assert df['HUMin500'].iloc[i] == -980, 'Phenotype not as expected'
        if (r == 'WholeLung' and t == wc):
            assert df['HUMax500'].iloc[i] == -800, 'Phenotype not as expected'
        if (r == 'WholeLung' and t == wc):
            assert df['Volume'].iloc[i] == 1.8e-05, 'Phenotype not as expected'
        if (r == 'RightLung' and t == wc):
            assert np.isclose(df['Mass'].iloc[i], 0.00247609596), \
                'Phenotype not as expected'

def test_execute2():
    c = ChestConventions()
    wc = c.GetChestWildCardName()

    paren_pheno = ParenchymaPhenotypes(chest_regions=['WholeLung'])
    df = paren_pheno.execute(lm, 'simple', np.array([1., 1., 1.]), ct=ct)
    assert len(df.index) == 1, "Unexpected number of rows in dataframe"
    assert df['Region'].iloc[0] == 'WholeLung', "Unexpected region in dataframe"
    assert df['Type'].iloc[0] == wc, "Unexpected type in dataframe"

def test_execute3():
    c = ChestConventions()
    wc = c.GetChestWildCardName()

    paren_pheno = ParenchymaPhenotypes(chest_types=['Vessel'])
    df = paren_pheno.execute(lm, 'simple', np.array([1., 1., 1.]), ct=ct)
    assert len(df.index) == 1, "Unexpected number of rows in dataframe"
    assert df['Region'].iloc[0] == wc, "Unexpected region in dataframe"
    assert df['Type'].iloc[0] == 'Vessel', "Unexpected type in dataframe"    

def test_execute4():
    c = ChestConventions()
    wc = c.GetChestWildCardName()

    paren_pheno = ParenchymaPhenotypes(pairs=[['LeftLung','Vessel']])
    df = paren_pheno.execute(lm, 'simple', np.array([1., 1., 1.]), ct=ct)

    assert len(df.index) == 1, "Unexpected number of rows in dataframe"
    assert df['Region'].iloc[0] == 'LeftLung', "Unexpected region in dataframe"
    assert df['Type'].iloc[0] == 'Vessel', "Unexpected type in dataframe"    

def test_execute5():
    c = ChestConventions()
    wc = c.GetChestWildCardName()

    paren_pheno = ParenchymaPhenotypes(chest_regions=['LeftLung'], \
                                       pheno_names=['LAA950'])
    df = paren_pheno.execute(lm, 'simple', np.array([1., 1., 1.]), ct)

    assert len(df.index) == 1, "Unexpected number of rows in dataframe"
    assert df['Region'].iloc[0] == 'LeftLung', "Unexpected region in dataframe"
    assert df['Type'].iloc[0] == wc, "Unexpected type in dataframe"   
    assert np.isnan(df.LAA910.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.LAA856.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HAA700.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HAA600.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HAA500.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HAA250.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.Perc15.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.Perc10.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HUMean.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HUStd.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HUKurtosis.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HUSkewness.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HUMode.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HUMedian.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HUMin.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HUMax.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HUMean500.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HUStd500.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HUKurtosis500.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HUSkewness500.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HUMode500.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HUMedian500.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HUMin500.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.HUMax500.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.Volume.iloc[0]), "Phenotype value should be NaN"
    assert np.isnan(df.Mass.iloc[0]), "Phenotype value should be NaN"

def test_execute6():
    paren_pheno = ParenchymaPhenotypes(pairs=[['LeftLung','Vessel']])
    df = paren_pheno.execute(lm, 'simple', np.array([1., 1., 1.]), 
                             pheno_names=['TypeFrac'])

    assert len(df.index) == 1, "Unexpected number of rows in dataframe"
    assert df['Region'].iloc[0] == 'LeftLung', "Unexpected region in dataframe"
    assert df['Type'].iloc[0] == 'Vessel', "Unexpected type in dataframe"    
    assert np.isclose(df['TypeFrac'].iloc[0], 1./9.), \
      'Phenotype value not as expected'

