import os.path
import pandas as pd
import nrrd
from cip_python.phenotypes.laa_phenotypes import *
import pdb

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

this_dir = os.path.dirname(os.path.realpath(__file__))
lm_name = this_dir + '/../../../Testing/Data/Input/simple_lm.nrrd'
lm, lm_header = nrrd.read(lm_name)
ct_name = this_dir + '/../../../Testing/Data/Input/simple_ct.nrrd'
ct, ct_header = nrrd.read(ct_name)

def test_execute_1():
    laa = LAAPhenotypes()    
    df = laa.execute(ct, lm, 'simple')

    for i in xrange(0, 18):
        r = df['Region'].iloc[i]
        t = df['Type'].iloc[i]
        val_950 = df['LAA-950'].iloc[i]
        val_910 = df['LAA-910'].iloc[i]
        val_856 = df['LAA-856'].iloc[i]
        if (r == 'UNDEFINEDREGION' and t == 'NaN') or \
            (r == 'NaN' and t == 'UNDEFINEDTYPE') or \
            (r == 'UNDEFINEDREGION' and t == 'UNDEFINEDTYPE'):
            assert val_950 == 'NaN', 'Phenotype not as expected'
            assert val_910 == 'NaN', 'Phenotype not as expected'
            assert val_856 == 'NaN', 'Phenotype not as expected'
        elif (r == 'WHOLELUNG' and t == 'NaN') or \
            (r == 'RIGHTLUNG' and t == 'NaN') or \
            (r == 'LEFTLUNG' and t == 'NaN'):
            assert np.isclose(val_950, 0.1111111), 'Phenotype not as expected'
            assert np.isclose(val_910, 0.1111111), 'Phenotype not as expected'
            assert np.isclose(val_856, 0.1111111), 'Phenotype not as expected'
        elif (r == 'NaN' and t == 'AIRWAY') or \
            (r == 'UNDEFINEDREGION' and t == 'AIRWAY') or \
            (r == 'WHOLELUNG' and t == 'AIRWAY') or \
            (r == 'RIGHTLUNG' and t == 'AIRWAY') or \
            (r == 'LEFTLUNG' and t == 'AIRWAY'):
            assert np.isclose(val_950, 1.0), 'Phenotype not as expected'
            assert np.isclose(val_910, 1.0), 'Phenotype not as expected'
            assert np.isclose(val_856, 1.0), 'Phenotype not as expected'
        elif (r == 'WHOLELUNG' and t == 'UNDEFINEDTYPE') or \
            (r == 'RIGHTLUNG' and t == 'UNDEFINEDTYPE') or \
            (r == 'LEFTLUNG' and t == 'UNDEFINEDTYPE') or \
            (r == 'WHOLELUNG' and t == 'VESSEL') or \
            (r == 'RIGHTLUNG' and t == 'VESSEL') or \
            (r == 'LEFTLUNG' and t == 'VESSEL') or \
            (r == 'NaN' and t == 'VESSEL'):
            assert np.isclose(val_950, 0.0), 'Phenotype not as expected'
            assert np.isclose(val_910, 0.0), 'Phenotype not as expected'
            assert np.isclose(val_856, 0.0), 'Phenotype not as expected'

def test_execute_2():
    laa = LAAPhenotypes()    
    df = laa.execute(ct, lm, 'simple', chest_regions=np.array([1]),
                     threshs=np.array([-960]))
    assert np.isclose(df['LAA-960'].iloc[0], 0.055555), \
        'Phenotype is not as expected'

