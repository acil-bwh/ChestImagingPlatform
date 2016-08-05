import numpy as np


from cip_python.input_output import ImageReaderWriter
from cip_python.common import ChestConventions, Paths
from cip_python.phenotypes import LAAPhenotypes

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
    
    laa = LAAPhenotypes()    
    df = laa.execute(ct, lm, 'simple')

    for i in xrange(0, 14):
        r = df['Region'].iloc[i]
        t = df['Type'].iloc[i]
        val_950 = df['LAA950'].iloc[i]
        val_910 = df['LAA910'].iloc[i]
        val_856 = df['LAA856'].iloc[i]
        if (r == 'UNDEFINEDREGION' and t == wc) or \
            (r == wc and t == 'UNDEFINEDTYPE') or \
            (r == 'UNDEFINEDREGION' and t == 'UNDEFINEDTYPE'):
            assert val_950 == wc, 'Phenotype not as expected'
            assert val_910 == wc, 'Phenotype not as expected'
            assert val_856 == wc, 'Phenotype not as expected'
        elif (r == 'WHOLELUNG' and t == wc) or \
            (r == 'RIGHTLUNG' and t == wc) or \
            (r == 'LEFTLUNG' and t == wc):
            assert np.isclose(val_950, 0.1111111), 'Phenotype not as expected'
            assert np.isclose(val_910, 0.1111111), 'Phenotype not as expected'
            assert np.isclose(val_856, 0.1111111), 'Phenotype not as expected'
        elif (r == wc and t == 'AIRWAY') or \
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
            (r == wc and t == 'VESSEL'):
            assert np.isclose(val_950, 0.0), 'Phenotype not as expected'
            assert np.isclose(val_910, 0.0), 'Phenotype not as expected'
            assert np.isclose(val_856, 0.0), 'Phenotype not as expected'


