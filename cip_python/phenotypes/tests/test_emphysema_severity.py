import numpy as np
import pandas as pd
import pdb
from cip_python.phenotypes.emphysema_severity import EmphysemaSeverityIndex
from cip_python.common import Paths


test_dataset = Paths.resources_file_path(\
    'EmphysemaSeverity/Testing_emphysemaClassificationPhenotypes.csv')
training_dataset = Paths.resources_file_path(\
    'EmphysemaSeverity/Training_emphysemaClassificationPhenotypes.csv')

in_df_training = pd.read_csv(training_dataset)
in_df_testing = pd.read_csv(test_dataset)

in_df_training = in_df_training[~pd.isnull(in_df_training).any(axis=1)]
in_df_testing = in_df_testing[~pd.isnull(in_df_testing).any(axis=1)]

in_training_features = np.array(in_df_training)[:,1:7]
in_testing_features = np.array(in_df_testing)[:,1:7]

#Test single path
def test_execute():
    """ """  
    reference_data=dict()
    
    reference_data['LHFeatures'] = \
      np.array([[0.003436, 0.005154, 0, 0.234385, 0.571481, 0.013621],
            [0.355349, 0, 0, 0.484651, 0.092713, 0]])
    
    reference_data['SN'] = np.array([[16.79392487, -1.76907544],
                                     [ 9.98989526, -3.40842286]])

    esi = EmphysemaSeverityIndex(single_path=True)

    esi.fit(in_training_features)

    test = esi.predict(reference_data['LHFeatures'])
                                    
    assert np.allclose(test,reference_data['SN']), \
      'Severity  metrics in single path mode not as expected'

def test_execute2():
    """ """
    reference_data = dict()
    reference_data['LHFeatures'] = \
      np.array([[0.003436, 0.005154, 0, 0.234385, 0.571481, 0.013621],
        [0.355349, 0, 0, 0.484651, 0.092713, 0]])
      
    reference_data['SN'] = np.array([[2.02124626e+01,  2.81663549e-05],
                                     [ 1.05977163e+01, -3.92402035e-06]])

    esi = EmphysemaSeverityIndex(interp_method='spline', single_path=False)

    esi.fit(in_training_features)

    test = esi.predict(reference_data['LHFeatures'])
#pdb.set_trace()
    assert np.allclose(test, reference_data['SN']), \
      'Severity metrics in multi path mode not as expected'


