import os
from cip_python.phenotypes.emphysema_severity import *
import pandas as pd

test_dataset=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../Resources/EmphysemaSeverity/Testing_emphysemaClassificationPhenotypes.csv')
training_dataset=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../Resources/EmphysemaSeverity/Training_emphysemaClassificationPhenotypes.csv')

in_df_training = pd.read_csv(training_dataset)
in_df_testing = pd.read_csv(test_dataset)

in_df_training=in_df_training[~pd.isnull(in_df_training).any(axis=1)]
in_df_testing=in_df_testing[~pd.isnull(in_df_testing).any(axis=1)]

in_training_features=np.array(in_df_training)[:,1:7]
in_testing_features=np.array(in_df_testing)[:,1:7]

#Test single path
def test_execute():
  
    reference_data=dict()
    reference_data['LHFeatures']=np.array([[0.003436,0.005154,0,0.234385,0.571481,0.013621],
                                           [0.355349,0,0,0.484651,0.092713,0]])
                                          
    reference_data['SN'] = np.array([[16.79109466,-1.75109238],
                                           [9.98064803,-3.36696504]])
                                    
    esi = EmphysemaSeverityIndex(single_path=True)

    esi.fit(in_training_features)

    test = esi.predict(reference_data['LHFeatures'])
                                    
    assert np.allclose(test,reference_data['SN']), 'Severity  metrics in single path mode not as expected'

def test_execute2():
  
    reference_data=dict()
    reference_data['LHFeatures']=np.array([[0.003436,0.005154,0,0.234385,0.571481,0.013621],
                                             [0.355349,0,0,0.484651,0.092713,0]])
      
    reference_data['SN'] = np.array([[ 19.41060621,  -0.66621507],
                                     [ 11.36870741,  -1.68309178]])

    esi = EmphysemaSeverityIndex(interp_method='spline',single_path=False)

    esi.fit(in_training_features)

    test = esi.predict(reference_data['LHFeatures'])

    assert np.allclose(test,reference_data['SN']), 'Severity  metrics in multi path mode not as expected'


