import numpy as np
import pdb
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from numpy.polynomial import polynomial as P
import math
from scipy.interpolate import PchipInterpolator    
 
class EmphysemaSeverityIndex:
    """General purpose class implementing the emphysema severity index.
    Training data with features consisting on 6 emphysema subtype (N, LH_PS,
    LH_PL, LH_CL1, LH_CL2, LH_CL3) are used in order to define either 1 or
    multiple paths (4 to be specific). 
    
    Parameters 
    ----------    
    interp_method: string
        Choice of interpolation method between the different paths
        Choice should be : linear, spline, nearest (so far only spline is
        defined)
        
    multipath: boolean
        Set to True if multiple paths (4 to be specific) as opposed to 1
        severity path is used. 
                             
    Attribues
    ---------
    final_model : dict()
        Contains the model (or models) used to compute the siverity index (lda basis, 
            polynomial, min and max values..).      
    """
    def __init__(self, interp_method="spline", single_path=False):

        self.interp_method = interp_method
        self.single_path = single_path
        self.final_model = dict()
    
        #Internal variables
        self.nClusters = 4
        self.dimSubspace = 3
        #Centers to initialize KNN: numClusters x numFeatures
        self.initial_centers = np.array([[0.7, 0, 0, 0.1, 0, 0],\
                                [0.5, 0.1, 0, 0.2, 0, 0],\
                                [0.2, 0.1, 0, 0.3, 0.2, 0],\
                                [0, 0.1, 0, 0.1, 0.4, 0.1]])
  
    def compute_emphysema_severity_index(self, the_features, the_model):
        """ Compute severity on testing data given a model. 
        
        Parameters
        ----------
        the_features: numpy array
            6 local histogram features used to compute the severity
            
        the_model: dict()
        Contains the model used to compute the siverity index (lda basis,
            polynomial, min and max values..) 
        
        """                                        
        embeddedData = the_model['ldaBasis'].transform(the_features)

        poly_a = the_model['poly'][2]
        poly_b = the_model['poly'][1]
        poly_c = the_model['poly'][0]
        x0 = the_model['min'] # =ldamin;
                      
        ST = np.zeros([np.shape(the_features)[0],2])
        for kk in range(0,np.shape(the_features)[0]):
            [ST[kk,0],ST[kk,1]] = self.computeSTcoordinates(embeddedData[kk,:],
                x0, poly_a, poly_b, poly_c)
        
        return ST
                                                                                                                                                                                                      
    def predict(self, testing_features):
        """Compute severity on testing data given models. Equivalent to
        computeemphysemaSeverityIndex
        
        Parameters
        ----------
        testing_features: Numpy array 6 x N
            6 local histogram features used to compute the severity        
        """                

        if (self.single_path):
            print(" predicting single path severity")
            ST = self.compute_emphysema_severity_index(testing_features,
                                                       self.final_model)
        else:
            print(" using interpolated model to compute severity")
            ST = self.computeSTCoordinateInterpModel(testing_features,
                                                   self.final_model)
            
        return ST
        
    def emphysemaSeverityIndexTrajectories(self, training,nClusters,
        dimSubspace, ctrsinit, model_ref=None):
        """ Build a severity model from training data
        
        Parameters
        ----------
        training: Numpy array 6 x N
            6 local histogram features used to compute the severity
            
        nClusters: int
            Number of k means clusters to be used
            
        dimSubspace: int
            Dimensions of the subspace sought
            
        ctrsinit: numpy array
            Centroids to initialize the k-meams
            
        model_ref: dict()
            A reference model from which to take the lda basis ...        
        """                 
        model = dict()

        # [ldacc,ldaBasis]=lda(cc,cidxpca,nClusters-1);
        # Compute LDA using clusters as groups 
        lda = LDA(n_components=2)
        if model_ref is None:
            mykmeans = KMeans(n_clusters=nClusters, init = ctrsinit)
        
            print("performing kmeans to training data")
            mykmeans_fit =  mykmeans.fit(training)
            result = mykmeans.predict(training)

            #original k means centers, non-reduced space
            model['ctrs'] = mykmeans_fit.cluster_centers_ 
            
            # Create clusters in the dimensional reduced space (PCA) 
            pca = PCA(n_components=dimSubspace)
            pca.fit(training)
            
            tcc = pca.transform(training)
            model['tctrs'] = pca.transform(model['ctrs'])
            
            mykmeans_pca = KMeans(n_clusters=nClusters, init = model['tctrs'])
            mykmeans_pca_fit = mykmeans_pca.fit(tcc)

            # this is almost identical to matlab results
            model['ctrspca'] = mykmeans_pca_fit.cluster_centers_ 
            
            cidxpca = mykmeans_pca_fit.predict(tcc)
            lda.fit(training, cidxpca)
        else:
            lda = model_ref['ldaBasis']

        ldacc = lda.transform(training)#, cidxpca) #close
        
        # Check if we have to flip axis. We assume that cluster 1 has less
        # severity and final cluster is more severity
        if model_ref is None:
            if np.mean(ldacc[cidxpca==0,0]) > \
              np.mean(ldacc[cidxpca==nClusters,0]):
                pdb.set_trace()
                ldacc[:,0]=-ldacc[:,0]
                model['flip']=-1
            else:
                model['flip']=1
        else:
            ldacc[:,0]=model_ref['flip']*ldacc[:,0]

        model['ldaBasis'] = lda
        
        # Fitting model to data: First two LDA components 
        ldamin=np.min(ldacc[:,0]);
        ldamax=np.max(ldacc[:,0]);
        
        flda = P.polyfit(ldacc[:,0], ldacc[:,1], 2)
        # p(x) = c0+c1*x+c2*x^2, coef = [c0,c1,c2]
        model['poly'] = [flda[2],flda[1], flda[0]] ; #?
        model['min'] =ldamin;
        model['max'] =ldamax;

        # Map all points to Severity space (S-T) Space
        x0 = ldamin
        poly_a = flda[0]  # equivalent to ['poly'][2] 
        poly_b = flda[1]
        poly_c = flda[2] # this way it multiplies x^2

        
        ST = np.zeros([np.shape(ldacc)[0],2])
        for kk in range(0,np.shape(ldacc)[0]):
            [ST[kk,0],ST[kk,1]] = self.computeSTcoordinates(ldacc[kk, :], x0,
                poly_a, poly_b, poly_c)
        
        model['minST'] = [np.min(ST[:,0]),np.min(ST[:,1])]
        model['maxST'] = [np.max(ST[:,0]),np.max(ST[:,1])]

        return model

    def computeSTcoordinates(self, p,x0,a,b,c):
        """ Compute the severity and normal values given polynomial
        coefficients
        
        Parameters
        ----------
        p: Numpy array 2 x 1
            pointin LDA space
            
        x0,a,b,c: each int an int
            polynomial coefficients, where the polynomial is:
            p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]    

        Returns
        -------
        
        [S,T]: List 2 x 1
            Severity and Normal values
                    
        """                             
        coeff = [2*np.square(c), 3*b*c, 1+2*c*a + np.square(b) - \
            2*c*p[1],a*b-p[1]*b-p[0]]

        # Get the roots of the polynomial
        res = np.roots(coeff)
        
        # Get index real solution 
        idx=np.where(res.imag==0)

        # Check if we don't have real roots 
        if (idx == []):
            S = np.nan
            T = np.nan
            return([S, T])
            
        xp = np.real(res[idx[0][0]])
        yp = a + b*xp + c*np.square(xp)

        T = (yp-p[1])*np.sqrt(1 + np.square(b+2*c*xp))

        # arclength between point projected on curve and curve starting point 
        S = self.arclengthpoly(xp, a, b, c) - self.arclengthpoly(x0, a, b, c)
        
        return ([S,T])

    def arclengthpoly(self, x,a,b,c):
        """ arc length of a second order polynomial
        
        Parameters
        ----------
        x: Numpy array 2 x 1
            pointin LDa space
            
        x0,a,b,c: each int an int
            polynomial coefficients, where the polynomial is:
            p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]    

        Returns
        -------
        
        ss: float
            arc length value
        """            
        ss=((b + 2 * c*x) * np.sqrt(1 + np.square(b + 2 * c * x)) + \
            math.asinh(b + 2* c * x))/(4*c)

        return ss
          
    def evalInterpModel(self, interp_model, t_val):
        """ Evaluate the interpolated model at a specific parameter point.
        
        Parameters
        ----------
        interp_model: dict
            An interpolation model, consisting of a set of coefficients to
            interpolate from
            
        t_val: float
            Interpolation parameter

        Returns
        -------
        
        modeli: dict()
            model interpolated at parameter t_val
        """              
        interp_method=self.interp_method
        modeli = dict()
        modeli['poly'] = [0,0,0]

        if (interp_method == 'spline'):
            modeli['poly'][0] = PchipInterpolator(interp_model['Ti'],
                interp_model['ai'], extrapolate=True)(t_val)[0]
            modeli['poly'][1] = PchipInterpolator(interp_model['Ti'],
                interp_model['bi'], extrapolate=True)(t_val)[0]
            modeli['poly'][2] = PchipInterpolator(interp_model['Ti'],
                interp_model['ci'], extrapolate=True)(t_val)[0]
 
        modeli['min']=interp_model['min']
        modeli['max']=interp_model['max']
        modeli['ldaBasis']=interp_model['ldaBasis']
        modeli['ctrs']=interp_model['ctrs'];

        return modeli
                                                    
    def optimize_interp_model(self,p,interp_model):                    
        """Perform the optimization in order to obtain the interpolated model
        at point p in LDA space.
        
        Parameters
        ----------
        p: Numpy array 2 x 1
            Point in LDA space
            
        interp_model: dict
            An interpolation model, consisting of a set of coefficients to
            interpolate from.
            
        Returns
        -------        
        t_param: float
            Optimal interpolation parameter for point
        """          
        global_t_param = 0
        import scipy.optimize as op

        res  = op.minimize(fun = self.compute_local_N_given_global_t_param,
            x0 = global_t_param, args = (p,interp_model),
            method='Nelder-Mead')
        t_param = res.x

        return t_param
        
    def compute_local_N_given_global_t_param(self, t_value, p, im):
        """Compute model specific N given global params
        
        Parameters
        ----------
        p: Numpy array 2 x 1
            Point in LDA space
        
        t_value: float
            Global param
                    
        im: dict
            The interpolated model 
            
        Returns
        -------        
        local_N: float
            Corresponding local normal
        """  
        
        model_ii = self.evalInterpModel(im, t_value)

        [Local_S,local_N] = self.computeSTcoordinates(p, model_ii['min'],
            model_ii['poly'][2], model_ii['poly'][1], model_ii['poly'][0])


        #computeSTcoordinates(self, p,x0,a,b,c)
        local_N=abs(local_N)
        return local_N    
        
    def computeSTCoordinateInterpModel(self, the_features, interp_model):
        """Compute ST values given the features and the interpolated model
        
        Parameters
        ----------
        the_features: Numpy array 6 x N
            6 local histogram features used to compute the severity
            
        interp_model: dict
            An interpolation model, consisting of a set of coefficients to
            interpolate from.
            
        Returns
        -------        
        [S,T]: List 2 x 1
            Severity and Normal values
        """  
        embeddedData = interp_model['ldaBasis'].transform(the_features)   
        ST = np.zeros([np.shape(the_features)[0],2])

        for kk in range(0,np.shape(the_features)[0]):
            if(np.mod(kk,500)==0):
                print(kk)
            p = embeddedData[kk,:]
            t_opt = self.optimize_interp_model(p,interp_model);

            model_ii = self.evalInterpModel(interp_model,t_opt);
            [ST[kk,0], ST[kk,1]] = self.computeSTcoordinates(p,
                model_ii['min'], model_ii['poly'][2], model_ii['poly'][1],
                model_ii['poly'][0])
            # [S,T]=computeSTcoordinates(p,model_ii.min,model_ii.poly.p3 ,model_ii.poly.p2, model_ii.poly.p1);

        return ST
                        
    def computeInterpModel(self, models,model0):
        """Compute ST values given the features and the interpolated model.
        
        Parameters
        ----------
        model0: dict()
            An initial model used to get certain elements from.
            
        models: dict
            An interpolation model, consisting of a set of coefficients to
            interpolate from.
            
        Returns
        -------        
        [S,T]: List 2 x 1
            Severity and Normal values
        """          
        ai = np.zeros(4, dtype = float)
        bi = np.zeros(4, dtype = float)
        ci = np.zeros(4, dtype = float)
        x0i = np.zeros(4, dtype = float) 
        ye = np.zeros(4, dtype = float) 
        
        for kk in range(0,4):
            ai[kk] = models[kk]['poly'][0]
            bi[kk] = models[kk]['poly'][1]
            ci[kk] = models[kk]['poly'][2]
            x0i[kk] = models[kk]['min']
         
        xe = -bi/(2*ai)
        for kk in range(0,4):
            ye[kk] = ci[kk] + bi[kk]*xe[kk] + ai[kk]*np.square(xe[kk])
            #yp=a+b*xp+c*np.square(xp), no opposite here
            
        #Compute ST values for the extremas using baseline model (model1)
        a = model0['poly'][2]
        b = model0['poly'][1]
        c = model0['poly'][0]
        x0 = model0['min']
        STe = np.zeros([4,2]);

        for kk in range(0,4):
            [STe[kk,0], STe[kk,1]] = \
              self.computeSTcoordinates([xe[kk], ye[kk]], x0, a, b, c)
        
        interp_model = dict()
        Ti=STe[:,1]
        
        # order Ti in ascending order, then order the remaining coefficients accordingly
        args = np.argsort(Ti)
        interp_model['ai'] = ai[args]
        interp_model['bi'] = bi[args]
        interp_model['ci'] = ci[args]
        interp_model['Ti'] = Ti[args]
        interp_model['min'] = model0['min']
        interp_model['max'] = model0['max']
        interp_model['ldaBasis'] = model0['ldaBasis']
        interp_model['ctrs'] = model0['ctrs']
        interp_model['model1'] = model0    

        return interp_model
                                                                                                                                                                                                                                                                
    def fit(self, training_features):
        """Obtain the models given the training data
        
        Parameters
        ----------
        training_features_df: pandas dataframe
            case_id followed by 6 local histogram features proportions used to
            compute the severity    
        """              
        
        # Build level 1 model  
        if (self.single_path):
            model = [None]
            ST = [None]
        else:
            model = [None]*31 # the last 4 are the ones we need
            ST = [None]*31
                        
        model0 = self.emphysemaSeverityIndexTrajectories(training_features,
            self.nClusters, self.dimSubspace, self.initial_centers)
        ST[0] = \
          self.compute_emphysema_severity_index(training_features, model0)

        # If multipath compute S values and split, twice. 
        if (self.single_path):
            self.final_model = model0 
            print("single path")
        else:
            print("building multipath model with the training data")
                    
            # Build 4 models with the training data model is now an
            # interpolated model
            model[0] = model0
            num_levels = 4
            for level_num in range (1, (num_levels+1)):
                for node_in_level in range (0, np.power(2,level_num)):
                    # Because we are starting at node 0 in python
                    current_node_number = np.power(2, level_num) + \
                      node_in_level - 1
                    current_ST = current_node_number
                    ST_to_split = \
                      int(np.floor((np.float(current_ST) + 1.0)/2.0)) - 1
                    st_inclusion = np.ones(np.shape(ST[0])[0])

                    # not counting root node, root node is 0
                    for current_split_level in range(level_num, 0, -1):
                        # to the left -> I am an uppser split -> set lower
                        # split to 0
                        if ((np.float(current_ST+1)/\
                             np.float(ST_to_split+1)) == 2):
                            st_inclusion[ST[ST_to_split][:,1]<0] = 0
                        else:
                            st_inclusion[ST[ST_to_split][:,1]>0] = 0

                        current_ST = ST_to_split;
                        ST_to_split = \
                          int(np.floor((np.float(current_ST)+1.0)/2.0)) - 1
                        
                    # Extract ST data based on the final inclusion values
                    ttfinal = st_inclusion>0 
                    
                    model[current_node_number] = \
                      self.emphysemaSeverityIndexTrajectories(\
                        training_features[ttfinal,:], self.nClusters,
                        self.dimSubspace, self.initial_centers, model0)

                    ST[current_node_number] = \
                      self.compute_emphysema_severity_index(training_features,
                        model[current_node_number])
                    
            final_models = [model[1+14], model[6+14], model[10+14], model[15+14]]

            # Compute the interpolated model from the 4 models
            im = self.computeInterpModel(final_models,model0)      

            self.final_model = im
            
if __name__ == "__main__":  
    from argparse import ArgumentParser
    import os
    desc = """Computes the emphysema severity index. Training data with
    features consisting on 6 emphysema subtype (N, LH_PS, LH_PL, LH_CL1,
    LH_CL2, LH_CL3) are used in order to define either 1 or multiple paths (4
    to be specific)."""
    
    test_dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        '../../Resources/EmphysemaSeverity/\
    Testing_emphysemaClassificationPhenotypes.csv')
    training_dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        '../../Resources/EmphysemaSeverity/\
    Training_emphysemaClassificationPhenotypes.csv')    
    
    parser = ArgumentParser(description=desc)
    parser.add_argument('-i', help='Input features file with the existing \
        features consisting on 6 emphysema subtypes (N, LH_PS, LH_PL, LH_CL1, \
        LH_CL2, LH_CL3).', dest='in_csv', metavar='<csv file>', type= str,
        default=test_dataset)
    parser.add_argument('-o', help='Output csv file with the severity values',
        type=str, dest='out_csv', metavar='<csv file>', default=None)
    parser.add_argument('--interp_method', help='Interpolation method using \
        for the patient-specific path. Only spline defined so far', \
        dest='interp_method', metavar='<string>', type=str, default="spline",
        choices=['linear', 'spline', 'nearest'])
    parser.add_argument('--single_path', action='store_true',
        dest='single_path', default=False, help='Set to True for using 1 \
        severity path instead of multiple (4) paths')
    parser.add_argument('--col_idx', metavar='min<int>,max<int>',type=lambda s: [int(item) for item in s.split(',')], default='1,6',dest='col_idx',\
                        help='column indexes (0-based and comman separated) for the LH quantities')
    parser.add_argument('--training',
                        help='Training features file with the existing features consisting on 6 emphysema\
                        subtypes (N, LH_PS, LH_PL, LH_CL1, LH_CL2, LH_CL3) .',
                        dest='training_csv', metavar='<csv file>', default= \
                        training_dataset)
    parser.add_argument('--training_col_idx', metavar='min<int>,max<int>',type=lambda s: [int(item) for item in s.split(',')], default='1,6',dest='training_col_idx',\
                        help='column indexes (0-based and comman separated) for the LH quantities in the training dataset')
                      
    options = parser.parse_args()

    in_df_training = pd.read_csv(options.training_csv)

    try:
        print ("Reading testing features file...")
        in_df = pd.read_csv(options.in_csv)  
    except IOError as e:
        print (e.errno)
        print (e)
        exit()
        
    if (options.interp_method != 'spline'):
        print("interpolation not defined")
        exit()
            
    """ mask bad values """
    in_df_training=in_df_training[~pd.isnull(in_df_training).any(axis=1)]
    in_df=in_df[~pd.isnull(in_df).any(axis=1)]
    
    in_df_testing = in_df 
    
    in_training_features = np.array(in_df_training)[:,int(options.training_col_idx[0]):int(options.training_col_idx[1])+1]
    in_testing_features = np.array(in_df_testing)[:,int(options.col_idx[0]):int(options.col_idx[1])+1]
    print (in_testing_features.shape)
    
    """ fit and predict """
    my_severity_index = EmphysemaSeverityIndex(interp_method=options.interp_method, single_path=options.single_path)
    
    my_severity_index.fit(in_training_features)
    
    severity_values = my_severity_index.predict(in_testing_features)

    """ build dataframe including case id, features, and st values """
    severity_values_df = pd.DataFrame(severity_values, index=in_df_testing.index.values, columns=['S', 'N'])
    out_df = pd.concat([in_df_testing, severity_values_df], axis=1)    
    
    if options.out_csv is not None:
        print ("Writing...")
        out_df.to_csv(options.out_csv, index=False)
        
