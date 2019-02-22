# -*- encoding: utf-8 -*-
##!/usr/bin/python
from __future__ import division

try:
    import copy_reg
except:
    import copyreg as copy_reg
import math
import os
import types
from multiprocessing import Pool
from cip_python.input_output import ImageReaderWriter
#import nrrd
import numpy as np
import scipy.fftpack
import scipy.io
import scipy.io
import scipy.optimize
from numpy.lib.stride_tricks import as_strided as ast
from scipy import interpolate
from sklearn import neighbors

import numpy as np
import scipy as sci
import scipy.fftpack
import scipy.io
import scipy.optimize

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


class LocalHistogramEmphysema():
  
    def __init__(self,frames,mask,ws=(31,31),ss=(5,5),off=5,num_threads=1,z=2,database=None):
      #Image and mask files 
      self.frames=frames
      self.mask = mask
      #Patch window size
      self.ws = [int(x) for x in ws]
      #Window offset
      self.ss = [int(x) for x in ss]
      #Offset between patches
      self.off = int(off)
      #Database for training the KKN classifier
      self.database =database

      self.X_plot = np.linspace(-1050, 3050, 4096)[:, np.newaxis]
      self.bb=botev_bandwidth(4096,-1050,3050)

      #Classifier: might be interested to test different kNN weight schemes: uniform vs distance
      self.clf = neighbors.KNeighborsClassifier(5, metric='l1',weights='distance')
      #Number of threads in multiprocessing
      self.num_threads = int(num_threads)
      #Offset along z direction
      self.z=int(z)


    ##Defining necessary functions to create the patches##
    def get_win_pixel_coords(self, grid_pos, win_shape, shift_size=None):
      if shift_size is None:
        shift_size = win_shape
      gr, gc = grid_pos
      sr, sc = shift_size
      wr, wc = win_shape
      top, bottom = gr * sr, (gr * sr) + wr
      left, right = gc * sc, (gc * sc) + wc
      return top, bottom, left, right

    def norm_shape(self,shape):
        '''
        Normalize numpy array shapes so they're always expressed as a tuple, 
        even for one-dimensional shapes.
         
        Parameters
            shape - an int, or a tuple of ints
         
        Returns
            a shape tuple
        '''
        try:
            i = int(shape)
            return (i,)
        except TypeError:
            # shape was not a number
            pass
     
        try:
            t = tuple(shape)
            return t
        except TypeError:
            # shape was not iterable
            pass
         
        raise TypeError('shape must be an int, or a tuple of ints')

    def sliding_window(self,a,ws,ss = None,flatten = True):
        '''
        Return a sliding window over a in any number of dimensions
         
        Parameters:
            a  - an n-dimensional numpy array
            ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
                 of each dimension of the window
            ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
                 amount to slide the window in each dimension. If not specified, it
                 defaults to ws.
            flatten - if True, all slices are flattened, otherwise, there is an 
                      extra dimension for each dimension of the input.
         
        Returns
            an array containing each n-dimensional window from a
        '''
         
        if None is ss:
            # ss was not provided. the windows will not overlap in any direction.
            ss = ws
        ws = self.norm_shape(ws)
        ss = self.norm_shape(ss)
         
        # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
        # dimension at once.
        ws = np.array(ws)
        ss = np.array(ss)
        shape = np.array(a.shape)
        
        ws=ws.astype(int)
        ss=ss.astype(int)

         
        # ensure that ws, ss, and a.shape all have the same number of dimensions
        ls = [len(shape),len(ws),len(ss)]
        if 1 != len(set(ls)):
            raise ValueError(\
            'a.shape, ws and ss must all have the same length. They were %s' % str(ls))
         
        # ensure that ws is smaller than a in every dimension
        if np.any(ws > shape):
            raise ValueError(\
            'ws cannot be larger than a in any dimension.\
     a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
         
        # how many slices will there be in each dimension?
        newshape = self.norm_shape(((shape - ws) // ss) + 1)
        # the shape of the strided array will be the number of slices in each dimension
        # plus the shape of the window (tuple addition)
        newshape += self.norm_shape(ws)
        # the strides tuple will be the array's strides multiplied by step size, plus
        # the array's strides (tuple addition)
        newstrides = self.norm_shape(np.array(a.strides) * ss) + a.strides
        strided = ast(a,shape = newshape,strides = newstrides)
        if not flatten:
            return strided
         
        # Collapse strided so that it has one more dimension than the window.  I.e.,
        # the new array is a flat list of slices.
        meat = len(ws) if ws.shape else 0
        firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
        dim = firstdim + (newshape[-meat:])
        # remove any dimensions with size 1
        dim = filter(lambda i : i != 1,dim)
        return strided.reshape(dim)

    def process_slice(self,k):
        print ('Processing slice '+str(k+1)+'/'+str(self.frames.shape[2]))
        image_patches=self.sliding_window(self.frames[:,:,k],self.ws, self.ss, flatten=False)
        m=self.mask[:,:,k]
        m=m>=1
        m=np.rot90(np.fliplr(m))
        
        #Predict each patch class
        labels=[]
        sz=(self.frames.shape[0],self.frames.shape[1])
        image=np.zeros(sz,dtype='uint16')
        for i in range(image_patches.shape[0]):
            for j in range(image_patches.shape[1]):
              t, b, l, r = self.get_win_pixel_coords((i,j), self.ss)
              centerX_p, centerY_p = int(np.rint(t+self.ws[0]/2)), int(np.rint(l+self.ws[1]/2))
              if m[centerY_p, centerX_p]==True:
                image_patch=image_patches[i,j,:,:].ravel()[:, np.newaxis]
                bw,mesh,dens=self.bb.kde_b(image_patch)
                dens1=dens[0:600]
                dens1=dens1/sum(dens1)
                sizeDens=dens.shape
                Dens1=dens1.tolist()
                dens2=sum(dens[600:sizeDens[0]])/sum(dens)  # Value representing the percentage of high-intensity pixels
                Z = self.clf.predict(np.array(Dens1).reshape((1, -1)))                 # Hierarchical classifier. First step
                if ((Z != 1) and (dens2>0.2843)):           # Hierarchical classifier. Second step
                  Z = 2
                labels.append(Z)
                image[int(centerX_p-(self.off/2)):int(centerX_p+(self.off/2+1)),int(centerY_p-(self.off/2)):int(centerY_p+(self.off/2+1))]=Z
    
        #Convert the output labels to cipChestConventions
        #palette = [0,1,2,3,4,5,6] #orginal labels
        palette = np.array([0,1,2,3,4,5,6],dtype='uint16') #orginal labels
        #key=np.array([0,1,67,69,16,17,18]) #cipChestConventions labels
        key=np.array([0,1,67,69,16,17,18],dtype='uint16') #cipChestTypeConventions labels
        key=key<<8  #Convert to Region-Type pair with Region=Undefined (0)

        index=np.digitize(image.ravel(), palette, right=True)
        image=key[index].reshape(image.shape).astype(dtype='uint16')

        image=np.rot90(np.fliplr(image))

        #image=image.astype(dtype='uint16')
        image[~m]=0
        #image=np.multiply(m*1.0,image)
        return image

    def train(self):
      self.database
      ##Creating training set for the KNN Classifier##
      print ('...Creating training set for the KNN Classifier...')
      mat = scipy.io.loadmat(self.database)
      mat2=mat['EmphyPatch'] #type: numpy.ndarray
      X=[] #histograms
      y=[] #corresponding class
      for i in range(6):
        if i!=1: #We train the classifier with 5 classes (all except PS class)
          print ('Running class '+str(i+1))
          patches_size=mat2[0][i].shape
          for j in range(patches_size[2]):
              patch=(mat2[0][i])[:,:,j]
              patch=patch.ravel()[:, np.newaxis]
              bw_train,mesh_train,dens_train=self.bb.kde_b(patch)
              dens1=dens_train[0:600]
              dens1=dens1/sum(dens1)
              sizeDens=dens_train.shape
              dens2=sum(dens_train[600:sizeDens[0]])/sum(dens_train)
              Dens1=dens1.tolist()
              X.append(Dens1)
              y.append(i+1)
      self.clf.fit(X, y)

    def execute(self):
      
        lh.train()
        #output_image=np.zeros((self.frames.shape[0],self.frames.shape[1],self.frames.shape[2]),dtype='uint16')
        
        if self.num_threads > 1:
          p = Pool(int(self.num_threads))
          pp=(p.map(self.process_slice, range(0,self.frames.shape[2],self.z)))
          pp = np.asarray(pp)
        else:
          pp = list()
          for kk in range(0,self.frames.shape[2],self.z):
            pp.append(self.process_slice(kk))
          pp = np.asarray(pp)
        
        pp=pp.transpose([1,2,0])
        ff=interpolate.interp1d(range(0,self.frames.shape[2],self.z),pp,kind='nearest',fill_value=0,axis=2)

        ppn=ff(range(0,self.frames.shape[2]-self.z)).astype('uint16')
        
        #Adding last slices to match the size
        last_slice=ppn[:,:,-1]
        last_slices=np.repeat(last_slice[:, :, np.newaxis], self.frames.shape[2]-(ppn.shape[2]), axis=2)
        output_image=np.append(ppn,last_slices,axis=2)
        
        return np.rot90(np.fliplr(output_image))



class botev_bandwidth():
    """
      Implementation of the KDE bandwidth selection method outline in:
      Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
      estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.
      Based on the implementation of Daniel B. Smith, PhD.
      The object is a callable returning the bandwidth for a 1D kernel.
    """

    def __init__(self, N=None, lower=np.nan, upper=np.nan):
        self.N = N
        self.MIN = lower
        self.MAX = upper

    def kde_b(self, data):
        # Parameters to set up the mesh on which to calculate
        N = 2 ** 14 if self.N is None else int(2 ** sci.ceil(sci.log2(self.N)))
        if self.MIN is None or self.MAX is None:
            minimum = min(data)
            maximum = max(data)
            Range = maximum - minimum
            self.MIN = minimum - Range / 10 if self.MIN is None else self.MIN
            self.MAX = maximum + Range / 10 if self.MAX is None else self.MAX
        # Range of the data
        R = self.MAX - self.MIN
        # Histogram the data to get a crude first approximation of the density
        M = len(data)
        DataHist, bins = sci.histogram(data, bins=N, range=(self.MIN, self.MAX))
        DataHist = DataHist / M
        DCTData = scipy.fftpack.dct(DataHist, norm=None)
        I = [iN * iN for iN in xrange(1, N)]
        SqDCTData = (DCTData[1:] / 2) ** 2
        # The fixed point calculation finds the bandwidth = t_star
        guess = 0.1
        try:
            t_star = scipy.optimize.brentq(self.fixed_point, 0, guess,
                                           args=(M, I, SqDCTData))
        except ValueError:
            print ('Oops!')
            return None
        # Smooth the DCTransformed data using t_star
        SmDCTData = DCTData * sci.exp(-sci.arange(N) ** 2 * sci.pi ** 2 * t_star / 2)
        # Inverse DCT to get density
        density = scipy.fftpack.idct(SmDCTData, norm=None) * N / R
        mesh = [(bins[i] + bins[i + 1]) / 2 for i in xrange(N)]
        bandwidth = sci.sqrt(t_star) * R
        density = density / sci.trapz(density, mesh)
        return bandwidth, mesh, density

    def fixed_point(self, t, M, I, a2):
        l = 7
        I = sci.float128(I)
        M = sci.float128(M)
        a2 = sci.float128(a2)
        f = 2 * sci.pi ** (2 * l) * sci.sum(I ** l * a2 * sci.exp(-I * sci.pi ** 2 * t))
        for s in range(l, 1, -1):
            K0 = sci.prod(xrange(1, 2 * s, 2)) / sci.sqrt(2 * sci.pi)
            const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
            time = (2 * const * K0 / M / f) ** (2 / (3 + 2 * s))
            f = 2 * sci.pi ** (2 * s) * sci.sum(I ** s * a2 * sci.exp(-I * sci.pi ** 2 * time))
        return t - (2 * M * sci.sqrt(sci.pi) * f) ** (-2 / 5)



from optparse import OptionParser
if __name__ == "__main__":
  
    print ("In main")
    desc = """Compute local histogram phenotypes"""
    
    parser = OptionParser(description=desc)
    parser.add_option('-i',help='Input image file',
                      dest='image_file',metavar='<string>',default=None)
    parser.add_option('-m',help='Input mask file',
                      dest='mask_file',metavar='<string>',default=None)
    parser.add_option('-p',help='Patch size. Default=31',
                      dest='patch_size',default=31)
    parser.add_option('-s', help='Offset between patches. Default=5',
                      dest='offset',default=5)
    parser.add_option('-o', help='Output image file', dest='output_image_file',default=None)
    parser.add_option('-t', help='Number of threads in multiprocessing. Default=1',
                      dest='num_threads',default=1)
    parser.add_option('-z', help='Offset along z direction. Default=2',
                      dest='z',default=2)
    (options,args) =  parser.parse_args()

    DB=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../Resources/EmphysemaPatches.mat')
    
    ##Check DB exists
    if os.path.isfile(DB) == False:
      raise ValueError('Database file '+DB+' does not exist')
    
    ##
    if options.image_file is None or options.mask_file is None:
      raise ValueError('Input image file or mask file have not been set')
    
    ##Processing image##
    print ('...Processing Image: '+options.image_file+'...')
    removed=[]


    # Old IO module
    #frames, frames_header = nrrd.read(options.image_file)
    #mask, mask_header = nrrd.read(options.mask_file)
    image_io = ImageReaderWriter()

    frames, frames_header = image_io.read_in_numpy(options.image_file)
    mask, mask_header = image_io.read_in_numpy(options.mask_file)

    masks=[mask>=1, mask!=512]
    mask=reduce(np.logical_and, masks)

    ws=(options.patch_size,options.patch_size)
    ss=(options.offset,options.offset)
    num_threads=options.num_threads
    z=options.z
    print ('## Offset: '+str(options.offset))
    print ('## Patch Size: '+str(options.patch_size))
    print ('## Offset along Z axis: '+str(z))
    print ('## Number of threads in multiprocessing: '+str(num_threads))

    lh = LocalHistogramEmphysema(frames,mask,ws,ss,options.offset,num_threads,z,database=DB)

    lh_image = lh.execute()
    
    ##Writing labeled image##
    print ('...Writing labeled image...')
    #nrrd.write(options.output_image_file,lh_image,options=frames_header)
    image_io.write_from_numpy(lh_image,frames_header,options.output_image_file)

  
    print ('')
    print ('DONE')
