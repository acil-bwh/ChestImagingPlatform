import SimpleITK as sitk

class ImageReaderWriter:
    """
    Interface to the ImageReaderWriter program
    
    Parameters
    ----------
    
    """
    
    def __init__(self):
        self._np_axes_order=[2,1,0]

    def read(self,file_name):
        """Read image file into SimpleITK image.
          
          Parameters
          ----------
          file_name: string
            Image file name
            
          Returns
          -------
          sitk_image: sitk.Image
            SimpleITK image object
        """
        
        sitk_image=sitk.ReadImage(file_name)
        return sitk_image

    def read_in_numpy(self,file_name):
        """Read image file into numpy array.
        
        Parameters
        ----------
          file_name: string
            Image file name
        
        Returns
        -------
          np_array: array
            Image in numpy array
          metainfo: dictionary
            Meta information dictionary that stores spacing,origin,direction
        """
          
        sitk_image=self.read(file_name)
        np_array = sitk.GetArrayFromImage(sitk_image)
        np_array = np_array.transpose(self._np_axes_order)
        metainfo=dict()
        metainfo['space origin']=sitk_image.GetOrigin()
        metainfo['spacing']=sitk_image.GetSpacing()
        metainfo['space directions']=sitk_image.GetDirection()
        #Include additional metadata specific to format
        for key in sitk_image.GetMetaDataKeys():
          metainfo[key]=sitk_image.GetMetaData(key)
        
        return np_array,metainfo

    def write(self,sitk_image,file_name):
        """Write SimpleITK image.
        
          Parameters
          ----------
            sitk_image: sitk.Image
              SimpleITK image object
            file_name: string
              File name
        
        """

        sitk.WriteImage(sitk_image,file_name,True)

    def numpy_to_sitkImage(self,npy_array,metainfo=None,sitk_image_tempate=None):
    
      sitk_image=sitk.GetImageFromArray(npy_array.transpose(self._np_axes_order))
      if sitk_image_tempate == None and metainfo == None:
          pass
      elif metainfo == None:
          sitk_image.CopyInformation(sitk_image_tempate)
      else:
          sitk_image.SetSpacing(metainfo['spacing'])
          sitk_image.SetOrigin(metainfo['space origin'])
          sitk_image.SetDirection(metainfo['space directions'])

      return sitk_image

    def sitkImage_to_numpy(self,sitk_image):
      np_array = sitk.GetArrayFromImage(sitk_image)
      np_array = np_array.transpose(self._np_axes_order)
      return np_array

    def write_from_numpy(self,npy_array,metainfo,file_name):
        """Write an image numpy array.
        
          Parameters
          ----------
            np_array: array
              Image numpy array
            metainfo: dictionary
              Meta information dictionary (spacing,origin,direction)
            file_name: string 
              Output image file name
        
        """
        sitk_image=sitk.GetImageFromArray(npy_array.transpose(self._np_axes_order))
        sitk_image.SetSpacing(metainfo['spacing'])
        sitk_image.SetOrigin(metainfo['space origin'])
        sitk_image.SetDirection(metainfo['space directions'])
        self.write(sitk_image,file_name)
