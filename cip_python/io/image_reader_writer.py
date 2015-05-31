import SimpleITK as sitk

class ImageReaderWriter:
    """
    Interface to the ReaderWriterNRRD program
    
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
        metainfo['origin']=sitk_image.GetOrigin()
        metainfo['spacing']=sitk_image.GetSpacing()
        metainfo['direction']=sitk_image.GetDirection()
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

        sitk.WriteImage(sitk_image,file_name,useCompression=True)

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
        sitk_image.SetOrigin(metainfo['origin'])
        sitk_image.SetDirection(metainfo['direction'])
        self.write(sitk_image,file_name)
