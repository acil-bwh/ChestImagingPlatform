import numpy as np

"""
Helper functions to extract patches of a certain size from a volume given a 
center and an extent.
@TODO: consider to move this to utils, as it looks like a pretty general utility
"""

class Patcher(object):
    @staticmethod
    def get_bounds_from_center(image, patch_center,extent):
        """
        Gets the bounding box of a patch given its center and extent

        Parameters
        ----------
        image: 3D numpy array, shape (L, M, N)
            Input CT or labelmap from which we want to extract a patch

        patch_center: list, shape (3,1)
            x,y,z coordinates of the center of the patch

        extent: list, shape (3,1)
            x,y,z patch extent

        Returns
        -------
        bounds: list, shape (6,1)
            x,y,z min and max patch coordinates. The max coordinate returned
            is actually max+1 (i.e. exclusive).
        """

        x_half_length = np.floor(extent[0]/2)
        y_half_length = np.floor(extent[1]/2)
        z_half_length = np.floor(extent[2]/2)

        assert ((x_half_length*2 <= np.shape(image)[0]) and \
                (y_half_length*2 <= np.shape(image)[1]) and \
                (z_half_length*2 <= np.shape(image)[2])), "region extent must \
                be less that image dimensions. extent="+str(np.shape(image)[1])+" "+str(y_half_length*2)

                #+x_half_length*2+","+\
                #y_half_length*2+","+z_half_length*2+" image shape="+np.shape(image)[0]+\
                #","+ np.shape(image)[1]+","+ np.shape(image)[2]

        xmin = max(patch_center[0]-x_half_length,0)
        xmax =  min(patch_center[0]+x_half_length+1,np.shape(image)[0])
        ymin = max(patch_center[1]-y_half_length,0)
        ymax = min(patch_center[1]+\
                        y_half_length+1,np.shape(image)[1])
        zmin = max(patch_center[2]-z_half_length,0)
        zmax = min(patch_center[2]+\
                        z_half_length+1,np.shape(image)[2])

        bounds = [int(xmin),int(xmax),int(ymin),int(ymax),int(zmin),int(zmax)]

        #if (ymax<ymin):#( ((xmax-xmin)>31) or ((ymax-ymin)>31)  or ((zmax-zmin)>31) ):
        #    pdb.set_trace()
        return bounds

    @staticmethod
    def get_patch_given_bounds(image, bounds):
        """
        Extract an image patch given the image and patch bounds

        Parameters
        ----------
        image: 3D numpy array, shape (L, M, N)
            Input CT or labelmap from which we want to extract a patch

        bounds: list, shape (6,1)
            x,y,z coordinates bounds of the patch


        Returns
        -------
        image_patch: 3D numpy array, shape (O, P, Q)
            Output patch. Outputs None if the bounds are invalid
        """

        image_patch = None

        # check that the bounds are between 0 and the image extent
        if ((bounds[0]>=0) and (bounds[0] <= np.shape(image)[0]) and \
            (bounds[1] >0) and (bounds[1] <= np.shape(image)[0]) and \
            (bounds[2]>=0) and (bounds[2] <= np.shape(image)[1]) and \
            (bounds[3]>=0) and (bounds[3] <= np.shape(image)[1]) and \
            (bounds[4]>=0) and (bounds[4] <= np.shape(image)[2]) and\
            (bounds[5]>=0) and (bounds[5] <= np.shape(image)[2])  )  :

            image_patch = np.squeeze(image[bounds[0]:bounds[1], bounds[2]:bounds[3],\
                bounds[4]:bounds[5]] )

        return image_patch
