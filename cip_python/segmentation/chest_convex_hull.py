from skimage.morphology import convex_hull_image,erosion,disk
from skimage import data, img_as_float
from skimage.util import invert
import SimpleITK as sitk
import numpy as np
from optparse import OptionParser



def convex_2d(im,footprint=disk(1)):
		footprint = disk(1)
		chull=convex_hull_image(im).astype('uint16')
		return erosion(chull-im,footprint)


desc = """Generates chest convex hull from lung label map"""

parser = OptionParser(description=desc)
parser.add_option('-i',
                  help='Input whole lung label map file', dest='in_lm', metavar='<string>',
                  default=None)
parser.add_option('-o',
                  help='Convex hull output label map', dest='out_ch', metavar='<string>',
                  default=None)

 
(options, args) = parser.parse_args()

#im_file='20121009_INSP_B65s_L1_Cotton_wl.nrrd'

im=sitk.ReadImage(options.in_lm)

im_np=sitk.GetArrayFromImage(im)


sz=im_np.shape
footprint = disk(1)
out=np.zeros(im_np.shape).astype('uint16')

## Axial parsing
for zz in range(sz[0]):
	if np.sum(im_np[zz,:,:].ravel())>0:
		out[zz,:,:]=convex_2d(im_np[zz,:,:])

##Sag parsing
#out2=np.zeros(im_np.shape).astype('uint16')
#for yy in range(sz[2]):
#	if np.sum(im_np[:,:,yy].ravel())>0:
#		out2[:,:,yy]=convex_2d(im_np[:,:,yy])

#Axial parsion
#for xx in range(sz[1]):
#	if np.sum(im_np[:,xx,:].ravel())>0:
#		chull=convex_hull_image(im_np[:,xx,:]).astype('uint16')
#		out2[:,xx,:]=erosion(chull-im_np[:,xx,:],footprint)


out=sitk.GetImageFromArray(out)
out.CopyInformation(im)

sitk.WriteImage(out,options.out_ch,True)