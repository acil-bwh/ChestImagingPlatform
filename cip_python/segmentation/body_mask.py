
import numpy as np
import cv2
from scipy.ndimage import generate_binary_structure
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from argparse import ArgumentParser
import SimpleITK as sitk
from cip_python.input_output import ImageReaderWriter


def getBodyMask(ct_arr):
    bd_msk = (ct_arr > -250).astype("uint8")
    struct = generate_binary_structure(3, 1)
    out = binary_erosion(bd_msk, structure=struct, iterations=3)
    bd_msk_close = binary_dilation(out, structure=struct, iterations=10).astype("uint8")
    bd_msk_filled = np.zeros_like(bd_msk_close)
    for ix in range(bd_msk_close.shape[-1]):
        slc = bd_msk_close[:, :, ix]
        contours, _ = cv2.findContours(slc, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cnt_mask = np.zeros(slc.shape, dtype="uint8")
        for cnt in contours:
            cv2.drawContours(image=cnt_mask, contours=[cnt], color=1, thickness=-1, contourIdx=-1)
        bd_msk_filled[:, :, ix] = cnt_mask
    struct[:,:,0] = 0
    struct[:,:,2] = 0
    bd_msk_filled = binary_erosion(bd_msk_filled, structure=struct, iterations=10)
    return bd_msk_filled.astype("uint16")




if __name__ == '__main__':
  desc = "Extract body mask from chest CT"

  parser = ArgumentParser(description=desc)
  parser.add_argument('-i', '--input_file', required=True, help='Input image volume file')
  parser.add_argument('-o', '--output_file', required=True, help='Output body mask')


  args = parser.parse_args()
  image_io = ImageReaderWriter()
  ct, ct_header = image_io.read_in_numpy(args.input_file)
  out_img=getBodyMask(ct)

  lm = image_io.write_from_numpy(out_img,ct_header,args.output_file)
