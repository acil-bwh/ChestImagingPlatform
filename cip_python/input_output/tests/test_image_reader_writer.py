import os.path
import numpy as np
from cip_python.input_output.image_reader_writer import ImageReaderWriter
import SimpleITK as sitk
import tempfile,shutil

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

this_dir = os.path.dirname(os.path.realpath(__file__))
file_name = this_dir + '/../../../Testing/Data/Input/simple_ct.nrrd'
tmp_dir=tempfile.mkdtemp()
output_filename = os.path.join(tmp_dir,'simple_ct.nrrd')

def test_reader():
  image_io = ImageReaderWriter()
  sitk_image=image_io.read(file_name)
  
  sitk_test=sitk.Image()
  
  assert type(sitk_image)==type(sitk_test), \
      "Reader does not produce the proper type"

def test_writer():
  image_io = ImageReaderWriter()
  sitk_image=image_io.read(file_name)
  image_io.write(sitk_image,output_filename)
  sitk_image2=image_io.read(output_filename)
  
  #Convert to numpy and evaluate
  np_image = sitk.GetArrayFromImage(sitk_image)
  np_image2 = sitk.GetArrayFromImage(sitk_image2)
  assert np.mean(np_image-np_image2) < np.spacing(1), \
      "Image does not match in read/write operation"

def test_reader_numpy():
  
  image_io = ImageReaderWriter()
  np_image,metainfo=image_io.read_in_numpy(file_name)

  np_test=np.array([])

  assert type(np_image) == type(np_test), \
    "Reader does not produce the proper type"

def test_writer_numpy():
  
  image_io = ImageReaderWriter()
  np_image,metainfo=image_io.read_in_numpy(file_name)
  image_io.write_from_numpy(np_image,metainfo,output_filename)
  np_image2,metainfo2=image_io.read_in_numpy(output_filename)
  
  assert np.mean(np_image-np_image2) < np.spacing(1) and metainfo==metainfo2, \
    "Image does not match in read/write operation"

def test_clean():
  try:
    assert 1==1
  finally:
    shutil.rmtree(tmp_dir)
