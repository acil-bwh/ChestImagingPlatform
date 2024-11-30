#!/usr/bin/python

import subprocess
import os
import sys
from lxml import etree
import time

from optparse import OptionParser

"""performs affine and non-rigid registreation between 2 CT volumes using ANTS
"""

#TODO:
#Give an option to resample outputs
#Give an option to compute a quality control image
#allow for other similarity measures than MI
#store similarity value 



class ANTSRegistration():
    def __init__(self):
        toolsPaths = ['TEEM_PATH','ITKTOOLS_PATH','ANTS_PATH']
        self.path=dict()
        for path_name in toolsPaths:
            self.path[path_name]=os.environ.get(path_name,False)
            if self.path[path_name] == False:
                print path_name + " environment variable is not set"
                    #Raise exception


  def transform_data ( reference_im, moving_im, transforms_list, output):
    """Transform the oving image to the reference image using each transform
          in the list of transforms.  
      
      Parameters
      ----------
      reference_im: String
              Reference image file name

      moving_im: String
              Moving image file name
              
      transforms_list : Liost of String
              Names of transform files to apply
          ...
          
      Returns
      -------
      None
      
    """
          
    tmp_command = "antsApplyTransforms -d 3 -i %(m)s -r %(f)s -n linear -o %(deformed)s"
    tmp_command = tmp_command % {'f':reference_im,'m':moving_im,'deformed':output}
    for tt in transforms_list:
      tmp_command = tmp_command + " -t %(tt)s"
      tmp_command = tmp_command % {'tt':tt}
     
      tmp_command = os.path.join(self.path['ANTS_PATH'],tmp_command)
      subprocess.call( tmp_command, shell=True, env=os.environ.copy())


  def register_images(self,fixed_cid,moving_cid,sigma,rate,tmp_dir,data_dir,rigid=True,affine=True,elastic=True,delete_cache=False,,use_lung_mask=True,use_body_mask=True,body_th=0,fast=True,resampled_out=True,generate_tfm=True,pec_registration=False):
          
    # Set up input, output and temp volumes
    fixed = os.path.join(data_dir,fixed_cid + ".nhdr")
    moving = os.path.join(data_dir,moving_cid + ".nhdr")
    fixed_tmp = os.path.join(tmp_dir,fixed_cid + ".nhdr")
    moving_tmp = os.path.join(tmp_dir,moving_cid + ".nhdr")
    moving_deformed = os.path.join(data_dir,moving_cid + "_to_" + fixed_cid + ".nhdr")
    fixed_deformed = os.path.join(data_dir,fixed_cid + "_to_"+ moving_cid + ".nhdr")
    moving_deformed_tmp = os.path.join(tmp_dir,moving_cid + "_to_" + fixed_cid + ".nhdr")
    fixed_deformed_tmp = os.path.join(tmp_dir,fixed_cid + "_to_"+ moving_cid + ".nhdr")


    deformation_prefix = os.path.join(data_dir,moving_cid + "_to_" + fixed_cid + "_tfm_")
    affine_tfm = deformation_prefix + "0GenericAffine.mat"
    affine_output_tfm = deformation_prefix + "0GenericAffine.tfm"
    affine_output_xml = deformation_prefix + "0GenericAffine.xml"
    elastic_tfm = deformation_prefix + "1Warp.nii.gz"
    elastic_inv_tfm = deformation_prefix + "1InverseWarp.nii.gz"

    fixed_mask = os.path.join(data_dir,fixed_cid + "_partialLungLabelMap.nhdr")
    moving_mask = os.path.join(data_dir,moving_cid + "_partialLungLabelMap.nhdr")
    fixed_mask_tmp = os.path.join(tmp_dir,fixed_cid + "_partialLungLabelMap.nhdr")
    moving_mask_tmp = os.path.join(tmp_dir,moving_cid + "_partialLungLabelMap.nhdr")

    fixed_body_mask_tmp = os.path.join(tmp_dir,fixed_cid + "_bodyMask.nhdr")
    moving_body_mask_tmp = os.path.join(tmp_dir,moving_cid + "_bodyMask.nhdr")

    fixed_pec_mask = os.path.join(data_dir,fixed_cid + "_pecsSubcutaneousFatClosedSlice_mask.nrrd") #dialiared pec mask
    moving_pec_mask = os.path.join(data_dir,fixed_cid + "_pecsSubcutaneousFatClosedSlice_mask.nrrd")
    fixed_pec_mask_tmp = os.path.join(tmp_dir,fixed_cid + "_pecsSubcutaneousFatClosedSlice_mask.nrrd")
    moving_pec_mask_tmp = os.path.join(tmp_dir,fixed_cid + "_pecsSubcutaneousFatClosedSlice_mask.nrrd")
    
    if pec_registration == True:
      fixed = os.path.join(data_dir,fixed_cid + "_pecSlice.nrrd")
      moving = os.path.join(data_dir,moving_cid + "_pecSlice.nrrd")
      fixed_tmp = os.path.join(tmp_dir,fixed_cid + "_pecSlice.nrrd")
      moving_tmp = os.path.join(tmp_dir,moving_cid + "_pecSlice.nrrd")
      moving_deformed = os.path.join(data_dir,moving_cid + "_to_" + fixed_cid + "_pecSlice.nrrd")
      fixed_deformed = os.path.join(data_dir,fixed_cid + "_to_"+ moving_cid + "_pecSlice.nrrd")
      moving_deformed_tmp = os.path.join(tmp_dir,moving_cid + "_to_" + fixed_cid + "_pecSlice.nrrd")
      fixed_deformed_tmp = os.path.join(tmp_dir,fixed_cid + "_to_"+ moving_cid + "_pecSlice.nrrd")
      affine_output_tfm = deformation_prefix + "0GenericAffine_pec.tfm"
      affine_output_xml = deformation_prefix + "0GenericAffine_pec.xml"    
      
      fixed_body_mask = os.path.join(data_dir,fixed_cid + "_partialLungLabelMap_pecSlice.nrrd") #dialiared pec mask
      moving_body_mask = os.path.join(data_dir,fixed_cid + "_partialLungLabelMap_pecSlice.nrrd")
      fixed_body_mask_tmp = os.path.join(tmp_dir,fixed_cid + "_partialLungLabelMap_pecSlice.nrrd")
      moving_body_mask_tmp = os.path.join(tmp_dir,fixed_cid + "_partialLungLabelMap_pecSlice.nrrd")
    
      fixed_mask = os.path.join(data_dir,fixed_cid + "_partialLungLabelMap_pecSlice.nrrd")
      moving_mask = os.path.join(data_dir,moving_cid + "_partialLungLabelMap_pecSlice.nrrd")
      fixed_mask_tmp = os.path.join(tmp_dir,fixed_cid + "_partialLungLabelMap_pecSlice.nrrd")
      moving_mask_tmp = os.path.join(tmp_dir,moving_cid + "_partialLungLabelMap_pecSlice.nrrd")
  # moving_mask = os.path.join(data_dir,moving_cid + "_pecsSubcutaneousFatClosedSlice_regmask.nrrd")

    #Conditions input images: Gaussian blurring to account for SHARP kernel
    #Compute tissue compartment volume (silly linear mapping)
    unu = os.path.join(self.path['TEEM_PATH'],"unu")
    for im in [fixed,moving]:
      out = os.path.join(tmp_dir,os.path.basename(im))
      if pec_registration == True:
          tmp_command = unu + " resample -i %(in)s -s x1 x1 = -k dgauss:%(sigma)f,3 | "+ unu + " resample -s x%(r)f x%(r)f x%(r)f -k tent"
      else:
          tmp_command = unu + " resample -i %(in)s -s x1 x1 = -k dgauss:%(sigma)f,3 | "+ unu + " resample -s x%(r)f x%(r)f x%(r)f -k tent"
      tmp_command = tmp_command + " | "+ unu + " 2op + - 1000 | " + unu + " 2op / - 1055 -t float -o %(out)s"
      tmp_command = tmp_command % {'in':im, 'out':out,'sigma':sigma,'r':rate}
      print tmp_command
      sys.stdout.flush()
      subprocess.call( tmp_command, shell=True)
      
    #Extract lung mask region and dilate result  
    if use_lung_mask == True:
      dilation_distance=[10,20]
      for kk,im in enumerate([fixed_mask,moving_mask]):
        out = os.path.join(tmp_dir,os.path.basename(im))
        tmp_command = unu + " 3op in_op 1 %(in)s 10000 | " + unu + " resample -s x%(r)f x%(r)f x%(r)f -k cheap -o %(out)s"
        tmp_command = tmp_command % {'in':im, 'out':out,'r':rate}
        print tmp_command
        sys.stdout.flush()
        subprocess.call( tmp_command, shell=True)
        tmp_command = "pxdistancetransform -m Maurer -in %(in)s -out %(out)s"
        tmp_command = tmp_command % {'in':out, 'out':out}
        tmp_command = os.path.join(self.path['ITKTOOLS_PATH'],tmp_command)
        print tmp_command
        sys.stdout.flush()
        subprocess.call( tmp_command, shell=True)
        tmp_command = "unu 2op lt %(in)s %(val)d -o %(out)s"
        tmp_command = tmp_command % {'in':out, 'out':out,'val':dilation_distance[kk]}
        tmp_command = os.path.join(self.path['TEEM_PATH'],tmp_command)
        print tmp_command
        sys.stdout.flush()
        subprocess.call( tmp_command, shell=True)

    if use_body_mask == True:
      for out_im,in_im in zip([fixed_body_mask_tmp,moving_body_mask_tmp],[fixed_tmp,moving_tmp]):
        tmp_command = unu + " 2op gt %(in)s %(th)f -t short -o %(out)s"
        tmp_command = tmp_command % {'in':in_im, 'out':out_im, 'th':(body_th-1000)/1055}
        print tmp_command
        sys.stdout.flush()
        subprocess.call( tmp_command, shell=True)

    #If pecs, extract pec portion and dilate result, no need to now.
    
    
    #Perform Affine registration
    if fast == True:
      percentage=0.1
      its="100x20x0,1e-6,5"
    else:
      percentage=0.3
      its="100x50x20,1e-6,5"

    tmp_command = "antsRegistration -d 3 -r [ %(f)s,%(m)s,0] \
                   -m mattes[  %(f)s, %(m)s , 1 , 32, regular, %(percentage)f ] \
                   -t  translation[ 0.1 ] \
                   -c [%(its)s,1.e-8,20]  \
                   -s 4x2x1vox \
                   -f 8x4x2 \
                   -l 1" 
    tmp_command = tmp_command % {'f':fixed_tmp,'m':moving_tmp, \
      'percentage':percentage,'its':its}
   
    print(tmp_command)
    
    if affine :
      tmp_command = tmp_command + " -m mattes[  %(f)s, %(m)s , 1 , 32, regular, %(percentage)f ]  \
                   -t affine[ 0.1 ] \
                   -c [%(its)s,1.e-8,20] \
                   -s 4x2x1vox \
                   -f 8x4x2 \
                   -l 1 -u 1 -z 1"
      
      tmp_command = tmp_command % {'f':fixed_tmp,'m':moving_tmp, \
      'percentage':percentage,'its':its,'nm':deformation_prefix}

    print(tmp_command)               
    if use_body_mask:
        tmp_command = tmp_command + " -x [%(f_mask)s,%(m_mask)s]"
        tmp_command = tmp_command % {'f_mask':fixed_body_mask_tmp,'m_mask':moving_body_mask_tmp}

    tmp_command = tmp_command + " -o [%(nm)s]"
    tmp_command = tmp_command % {'nm':deformation_prefix}   
    print(tmp_command)
    tmp_command = os.path.join(self.path['ANTS_PATH'],tmp_command)
    
     
    if pec_registration == True:
      tmp_command = "antsRegistration -d 2 -r [ %(f)s,%(m)s,0] \
               -m mattes[  %(f)s, %(m)s , 1 , 32, regular, %(percentage)f ] \
               -t  translation[ 0.1 ] \
               -c [%(its)s,1.e-8,20]  \
               -s 4x2x1vox \
               -f 8x4x2 \
               -l 1"
      if affine == True:
        tmp_command = tmp_command + " -m mattes[  %(f)s, %(m)s , 1 , 32, regular, %(percentage)f ] \
                    -t affine[ 0.1 ] \
                    -c [%(its)s,1.e-8,20] \
                    -s 4x2x1vox \
                    -f 8x4x2 \
                    -l 1 -u 1 -z 1"
        tmp_command = tmp_command + " -o [%(nm)s]"
        tmp_command = tmp_command % {'f':fixed_tmp,'m':moving_tmp, \
          'percentage':percentage,'its':its,'nm':deformation_prefix}

        if use_body_mask == True:
          tmp_command = tmp_command + " -x [%(f_mask)s,%(m_mask)s]"
          tmp_command = tmp_command % {'f_mask':fixed_body_mask_tmp,'m_mask':moving_body_mask_tmp}

    tmp_command = os.path.join(self.path['ANTS_PATH'],tmp_command)
          
    print tmp_command
    sys.stdout.flush()
    subprocess.call( tmp_command, shell=True, env=os.environ.copy())
      
      #transform to tfm according to cip naming conventions and generate xml file
    if generate_tfm == True:
        #transform the .mat file
        tmp_command = "ConvertTransformFile 3 " +affine_tfm + " " + affine_output_tfm
        if pec_registration == True:
            tmp_command = "ConvertTransformFile 2 " +affine_tfm + " " + affine_output_tfm
        tmp_command = os.path.join(self.path['ANTS_PATH'],tmp_command)
        subprocess.call( tmp_command, shell=True )
        
        #write the xml file
        regID = "Registration"+str(int(time.time()))+"_"+moving_cid+"_to_"+fixed_cid
        registration = etree.Element('Registration', Registration_ID = regID)
        doc = etree.ElementTree(registration)
        transformation = etree.SubElement(registration, 'transformation')
        transformation.text = 'GenericAffinee'
        movingID = etree.SubElement(registration, 'movingID')
        movingID.text = moving_cid
        fixedID = etree.SubElement(registration, 'fixedID')
        fixedID.text = fixed_cid
        
        similarityMeasure = etree.SubElement(registration, 'SimilarityMeasure')
        similarityMeasure.text = "Mattes mutual information"
        similarityValue = etree.SubElement(registration, 'similarityValue')
        similarityValue.text = 'NA'
        
        with open(affine_output_xml, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8" ?>')
            f.write('<!DOCTYPE root SYSTEM "RegistrationOutput_v1.dtd">')
            doc.write(f,  pretty_print=True)

    if elastic == True:
      #Perform Diffeomorphic registration
      if fast == True:
        syn="100x20x0,1e-5,5"
      else:
        syn="100x50x40,1e-6,5"

      tmp_command = "antsRegistration -d 3 -r %(affine)s \
      -m cc[ %(f)s, %(m)s , 0.5 , 2 ] \
      -t SyN[ .20, 3, 0 ] \
      -c [ %(syn)s ]  \
      -s 8x4x2vox  \
      -f 8x4x2 \
      -l 1 -z 1 \
      -o [%(nm)s,%(out)s,%(out_inv)s]"
      tmp_command = tmp_command % {'affine':affine_tfm, 'f':fixed_tmp, 'm':moving_tmp,'syn':syn, \
      'nm':deformation_prefix,'out':moving_deformed_tmp,'out_inv':fixed_deformed_tmp}
      
      

      if use_lung_mask:
        tmp_command = tmp_command + " -x [%(f_mask)s,%(m_mask)s]"
        tmp_command = tmp_command % {'f_mask':fixed_mask_tmp,'m_mask':moving_mask_tmp}

        tmp_command = os.path.join(self.path['ANTS_PATH'],tmp_command)
        print tmp_command
        sys.stdout.flush()
        subprocess.call( tmp_command, shell=True, env=os.environ.copy())


    #Apply composite transform (if necessary)
    # The previous command already outputs the transformed images composed with both
    # transformations for the tmp images used in the registration. Here we deformed the original ones


    if resampled_out == True:
      if elastic == True:
        tt_list = [elastic_tfm,affine_tfm]
        transform_data(fixed,moving,tt_list,moving_deformed_tmp)
        tt_list = ["["+affine_tfm+",1]",elastic_inv_tfm]
        transform_data(moving,fixed,tt_list,fixed_deformed_tmp)
      else:
        tt_list = [affine_tfm]
        transform_data(fixed,moving,tt_list,moving_deformed_tmp)
        tt_list = ["[" + affine_tfm + ",1]"]
        transform_data(moving,fixed,tt_list,fixed_deformed_tmp)
        
      # Do nrrd gzip compression and type casting for the output images
      for im_out,im_in in zip([fixed_deformed,moving_deformed],[fixed_deformed_tmp,moving_deformed_tmp]):
        tmp_command = unu + " convert -i %(in)s -t short | "+ unu + " save -f nrrd -e gzip -o %(out)s"
        tmp_command = tmp_command % {'in':im_in,'out':im_out}
        subprocess.call( tmp_command, shell=True)

    #tmp_command = "antsApplyTransforms -d 3 -i %(m)s -r %(f)s -n linear \
    #   -t %(elastic-tfm)s -t %(affine-tfm)s -o %(deformed)s"
    #tmp_command = tmp_command % {'f':fixed,'m':moving, \
    #'elastic-tfm':elastic_tfm,'affine-tfm':affine_tfm,'deformed':moving_deformed_tmp}
    #tmp_command = os.path.join(self.path['ANTS_PATH'],tmp_command)
    #subprocess.call( tmp_command, shell=True, env=os.environ.copy())

    #tmp_command = "antsApplyTransforms -d 3 -i %(f)s -r %(m)s -n linear \
    #-t [%(affine-tfm)s,1] -t %(elastic-inv-tfm)s  -o %(deformed)s"
    #tmp_command = tmp_command % {'f':fixed,'m':moving, \
    #'elastic-inv-tfm':elastic_inv_tfm,'affine-tfm':affine_tfm,'deformed':fixed_deformed_tmp}
    #tmp_command = os.path.join(self.path['ANTS_PATH'],tmp_command)
    #subprocess.call( tmp_command, shell=True, env=os.environ.copy())

      
    if delete_cache == True:
      print "Cleaning tempoarary directory..."
      tmp_command = "\\rm " + os.path.join(tmp_dir, "*")
      subprocess.call( tmp_command, shell=True )

if __name__ == "__main__":  
    
    parser = OptionParser()

    parser.add_option("-f",dest="fixedCID",help="case ID for fixed image")
    parser.add_option("-m",dest="movingCID",help="case ID for moving image")
    parser.add_option("-s",dest="sigma",help="sigma for initial smoothing [default: %default]", type="float",default=1)
    parser.add_option("-r",dest="rate",help="initial upsampling/downsampling rate [default: %default]", type="float", default=0.5)
    parser.add_option("--tmpDir",dest="tmpDirectory",help="temporary directory" )
    parser.add_option("--dataDir", dest="dataDirectory",help="data directory for the case IDs")
    parser.add_option("-d",dest="delete_cache",help="delete cache in temp directory [default: %default]", action="store_true")
    parser.add_option("-a",dest="affine",help="run affine", action="store_true")
    parser.add_option("-e",dest="elastic",help="run elastic", action="store_true")
    parser.add_option("-b",dest="use_body_mask",help="compute body mask for affine registration step", action="store_true")
    parser.add_option("-l",dest="use_lung_mask",help="employs lung mask for elastic registration step", action="store_true")
    parser.add_option("--bTh", dest="body_th", help="body intesity threshold [default: %default]", type="float", default=0)
    parser.add_option("--fast", dest="fast", help="fast (less accurate) registration", action="store_true")
    parser.add_option("--resampledOut", dest="resampled_out", help="produce resample outputs into the fixed and moving image space", action="store_true")
    parser.add_option("--generate_tfm", dest="generate_tfm", help="generate .tfm transformation file and corresponding xml file", action="store_true")


    (options, args) = parser.parse_args()
    
    fixed_cid = options.fixedCID
    moving_cid = options.movingCID
    sigma = options.sigma
    rate = options.rate
    tmp_dir = options.tmpDirectory
    data_dir = options.dataDirectory
    delete_cache = options.delete_cache
    affine = options.affine
    elastic = options.elastic
    use_lung_mask = options.use_lung_mask
    use_body_mask = options.use_body_mask
    body_th = options.body_th
    fast= options.fast
    resampled_out = options.resampled_out
    generate_tfm = options.generate_tfm
    
    register_images(fixed_cid,moving_cid,sigma,rate,tmp_dir,data_dir,delete_cache,affine,elastic,use_lung_mask,use_body_mask,body_th,fast,resampled_out,generate_tfm, False)









