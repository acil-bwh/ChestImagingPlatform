import subprocess 
import os

class quickAirwayVesselSegmentator:
    """General purpose class for detecting vessels and airways.

    This class extracts vessels and airways and produces a labelmap consisting of the
        whole lung but excluding the vessels and airways.         
              
    Scale limits depend on structure size:
    For large, use: minscale=2  maxscale=10
    For small, use: minscale=0.7 maxscale=2
    Both: minscale=0.7 maxscale=10
    
       
    Parameters 
    ----------    
    
    min_scale: float 
        lower scale limit
    max_scale: float 
        lower scale limit        
    vesselness_th: float 
        vesselness thinkness  
    airwayness_th: float 
        airwayness thinkness  
    extract_airways: boolean
       set to true if airways are to be extracted                                         
    extract_vessels: boolean
       set to true if vessels are to be extracted  
    C_airway: integer
        C parameter for the airways
        The C param is based on the image contrast and depends on the amount of noise. 

    C_vessel: integer
        C parameter for the vessels
    """
    def __init__(self, min_scale=0.7, max_scale=4, vesselness_th=0.28, airwayness_th=0.4, extract_airways=True, extract_vessels=False,
        C_airway=80, C_vessel=245):
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._vesselness_th = vesselness_th
        self._airwayness_th = airwayness_th
        self._extract_airways= extract_airways
        self._extract_vessels= extract_vessels
        self._C_airway = C_airway
        self._C_vessel = C_vessel
        
    def execute(self, ct_file_name, maskFileName, temp_dir, pl_file_name):
        """
        executes the airway / vessel segmentation
        
        Inputs
        -----
        ct_file_name : String
            Full path to input CT filename
        pl_file_name : String
            full path to input partial lung labelmap filename
        maskFileName :String
            full path to output mask file name. The mask is set to 0 where there are 
            vessels / airways.   
        """

        #Check required tools path enviroment variables for tools path        
        os.environ['CIP_PATH']=os.path.expanduser('/Users/rolaharmouche/ChestImagingPlatform-build/CIP-build/bin')
        os.environ['TEEM_PATH']=os.path.expanduser('/Users/rolaharmouche/teem-code/teem-build/bin')
        os.environ['ITKTOOLS_PATH']=os.path.expanduser('/Users/rolaharmouche/Downloads/ITKTools/build/bin')
    
        toolsPaths = ['CIP_PATH','TEEM_PATH','ITKTOOLS_PATH'];
        path=dict()
        for path_name in toolsPaths:
            path[path_name]=os.environ.get(path_name,False)
            if path[path_name] == False:
                print (path_name + " environment variable is not set")
                exit()

        unuop = os.path.join(path['TEEM_PATH'],'unu')

        case_id = ct_file_name.split('/')[-1].split('.')[0]
        print(case_id)
        featureMapFileName = os.path.join(temp_dir,case_id +   "_featureMap.nrrd")
        maskFileNameVessel = os.path.join(temp_dir,case_id +   "_vesselMask.nrrd")
        maskFileNameAirway = os.path.join(temp_dir,case_id +   "_airwayMask.nrrd")
        
        #Vessel (only airways working properly now...)
        if (self._extract_vessels):
            print("Extracting vessels..")
            # The C param is based on the image contrast and defendants on the amount of noise. 
            # alpha and beta are frangi parameters
            tmpCommand = "ComputeFeatureStrength -i %(in)s -m Frangi -f RidgeLine --std %(minscale)f,7,%(maxscale)f --ssm 1 --alpha 0.63 --beta 0.51 --C "+ \
                str(self._C_vessel)+" -o %(out)s"
            tmpCommand = tmpCommand % {'in':ct_file_name,'out':featureMapFileName,'minscale':self._min_scale,'maxscale':self._max_scale}
            tmpCommand  = os.path.join(path['CIP_PATH'],tmpCommand)
            print (tmpCommand)
            subprocess.call( tmpCommand, shell=True )
                    
            #Hist equalization, threshold Feature strength and masking
            tmpCommand = unuop+" 2op x %(feat)s %(mask)s -t float | "+unuop+" heq -b 10000 -a 0.96 -s 5 |  "+unuop+" 2op gt - %(vesselness_th)f  |  "+unuop+" convert -t short -o %(out)s"
            tmpCommand = tmpCommand % {'feat':featureMapFileName,'mask':pl_file_name,'vesselness_th':self._vesselness_th,'out':maskFileNameVessel}
            print (tmpCommand)
            subprocess.call( tmpCommand , shell=True)

        if (self._extract_airways):
            #Airway
            print("Extracting Airways..")
            tmpCommand = "ComputeFeatureStrength -i %(in)s -m Frangi -f ValleyLine --std %(minscale)f,4,%(maxscale)f --ssm 1 --alpha 0.5 --beta 0.5 --C "+ \
                str(self._C_airway)+" -o %(out)s"
            tmpCommand = tmpCommand % {'in':ct_file_name,'out':featureMapFileName,'minscale':self._min_scale,'maxscale':self._max_scale}
            tmpCommand  = os.path.join(path['CIP_PATH'],tmpCommand)
            print (tmpCommand)
            subprocess.call( tmpCommand, shell=True )

            #Hist equalization, threshold Feature strength and masking
            tmpCommand = unuop+" 2op x %(feat)s %(mask)s -t float | "+unuop+" heq -b 10000 -a 0.5 -s 2 | "+unuop+" 2op gt - %(airwayness_th)f  | "+unuop+" convert -t short -o %(out)s"
            tmpCommand = tmpCommand % {'feat':featureMapFileName,'mask':pl_file_name,'airwayness_th':self._airwayness_th,'out':maskFileNameAirway}
            print (tmpCommand)
            subprocess.call( tmpCommand , shell=True)
        
        # Combine both or copy one of them to final output
        if (self._extract_vessels and self._extract_airways):
            tmpCommand = unuop+" 2op min %(in1)s %(in2)s -o %(out)s"
            tmpCommand = tmpCommand % {'in1':maskFileNameVessel,'in2':maskFileNameAirway, 'out':maskFileName}
            print (tmpCommand)
            subprocess.call( tmpCommand , shell=True)
        elif(self._extract_airways ):        
            tmpCommand = "cp "+maskFileNameAirway+" "+maskFileName 
            subprocess.call( tmpCommand , shell=True)
        elif(self._extract_vessels ):        
            tmpCommand = "cp "+maskFileNameVessel+" "+maskFileName 
            subprocess.call( tmpCommand , shell=True)
            

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description='Airway and vessel extraction pipeline.')
  parser.add_argument("--in_ct", dest="ct_file_name",required=True, help='Input CT filename.')
  parser.add_argument("--in_partial", dest="pl_file_name",required=True, help='Input partial lung labelmap filename.')
  parser.add_argument("--out_mask", dest="out_mask_file_name",required=True, help='Output mask filename. Extracted structures\
    (vessels, airways) will have a value of 0.')
  parser.add_argument("--temp_dir", dest="temp_dir",required=True, help='Directory to store the temp data.')
    
  parser.add_argument("--structure_size",dest="structure_size",default="all", help='Options are small, large, all \
    (for small and large structures).')

  parser.add_argument("--vesselness_th",dest="vesselness_th",type=float,default=0.38, help='Vesselness thickness.')
  parser.add_argument("--airwayness_th",dest="airwayness_th",type=float,default=0.5, help='Airwayness thickness.')
  parser.add_argument("--C_airway",dest="C_airway",default=80, help='The C param is based on the image contrast and depends on the amount of noise. ')
  parser.add_argument("--C_vessel",dest="C_vessel",default=245, help='The C param is based on the image contrast and depends on the amount of noise. ')

  parser.add_argument("--extract_airways", help='Set to True if we want to extract airways. Default: True.', dest="extract_airways", action='store_true')
  parser.add_argument("--extract_vessels", help='Set to True if we want to extract airways. Default: False.', dest="extract_vessels", action='store_true')

  options  = parser.parse_args()
  minscale=0.7 
  maxscale=10 
  if (options.structure_size =="all"):
      minscale=0.7 
      maxscale=10
  elif (options.structure_size =="large"):
      minscale=2  
      maxscale=10
  elif (options.structure_size =="small"):
      minscale=0.7 
      maxscale=2
  else:
      raise ValueError("wrong structure_size option.")
 
  """
  Example usage:  ~/ChestImagingPlatform-build/CIPPython-install/bin/python ~/ChestImagingPlatform/cip_python/utils/quick_airway_vessel_segmentation.py  
    --in_ct /Users/rolaharmouche/Documents/Data/PLuSS/00694-4/00694-4_INSP_LUNG_T00_PLuSS/00694-4_INSP_LUNG_T00_PLuSS.nrrd 
    --in_partial /Users/rolaharmouche/Documents/Data/PLuSS/00694-4/00694-4_INSP_LUNG_T00_PLuSS/00694-4_INSP_LUNG_T00_PLuSS_partialLungLabelMap.nrrd 
    --out_mask /Users/rolaharmouche/Documents/Data/PLuSS/00694-4/00694-4_INSP_LUNG_T00_PLuSS/00694-4_INSP_LUNG_T00_PLuSS_airwayVesselMask.nrrd
    --temp_dir ~/Documents/Data/tmp_data_ILD_classifications/ --extract_airways 
  """
  
        
  vad = quickAirwayVesselSegmentator(min_scale=minscale, max_scale=maxscale, \
    vesselness_th=options.vesselness_th, airwayness_th=options.airwayness_th, extract_airways=options.extract_airways, \
    extract_vessels=options.extract_vessels, C_airway=options.C_airway, C_vessel=options.C_vessel)
    
  vad.execute(options.ct_file_name, options.out_mask_file_name, options.temp_dir, options.pl_file_name)
  
  
