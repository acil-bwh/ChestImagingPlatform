#!/usr/bin/python

import sys
import string
import subprocess
import os
import nrrd
import SimpleITK as sitk

from cip_python.particles.vessel_particles import VesselParticles
from cip_python.particles.multires_vessel_particles import MultiResVesselParticles
from cip_python.utils.interpolate_volume import InterpolateVolume


class VesselParticlesPipeline:
    """Class that implements a vessel particle extraction pipeline for a given region
        This tools currently depends on Teem, ITKTools and CIP
        Parameters
        ---------
        
        """
    def __init__(self,ct_file_name,pl_file_name,regions,tmp_dir,output_prefix,init_method='Frangi',
                 vessel_mask=None,resampling_method='Linear',lth=-95,sth=-70,voxel_size=0,min_scale=0.7,max_scale=4,
                 vesselness_th=0.38,crop=0,rate=1,multires=False,justparticles=False,clean_cache=True):
        
        assert init_method == 'Frangi' or init_method == 'Threshold' or init_method == 'StrainEnergy' \
               or init_method == 'VesselMask'
        
        self._ct_file_name=ct_file_name
        self._pl_file_name=pl_file_name
        self._regions=regions
        self._tmp_dir=tmp_dir
        self._output_prefix=output_prefix
        self._init_method=init_method
        self._resampling_method=resampling_method
        self._lth=lth
        self._sth=sth
        self._voxel_size=voxel_size
        self._min_scale=min_scale
        self._max_scale=max_scale
        self._crop=crop
        self._rate=rate
        self._multires=multires
        self._justparticles=justparticles
        self._clean_cache=clean_cache
        self._vessel_mask=vessel_mask
        
        self._case_id = str.split(os.path.basename(ct_file_name),'.')[0]
        
        #Internal params
        #Distance from wall that we don't want to consider in the initialization (negative= inside the lung, positive= outside the lung)
        self._distance_from_wall = 2.0
        #Threshold on the vesselness map (for particles initialization mask)
        self._vesselness_th = vesselness_th
        #Intensity threshold (for particles initialization mask)
        self._intensity_th = -700
    
    def execute(self):
        
        header=nrrd.read_header(open(self._ct_file_name))
        
        max_z=int(header['sizes'][2])
        spacing=[float(header['space directions'][kk][kk]) for kk in xrange(3)]
        
        if len(crop) < 2:
            crop_flag = False
        else:
            if crop[0]<0:
                crop[0]=0
            elif crop[1]>max_z:
                crop_flag = False
            
            crop_flag = True
        
        ct_file_name = self._ct_file_name
        pl_file_name = self._pl_file_name
        
        #Data preprocessing (cropping and/or resampling) before region based analysis
        if self._justparticles == False:
            
            #Crop volume if flag is set
            ct_file_name_crop = os.path.join(self._tmp_dir,self._case_id + "_crop.nrrd")
            pl_file_name_crop = os.path.join(self._tmp_dir,self._case_id + "_croppartialLungLabelMap.nrrd")
            if crop_flag == True:
                if self._rate == 1:
                    ratestr = "="
                else:
                    ratestr = 'x%f' % rate
                
                tmpCommand = "unu crop -min 0 0 %(z1)d -max M M %(z2)d -i %(in)s | unu resample -k %(kernel)s -s = = %(ratestr)s -o %(out)s"
                
                tmpCommandCT = tmpCommand % {'z1':crop[0],'z2':crop[1],'in':ct_file_name,'out':ct_file_name_crop,'ratestr':ratestr,'kernel':"cubic:1,0"}
                tmpCommandPL = tmpCommand % {'z1':crop[0],'z2':crop[1],'in':pl_file_name,'out':pl_file_name_crop,'ratestr':ratestr,'kernel':"cheap"}
                print (tmpCommandCT)
                print (tmpCommandPL)
                
                subprocess.call(tmpCommandCT,shell=True)
                subprocess.call(tmpCommandPL,shell=True)
                ct_file_name = ct_file_name_crop
                pl_file_name = pl_file_name_crop
            
            #Do resampling to isotropic voxel size to compute accurate scale measurements
            ct_file_name_resample = os.path.join(self._tmp_dir,self._case_id + "_resample.nrrd")
            pl_file_name_resample = os.path.join(self._tmp_dir,self._case_id + "_resamplepartialLungLabelMap.nrrd")
            
            if self._voxel_size>0:
                
                tmpCommandUnuResampling = "unu resample -k %(kernel)s -s x%(f1)f x%(f2)f x%(f3)f -i %(in)s -o %(out)s -c cell"

                #CT Resampling: Three approches: Linear, cubic, registration and hybrid (registration and cubic)
                if self._resampling_method=='Linear':
                    tmpCommandCT = tmpCommandUnuResampling % {'in':ct_file_name,'out':ct_file_name_resample,'kernel':"tent",'f1':spacing[0]/self._voxel_size,'f2':spacing[1]/self._voxel_size,'f3':spacing[2]/self._voxel_size}
                    print (tmpCommandCT)
                    subprocess.call(tmpCommandCT,shell=True)
                elif self._resampling_method=='Cubic':
                    tmpCommandCT = tmpCommandUnuResampling % {'in':ct_file_name,'out':ct_file_name_resample,'kernel':"cubic:0,0.5",'f1':spacing[0]/self._voxel_size,'f2':spacing[1]/self._voxel_size,'f3':spacing[2]/self._voxel_size}
                    print (tmpCommandCT)
                    subprocess.call(tmpCommandCT,shell=True)
                elif self._resampling_method=='Registration':
                    ct_image = sitk.ReadImage(ct_file_name)
                    interpolator = InterpolateVolume(self._voxel_size)
                    ct_interp=interpolator.execute(ct_image)
                    sitk.WriteImage(ct_interp,ct_file_name_resample,True)
                    #Now Fix x,y
                    tmpCommandCT = tmpCommandUnuResampling % {'in':ct_file_name_resample,'out':ct_file_name_resample,'kernel':"tent",'f1':spacing[0]/self._voxel_size,'f2':spacing[1]/self._voxel_size,'f3':1.0}
                    print (tmpCommandCT)
                    subprocess.call(tmpCommandCT,shell=True)
                elif self._resampling_method=='Hybrid':
                    print ("Hybrid resampling approach...")
                    ct_image = sitk.ReadImage(ct_file_name)
                    interpolator = InterpolateVolume(2.0*self._voxel_size)
                    ct_interp=interpolator.execute(ct_image)
                    sitk.WriteImage(ct_interp,ct_file_name_resample,True)
                    #Now cubic for x,y and z
                    tmpCommandCT = tmpCommandUnuResampling % {'in':ct_file_name_resample,'out':ct_file_name_resample,'kernel':"cubic:0,0.5",'f1':spacing[0]/self._voxel_size,'f2':spacing[1]/self._voxel_size,'f3':2.0}
                    print (tmpCommandCT)
                    subprocess.call(tmpCommandCT,shell=True)

                tmpCommandPL = tmpCommandUnuResampling % {'in':pl_file_name,'out':pl_file_name_resample,'kernel':"cheap",'f1':spacing[0]/self._voxel_size,'f2':spacing[1]/self._voxel_size,'f3':spacing[2]/self._voxel_size}
                print (tmpCommandPL)
                subprocess.call(tmpCommandPL,shell=True)
                
                ct_file_name = ct_file_name_resample
                pl_file_name = pl_file_name_resample
        
        
        for ii in self._regions:
            #Lower camel case region tag
            rtag = ii[0].lower()+ii[1:]
            tmpDir = os.path.join(self._tmp_dir,rtag)
            if os.path.exists(tmpDir) == False:
                os.mkdir(tmpDir)
            
            # Define FileNames that will be used
            pl_file_nameRegion= os.path.join(tmpDir,self._case_id + "_" + rtag + "_partialLungLabelMap.nrrd")
            pl_file_nameRegion2= os.path.join(tmpDir,self._case_id + "_" + rtag + "_partialLungLabelMap2.nrrd")

            ct_file_nameRegion= os.path.join(tmpDir,self._case_id + "_" + rtag + ".nrrd")
            featureMapFileNameRegion = os.path.join(tmpDir,self._case_id + "_" + rtag + "_featureMap.nrrd")
            distanceMapFileNameRegion = os.path.join(tmpDir,self._case_id + "_" + rtag + "_distanceMap.nrrd")
            maskFileNameRegion = os.path.join(tmpDir,self._case_id + "_" + rtag + "_mask.nrrd")
            particlesFileNameRegion = os.path.join(self._output_prefix+ "_" + rtag + "VesselParticles.vtk")
            
            if self._justparticles == False:
                
                #Create SubVolume Region
                tmpCommand = "CropLung --cipr %(region)s -m 0 -v -1000 --ict %(ct-in)s --ilm %(lm-in)s --oct %(ct-out)s --olm %(lm-out)s"
                tmpCommand = tmpCommand % {'region':ii,'ct-in':ct_file_name,'lm-in':pl_file_name,'ct-out':ct_file_nameRegion,'lm-out':pl_file_nameRegion}
                tmpCommand = os.path.join(path['CIP_PATH'],tmpCommand)
                print (tmpCommand)
                subprocess.call( tmpCommand, shell=True )
                
                #Extract Lung Region + Distance map to peel lung
                tmpCommand = "ExtractChestLabelMap -r %(region)s -i %(lm-in)s -o %(lm-out)s"
                tmpCommand = tmpCommand % {'region':ii,'lm-in':pl_file_nameRegion,'lm-out':pl_file_nameRegion}
                tmpCommand = os.path.join(path['CIP_PATH'],tmpCommand)
                print (tmpCommand)
                subprocess.call( tmpCommand, shell=True )
                
                tmpCommand = "unu 2op gt %(lm-in)s 0.5 -o %(lm-out)s"
                tmpCommand = tmpCommand % {'lm-in':pl_file_nameRegion,'lm-out':pl_file_nameRegion}
                print (tmpCommand)
                subprocess.call( tmpCommand, shell=True )
                
                #Trimming of chest wall using CIP distance map approach
                tmpCommand ="ComputeDistanceMap -l %(lm-in)s -d %(distance-map)s -s 1 -p -m Maurer"
                tmpCommand = tmpCommand % {'lm-in':pl_file_nameRegion,'distance-map':distanceMapFileNameRegion}
                tmpCommand = os.path.join(path['CIP_PATH'],tmpCommand)
                print (tmpCommand)
                subprocess.call( tmpCommand, shell=True )
                print ("Distance from wall: "+str(self._distance_from_wall))
                tmpCommand ="unu 2op gt %(distance-map)s %(distance)f | unu convert -t short -o %(lm-out)s"
                tmpCommand = tmpCommand % {'distance-map':distanceMapFileNameRegion,'distance':self._distance_from_wall,'lm-out':pl_file_nameRegion}
                print (tmpCommand)
                subprocess.call( tmpCommand, shell=True )

                #Trimming of chest wall using ITKTools distance map approach
#                tmpCommand ="pxdistancetransform -in %(lm-in)s -out %(distance-map)s"
#                tmpCommand = tmpCommand % {'lm-in':pl_file_nameRegion,'distance-map':pl_file_nameRegion}
#                tmpCommand = os.path.join(path['ITKTOOLS_PATH'],tmpCommand)
#                print (tmpCommand)
#                subprocess.call( tmpCommand, shell=True )
#
#                tmpCommand ="unu 2op lt %(distance-map)s %(distance)f -t short -o %(lm-out)s"
#                tmpCommand = tmpCommand % {'distance-map':pl_file_nameRegion,'distance':-1.0*self._distance_from_wall,'lm-out':pl_file_nameRegion}
#                print (tmpCommand)
#                subprocess.call( tmpCommand, shell=True )

                # Compute Frangi
                if self._init_method == 'Frangi':
                    tmpCommand = "ComputeFeatureStrength -i %(in)s -m Frangi -f RidgeLine --std %(minscale)f,%(maxscale)f,7 --ssm 1 --alpha 0.63 --beta 0.51 --C 245 -o %(out)s"
                    tmpCommand = tmpCommand % {'in':ct_file_nameRegion,'out':featureMapFileNameRegion,'minscale':self._min_scale,'maxscale':self._max_scale}
                    tmpCommand  = os.path.join(path['CIP_PATH'],tmpCommand)
                    print (tmpCommand)
                    subprocess.call( tmpCommand, shell=True )
                    
                    #Hist equalization, threshold Feature strength and masking
                    tmpCommand = "unu 2op x %(feat)s %(mask)s -t float | unu heq -b 10000 -a 0.96 -s 5 | unu 2op gt - %(vesselness_th)f  | unu convert -t short -o %(out)s"
                    tmpCommand = tmpCommand % {'feat':featureMapFileNameRegion,'mask':pl_file_nameRegion,'vesselness_th':self._vesselness_th,'out':maskFileNameRegion}
                    print (tmpCommand)
                    subprocess.call( tmpCommand , shell=True)
                elif self._init_method == 'StrainEnergy':
                    tmpCommand = "ComputeFeatureStrength -i %(in)s -m StrainEnergy -f RidgeLine --std %(minscale)f,%(maxscale)f,7 --ssm 1 --alpha 0.2 --beta 0.1 --kappa 0.5 --nu 0.1 -o %(out)s"
                    tmpCommand = tmpCommand % {'in':ct_file_nameRegion,'out':featureMapFileNameRegion,'minscale':self._min_scale,'maxscale':self._max_scale}
                    tmpCommand  = os.path.join(path['CIP_PATH'],tmpCommand)
                    print (tmpCommand)
                    subprocess.call( tmpCommand, shell=True )
                    
                    #Hist equalization, threshold Feature strength and masking
                    tmpCommand = "unu 2op x %(feat)s %(mask)s -t float | unu heq -b 10000 -a 0.95 -s 5 | unu 2op gt - %(vesselness_th)f  | unu convert -t short -o %(out)s"
                    tmpCommand = tmpCommand % {'feat':featureMapFileNameRegion,'mask':pl_file_nameRegion,'vesselness_th':self._vesselness_th,'out':maskFileNameRegion}
                    print (tmpCommand)
                    subprocess.call( tmpCommand , shell=True)
                elif self._init_method == 'Threshold':
                    tmpCommand = "unu 2op gt %(in)s %(intensity_th)f | unu 2op x - %(mask)s -o %(out)s"
                    tmpCommand = tmpCommand % {'in':ct_file_nameRegion,'mask':pl_file_nameRegion,'intensity_th':self._intensity_th,'out':maskFileNameRegion}
                    print (tmpCommand)
                    subprocess.call( tmpCommand , shell=True)
                elif self._init_method == 'VesselMask':
                    # vessel_file=os.path.join(self._tmp_dir,self._vessel_mask)
                    vessel_file=self._vessel_mask
                    vessel_file_resampled=os.path.join(self._tmp_dir,self._case_id + "_" + "vesselMaskResampled" + ".nrrd")
                    tmpCommandUnuResampling = "unu resample -k %(kernel)s -s x%(f1)f x%(f2)f x%(f3)f -i %(in)s -o %(out)s -c cell"
                    tmpCommandPL = tmpCommandUnuResampling % {'in':vessel_file,'out':vessel_file_resampled,'kernel':"cheap",'f1':spacing[0]/self._voxel_size,'f2':spacing[1]/self._voxel_size,'f3':spacing[2]/self._voxel_size}
                    print (tmpCommandPL)
                    subprocess.call(tmpCommandPL,shell=True)
                    tmpCommand = "CropLung --cipr %(region)s -m 0 -v 0 --ict %(mask-in)s --ilm %(lm-in)s --oct %(mask-out)s --olm %(lm-out)s"
                    tmpCommand = tmpCommand % {'region': ii, 'mask-in': vessel_file_resampled, 'lm-in': pl_file_name,
                                               'mask-out': maskFileNameRegion, 'lm-out': pl_file_nameRegion}
                    tmpCommand = os.path.join(path['CIP_PATH'], tmpCommand)
                    # print tmpCommand
                    subprocess.call(tmpCommand, shell=True)
                
                # Binary Thinning
                tmpCommand = "GenerateBinaryThinning3D -i %(in)s -o %(out)s"
                tmpCommand = tmpCommand % {'in':maskFileNameRegion,'out':maskFileNameRegion}
                tmpCommand = os.path.join(path['CIP_PATH'],tmpCommand)
                subprocess.call( tmpCommand, shell=True)
            
            # Vessel Particles For the Region
            if self._multires==False:
                particlesGenerator = VesselParticles(ct_file_nameRegion,particlesFileNameRegion,tmpDir,maskFileNameRegion,live_thresh=self._lth,seed_thresh=self._sth,min_intensity=-950,max_intensity=200)
                particlesGenerator._clean_tmp_dir=self._clean_cache
                #particlesGenerator._interations_phase3 = 70
                #particlesGenerator._irad_phase3 = 0.9
                #particlesGenerator._srad_phase3 = 4
                particlesGenerator._verbose = 0
                
                #Adjust default irad param to accomodate different scales. Those params were optimized for a nominal resolution of 0.625 mm
                adjust_irad_factor=0.625/self._voxel_size
                if adjust_irad_factor is not 1.0:
                    phase_irads=particlesGenerator._phase_irads
                    new_phase_irads=list()
                    for val in phase_irads:
                        new_phase_irads.append(val/adjust_irad_factor)
                    particlesGenerator._phase_irads = new_phase_irads
            
                #particlesGenerator._ppv = 1
                particlesGenerator.execute()
            
            
            else:
                particlesGenerator = MultiResVesselParticles(ct_file_nameRegion,particlesFileNameRegion,tmpDir,maskFileNameRegion,live_thresh=-600,seed_thresh=-600)
                particlesGenerator._clean_tmp_dir=self._clean_cache
                particlesGenerator.execute()




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Vessel particle extraction pipeline')
    parser.add_argument("-i", dest="ct_file_name",required=True)
    parser.add_argument("-l", dest="pl_file_name",required=True)
    parser.add_argument("-o", dest="output_prefix",required=True)
    parser.add_argument("--tmpDir", dest="tmp_dir",required=True)
    parser.add_argument("-r", dest="regions",required=True)
    parser.add_argument("--liveTh", dest="lth",type=float,default=-90)
    parser.add_argument("--seedTh", dest="sth",type=float,default=-70)
    parser.add_argument("-s", dest="voxel_size",type=float,default=0)
    parser.add_argument("--crop", dest="crop",default="0")
    parser.add_argument("--rate", dest="rate",type=float,default=1)
    parser.add_argument("--minscale", dest="min_scale",type=float,default=0.7)
    parser.add_argument("--maxscale", dest="max_scale",type=float,default=4)
    parser.add_argument("--init", dest="init_method",default="Frangi")
    parser.add_argument("--vmask", dest="vessel_mask", default=None)
    parser.add_argument("--dwall", dest="distance_from_wall", type=float,default=2.0,help="Distance from wall (in mm) to exclude for the vessel extraction to avoid chest wall artifacts")
    parser.add_argument("--vesselness_th", dest="vesselness_th",type=float,default=0.38)
    parser.add_argument("--resampling", dest="resampling_method",default="Linear",help="Resampling method for CT image: linear, cubic, registration (demons-based approach), hybrid (registration + cubic)")
    parser.add_argument("--multires", dest="multires",action="store_true", default = False)
    parser.add_argument("--justparticles", dest="justparticles",action="store_true", default=False)
    parser.add_argument("--cleanCache", action="store_true", dest="clean_cache", default=False)

    # Check required tools path enviroment variables for tools path
    toolsPaths = ['CIP_PATH','TEEM_PATH','ITKTOOLS_PATH'];
    path=dict()
    for path_name in toolsPaths:
        path[path_name] = os.environ.get(path_name,False)
        if path[path_name] == False:
            print (path_name + " environment variable is not set")
            exit()
    
    op = parser.parse_args()
    assert op.init_method == 'Frangi' or op.init_method == 'Threshold' or op.init_method == 'StrainEnergy' or \
           op.init_method == 'VesselMask'
    
    # region = [2,3]
    # region=[2]
    # regionTag = ['right','left']
    crop = [int(kk) for kk in str.split(op.crop,',')]
    regions = [kk for kk in str.split(op.regions,',')]
    
    vp = VesselParticlesPipeline(op.ct_file_name,op.pl_file_name,regions,op.tmp_dir,op.output_prefix,op.init_method,
                               op.vessel_mask, op.resampling_method, op.lth,op.sth,op.voxel_size,op.min_scale,
                               op.max_scale,op.vesselness_th,crop,op.rate,op.multires,op.justparticles,op.clean_cache)

    vp._distance_from_wall=op.distance_from_wall

    vp.execute()
