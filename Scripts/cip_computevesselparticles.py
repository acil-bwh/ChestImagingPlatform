#!/usr/bin/python

import sys
import string
import subprocess
import os

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-c", dest="caseId")
parser.add_option("--cipPython", dest="cipPythonDirectory")
parser.add_option("--tmpDir", dest="tmpDirectory")
parser.add_option("--dataDir", dest="dataDirectory")
parser.add_option("--cleanCache", action="store_true", dest="cleanCache", default=False)
parser.add_option("-r",dest="region")
parser.add_option("--rate",dest="rate")
parser.add_option("--init",dest="init_method",default="Frangi")
parser.add_option("--crop",dest="crop",default="0")
parser.add_option("--multires",dest="multires",action="store_true", default = False)
parser.add_option("--justparticles",dest="justparticles",action="store_true", default=False)

(options, args) = parser.parse_args()

sys.path.append(options.cipPythonDirectory)
from cip_python import GenerateAirwayParticles
from cip_python import GenerateFeatureStrengthMap
from cip_python.vessel_particles import VesselParticles
from cip_python.multires_vessel_particles import MultiResVesselParticles

caseId = options.caseId
init_method = options.init_method
assert init_method == 'Frangi' or init_method == 'Threshold'

rate=float(options.rate)

#region = [2,3]
#region=[2]
#regionTag = ['right','left']
region = [int(i) for i in str.split(options.region,',')]

regionTag = dict()
regionTag[2]='right'
regionTag[3]='left'

if options.crop == 0:
    crop_flag = False
else:
    crop = [int(i) for i in str.split(options.crop,',')]
    crop_flag = True

multires = options.multires
justparticles  = options.justparticles
print justparticles


#Check required tools path enviroment variables for tools path
toolsPaths = ['CIP_PATH','TEEM_PATH','ITKTOOLS_PATH'];
path=dict()
for path_name in toolsPaths:
    path[path_name]=os.environ.get(path_name,False)
    if path[path_name] == False:
        print path_name + " environment variable is not set"
        exit()


plFileName = os.path.join(options.dataDirectory,caseId + "_partialLungLabelMap.nhdr")
ctFileName = os.path.join(options.dataDirectory,caseId + ".nhdr")

minFeatureStrength = 60

#Crop volume if flag is set
ctFileNameCrop = os.path.join(options.tmpDirectory,caseId + "_crop.nhdr")
plFileNameCrop = os.path.join(options.tmpDirectory,caseId + "_croppartialLungLabelMap.nhdr")

if justparticles == False:

    if crop_flag == True:
        tmpCommand = "unu crop -min 0 0 %(z1)d -max M M %(z2)d -i %(in)s | unu resample -k %(kernel)s -s = = x%(factor)f -o %(out)s"

        tmpCommandCT = tmpCommand % {'z1':crop[0],'z2':crop[1],'in':ctFileName,'out':ctFileNameCrop,'factor':rate,'kernel':"cubic:1,0"}
        tmpCommandPL = tmpCommand % {'z1':crop[0],'z2':crop[1],'in':plFileName,'out':plFileNameCrop,'factor':rate,'kernel':"cheap"}
        print tmpCommandCT
        print tmpCommandPL
        subprocess.call(tmpCommandCT,shell=True)
        subprocess.call(tmpCommandPL,shell=True)
        ctFileName = ctFileNameCrop
        plFileName = plFileNameCrop

for ii in region:
    tmpDir = os.path.join(options.tmpDirectory,regionTag[ii])
    if os.path.exists(tmpDir) == False:
        os.mkdir(tmpDir)

    # Define FileNames that will be used
    plFileNameRegion= os.path.join(tmpDir,caseId + "_" + regionTag[ii] + "_partialLungLabelMap.nhdr")
    ctFileNameRegion= os.path.join(tmpDir,caseId + "_" + regionTag[ii] + ".nhdr")
    featureMapFileNameRegion = os.path.join(tmpDir,caseId + "_" + regionTag[ii] + "_featureMap.nhdr")
    maskFileNameRegion = os.path.join(tmpDir,caseId + "_" + regionTag[ii] + "_mask.nhdr")
    particlesFileNameRegion = os.path.join(options.dataDirectory,caseId + "_" + regionTag[ii] + "VesselParticles.vtk")

    if justparticles == False:

        #Create SubVolume Region
        tmpCommand ="CropLung -r %(region)d -m 1 -v 0 -i %(ct-in)s --plf %(lm-in)s -o %(ct-out)s --opl %(lm-out)s"
        tmpCommand = tmpCommand % {'region':ii,'ct-in':ctFileName,'lm-in':plFileName,'ct-out':ctFileNameRegion,'lm-out':plFileNameRegion}
        tmpCommand = os.path.join(path['CIP_PATH'],tmpCommand)
        print tmpCommand
        subprocess.call( tmpCommand, shell=True )

        #Extract Lung Region + Distance map to peel lung
        tmpCommand ="ExtractChestLabelMap -r %(region)d -i %(lm-in)s -o %(lm-out)s"
        tmpCommand = tmpCommand % {'region':ii,'lm-in':plFileNameRegion,'lm-out':plFileNameRegion}
        tmpCommand = os.path.join(path['CIP_PATH'],tmpCommand)
        print tmpCommand
        subprocess.call( tmpCommand, shell=True )

        tmpCommand ="unu 2op gt %(lm-in)s 0.5 -o %(lm-out)s"
        tmpCommand = tmpCommand % {'lm-in':plFileNameRegion,'lm-out':plFileNameRegion}
        print tmpCommand
    ###subprocess.call( tmpCommand, shell=True )

        tmpCommand ="pxdistancetransform -in %(lm-in)s -out %(distance-map)s"
        tmpCommand = tmpCommand % {'lm-in':plFileNameRegion,'distance-map':plFileNameRegion}
        tmpCommand = os.path.join(path['ITKTOOLS_PATH'],tmpCommand)
        print tmpCommand
        subprocess.call( tmpCommand, shell=True )

        tmpCommand ="unu 2op lt %(distance-map)s -2 -t short -o %(lm-out)s"
        tmpCommand = tmpCommand % {'distance-map':plFileNameRegion,'lm-out':plFileNameRegion}
        print tmpCommand
        subprocess.call( tmpCommand, shell=True )


        if False:
            # Create Feature Strength Map for the Region
            strengthMap = FeatureStrengthMap("ridge_line",ctFileNameRegion,featureMapFileNameRegion,tmpDir)
            strengthMap._clean_tmp_dir=False
            strengthMap._max_scale = 4
            strengthMap._scale_samples = 5
            strengthMap._max_feature_strength = 10000
            strengthMap._probe_scales = [0.75,1,1.75,3]

            strengthMap.execute()

            # Threshold Feature Strength Map
            tmpCommand ="unu 2op gt %(input)s %(min-strength)f -t short -o %(output)s; unu 2op gt %(lm-input)s 0.5 | unu 2op x - %(output)s -o %(output)s"
            tmpCommand = tmpCommand % {'input':featureMapFileNameRegion,'min-strength':minFeatureStrength,'lm-input':plFileNameRegion,'output':maskFileNameRegion}
            print tmpCommand
            subprocess.call( tmpCommand, shell=True )

    # Compute Frangi
        if init_method == 'Frangi':
            tmpCommand = "pxenhancement -in %(in)s -m FrangiVesselness -std %(minscale)f 3 4 -ssm 0 -alpha 0.5 -beta 0.5 -C 250 -out %(out)s"
            minscale=0.7/rate
            tmpCommand = tmpCommand % {'in':ctFileNameRegion,'out':featureMapFileNameRegion,'minscale':minscale}
            tmpCommand  = os.path.join(path['ITKTOOLS_PATH'],tmpCommand)
            print tmpCommand
            subprocess.call( tmpCommand, shell=True )

            #Hist equalization, threshold Feature strength and masking
            tmpCommand = "unu 2op x %(feat)s %(mask)s -t float | unu heq -b 10000 -a 0.5 | unu 2op gt - 0.55  | unu convert -t short -o %(out)s"
            tmpCommand = tmpCommand % {'feat':featureMapFileNameRegion,'mask':plFileNameRegion,'out':maskFileNameRegion}
            print tmpCommand
            subprocess.call( tmpCommand , shell=True)
        elif init_method == 'Threshold':
            tmpCommand = "unu 2op gt %(in)s -700 | unu 2op x - %(mask)s -o %(out)s"
            tmpCommand = tmpCommand % {'in':ctFileNameRegion,'mask':plFileNameRegion,'out':maskFileNameRegion}
            print tmpCommand
            subprocess.call( tmpCommand , shell=True)
        
        #Binary Thinning
        tmpCommand = "/Users/rjosest/src/external/itkBinaryThinnking3D/Source/bin/BinaryThinning3D %(in)s %(out)s"
        tmpCommand = tmpCommand % {'in':maskFileNameRegion,'out':maskFileNameRegion}
        print tmpCommand
        subprocess.call( tmpCommand, shell=True)

    # Airway Particles For the Region
    if multires==False:
        lt=-500*float(rate)
        st=-400*float(rate)
        lt=float(-400)
        st=float(-300)
        particlesGenerator = VesselParticles(ctFileNameRegion,particlesFileNameRegion,tmpDir,maskFileNameRegion,live_thresh=lt,seed_thresh=st)
        particlesGenerator._clean_tmp_dir=options.cleanCache
        particlesGenerator.execute()
    else:
        particlesGenerator = MultiResVesselParticles(ctFileNameRegion,particlesFileNameRegion,tmpDir,maskFileNameRegion,live_thresh=-600,seed_thresh=-600)
        particlesGenerator._clean_tmp_dir=options.cleanCache
        particlesGenerator.execute()
