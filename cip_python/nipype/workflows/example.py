import cip_python.nipype.interfaces.cip as cip


myfilter = cip.ConvertDicom()

myfilter.run(inputDicomDirectory='/Users/rolaharmouche/Documents/Data/STATCOPE/107896H55/107896H55_INSP_B31f_TEMPLE_STATCOPE',output='/Users/rolaharmouche/Documents/Data/STATCOPE/107896H55/test.nrrd')

#myfilter2 = cip.GeneratePartialLungLabelMap()



#myfilter2.run(ct='/var/tmp/test.nrrd',out='/var/tmp/test2.nrrd')



