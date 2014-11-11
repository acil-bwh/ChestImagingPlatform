import cip_python.nipype.interfaces.cip as cip


myfilter = cip.ConvertDicom()

myfilter.run(inputDicomDirectory='/Users/rjosest/Data/COPDGene/Dicom/10766M_INSP_STD_NJC_COPD',output='/var/tmp/test.nrrd')

myfilter2 = cip.GeneratePartialLungLabelMap()


myfilter2.run(ct='/var/tmp/test.nrrd',out='/var/tmp/test2.nrrd')



