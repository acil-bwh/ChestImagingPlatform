import cip_python.nipype.interfaces.cip as cip


input_file = ""
output_file = ""
x_radius = 3
y_radius = 3
z_radius = 0


# first perform morphological closing


# then compute body composition phenotypes

my_perform_morphological_filter = cip.PerformMorphological()

my_perform_morphological_filter.run(in= input_file,out=output_file,\
      radx, x_radius,
      rady, y_radius,
      radz= z_radius,
      cl=True)





#my_body_composition_phenotypes_filter = cip.GeneratePartialLungLabelMap()


#myfilter2.run(ct='/var/tmp/test.nrrd',out='/var/tmp/test2.nrrd')



# wrap python. xml definition? 