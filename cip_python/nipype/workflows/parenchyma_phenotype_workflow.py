import cip_python.nipype.interfaces.cip as cip
import cip_python.nipype.interfaces.cip.cip_pythonWrap as cip_python_interfaces
import nipype.interfaces.spm as spm         # the spm interfaces
import nipype.pipeline.engine as pe         # the workflow and node wrappers
import sys
import os 
from nipype import SelectFiles, Node

sys.path.append("/Users/rolaharmouche/ChestImagingPlatformPrivate/")



the_pheno_names =  "Volume"#,Mass"
regions = "WholeLung"
median_filter_radius = [1,2]
# http://nipy.sourceforge.net/nipype/users/tutorial_101.html
# wrap python. xml definition? 


# Read caselist
case_list = "/Users/rolaharmouche/Documents/Caselists/test_nipype_caselist.txt"
study = "COPDGene"
root_dir = "/Users/rolaharmouche/Documents/Data/"



with open(case_list) as f:
    cases = f.readlines()
for the_case in cases:
    print("Executing parenchyma workflow for case: "+the_case)
    
    the_cid = the_case.strip('\n') 
    sid = the_cid.split('_')[0]

    case_dir = os.path.join(root_dir,study,sid,the_cid)
    input_CT = os.path.join(case_dir,the_cid+".nhdr")
    lung_labelmap = os.path.join(case_dir,the_cid+"_partialLungLabelMapNipype.nhdr")
    phenotype_csv = os.path.join(case_dir,the_cid+"_parenchyma_phenotypesNipype.csv")
    median_filtered_file = os.path.join(case_dir,the_cid+"_medianFiltered_nipype.nhdr")


    # create nodes for a true pipeline           
    median_filter_generator = pe.Node(interface=cip.GenerateMedianFilteredImage(inputFile= input_CT,outputFile=median_filtered_file, Radius=median_filter_radius[0] ),
                   name='generate_median_filtered_image') # (the input names come from argstr from the interface)

    label_map_generator = pe.Node(interface=cip.GeneratePartialLungLabelMap(out=lung_labelmap),
                   name='generate_label_map') # no need to define the input ct because it is an output of another node
                    
    lung_parenchyma_generator = pe.Node(interface=cip_python_interfaces.parenchyma_phenotypes(\
    in_ct =input_CT, pheno_names=the_pheno_names,chest_regions=regions,out_csv = phenotype_csv,cid = the_cid),
                    name='generate_lung_parenchyma') # no need to define the input labelmap because it is an output of another node

    # create the workflow            
    workflow = pe.Workflow(name='generate_lung_phenotype_workflow')
    workflow.base_dir = '.'
    
    #generate label map then parenchyma
    workflow.add_nodes([median_filter_generator, label_map_generator, lung_parenchyma_generator]) # not necessary if connecting all nodes                    
    workflow.connect(label_map_generator, 'out', lung_parenchyma_generator, 'in_lm')
    workflow.connect(median_filter_generator, 'outputFile', label_map_generator, 'ct')
    
    # for the thing below to work: https://www.drupal.org/project/graphviz_filter
    workflow.write_graph(dotfilename=os.path.join("/Users/rolaharmouche/Documents/Data/parenchyma_workflow_graph.dot"))
    workflow.run()
    
    # different params / caseIDs
    ##http://nipy.sourceforge.net/nipype/users/tutorial_102.html
    
