import cip_python.nipype.interfaces.cip as cip
import cip_python.nipype.interfaces.cip.cip_python_interfaces \
  as cip_python_interfaces
import nipype.interfaces.spm as spm         # the spm interfaces
import nipype.pipeline.engine as pe         # the workflow and node wrappers
import sys
import os 
from nipype import SelectFiles, Node
import nipype.interfaces.utility as util     # utility
import nipype.interfaces.io as nio           # Data i/o

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

# iterables : http://nipy.sourceforge.net/nipype/users/mapnode_and_iterables.html

with open(case_list) as f:
    cases = f.readlines()

cases = [s.strip('\n') for s in cases]
pids = [s.split('_')[0] for s in cases]

#http://nipy.sourceforge.net/nipype/users/examples/fmri_spm.htmlinfo

#/Users/rolaharmouche/Documents/Data/COPDGene/10138J/10138J_INSP_STD_NJC_COPD/10138J_INSP_STD_NJC_COPD_pecSlice.nrrd
# We will be connecting infosource to datasoure through subject_id
#info = dict('subject_id')

infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")
infosource.iterables = ('subject_id', cases)



datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id']),
                     name = 'datasource')
datasource.inputs.base_directory = root_dir

datasource.inputs.template = '*/%s/*.nhdr'
#datasource.inputs.template_args = info # not necessary in simple case
#datasource.inputs.sort_filelist = True

#median_filter_templates={"inputFile": "/{root_dir}/{study}/{the_cid}.split('_')[0]/{the_cid}/{the_cid}.nhdr",
#            "outputFile": "/{root_dir}/{study}/{the_cid}.split('_')[0]/{the_cid}/{the_cid}_medianFiltered_nipype.nhdr"}
            
median_filter_generator = pe.Node(interface=cip.GenerateMedianFilteredImage(Radius=median_filter_radius[0] ), # do I need to specify output file name?
                   name='generate_median_filtered_image') # (the input names come from argstr from the interface)

                      
#median_filter_generator_node = Node(SelectFiles(median_filter_templates), "generate_median_filtered_image")
#median_filter_generator_node.iterables = ('the_cid', cases)


# then for each case
#median_filter_generator_node.inputs.subject_id = "subj1"
#print(median_filter_generator_node.inputs.get())

    # create the workflow            
workflow = pe.Workflow(name='generate_lung_phenotype_workflow')
workflow.base_dir = '.'
    
#generate label map then parenchyma
#workflow.add_nodes([datasource, infosource, median_filter_generator_node]) # not necessary if connecting all nodes                    
workflow.connect(infosource, datasource, [('subject_id', 'subject_id')])    
workflow.connect(datasource, median_filter_generator, [('func', 'input_CT')])  

#    workflow.connect(median_filter_generator, 'outputFile', label_map_generator, 'ct')


    # for the thing below to work: https://www.drupal.org/project/graphviz_filter
    #workflow.write_graph(dotfilename=os.path.join("/Users/rolaharmouche/Documents/Data/parenchyma_workflow_graph.dot"))
workflow.run()

# Still need to grab outputs
    
    # different params / caseIDs
    ##http://nipy.sourceforge.net/nipype/users/tutorial_102.html
    
