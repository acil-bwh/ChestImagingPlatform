
import sys
import os
import nipype.interfaces.spm as spm         # the spm interfaces
import nipype.pipeline.engine as pe         # the workflow and node wrappers
from nipype import SelectFiles, Node
from cip_python.nipype.cip_node import CIPNode
from cip_python.nipype.cip_convention_manager import CIPConventionManager
import cip_python.nipype.interfaces.cip as cip
import cip_python.nipype.interfaces.cip.cip_pythonWrap as cip_python_interfaces
import nipype.interfaces.utility as util     # utility
import nipype.interfaces.io as nio           # Data i/o
from nipype import config, logging
#from cip_workflow import CIPWorkflow

# http://nipy.sourceforge.net/nipype/users/tutorial_101.html
# wrap python. xml definition?              
               
class ParenchymaPhenotypesWorkflow(pe.Workflow):

    def __init__(self, cid, case_dir, filter_image = False, 
                 chest_regions = None, chest_types = None, pairs = None, 
                 pheno_names = None, median_filter_radius=None):
                    
        """ set up inputs to workflow"""
        self.cid_ = cid
        self.root_dir = case_dir
        self.median_filter_radius = median_filter_radius 
        self.regions = chest_regions 
        self.types = chest_types
        self.pairs = pairs 
        self.pheno_names = pheno_names 
        self.filter_image = filter_image
        
        pe.Workflow.__init__(self, "ParenchymaPhenotypesWorkflow")  
        

    def getstripdir(subject_id):
        sid = subject_id.split('_')[0]
        return os.path.join(os.path.abspath('.'),sid, subject_id)

                    
    def get_cid(self):
        """Get the case ID (CID)

        Returns
        -------
        cid : string
            The case ID (CID)
        """
        return self.cid_
          
    def set_up_workflow(self):
        """ Set up nodes that will make the wf  
        """
        
        #config.set('execution', 'remove_unnecessary_outputs', 'false')
        
        self.config['execution'] = {'remove_unnecessary_outputs': 'False'}
        # node.output_dir( returns the output directory if we want to remove temp files)
                                   
        subject_list= [self.cid_]#,"10633T_INSP_STD_NJC_COPD"]
        print(subject_list)
        #info = dict(normal_run = [['subject_id', self.cid_]])
        infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")
        infosource.iterables = ('subject_id', subject_list)

        datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'], outfields=['normal_run']),
                     name = 'datasource')
        datasource.inputs.base_directory = self.root_dir
        #datasource.inputs.template = '%s.nhdr' #CIPConventionManager.CT
        ##datasource.inputs.field_template = dict(ct='%s.nhdr')
        #datasource.inputs.template_args = info #dict(ct=[['subject_id',self.cid_]])
        #
                                       
        datasource.inputs.template = '%s.nhdr'
        datasource.inputs.field_template = dict(normal_run='%s.nhdr')
        datasource.inputs.template_args = dict(ct_input=[['subject_id']])
        datasource.inputs.sort_filelist = True
                                                                                                                
        if self.filter_image is True:
            median_filter_generator_node = \
              CIPNode(interface=cip.GenerateMedianFilteredImage(),
                      name='generate_median_filtered_image') # (the input names come from argstr from the interface)
            median_filter_generator_node.set_input('outputFile', 'temp', CIPConventionManager.MedianFilteredImage) # will it work without this?
            median_filter_generator_node.set_input('Radius', self.median_filter_radius, CIPConventionManager.NONE)
            self.add_nodes([median_filter_generator_node])

        print(self.get_node('generate_median_filtered_image'))
            
        label_map_generator_node = CIPNode(cip.GeneratePartialLungLabelMap(),
            'generate_label_map') # no need to define the input ct if it is an output of another node.      
        self.add_nodes([label_map_generator_node])
            
        label_map_generator_node.set_input( 'out', 'temp', CIPConventionManager.PartialLungLabelmap)
  
       
        parenchyma_phenotype_generator_node = CIPNode(interface=cip_python_interfaces.parenchyma_phenotypes(),
            name='generate_lung_phenotypes') 
            
        self.add_nodes([parenchyma_phenotype_generator_node])  
        #parenchyma_phenotype_generator_node.set_input('in_ct', os.path.join(self.root_dir,self.cid_), CIPConventionManager.CT)
        parenchyma_phenotype_generator_node.set_input('pheno_names', self.pheno_names, CIPConventionManager.NONE)
        parenchyma_phenotype_generator_node.set_input('out_csv', 'temp', CIPConventionManager.ParenchymaPhenotypes)
        
        parenchyma_phenotype_generator_node.set_input('cid', self.cid_, CIPConventionManager.NONE)
        
        if (self.regions is not None):
            parenchyma_phenotype_generator_node.set_input('chest_regions', self.regions, CIPConventionManager.NONE)
 
        if (self.types is not None):
            parenchyma_phenotype_generator_node.set_input('chest_types', self.types, CIPConventionManager.NONE)

        if (self.pairs is not None):
            parenchyma_phenotype_generator_node.set_input('pairs', self.pairs, CIPConventionManager.NONE)                                                      


        ## nodes to handle the saving of raw.gz files
        median_filter_rename_node = CIPNode(interface=cip_python_interfaces.nhdr_handler(),
            name='median_filter_rename_node') 
            
        #median_filter_rename_node.set_input('in_ct','temp', CIPConventionManager.MedianFilteredImage) ###actually connect to output of medianfiltered image
        #median_filter_rename_node.set_input('caseid_ct','temp', CIPConventionManager.NONE) 
        
        datasink = pe.Node(interface=nio.DataSink(), name="datasink")
        datasink.inputs.base_directory = '/Users/rolaharmouche/Documents/Data/COPDGeneNP'
 
        #getstripdir(subject_id)
        
        self.base_dir = self.root_dir

         
        """connect nodes based on whether we are performing median filtering or not"""
       
        self.connect(infosource, 'subject_id', datasource, 'subject_id')    
        if self.filter_image is True:
            self.connect(datasource, 'ct_input', median_filter_generator_node,  'inputFile')
            #self.connect(datasource, 'ct_input', median_filter_generator_node,  'outputFile')     # HACK LIKE A CASTLE    
                 
            #self.connect(median_filter_generator_node, 'outputFile', label_map_generator_node, 'ct')
            self.connect(datasource, 'ct_input', median_filter_rename_node, 'caseid_ct')
            self.connect(median_filter_generator_node, 'outputFile', median_filter_rename_node, 'in_ct')
            self.connect(median_filter_rename_node, 'out_nhdr', label_map_generator_node, 'ct')
        else:
            self.connect(datasource, 'ct_input', label_map_generator_node, 'ct')              
           
        self.connect(label_map_generator_node, 'out', parenchyma_phenotype_generator_node, 'in_lm')
        self.connect(datasource, 'ct_input', parenchyma_phenotype_generator_node,  'in_ct')                 

        self.connect(median_filter_rename_node, 'out_nhdr', datasink, 'medianFilter')
        self.connect(median_filter_rename_node, 'out_rawgz', datasink, 'medianFilterraw')
        #self.connect(median_filter_generator_node,'outputFile', datasink,'medianFilter')
                                                
        self.write_graph(dotfilename="parenchyma_workflow_graph.dot")


if __name__ == "__main__": 
    
    the_pheno_names =  "Volume"#,Mass"
    regions = "WholeLung"
    median_filter_radius = [1,2]
    my_workflow = ParenchymaPhenotypesWorkflow('11088Z_INSP_STD_TEM_COPD', '/Users/rolaharmouche/Documents/Data/COPDGene/data_for_nypipe/', filter_image = True, 
                 chest_regions = regions, chest_types = None, pairs = None, pheno_names = the_pheno_names, median_filter_radius=1.0)
    my_workflow.set_up_workflow()
    my_workflow.run()             
                    