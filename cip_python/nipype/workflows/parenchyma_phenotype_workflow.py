
import sys
import os
import nipype.interfaces.spm as spm         # the spm interfaces
import nipype.pipeline.engine as pe         # the workflow and node wrappers
from nipype import SelectFiles, Node
from cip_python.nipype.cip_node import CIPNode
from cip_python.nipype.cip_convention_manager import CIPConventionManager
import cip_python.nipype.interfaces.cip as cip
import cip_python.nipype.interfaces.cip.cip_pythonWrap as cip_python_interfaces
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
        if self.filter_image is True:
            median_filter_generator_node = \
              CIPNode(interface=cip.GenerateMedianFilteredImage(),
                      name='generate_median_filtered_image') # (the input names come from argstr from the interface)
            median_filter_generator_node.set_input('inputFile', os.path.join(self.root_dir,self.cid_), CIPConventionManager.CT)
            median_filter_generator_node.set_input('outputFile', os.path.join(self.root_dir,self.cid_), CIPConventionManager.MedianFilteredImage)
            median_filter_generator_node.set_input('Radius', self.median_filter_radius, CIPConventionManager.NONE)
            self.add_nodes([median_filter_generator_node])

        print(self.get_node('generate_median_filtered_image'))
            
        label_map_generator_node = CIPNode(cip.GeneratePartialLungLabelMap(),
            'generate_label_map') # no need to define the input ct if it is an output of another node.      
        self.add_nodes([label_map_generator_node])
            
        if self.filter_image is False:
            label_map_generator_node.set_input('ct', os.path.join(self.root_dir,self.cid_), CIPConventionManager.CT)      
            #else:
                  #self.set_input('generate_label_map', 'ct', self.median_filtered_file)

        label_map_generator_node.set_input( 'out', os.path.join(self.root_dir,self.cid_), CIPConventionManager.PartialLungLabelmap)
  
       
        lung_parenchyma_generator_node = CIPNode(interface=cip_python_interfaces.parenchyma_phenotypes(),
            name='generate_lung_parenchyma') 
            
        self.add_nodes([lung_parenchyma_generator_node])  
        lung_parenchyma_generator_node.set_input('in_ct', os.path.join(self.root_dir,self.cid_), CIPConventionManager.CT)
        lung_parenchyma_generator_node.set_input('pheno_names', self.pheno_names, CIPConventionManager.NONE)
        lung_parenchyma_generator_node.set_input('out_csv', os.path.join(self.root_dir,self.cid_), CIPConventionManager.ParenchymaPhenotypes)
        lung_parenchyma_generator_node.set_input('cid', self.cid_, CIPConventionManager.NONE)
        
        if (self.regions is not None):
            lung_parenchyma_generator_node.set_input('chest_regions', self.regions, CIPConventionManager.NONE)

        if (self.types is not None):
            lung_parenchyma_generator_node.set_input('chest_types', self.types, CIPConventionManager.NONE)

        if (self.pairs is not None):
            lung_parenchyma_generator_node.set_input('pairs', self.pairs, CIPConventionManager.NONE)                                                      

        self.base_dir = self.root_dir

        """connect nodes based on whether we are performing median filtering or not"""
        self.connect(label_map_generator_node, 'out', lung_parenchyma_generator_node, 'in_lm')

        if self.filter_image is True:
            self.connect(median_filter_generator_node, 'outputFile', label_map_generator_node, 'ct')
#        else:
#            self.connect(label_map_generator_node, 'out', lung_parenchyma_generator_node, 'in_lm')
#            
        self.write_graph(dotfilename="parenchyma_workflow_graph.dot")


if __name__ == "__main__": 
    
    the_pheno_names =  "Volume"#,Mass"
    regions = "WholeLung"
    median_filter_radius = [1,2]
    my_workflow = ParenchymaPhenotypesWorkflow('11088Z_INSP_STD_TEM_COPD', '/Users/rolaharmouche/Documents/Data/COPDGene/11088Z/11088Z_INSP_STD_TEM_COPD/', filter_image = False, 
                 chest_regions = regions, chest_types = None, pairs = None, pheno_names = the_pheno_names, median_filter_radius=None)
    my_workflow.set_up_workflow()
    my_workflow.run()             
                    