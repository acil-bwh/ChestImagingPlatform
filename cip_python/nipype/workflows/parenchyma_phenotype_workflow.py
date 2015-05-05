import cip_python.nipype.interfaces.cip as cip
import cip_python.nipype.interfaces.cip.cip_pythonWrap as cip_python_interfaces
import nipype.interfaces.spm as spm         # the spm interfaces
import nipype.pipeline.engine as pe         # the workflow and node wrappers
import sys
import os 
from nipype import SelectFiles, Node

# http://nipy.sourceforge.net/nipype/users/tutorial_101.html
# wrap python. xml definition?              
               
class ParenchymaPhenotypesWorkflow(pe.Workflow):

    def __init__(self, cid, case_dir, input_ct, lung_labelmap, phenotype_csv, 
                 median_filtered_file, filter_image = False, 
                 chest_regions = None, chest_types = None, pairs = None, 
                 pheno_names = None, median_filter_radius=None):
                    
        """ set up inputs to workflow"""
        self.cid = cid
        self.input_CT = input_ct 
        self.root_dir = case_dir
        self.lung_labelmap= lung_labelmap 
        self.phenotype_csv = phenotype_csv 
        self.median_filtered_file = median_filtered_file 
        self.median_filter_radius = median_filter_radius 
        self.regions = chest_regions 
        self.types = chest_types 
        self.pairs = pairs 
        self.pheno_names = pheno_names 
        self.filter_image = filter_image
        
        CipWorkflow.__init__(self, "ParenchymaPhenotypesWorkflow")  
        
        #""" create the nodes and add them to the workflow"""
        #self.param.node_name.append('GenerateMedianFilteredImage')
        #self.param.name.append('Radius')
        #
        #if median_filter_radius is not None:
        #    self.param.value.append(int(median_filter_radius))
        #else:
        #    self.param.value.append(1.0)
        #    
        #node = Node(interface=int)
        #node.params.add(name,value)
              
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
            median_filter_generator = \
              pe.Node(interface=cip.GenerateMedianFilteredImage(),
                      name='generate_median_filtered_image') # (the input names come from argstr from the interface)
        
            test_workflow = pe.Workflow(name='generate_lung_phenotype_workflo2w')
            test_workflow.add_nodes([median_filter_generator]) 
        
            self.add_nodes([median_filter_generator])
            param = ParameterFactory.buildParameter('inputFile')
            self.add_node_param('generate_median_filtered_image', 'inputFile', self.input_CT)
            self.add_node_param('generate_median_filtered_image', 'outputFile', self.median_filtered_file)
            self.add_node_param('generate_median_filtered_image', 'Radius', self.median_filter_radius)
        
        print(self.get_node('generate_median_filtered_image'))
            
        label_map_generator = pe.Node(interface=cip.GeneratePartialLungLabelMap(out=self.lung_labelmap),
            name='generate_label_map') # no need to define the input ct because it is an output of another node. hmm, not always      
        self.add_nodes([label_map_generator])
        self.add_node_param('generate_label_map', 'out', self.lung_labelmap)
  
        if self.filter_image is False:
            self.add_node_param('generate_label_map', 'ct', self.input_CT)
        else:
            self.add_node_param('generate_label_map', 'ct', self.median_filtered_file)
        
        lung_parenchyma_generator = pe.Node(interface=cip_python_interfaces.parenchyma_phenotypes(),
            name='generate_lung_parenchyma') 
            
        self.add_nodes([lung_parenchyma_generator])  
        self.add_node_param('generate_lung_parenchyma', 'in_ct', self.input_CT)
        self.add_node_param('generate_lung_parenchyma', 'pheno_names', self.pheno_names)
        self.add_node_param('generate_lung_parenchyma', 'out_csv', self.phenotype_csv)
        self.add_node_param('generate_lung_parenchyma', 'cid', self.cid)
        
        if (self.regions is not None):
             self.add_node_param('generate_lung_parenchyma', 'chest_regions', self.regions)
        if (self.types is not None):
             self.add_node_param('generate_lung_parenchyma', 'chest_types', self.types)
        if (self.pairs is not None):
             self.add_node_param('generate_lung_parenchyma', 'chest_regions', self.pairs)
                                                      
        #in_ct =self.input_CT, pheno_names=self.pheno_names,chest_regions=self.regions,\
        #    chest_types=self.types, out_csv = self.phenotype_csv,cid = self.cid),
                                        
        self.base_dir = self.root_dir

        """ connect nodes based on whether we are performing median filtering or not"""
        if self.filter_image is True:
            self.connect(label_map_generator, 'out', lung_parenchyma_generator, 'in_lm')
            self.connect(median_filter_generator, 'outputFile', label_map_generator, 'ct')
        else:
            self.connect(label_map_generator, 'out', lung_parenchyma_generator, 'in_lm')
            
        self.write_graph(dotfilename="parenchyma_workflow_graph.dot")

