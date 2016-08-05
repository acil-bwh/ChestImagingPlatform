import sys
import os
import nipype.pipeline.engine as pe         # the workflow and node wrappers
from nipype import SelectFiles, Node
from cip_python.nipype.cip_node import CIPNode
from cip_python.nipype.cip_convention_manager import CIPConventionManager as CM
from ..interfaces import cip
from ..interfaces.cip import cip_python_interfaces
# import cip_python.nipype.interfaces.cip.cip_python_interfaces \
#   as cip_python_interfaces
import nipype.interfaces.utility as util     # utility
import nipype.interfaces.io as nio           # Data i/o
from nipype import config, logging
#from cip_workflow import CIPWorkflow
import pdb
import nipype.interfaces.utility as niu
from optparse import OptionParser
                               
class ParenchymaPhenotypesWorkflow(pe.Workflow):
    def __init__(self, tmp_dir, in_ct=None,  out_lm=None, out_csv= None,  filter_image = False, 
                  cid=None, chest_regions = None, chest_types = None, pairs = None, 
                 pheno_names = None, median_filter_radius=None, save_graph=False):                                                                                             
        """ set up inputs to workflow"""
        self._in_ct = in_ct
        self._out_lm = out_lm
        self._out_csv = out_csv
        self._tmp_dir = tmp_dir
        self._median_filter_radius = median_filter_radius 
        self._regions = chest_regions 
        self._types = chest_types
        self._pairs = pairs 
        self._pheno_names = pheno_names 
        self._filter_image = filter_image
        self._save_graph = save_graph
        self._cid = cid
        
        pe.Workflow.__init__(self, "ParenchymaPhenotypesWorkflow")  
        
        if cid is None:
            self._cid = in_ct[max([in_ct.rfind('/'), 0])+1:\
                                 in_ct.rfind('.')]
        
        if in_ct.rfind('/') != -1:
            self._dir = in_ct[0:in_ct.rfind('/')]
        else:
            self._dir = '.'
            
        if out_lm is None:
            self._out_lm = \
            os.path.join(self._dir, self._cid + CM._partialLungLabelmap)
        else:
            self._out_lm = out_lm
        if out_csv is None:
            self._out_csv = \
            os.path.join(self._dir, self._cid + CM._parenchymaPhenotypes)
        else:
            self._out_csv = out_csv 
            
        self._median_filtered_file = os.path.join(self._tmp_dir, self._cid+CM._medianFilteredImage)       
                      

    def myfunction(case_id):
        return                 
          
    def set_up_workflow(self):
        """ Set up nodes that will make the wf  
        """
                
        self.config['execution'] = {'remove_unnecessary_outputs': 'False'}                                   
        
                                                                                                                                                                                                                                                                                                                                                                                                                                        
        if self._filter_image is True:
            median_filter_generator_node = \
              pe.Node(interface=cip.GenerateMedianFilteredImage(),
                      name='generate_median_filtered_image') 
            median_filter_generator_node.set_input('inputFile', self._in_ct) #inputFile
            median_filter_generator_node.set_input('outputFile', self._median_filtered_file)   #outputFile         
            median_filter_generator_node.set_input('Radius', self._median_filter_radius)
            #self.add_nodes([median_filter_generator_node])
             
        label_map_generator_node = pe.Node(cip.GeneratePartialLungLabelMap(),
            'generate_label_map') # no need to define the input ct if it is an output of another node.      
        #self.add_nodes([label_map_generator_node])
            
        label_map_generator_node.set_input( 'out', self._out_lm)
        
        parenchyma_phenotype_generator_node = CIPNode(interface=cip_python_interfaces.parenchyma_phenotypes(),
            name='generate_lung_phenotypes') 
            
        self.add_nodes([parenchyma_phenotype_generator_node])  
        parenchyma_phenotype_generator_node.set_input('pheno_names', self._pheno_names)
        parenchyma_phenotype_generator_node.set_input('out_csv', self._out_csv)
        parenchyma_phenotype_generator_node.set_input('in_ct', self._in_ct) 
        parenchyma_phenotype_generator_node.set_input('cid', self._cid) 
                        
        if (self._regions is not None):
            parenchyma_phenotype_generator_node.set_input('chest_regions', self._regions)
 
        if (self._types is not None):
            parenchyma_phenotype_generator_node.set_input('chest_types', self._types)

        if (self._pairs is not None):
            parenchyma_phenotype_generator_node.set_input('pairs', self._pairs)                                                      
         
        self.base_dir = self._tmp_dir
        
        """connect nodes based on whether we are performing median filtering or not"""
       
        if self._filter_image is True:
            self.connect(median_filter_generator_node, 'outputFile', label_map_generator_node, 'ct')   
        else:
            label_map_generator_node.set_input('ct', self._in_ct) 

        self.connect(label_map_generator_node, 'out', parenchyma_phenotype_generator_node, 'in_lm')
        
        # required to write graph : https://www.drupal.org/project/graphviz_filter   
        if self._save_graph: 
            self.write_graph(dotfilename="parenchyma_workflow_graph.dot")


if __name__ == "__main__": 
    
    desc = """Invokes body parenchyma phenotype computation workflow"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_ct',
                  help='Input CT filename',
                  dest='in_ct', metavar='<string>')
    parser.add_option('--out_lm',
                  help='Output lung labelmap filename. If none is specified, \
                      a file name will be created using the CT file name \
                      prefix with the suffix _partialLungLabelmap.nhdr. ',
                  dest='out_lm', metavar='<string>') 
    parser.add_option('--out_csv',
                  help='Output parenchyma phenotypes filename. If none is \
                      specified, a file name will be created using the CT file \
                      name prefix with the suffix _parenchymaPhenotypes.csv.',
                  dest='out_csv', metavar='<string>' )
    parser.add_option('--temp_dir',
                  help='Directory where to store temporary files.',
                  dest='temp_dir', metavar='<string>')                
    parser.add_option('--save_graph',
                  help='generate a graph of the wrorkflow. Requires \
                  graphviz_filter',
                  action="store_true", dest="save_graph", default=False)                          
    parser.add_option('--cid',
                  help='String identifying the case id of the file being \
                processed. This is needed ot be saved in the parenchyma \
                phenotypes workflow.',
                  dest='cid', metavar='<string>', default=None )                                                           
    (options, args) = parser.parse_args()


    """ hard code the variables that we don't want to have input by the user"""
    filter_image_bool = True
    the_pheno_names =  "Volume,Mass"
    regions = "WholeLung" 
    chest_types = None
    pairs = None
    median_filter_radius = 1.0
    output_dir=""
    
    my_workflow = ParenchymaPhenotypesWorkflow(options.temp_dir, in_ct=options.in_ct, 
                    out_lm=options.out_lm, out_csv=options.out_csv, filter_image = filter_image_bool, 
                    cid=options.cid, chest_regions = regions, chest_types = chest_types, 
                    pairs = pairs, pheno_names = the_pheno_names, 
                    median_filter_radius=median_filter_radius, 
                    save_graph=options.save_graph)
    my_workflow.set_up_workflow()
    my_workflow.run()           
