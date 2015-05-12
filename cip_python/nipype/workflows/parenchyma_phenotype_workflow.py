
import sys
import os
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
import pdb
import nipype.interfaces.utility as niu
from optparse import OptionParser

                    
               
class ParenchymaPhenotypesWorkflow(pe.Workflow):

    def __init__(self, list_of_cases_, case_dir, output_dir, filter_image = False, 
                 chest_regions = None, chest_types = None, pairs = None, 
                 pheno_names = None, median_filter_radius=None, save_graph=False):
                    
        """ set up inputs to workflow"""
        self.list_of_cases_ = list_of_cases_
        self.root_dir = case_dir
        self.median_filter_radius = median_filter_radius 
        self.regions = chest_regions 
        self.types = chest_types
        self.pairs = pairs 
        self.pheno_names = pheno_names 
        self.filter_image = filter_image
        self.save_graph = save_graph
        self.output_dir = output_dir
        #self.cid = <object of some kind>
        pe.Workflow.__init__(self, "ParenchymaPhenotypesWorkflow")  
        

    #def getstripdir(subject_id):
    #    sid = subject_id.split('_')[0]
    #    return os.path.join(os.path.abspath('.'),sid, subject_id) # too acilish

    def myfunction(case_id):
        return                 
          
    def set_up_workflow(self):
        """ Set up nodes that will make the wf  
        """
                
        self.config['execution'] = {'remove_unnecessary_outputs': 'False'} 
        # node.output_dir( returns the output directory if we want to remove temp files)
                                   
        subject_list= self.list_of_cases_
        print(subject_list)
        infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")
        infosource.iterables = ('subject_id', subject_list)

        datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'], outfields=['normal_run']),
                     name = 'datasource')
        datasource.inputs.base_directory = self.root_dir
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
           
        label_map_generator_node = CIPNode(cip.GeneratePartialLungLabelMap(),
            'generate_label_map') # no need to define the input ct if it is an output of another node.      
        self.add_nodes([label_map_generator_node])
            
        label_map_generator_node.set_input( 'out', 'temp', CIPConventionManager.PartialLungLabelmap)
        
        parenchyma_phenotype_generator_node = CIPNode(interface=cip_python_interfaces.parenchyma_phenotypes(),
            name='generate_lung_phenotypes') 
            
        self.add_nodes([parenchyma_phenotype_generator_node])  
        parenchyma_phenotype_generator_node.set_input('pheno_names', self.pheno_names, CIPConventionManager.NONE)
        parenchyma_phenotype_generator_node.set_input('out_csv', 'temp', CIPConventionManager.ParenchymaPhenotypes)
                
        #pdb.set_trace()
        if (self.regions is not None):
            parenchyma_phenotype_generator_node.set_input('chest_regions', self.regions, CIPConventionManager.NONE)
 
        if (self.types is not None):
            parenchyma_phenotype_generator_node.set_input('chest_types', self.types, CIPConventionManager.NONE)

        if (self.pairs is not None):
            parenchyma_phenotype_generator_node.set_input('pairs', self.pairs, CIPConventionManager.NONE)                                                      

        lung_labelmap_nrrd_identity_node = CIPNode(interface=cip_python_interfaces.nhdr_handler(),
            name='lung_labelmap_identity_node')      
        
        lung_labelmap_rename_node = CIPNode(cip.ReadWriteImageData(),
            name='lung_labelmap_rename_node')          
        
 
        csv_rename = pe.Node(niu.Rename(format_string='%(subject_id)s_parenchymaPhenotypes',
                            keep_ext=True),
                    name='namer')
        
        datasink = pe.Node(interface=nio.DataSink(), name="datasink")
        datasink.inputs.base_directory = self.output_dir
        datasink.inputs.parameterization = False # to remove the creation of a subdirectory for each run

        #getstripdir(subject_id)
        
        self.base_dir = self.root_dir
        """connect nodes based on whether we are performing median filtering or not"""
       
        self.connect(infosource, 'subject_id', datasource, 'subject_id')    
        if self.filter_image is True:
            self.connect(datasource, 'ct_input', median_filter_generator_node,  'inputFile')            
            self.connect(median_filter_generator_node, 'outputFile', label_map_generator_node, 'ct')   
        else:
            self.connect(datasource, 'ct_input', label_map_generator_node, 'ct')              

        self.connect(label_map_generator_node, 'out', parenchyma_phenotype_generator_node, 'in_lm')
        self.connect(datasource, 'ct_input', parenchyma_phenotype_generator_node,  'in_ct')    
        self.connect(infosource, 'subject_id', parenchyma_phenotype_generator_node,  'cid')
        
        # Renaming for output, worked before Jorge's changes
        self.connect(infosource, 'subject_id', lung_labelmap_nrrd_identity_node,  'case_id')
        self.connect(label_map_generator_node, 'out', lung_labelmap_nrrd_identity_node,  'in_nhdr')     
        self.connect(label_map_generator_node, 'out', lung_labelmap_rename_node, 'il')    
           
        self.connect(lung_labelmap_nrrd_identity_node, 'out_nhdr', lung_labelmap_rename_node, 'ol')  # il, ol
               
        self.connect(parenchyma_phenotype_generator_node, 'out_csv', csv_rename, 'in_file')  
        self.connect(infosource, 'subject_id', csv_rename, 'subject_id')
        
        
        # data sink
        self.connect(csv_rename, 'out_file', datasink, '@case')                          
        self.connect(lung_labelmap_rename_node, 'ol', datasink, '@caserename')
        self.connect(lung_labelmap_nrrd_identity_node, 'out_rawgz', datasink, '@caserename2') 
        
        # required to write graph : https://www.drupal.org/project/graphviz_filter   
        if self.save_graph: 
            self.write_graph(dotfilename="parenchyma_workflow_graph.dot")

        print("****output directory***********")
        print(datasink.inputs.base_directory)
        print("****output directory***********")
        print(self.output_dir)        
        #self.connect(infosource, 'subject_id', datasink, 'container') 
        # creates a subject_id subdirectory in the sink folder

if __name__ == "__main__": 
    
    desc = """Invokes body parenchyma phenotype computation workflow"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--cl',
                  help='Case list (txt file) containing a list of case IDs \
                  (one per line, no other formatting). This assumes that the \
                  input CT file will be of the format case_id.nhdr',
                  dest='case_list', metavar='<string>')
                  #, default= \
                  #['11088Z_INSP_STD_TEM_COPD','10633T_INSP_STD_NJC_COPD']
    parser.add_option('--input_data_dir',
                  help='Flat directory where all the input files reside',
                  dest='input_data_dir', metavar='<string>') 
                  #'/Users/rolaharmouche/Documents/Data/COPDGene/data_for_nypipe/'
    parser.add_option('--output_data_dir',
                  help='Flat directory where all the input files reside',
                  dest='output_data_dir', metavar='<string>' )
                  # '/Users/rolaharmouche/Documents/Data/COPDGeneNP'    
    parser.add_option('--save_graph',
                  help='generate a graph of the wrorkflow. Requires \
                  graphviz_filter',
                  action="store_true", dest="save_graph", default=False)                          
                                                               
    (options, args) = parser.parse_args()


    """ hard code the variables that we don't want to have input by the user"""
    filter_image_bool = True
    the_pheno_names =  "Volume,Mass"
    regions = "WholeLung" #LeftLung,RightLung
    chest_types = None
    pairs = None
    median_filter_radius = 1.0
    output_dir=""
    
    """ read caselist"""
    with open(options.case_list) as f2:
        case_list = f2.readlines()

    for r in range(0,len(case_list)):
        case_list[r] = case_list[r].rstrip('\n')
        
    my_workflow = ParenchymaPhenotypesWorkflow(case_list, options.input_data_dir, options.output_data_dir,filter_image = filter_image_bool, 
                 chest_regions = regions, chest_types = chest_types, pairs = pairs, pheno_names = the_pheno_names, median_filter_radius=median_filter_radius, save_graph=options.save_graph)
    my_workflow.set_up_workflow()
    my_workflow.run(plugin='MultiProc')           
                    
    #directories needed: 
    
    # args: --generate_graph                 