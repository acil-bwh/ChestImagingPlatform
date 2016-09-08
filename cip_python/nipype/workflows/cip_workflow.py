# import cip_python.nipype.interfaces.cip as cip
# import cip_python.nipype.interfaces.cip.cip_python_interfaces \
#   as cip_python_interfaces
import nipype.interfaces.spm as spm         # the spm interfaces
import nipype.pipeline.engine as pe         # the workflow and node wrappers
# from nipype import SelectFiles, Node

class CipWorkflow(pe.Workflow):      

    def add_node_param(self, node_name, param_name, param_value):    
        node = self.get_node(node_name)
        #print(node)
        #print(param_name)
        #eval(str(node.inputs)+"."+param_name)
        #node.param_var = param_value # is staying empty
        node.set_input(param_name,param_value)
        #print(node.param_var)

    def get_nodes(self):
        return self.list_node_names()     
        
    def __init__(self, name):
        pe.Workflow.__init__(self, name)            
    
    def run(self):
        return super(CipWorkflow, self).run() 
