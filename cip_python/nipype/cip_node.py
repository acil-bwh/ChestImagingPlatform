import nipype.pipeline.engine as pe
from cip_python.nipype import cip_convention_manager as cm

class CIPNode(pe.Node):
    """
    """
    def __init__(self, interface, name, iterables=None, itersource=None, synchronize=False, overwrite=None, needed_outputs=None, run_without_submitting=False):
        pe.Node.__init__(self, interface, name=name, iterables=iterables, itersource=itersource, synchronize=synchronize, overwrite=overwrite, needed_outputs=needed_outputs, run_without_submitting=run_without_submitting)

    def set_input(self, name, value, convention_id = 0):
        """

        Parameters
        ----------
        name :

        value :

        conventions_id :
        
        """
        if convention_id != cm.CIPConventionManager.NONE:
            value = cm.CIPConventionManager.applyConvention(value, convention_id)
        
        super(CIPNode, self).set_input(name, value)        
        #self.set_input(name, value)

