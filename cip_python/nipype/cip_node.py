import nipype.pipeline.engine as pe
import CIPConventionManager

class CIPNode(pe.Node):
    """
    """
    def __init__(self):
        self._convention_manager = CIPConventionManager.CIPConventionManager()

    def addParameter(self, name, value, convention_id = 0):
        """

        Parameters
        ----------
        name :

        value :

        conventions_id :
        
        """
        if conventionId != self.conventionManager.UNKNOWN:
            value = self.conventionManager.applyConvention(value, conventionId)
        pe.Node.addParameter(name, value)

