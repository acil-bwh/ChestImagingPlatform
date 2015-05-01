import nipype.pipeline.engine as pe
import CIPConventionManager


class CIPNode(pe.Node):
    def __init__(self):
        self.conventionManager = CIPConventionManager.CIPConventionManager()

    def addParameter(self, name, value, conventionId = 0):
        if conventionId != self.conventionManager.UNKNOWN:
            value = self.conventionManager.applyConvention(value, conventionId)
        pe.Node.addParameter(name, value)

