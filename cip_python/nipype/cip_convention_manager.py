

class CIPConventionManager:
    UNKNOWN = 0
    PartialLungLabelmap = 1

    # .....
    def __init__(self):
        pass

    def applyConvention(self, value, conventionId):
        if conventionId == CIPConventionManager.PartialLungLabelmap:
            return value + "_partialLungLabelmap.nhdr"
        # ....

        return value

