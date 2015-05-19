
class CIPConventionManager:
    NONE = 0
    CT = 1
    MedianFilteredImage = 2
    PartialLungLabelmap = 100
    ParenchymaPhenotypes = 200

    _vesselSeedsMask = '_vesselSeedsMask.nhdr'
    
    @staticmethod
    def applyConvention(value, conventionId):
        # Images
        if conventionId == CIPConventionManager.CT:
            return value + ".nhdr"

        if conventionId == CIPConventionManager.MedianFilteredImage:
            return value + "_medianFilteredImage.nhdr"
                
        # Labelmaps
        if conventionId == CIPConventionManager.PartialLungLabelmap:
            return value + "_partialLungLabelmap.nhdr"
                
        # phenotypes
        if conventionId == CIPConventionManager.ParenchymaPhenotypes:
            return value + "_parenchymaPhenotypes.csv"
        # ....

        return value

