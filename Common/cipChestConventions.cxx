#include "cipChestConventions.h"
#include <algorithm>

cip::ChestConventions::ChestConventions() {}

cip::ChestConventions::~ChestConventions()
{
    for ( unsigned int i=0; i<this->s_ChestConventions.ChestRegionColors.size(); i++ )
    {
        delete[] this->s_ChestConventions.ChestRegionColors[i];
    }

    for ( unsigned int i=0; i<this->s_ChestConventions.ChestTypeColors.size(); i++ )
    {
        delete[] this->s_ChestConventions.ChestTypeColors[i];
    }
}

unsigned char cip::ChestConventions::GetCollectionValueFromName(std::vector<unsigned char> collectionValues,
                                                                std::vector<std::string> collectionNames,
                                                                std::string key) const {

    std::string upperKeyString(key);
    std::transform(upperKeyString.begin(), upperKeyString.end(), upperKeyString.begin(), ::toupper);

    for (int i = 0; i < collectionValues.size(); i++) {
        std::string currentKey(collectionNames[i]);
        std::transform(currentKey.begin(), currentKey.end(), currentKey.begin(), ::toupper);

        if (!upperKeyString.compare(currentKey)) {
            return collectionValues[i];
        }
    }
    std::cout << "Key not found: " << key << std::endl;
    throw ("Key not found: " + key);
}

unsigned char cip::ChestConventions::GetNumberOfEnumeratedChestRegions() const {
    //return m_NumberOfEnumeratedChestRegions;
    return s_ChestConventions.ChestRegions.size();
}

unsigned char cip::ChestConventions::GetNumberOfEnumeratedChestTypes() const {
    //return m_NumberOfEnumeratedChestTypes;
    return s_ChestConventions.ChestTypes.size();
}

unsigned char cip::ChestConventions::GetNumberOfEnumeratedImageFeatures() const {
//  return m_NumberOfEnumeratedImageFeatures;
    return s_ChestConventions.ImageFeatures.size();
}

unsigned char cip::ChestConventions::GetNumberOfEnumeratedPlanes() const {
    return s_ChestConventions.Planes.size();
}

/** This method checks if the chest region 'subordinate' is within
 *  the chest region 'superior'. */
bool cip::ChestConventions::CheckSubordinateSuperiorChestRegionRelationship(unsigned char subordinate,
                                                                            unsigned char superior) {
    // No matter what the superior and subordinate regions are (even
    // if they are undefined regions), if they are the same then by
    // convention the subordinate is a subset of the superior, so
    // return true
    if (subordinate == superior) {
        return true;
    }

    // The undefined region does not belong to any other
    // region (except the undefined region itself). Similarly,
    // nothing belongs to the undefined region (except the undefined
    // region). So if the above test failed, then we're considering
    // the relationship between a defined region and and undefined
    // region. Therefore return false.
    if (subordinate == (unsigned char) (UNDEFINEDREGION) ||
        superior == (unsigned char) (UNDEFINEDREGION)) {
        return false;
    }

    std::map< unsigned char, std::vector<unsigned char> >::iterator it;
    it = s_ChestConventions.ChestRegionHierarchyMap.find(subordinate);

    if (it == s_ChestConventions.ChestRegionHierarchyMap.end())
        // Bastard child (no known parents)
        return false;
    // Initialize a vector as a copy of of the child node parents
    std::vector<unsigned char> parents(s_ChestConventions.ChestRegionHierarchyMap[subordinate]);
    while (!parents.empty()) {
        // Extract last element of the list (Deep breath search)
        unsigned char parent = parents.back();
        parents.pop_back();
        if (parent == superior)
            return true;
        // Check if we have to keep searching in the hierarchy bottom-up
        it = s_ChestConventions.ChestRegionHierarchyMap.find(parent);
        if (it != s_ChestConventions.ChestRegionHierarchyMap.end()){
            // Add new parents of the current node
            std::vector<unsigned char> elems = s_ChestConventions.ChestRegionHierarchyMap[parent];
            for (std::vector<unsigned char>::iterator it=elems.begin(); it!=elems.end(); it++)
                parents.push_back(*it);
        }
    };
    return false;
}

std::string cip::ChestConventions::GetChestWildCardName() const {
    return std::string("WildCard");
}

/** Given an unsigned short value, this method will compute the
 *  8-bit region value corresponding to the input */
unsigned char cip::ChestConventions::GetChestRegionFromValue(unsigned short value) const {
    return value - ((value >> 8) << 8);
}

/** The 'color' param is assumed to have three components, each in
 *  the interval [0,1]. All chest type colors will be tested until a
 *  color match is found. If no match is found, 'UNDEFINEDTYPYE'
 *  will be returned */
unsigned char cip::ChestConventions::GetChestTypeFromColor(double *color) const {
    for (unsigned int i = 0; i < GetNumberOfEnumeratedChestTypes(); i++) {
        if (s_ChestConventions.ChestTypeColors[i][0] == color[0] &&
            s_ChestConventions.ChestTypeColors[i][1] == color[1] &&
            s_ChestConventions.ChestTypeColors[i][2] == color[2]) {
            return (unsigned char) (i);
        }
    }
    return (unsigned char) (UNDEFINEDTYPE);
}

/** The 'color' param is assumed to have three components, each in
 *  the interval [0,1]. All chest region colors will be tested until a
 *  color match is found. If no match is found, 'UNDEFINEDTYPYE'
 *  will be returned */
unsigned char cip::ChestConventions::GetChestRegionFromColor(double *color) const {
    for (unsigned int i = 0; i < GetNumberOfEnumeratedChestRegions(); i++) {
        if (s_ChestConventions.ChestRegionColors[i][0] == color[0] &&
            s_ChestConventions.ChestRegionColors[i][1] == color[1] &&
            s_ChestConventions.ChestRegionColors[i][2] == color[2]) {
            return (unsigned char) (i);
        }
    }
    return (unsigned char) (UNDEFINEDTYPE);
}

/** Given an unsigned short value, this method will compute the
 *  8-bit type value corresponding to the input */
unsigned char cip::ChestConventions::GetChestTypeFromValue(unsigned short value) const {
    return (value >> 8);
}

/** Given an unsigned char value corresponding to a chest type, this
 *  method will return the string name equivalent. */
std::string cip::ChestConventions::GetChestTypeName(unsigned char whichType) const {
    return s_ChestConventions.ChestTypeNames[int(whichType)];
}


/** Get the chest type color. 'color' param is assumed to be an
 * allocated 3 dimensional double pointer */
void cip::ChestConventions::GetChestTypeColor(unsigned char whichType, double *color) const {
    color[0] = s_ChestConventions.ChestTypeColors[int(whichType)][0];
    color[1] = s_ChestConventions.ChestTypeColors[int(whichType)][1];
    color[2] = s_ChestConventions.ChestTypeColors[int(whichType)][2];
}

/** Get the chest region color. 'color' param is assumed to be an
 * allocated 3 dimensional double pointer */
void cip::ChestConventions::GetChestRegionColor(unsigned char whichRegion, double *color) const {
    color[0] = s_ChestConventions.ChestRegionColors[int(whichRegion)][0];
    color[1] = s_ChestConventions.ChestRegionColors[int(whichRegion)][1];
    color[2] = s_ChestConventions.ChestRegionColors[int(whichRegion)][2];
}

/** Get the color corresponding to the chest-region chest-pair pair. The
 * color is computed as the average of the two corresponding region and type
 * colors unless the region or type is undefined, in which case the color of
 * the defined region or type is returned. The 'color' param is assumed to be
 * an allocated 3 dimensional double pointer */
void cip::ChestConventions::GetColorFromChestRegionChestType(unsigned char whichRegion, unsigned char whichType,
                                                             double *color) const {
    if (whichRegion == (unsigned char) (cip::UNDEFINEDREGION)) {
        color[0] = s_ChestConventions.ChestTypeColors[int(whichType)][0];
        color[1] = s_ChestConventions.ChestTypeColors[int(whichType)][1];
        color[2] = s_ChestConventions.ChestTypeColors[int(whichType)][2];
    }
    else if (whichType == (unsigned char) (cip::UNDEFINEDTYPE)) {
        color[0] = s_ChestConventions.ChestRegionColors[int(whichRegion)][0];
        color[1] = s_ChestConventions.ChestRegionColors[int(whichRegion)][1];
        color[2] = s_ChestConventions.ChestRegionColors[int(whichRegion)][2];
    }
    else {
        color[0] = (s_ChestConventions.ChestRegionColors[int(whichRegion)][0] +
                    s_ChestConventions.ChestTypeColors[int(whichType)][0]) / 2.0;
        color[1] = (s_ChestConventions.ChestRegionColors[int(whichRegion)][1] +
                    s_ChestConventions.ChestTypeColors[int(whichType)][1]) / 2.0;
        color[2] = (s_ChestConventions.ChestRegionColors[int(whichRegion)][2] +
                    s_ChestConventions.ChestTypeColors[int(whichType)][2]) / 2.0;
    }
}

/** Given an unsigned char value corresponding to a chest region, this
 *  method will return the string name equivalent. */
std::string cip::ChestConventions::GetChestRegionName(unsigned char whichRegion) const {
    return s_ChestConventions.ChestRegionNames[int(whichRegion)];
}

/** Given an unsigned short value, this method will return the
 *  string name of the corresponding chest region */
std::string cip::ChestConventions::GetChestRegionNameFromValue(unsigned short value) const {
    unsigned char regionValue = 0;

    regionValue = this->GetChestRegionFromValue(value);

    return this->GetChestRegionName(regionValue);
};

/** Given an unsigned short value, this method will return the
 *  string name of the corresponding chest type */
std::string cip::ChestConventions::GetChestTypeNameFromValue(unsigned short value) const {
    unsigned char typeValue = 0;

    typeValue = this->GetChestTypeFromValue(value);
    return this->GetChestTypeName(typeValue);
}

unsigned short cip::ChestConventions::GetValueFromChestRegionAndType(unsigned char region,
                                                                     unsigned char type) const {
    unsigned short regionValue = (unsigned short) region;
    unsigned short tmp = (unsigned short) type;
    unsigned short regionType = (tmp << 8);
    unsigned short combinedValue = regionValue + regionType;
    return combinedValue;
}

unsigned char cip::ChestConventions::GetChestRegionValueFromName(std::string regionString) const {
    return this->GetCollectionValueFromName(s_ChestConventions.ChestRegions, s_ChestConventions.ChestRegionNames, regionString);
}

unsigned char cip::ChestConventions::GetChestTypeValueFromName(std::string typeString) const {
    return this->GetCollectionValueFromName(s_ChestConventions.ChestTypes, s_ChestConventions.ChestTypeNames, typeString);
}

unsigned char cip::ChestConventions::GetPlaneValueFromName(std::string key) const {
   return this->GetCollectionValueFromName(s_ChestConventions.Planes, s_ChestConventions.PlaneNames, key);
}

/** Get the ith chest region */
unsigned char cip::ChestConventions::GetChestRegion(unsigned int i) const{
    return (unsigned char) (s_ChestConventions.ChestRegions[i]);
}

/** Get the ith chest type */
unsigned char cip::ChestConventions::GetChestType(unsigned int i) const {
    return (unsigned char) (s_ChestConventions.ChestTypes[i]);
}


/** Get the ith image type */
unsigned char cip::ChestConventions::GetImageFeature(unsigned int i) const {
    return (unsigned char) (s_ChestConventions.ImageFeatures[i]);
}

/** Get the ith plane */
unsigned char cip::ChestConventions::GetPlane(unsigned int i) const {
    return (unsigned char) (s_ChestConventions.Planes[i]);
}

/** Given an unsigned char value corresponding to an ImageFeature, this
 *  method will return the string name equivalent. */
std::string cip::ChestConventions::GetImageFeatureName(unsigned char whichFeature) const {
    return s_ChestConventions.ImageFeatureNames[int(whichFeature)];
}

/** Given an unsigned char value corresponding to Plane, this
 *  method will return the string name equivalent. */
std::string cip::ChestConventions::GetPlaneName(unsigned char whichPlane) const {
    return s_ChestConventions.PlaneNames[int(whichPlane)];
}

/** Returns true if the passed string name is among the allowed body composition
 *  phenotype names and returns false otherwise */
bool cip::ChestConventions::IsBodyCompositionPhenotypeName(std::string pheno) const {
    for (int i = 0; i < s_ChestConventions.BodyCompositionPhenotypeNames.size(); i++) {
        if (!s_ChestConventions.BodyCompositionPhenotypeNames[i].compare(pheno)) {
            return true;
        }
    }

    return false;
}

/** Returns true if the passed string name is among the allowed parenchyma
 *  phenotype names and returns false otherwise */
bool cip::ChestConventions::IsParenchymaPhenotypeName(std::string pheno) const {
    for (int i = 0; i < s_ChestConventions.ParenchymaPhenotypeNames.size(); i++) {
        if (!s_ChestConventions.ParenchymaPhenotypeNames[i].compare(pheno)) {
            return true;
        }
    }

    return false;
}

/** Returns true if the passed string name is among the allowed histogram
 *  phenotype names and returns false otherwise */
bool cip::ChestConventions::IsHistogramPhenotypeName(std::string pheno) const {
    for (int i = 0; i < s_ChestConventions.HistogramPhenotypeNames.size(); i++) {
        if (!s_ChestConventions.HistogramPhenotypeNames[i].compare(pheno)) {
            return true;
        }
    }

    return false;
}

/** Returns true if the passed string name is among the allowed pulmonary vasculature
 *  phenotype names and returns false otherwise */
bool cip::ChestConventions::IsPulmonaryVasculaturePhenotypeName(std::string pheno) const {
    for (int i = 0; i < s_ChestConventions.PulmonaryVasculaturePhenotypeNames.size(); i++) {
        if (!s_ChestConventions.PulmonaryVasculaturePhenotypeNames[i].compare(pheno)) {
            return true;
        }
    }

    return false;
}


/** Returns true if the passed string name is among the allowed pulmonary vasculature
 *  phenotype names and returns false otherwise */
bool cip::ChestConventions::IsAirwayPhenotypeName(std::string pheno) const {
    for (int i = 0; i < s_ChestConventions.AirwayPhenotypeNames.size(); i++) {
        if (!s_ChestConventions.AirwayPhenotypeNames[i].compare(pheno)) {
            return true;
        }
    }

    return false;
}

/** Returns true if the passed string name is among the allowed pulmonary vasculature
 *  phenotype names and returns false otherwise */
bool cip::ChestConventions::IsBiomechanicalPhenotypeName(std::string pheno) const {
    for (int i = 0; i < s_ChestConventions.BiomechanicalPhenotypeNames.size(); i++) {
        if (!s_ChestConventions.BiomechanicalPhenotypeNames[i].compare(pheno)) {
            return true;
        }
    }

    return false;
}

/** Returns true if the passed string name is among the allowed fissure
 *  phenotype names and returns false otherwise */
bool cip::ChestConventions::IsFissurePhenotypeName(std::string pheno) const {
    for (int i = 0; i < s_ChestConventions.FissurePhenotypeNames.size(); i++) {
        if (!s_ChestConventions.FissurePhenotypeNames[i].compare(pheno)) {
            return true;
        }
    }

    return false;
}

/** Returns true if the passed string name is among the allowed
 *  phenotype names and returns false otherwise */
bool cip::ChestConventions::IsPhenotypeName(std::string pheno) const {
    for (int i = 0; i < s_ChestConventions.ParenchymaPhenotypeNames.size(); i++) {
        if (!s_ChestConventions.ParenchymaPhenotypeNames[i].compare(pheno)) {
            return true;
        }
    }

    for (int i = 0; i < s_ChestConventions.BodyCompositionPhenotypeNames.size(); i++) {
        if (!s_ChestConventions.BodyCompositionPhenotypeNames[i].compare(pheno)) {
            return true;
        }
    }

    for (int i = 0; i < s_ChestConventions.HistogramPhenotypeNames.size(); i++) {
        if (!s_ChestConventions.HistogramPhenotypeNames[i].compare(pheno)) {
            return true;
        }
    }

    for (int i = 0; i < s_ChestConventions.PulmonaryVasculaturePhenotypeNames.size(); i++) {
        if (!s_ChestConventions.PulmonaryVasculaturePhenotypeNames[i].compare(pheno)) {
            return true;
        }
    }

    for (int i = 0; i < s_ChestConventions.AirwayPhenotypeNames.size(); i++) {
        if (!s_ChestConventions.AirwayPhenotypeNames[i].compare(pheno)) {
            return true;
        }
    }

    for (int i = 0; i < s_ChestConventions.BiomechanicalPhenotypeNames.size(); i++) {
        if (!s_ChestConventions.BiomechanicalPhenotypeNames[i].compare(pheno)) {
            return true;
        }
    }

    for (int i = 0; i < s_ChestConventions.FissurePhenotypeNames.size(); i++) {
        if (!s_ChestConventions.FissurePhenotypeNames[i].compare(pheno)) {
            return true;
        }
    }
    
    return false;
}

/** Returns true if the passed string name is among the enumerated chest
 *  types and returns false otherwise */
bool cip::ChestConventions::IsChestType(std::string chestType) const {
    for (int i = 0; i < GetNumberOfEnumeratedChestTypes(); i++) {
        if (!s_ChestConventions.ChestTypeNames[i].compare(chestType)) {
            return true;
        }
    }

    return false;
}

/** Returns true if the passed string name is among the enumerated chest
 *  regions and returns false otherwise */
bool cip::ChestConventions::IsChestRegion(std::string chestRegion) const {
    for (int i = 0; i < GetNumberOfEnumeratedChestRegions(); i++) {
        if (!s_ChestConventions.ChestRegionNames[i].compare(chestRegion)) {
            return true;
        }
    }

    return false;
}


