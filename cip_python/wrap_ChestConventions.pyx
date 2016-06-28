from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "cipChestConventions.h" namespace "cip":
    cdef cppclass _ChestConventions "cip::ChestConventions":
        _ChestConventions()
        unsigned char GetNumberOfEnumeratedChestRegions() const
        unsigned char GetNumberOfEnumeratedChestTypes() const
        unsigned char GetChestRegionFromValue(unsigned short value) const
        unsigned char GetChestTypeFromValue(unsigned short value) const
        string GetChestTypeName(unsigned char whichType) const
        string GetChestRegionName(unsigned char whichType) const
        string GetChestRegionNameFromValue(unsigned short value) const
        string GetChestTypeNameFromValue(unsigned short value) const
        string GetChestWildCardName() const
        unsigned short GetValueFromChestRegionAndType(unsigned char region, unsigned char type) const
        unsigned char GetChestRegionValueFromName(string regionString) const
        unsigned char GetChestTypeValueFromName(string typeString) const
        bool CheckSubordinateSuperiorChestRegionRelationship(unsigned char subordinate, unsigned char superior)
        bool IsBodyCompositionPhenotypeName(string name) const
        bool IsParenchymaPhenotypeName(string name) const
        bool IsHistogramPhenotypeName(string name) const
        bool IsPulmonaryVasculaturePhenotypeName(string name) const
        bool IsPhenotypeName(string name) const
        bool IsChestRegion(string name) const
        bool IsChestType(string name) const

cdef class ChestConventions:
    cdef _ChestConventions *thisptr

    def __cinit__(self):
        self.thisptr = new _ChestConventions()

    def __dealloc__(self):
        del self.thisptr

    cpdef unsigned char GetNumberOfEnumeratedChestRegions(self):
        return self.thisptr.GetNumberOfEnumeratedChestRegions()

    cpdef unsigned char GetNumberOfEnumeratedChestTypes(self):
        return self.thisptr.GetNumberOfEnumeratedChestTypes()

    cpdef unsigned char GetChestRegionFromValue(self, unsigned short value):
        return self.thisptr.GetChestRegionFromValue(value)

    cpdef unsigned char GetChestTypeFromValue(self, unsigned short value):
        return self.thisptr.GetChestTypeFromValue(value)

    cpdef string GetChestWildCardName(self):
        return self.thisptr.GetChestWildCardName()

    cpdef string GetChestTypeName(self, unsigned char whichType):
        return self.thisptr.GetChestTypeName(whichType)

    cpdef string GetChestRegionName(self, unsigned char whichRegion):
        return self.thisptr.GetChestRegionName(whichRegion)

    cpdef string GetChestRegionNameFromValue(self, unsigned short value):
        return self.thisptr.GetChestRegionNameFromValue(value)

    cpdef string GetChestTypeNameFromValue(self, unsigned short value):
        return self.thisptr.GetChestTypeNameFromValue(value)

    cpdef unsigned short GetValueFromChestRegionAndType(self, unsigned char region, unsigned char type):
        return self.thisptr.GetValueFromChestRegionAndType(region, type)

    cpdef unsigned char GetChestRegionValueFromName(self, string regionString):
        return self.thisptr.GetChestRegionValueFromName(regionString)

    cpdef unsigned char GetChestTypeValueFromName(self, string typeString):
        return self.thisptr.GetChestTypeValueFromName(typeString)

    cpdef bool CheckSubordinateSuperiorChestRegionRelationship(self, unsigned char subordinate, unsigned char superior):    
        return self.thisptr.CheckSubordinateSuperiorChestRegionRelationship(subordinate, superior)

    cpdef bool IsBodyCompositionPhenotypeName(self, string name):
        return self.thisptr.IsBodyCompositionPhenotypeName(name)

    cpdef bool IsParenchymaPhenotypeName(self, string name):
        return self.thisptr.IsParenchymaPhenotypeName(name)

    cpdef bool IsHistogramPhenotypeName(self, string name):
        return self.thisptr.IsHistogramPhenotypeName(name)

    cpdef bool IsPulmonaryVasculaturePhenotypeName(self, string name):
        return self.thisptr.IsPulmonaryVasculaturePhenotypeName(name)

    cpdef bool IsPhenotypeName(self, string name):
        return self.thisptr.IsPhenotypeName(name)

    cpdef bool IsChestRegion(self, string name):    
        return self.thisptr.IsChestRegion(name)

    cpdef bool IsChestType(self, string name):    
        return self.thisptr.IsChestType(name)