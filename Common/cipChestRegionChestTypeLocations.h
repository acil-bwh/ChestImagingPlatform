/**
 *  \class cipChestRegionChestTypeLocations
 *  \ingroup common
 *  \brief This class provides a container for defined chest region -
 *  chest type pairs and their associated spatial locations. Used,
 *  e.g., to contain manually labeled points. Note that no
 *  disctinction is made between points and indices. It is up to the
 *  user to interpret them as physical points or indices.
 *
 *  $Date: 2012-10-02 15:54:43 -0400 (Tue, 02 Oct 2012) $
 *  $Revision: 283 $
 *  $Author: jross $
 *
 *  TODO:
 *  
 */

#ifndef __cipChestRegionChestTypeLocations_h
#define __cipChestRegionChestTypeLocations_h

#include "cipChestConventions.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include <vector>

class cipChestRegionChestTypeLocations
{
public:
  ~cipChestRegionChestTypeLocations();
  cipChestRegionChestTypeLocations();

  /**  Pointers are assumed to have 3 elements */
  void SetChestRegionChestTypeLocation( unsigned char, unsigned char, double const* );

  /**  Pointers are assumed to have 3 elements */
  void SetChestRegionChestTypeLocation( unsigned char, unsigned char, unsigned int const* );

  /** */
  unsigned char GetChestRegionValue( unsigned int ) const;

  /** */
  unsigned char GetChestTypeValue( unsigned int ) const;

  /** */
  std::string GetChestRegionName( unsigned int ) const;

  /** */
  std::string GetChestTypeName( unsigned int ) const;

  /** */
  void GetLocation( unsigned int, cip::PointType& ) const;

  /** */
  void GetLocation( unsigned int, unsigned int* ) const;

  /** Get the number of region-type-location tuples */
  unsigned int GetNumberOfTuples() const
    {
      return NumberOfTuples;
    };

  /** For the specified chest region and chest type, get a 
      polydata representation */
  void GetPolyDataFromChestRegionChestTypeDesignation( vtkSmartPointer< vtkPolyData >, unsigned char, unsigned char );

  /** For the specified chest region , get a polydata representation */
  void GetPolyDataFromChestRegionDesignation( vtkSmartPointer< vtkPolyData >, unsigned char );
  
  /** For the specified chest type, get a polydata representation */
  void GetPolyDataFromChestTypeDesignation( vtkSmartPointer< vtkPolyData >, unsigned char );

private:
  cip::ChestConventions Conventions;

  std::vector< double* >        Locations;
  std::vector< unsigned char >  ChestRegions;
  std::vector< unsigned char >  ChestTypes;

  unsigned int NumberOfTuples;
};

#endif
