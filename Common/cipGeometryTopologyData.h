/**
 *  \class cipGeometryTopologyData
 *  \ingroup common
 *  \brief This class manages geometry and topology entities used throughout the
 *  ChestImagingPlatform. These include, e.g., points and bounding boxes. Each 
 *  entity is associated with a chest region and a chest type (UNDEFINEDREGION
 *  and UNDEFINEDTYPE by default) as well as the necessary geometry / topology
 *  information needed to represent that entity. The distinction between 
 *  geometry and topology is that geometry is just a set of points (as in the 
 *  point entity), while topology corresponds to points and relationships 
 *  between those points (as in bounding box entity).
 *
 */

#ifndef __cipGeometryTopologyData_h
#define __cipGeometryTopologyData_h

#include <vector>
#include "cipChestConventions.h"

namespace cip {
  class GeometryTopologyData
  {
  public:
    GeometryTopologyData();
    GeometryTopologyData& operator= (const GeometryTopologyData &geometryTopology);
    ~GeometryTopologyData();
    
    typedef std::vector< float > CoordinateType;
    typedef std::vector< float > StartType;
    typedef std::vector< float > SizeType;

    /** Insert a new bounding box. 'start' is the 3d physical coordinate of the
     *  bounding box's start coordinate. 'size' indicates the extent of the bounding box
     *  in the x, y, and z direction, respectively (in physical units). 'cipRegion' and
     *  'cipType' are UNDEFINEDREGION and UNDEFINEDTYPE by default. */
    void InsertBoundingBox( float start[3], float size[3], unsigned char cipRegion, 
			    unsigned char cipType );

    unsigned int GetNumberOfBoundingBoxes() const
    {
      return m_BoundingBoxes.size();
    }

    /** Returns the chest-region of the bounding box given the specified index
     *  in the vector of bounding boxes. */
    unsigned char GetBoundingBoxChestRegion( unsigned int ) const;

    /** Returns the chest-region of the bounding box given the specified index
     *  in the vector of bounding boxes. */
    unsigned char GetBoundingBoxChestType( unsigned int ) const;

    /** Returns bounding box start location given the specified index
     *  in the vector of bounding boxes. */
    StartType GetBoundingBoxStart( unsigned int ) const;

    /** Returns bounding size given the specified index in the vector of 
     *	bounding boxes. */
    SizeType GetBoundingBoxSize( unsigned int ) const;

    /** Insert a new point. 'coordinate' is the 3d physical coordinate of the point. 
     *  'cipRegion' and 'cipType' are UNDEFINEDREGION and UNDEFINEDTYPE by default. */
    void InsertPoint( float coordinate[3], unsigned char cipRegion, 
		      unsigned char cipType );
    
    /** Returns the chest-region of the point given the specified index
     *  in the vector of point. */
    unsigned char GetPointChestRegion( unsigned int ) const;

    /** Returns the chest-region of the point given the specified index
     *  in the vector of point. */
    unsigned char GetPointChestType( unsigned int ) const;

    unsigned int GetNumberOfPoints() const
    {
      return m_Points.size();
    }

    /** Get the spatial coordinate of the point indicated with the specified index */
    CoordinateType GetPointCoordinate( unsigned int ) const;

  private:
    struct BOUNDINGBOX
    {
      StartType start;
      SizeType size;
      unsigned char cipRegion;
      unsigned char cipType;
    };

    struct POINT
    {
      CoordinateType coordinate;
      unsigned char cipRegion;
      unsigned char cipType;
    };
    
    std::vector< BOUNDINGBOX > m_BoundingBoxes;
    std::vector< POINT > m_Points;
  };
  
} // namespace cip

#endif
