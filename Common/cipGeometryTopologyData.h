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
    bool operator== (const GeometryTopologyData &geometryTopology) const;
    bool operator!= (const GeometryTopologyData &geometryTopology) const;
    ~GeometryTopologyData();
    
    typedef std::vector< double > CoordinateType;
    typedef std::vector< double > StartType;
    typedef std::vector< double > SizeType;
    
    struct BOUNDINGBOX
    {
      StartType start;
      SizeType size;
      unsigned char cipRegion;
      unsigned char cipType;
      unsigned char cipImageFeature;
      std::string description;
    };
    
    struct POINT
    {
      CoordinateType coordinate;
      unsigned char cipRegion;
      unsigned char cipType;
      unsigned char cipImageFeature;
      std::string description;
    };
    
    typedef std::vector< BOUNDINGBOX > BoundingBoxVectorType;
    typedef std::vector< POINT > PointVectorType;

    /** Insert a new bounding box. 'start' is the 3d physical coordinate of the
     *  bounding box's start coordinate. 'size' indicates the extent of the bounding box
     *  in the x, y, and z direction, respectively (in physical units). 'cipRegion' and
     *  'cipType' are UNDEFINEDREGION and UNDEFINEDTYPE by default. */
    void InsertBoundingBox( StartType start, SizeType size, unsigned char cipRegion, 
			    unsigned char cipType, unsigned char cipImageFeature, std::string );

    BoundingBoxVectorType::size_type GetNumberOfBoundingBoxes() const
    {
      return m_BoundingBoxes.size();
    }

    /** Returns the chest-region of the bounding box given the specified index
     *  in the vector of bounding boxes. */
    unsigned char GetBoundingBoxChestRegion( unsigned int ) const;

    /** Returns the chest-type of the bounding box given the specified index
     *  in the vector of bounding boxes. */
    unsigned char GetBoundingBoxChestType( unsigned int ) const;

    /** Returns the image feature of the bounding box given the specified index
     *  in the vector of bounding boxes. */
    unsigned char GetBoundingBoxImageFeature( unsigned int ) const;

    /** Returns bounding box start location given the specified index
     *  in the vector of bounding boxes. */
    StartType GetBoundingBoxStart( unsigned int ) const;

    /** Returns bounding size given the specified index in the vector of 
     *	bounding boxes. */
    SizeType GetBoundingBoxSize( unsigned int ) const;

    /** Returns the bounding box description given the specified index in the vector of 
     *	bounding boxes. */
    std::string GetBoundingBoxDescription( unsigned int ) const;

    /** Insert a new point. 'coordinate' is the 3d physical coordinate of the point. 
     *  'cipRegion' and 'cipType' are UNDEFINEDREGION and UNDEFINEDTYPE by default. */
    void InsertPoint( CoordinateType coordinate, unsigned char cipRegion, 
		      unsigned char cipType, unsigned char cipImageFeature, std::string );
    
    /** Returns the chest-region of the point given the specified index
     *  in the vector of point. */
    unsigned char GetPointChestRegion( unsigned int ) const;

    /** Returns the chest-type of the point given the specified index
     *  in the vector of point. */
    unsigned char GetPointChestType( unsigned int ) const;

    /** Returns the image feature of the point given the specified index
     *  in the vector of point. */
    unsigned char GetPointImageFeature( unsigned int ) const;

    /** Returns the point description given the specified index in the vector of 
     *	points. */
    std::string GetPointDescription( unsigned int ) const;

    PointVectorType::size_type GetNumberOfPoints() const
    {
      return m_Points.size();
    }

    /** Get the spatial coordinate of the point indicated with the specified index */
    CoordinateType GetPointCoordinate( unsigned int ) const;

  private:
    
    BoundingBoxVectorType m_BoundingBoxes;
    PointVectorType m_Points;
  };
  
} // namespace cip

#endif
