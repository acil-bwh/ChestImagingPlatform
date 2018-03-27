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

    /** Coordinate system: UNKNOWN, IJK, RAS, LPS **/
    std::string CoordinateSystem;
    /** Transformation matrix to go from LPS coordinates to IJK **/
    std::vector<double> LPS_to_IJK_TransformationMatrix;

    typedef std::vector< double > SpacingType;
    typedef std::vector< double > OriginType;
    typedef std::vector< unsigned int > DimensionsType;

    SpacingType m_Spacing;          // Spacing of the volume
    OriginType m_Origin;            // Origin of the volume
    DimensionsType m_Dimensions;    // Dimensions (int size) of the volume

    typedef std::vector< double > CoordinateType;
    typedef std::vector< double > StartType;
    typedef std::vector< double > SizeType;
    
    struct BOUNDINGBOX
    {
      unsigned int id;
      StartType start;
      SizeType size;
      unsigned char cipRegion;
      unsigned char cipType;
      unsigned char cipImageFeature;
      std::string description;
      std::string machineName;
      std::string userName;
      std::string timestamp;
    };
    
    struct POINT
    {
      unsigned int id;
      CoordinateType coordinate;
      unsigned char cipRegion;
      unsigned char cipType;
      unsigned char cipImageFeature;
      std::string description;
      std::string machineName;
      std::string userName;
      std::string timestamp;
    };
    
    typedef std::vector< BOUNDINGBOX > BoundingBoxVectorType;
    typedef std::vector< POINT > PointVectorType;

    /** Insert a bounding box where the id, username, etc. will be calculated automatically **/
    cip::GeometryTopologyData::BOUNDINGBOX* InsertBoundingBox( unsigned char cipRegion,
                                                               unsigned char cipType,
                                                               unsigned char cipImageFeature,
                                                               StartType start,
                                                               SizeType size,
                                                               std::string description);

    /** Insert a bounding box filling all the fileds (used when parsing a XML file **/
    cip::GeometryTopologyData::BOUNDINGBOX* InsertBoundingBox( int id,
                                                               unsigned char cipRegion,
                                                               unsigned char cipType,
                                                               unsigned char cipImageFeature,
                                                               StartType start,
                                                               SizeType size,
                                                               std::string description,
                                                               std::string timestamp,
                                                               std::string userName,
                                                               std::string machineName);

    BoundingBoxVectorType::size_type GetNumberOfBoundingBoxes() const
    {
      return m_BoundingBoxes.size();
    }

    /** Get an instance of the i-th Bounding box */
    cip::GeometryTopologyData::BOUNDINGBOX GetBoundingBox( unsigned int ) const;

    /** Get the id of the i-th Bounding box */
    unsigned int GetBoundingBoxId(unsigned int index) const;

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

    /** Insert a point where the id, username, etc. will be calculated automatically **/
    cip::GeometryTopologyData::POINT* InsertPoint( unsigned char cipRegion,
                                                   unsigned char cipType,
                                                   unsigned char cipImageFeature,
                                                   CoordinateType coordinate,
                                                   std::string description);

    /** Insert a point filling all the fileds (used when parsing a XML file **/
    cip::GeometryTopologyData::POINT* InsertPoint(  int id,
                                                    unsigned char cipRegion,
                                                    unsigned char cipType,
                                                    unsigned char cipImageFeature,
                                                    CoordinateType coordinate,
                                                    std::string description,
                                                    std::string timestamp,
                                                    std::string userName,
                                                    std::string machineName);


    /** Get an instance of the i-th Point */
    cip::GeometryTopologyData::POINT GetPoint( unsigned int ) const;

    /** Get the id of the i-th Point */
    unsigned int GetPointId(unsigned int index) const;

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

    /** Refresh the seed id that will be used when adding new points/bounding boxes */
    void UpdateSeed();
  private:
    unsigned int m_seedId;
    BoundingBoxVectorType m_BoundingBoxes;
    PointVectorType m_Points;
    void FillMetaFieldsPoint(POINT*);
    void FillMetaFieldsBoundingBox(BOUNDINGBOX*);
  };
  
} // namespace cip

#endif
