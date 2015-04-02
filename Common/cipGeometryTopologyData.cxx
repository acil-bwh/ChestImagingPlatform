#include "cipGeometryTopologyData.h"
#include "cipExceptionObject.h"

cip::GeometryTopologyData::GeometryTopologyData()
{
}

void cip::GeometryTopologyData::InsertBoundingBox( float start[3], float size[3], 
						   unsigned char cipRegion = (unsigned char)(cip::UNDEFINEDREGION), 
						   unsigned char cipType = (unsigned char)(cip::UNDEFINEDTYPE) )
{
  BOUNDINGBOX bb;
  bb.cipRegion = cipRegion;
  bb.cipType = cipType;

  for ( unsigned int i=0; i<3; i++ )
    {
      bb.start.push_back( start[i] );
      bb.size.push_back( size[i] );
    }

  this->m_BoundingBoxes.push_back( bb );
}

void cip::GeometryTopologyData::InsertPoint( float coordinate[3],
					     unsigned char cipRegion = (unsigned char)(cip::UNDEFINEDREGION), 
					     unsigned char cipType = (unsigned char)(cip::UNDEFINEDTYPE) )
{
  POINT p;
  p.cipRegion = cipRegion;
  p.cipType = cipType;

  for ( unsigned int i=0; i<3; i++ )
    {
      p.coordinate.push_back( coordinate[i] );
    }

  this->m_Points.push_back( p );
}

unsigned char cip::GeometryTopologyData::GetBoundingBoxChestRegion( unsigned int index )
{
  if ( index > this->m_BoundingBoxes.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::GetBoundingBoxChestRegion", 
				  "Index of range for m_BoundingBoxes" );
    }

  return this->m_BoundingBoxes[index].cipRegion;
}

unsigned char cip::GeometryTopologyData::GetBoundingBoxChestType( unsigned int index )
{
  if ( index > this->m_BoundingBoxes.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::GetBoundingBoxChestType", 
				  "Index of range for m_BoundingBoxes" );
    }

  return this->m_BoundingBoxes[index].cipType;
}

unsigned char cip::GeometryTopologyData::GetPointChestRegion( unsigned int index )
{
  if ( index > this->m_Points.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::GetPointChestRegion", 
				  "Index of range for m_Points" );
    }

  return this->m_Points[index].cipRegion;
}

unsigned char cip::GeometryTopologyData::GetPointChestType( unsigned int index )
{
  if ( index > this->m_Points.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::GetPointChestType", 
				  "Index of range for m_Points" );
    }

  return this->m_Points[index].cipType;
}

cip::GeometryTopologyData::StartType cip::GeometryTopologyData::GetBoundingBoxStart( unsigned int index )
{
  if ( index > this->m_BoundingBoxes.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::GetBoundingBoxStart", 
				  "Index of range for m_Points" );
    }

  return this->m_BoundingBoxes[index].start;
}

cip::GeometryTopologyData::SizeType cip::GeometryTopologyData::GetBoundingBoxSize( unsigned int index )
{
  if ( index > this->m_BoundingBoxes.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::GetBoundingBoxSize", 
				  "Index of range for m_BoundingBoxes" );
    }

  return this->m_BoundingBoxes[index].size;
}

cip::GeometryTopologyData::CoordinateType cip::GeometryTopologyData::GetPointCoordinate( unsigned int index )
{
  if ( index > this->m_Points.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::GetPointCoordinate", 
				  "Index of range for m_Points" );
    }

  return this->m_Points[index].coordinate;
}

