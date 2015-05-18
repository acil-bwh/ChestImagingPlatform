#include "cipGeometryTopologyData.h"
#include "cipExceptionObject.h"
#include <string.h>

cip::GeometryTopologyData::GeometryTopologyData()
{
}

cip::GeometryTopologyData::~GeometryTopologyData(void)
{
  this->m_BoundingBoxes.clear();
  this->m_Points.clear();
}

cip::GeometryTopologyData& cip::GeometryTopologyData::operator= (const cip::GeometryTopologyData &geometryTopology)
{
  // Check for self-assignment by comparing the address of the
  // implicit object and the parameter
  if (this == &geometryTopology)
    {
      return *this;
    }
  
  // Do the copy
  for ( unsigned int i=0; i<geometryTopology.GetNumberOfBoundingBoxes(); i++ )
    {
      BOUNDINGBOX bb;
      bb.cipRegion = geometryTopology.GetBoundingBoxChestRegion(i);
      bb.cipType = geometryTopology.GetBoundingBoxChestType(i);
      bb.description = geometryTopology.GetBoundingBoxDescription(i);

      for ( unsigned int j=0; j<geometryTopology.GetBoundingBoxStart(i).size(); j++ )
      	{
      	  bb.start.push_back(geometryTopology.GetBoundingBoxStart(i)[j]);
      	}
      for ( unsigned int j=0; j<geometryTopology.GetBoundingBoxSize(i).size(); j++ )
      	{
      	  bb.size.push_back(geometryTopology.GetBoundingBoxSize(i)[j]);
      	}
      m_BoundingBoxes.push_back( bb );
    }

  for ( unsigned int i=0; i<geometryTopology.GetNumberOfPoints(); i++ )
    {
      POINT p;
      p.cipRegion = geometryTopology.GetPointChestRegion(i);
      p.cipType = geometryTopology.GetPointChestType(i);
      p.description = geometryTopology.GetPointDescription(i);

      for ( unsigned int j=0; j<geometryTopology.GetPointCoordinate(i).size(); j++ )
  	{
  	  p.coordinate.push_back(geometryTopology.GetPointCoordinate(i)[j]);
  	}

      m_Points.push_back( p );
    }
  
  // Return the existing object
  return *this;
}

void cip::GeometryTopologyData::InsertBoundingBox( StartType start, SizeType size, 
						   unsigned char cipRegion = (unsigned char)(cip::UNDEFINEDREGION), 
						   unsigned char cipType = (unsigned char)(cip::UNDEFINEDTYPE),
						   std::string description = "NA" )
{
  BOUNDINGBOX bb;
  bb.cipRegion = cipRegion;
  bb.cipType = cipType;

  if ( start.size() != size.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::InsertBoundingBox", 
				  "start dimension does not equal size dimension" );
    }

  if ( size.size() != 2 && size.size() != 3  )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::InsertBoundingBox", 
				  "Unexpected bounding box size dimension" );
    }

  if ( start.size() != 2 && start.size() != 3  )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::InsertBoundingBox", 
				  "Unexpected bounding box start dimension" );
    }

  for ( unsigned int i=0; i<start.size(); i++ )
    {
      bb.start.push_back( start[i] );
      bb.size.push_back( size[i] );
    }

  bb.description = description;

  this->m_BoundingBoxes.push_back( bb );
}

void cip::GeometryTopologyData::InsertPoint( CoordinateType coordinate,
					     unsigned char cipRegion = (unsigned char)(cip::UNDEFINEDREGION), 
					     unsigned char cipType = (unsigned char)(cip::UNDEFINEDTYPE),
					     std::string description = "NA" )
{
  POINT p;
  p.cipRegion = cipRegion;
  p.cipType = cipType;

  if ( coordinate.size() != 2 && coordinate.size() != 3 )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::InsertPoint", 
				  "Unexpected coordinate dimension" );
    }

  for ( unsigned int i=0; i<coordinate.size(); i++ )
    {
      p.coordinate.push_back( coordinate[i] );
    }

  p.description = description;

  this->m_Points.push_back( p );
}

unsigned char cip::GeometryTopologyData::GetBoundingBoxChestRegion( unsigned int index ) const
{
  if ( index > this->m_BoundingBoxes.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::GetBoundingBoxChestRegion", 
				  "Index of range for m_BoundingBoxes" );
    }

  return this->m_BoundingBoxes[index].cipRegion;
}

unsigned char cip::GeometryTopologyData::GetBoundingBoxChestType( unsigned int index ) const
{
  if ( index > this->m_BoundingBoxes.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::GetBoundingBoxChestType", 
				  "Index of range for m_BoundingBoxes" );
    }

  return this->m_BoundingBoxes[index].cipType;
}

unsigned char cip::GeometryTopologyData::GetPointChestRegion( unsigned int index ) const
{
  if ( index > this->m_Points.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::GetPointChestRegion", 
				  "Index of range for m_Points" );
    }

  return this->m_Points[index].cipRegion;
}

unsigned char cip::GeometryTopologyData::GetPointChestType( unsigned int index ) const
{
  if ( index > this->m_Points.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::GetPointChestType", 
				  "Index of range for m_Points" );
    }

  return this->m_Points[index].cipType;
}

std::string cip::GeometryTopologyData::GetBoundingBoxDescription( unsigned int index ) const
{
  if ( index > this->m_BoundingBoxes.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::GetBoundingBoxDescription", 
				  "Index of range for m_Points" );
    }

  return this->m_BoundingBoxes[index].description;
}

std::string cip::GeometryTopologyData::GetPointDescription( unsigned int index ) const
{
  if ( index > this->m_Points.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::GetPointDescription", 
				  "Index of range for m_Points" );
    }

  return this->m_Points[index].description;
}

cip::GeometryTopologyData::StartType cip::GeometryTopologyData::GetBoundingBoxStart( unsigned int index ) const
{
  if ( index > this->m_BoundingBoxes.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::GetBoundingBoxStart", 
				  "Index of range for m_Points" );
    }

  return this->m_BoundingBoxes[index].start;
}

cip::GeometryTopologyData::SizeType cip::GeometryTopologyData::GetBoundingBoxSize( unsigned int index ) const
{
  if ( index > this->m_BoundingBoxes.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::GetBoundingBoxSize", 
				  "Index of range for m_BoundingBoxes" );
    }

  return this->m_BoundingBoxes[index].size;
}

cip::GeometryTopologyData::CoordinateType cip::GeometryTopologyData::GetPointCoordinate( unsigned int index ) const
{
  if ( index > this->m_Points.size() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cip::GeometryTopologyData::GetPointCoordinate", 
				  "Index of range for m_Points" );
    }

  return this->m_Points[index].coordinate;
}

bool cip::GeometryTopologyData::operator== (const GeometryTopologyData &geometryTopology) const
{
  if ( this->GetNumberOfBoundingBoxes() != geometryTopology.GetNumberOfBoundingBoxes() )
    {
      return false;
    }
  if ( this->GetNumberOfPoints() != geometryTopology.GetNumberOfPoints() )
    {
      return false;
    }

  bool boundingBoxesEqual = true;
  for ( unsigned int i=0; i<this->GetNumberOfBoundingBoxes(); i++ )
    {
      bool found = false;
      for ( unsigned int j=0; j<geometryTopology.GetNumberOfBoundingBoxes(); j++ )
	{
	  bool startSizeEqual = false;
	  if ( this->GetBoundingBoxStart(i).size() == geometryTopology.GetBoundingBoxStart(j).size() )
	    {
	      startSizeEqual = true;
	    }

	  bool sizeSizeEqual = false;
	  if ( this->GetBoundingBoxSize(i).size() != geometryTopology.GetBoundingBoxSize(j).size() )
	    {
	      sizeSizeEqual = true;
	    }

	  bool sizeEqual = true;
	  if ( sizeSizeEqual )
	    {
	      for ( unsigned int k=0; k<this->GetBoundingBoxSize(i).size(); k++ )
		{
		  if ( this->GetBoundingBoxSize(i)[k] != geometryTopology.GetBoundingBoxSize(j)[k] )
		    {
		      sizeEqual = false;
		    } 
		}
	    }

	  bool startEqual = true;
	  if ( startSizeEqual )
	    {
	      for ( unsigned int k=0; k<this->GetBoundingBoxStart(i).size(); k++ )
		{
		  if ( this->GetBoundingBoxStart(i)[k] != geometryTopology.GetBoundingBoxStart(j)[k] )
		    {
		      startEqual = false;
		    } 
		}
	    }

	  if ( this->GetBoundingBoxChestType(i) == geometryTopology.GetBoundingBoxChestType(j) &&
	       this->GetBoundingBoxChestRegion(i) == geometryTopology.GetBoundingBoxChestRegion(j) &&
	       sizeEqual && startEqual && startSizeEqual && sizeSizeEqual &&
	       geometryTopology.GetBoundingBoxDescription(j).compare(this->GetBoundingBoxDescription(i)) == 0)
	    {
	      found = true;
	      break;
	    }
	  found = true;
	  break;	   
	}
      if ( !found )
	{
	  boundingBoxesEqual = false;
	}
    }

  bool pointsEqual = true;
  for ( unsigned int i=0; i<this->GetNumberOfPoints(); i++ )
    {
      bool found = false;
      for ( unsigned int j=0; j<geometryTopology.GetNumberOfPoints(); j++ )
	{
	  bool coordinateSizeEqual = false;
	  if ( this->GetPointCoordinate(i).size() == geometryTopology.GetPointCoordinate(j).size() )
	    {
	      coordinateSizeEqual = true;
	    }

	  bool coordinateEqual = true;
	  if ( coordinateSizeEqual )
	    {
	      for ( unsigned int k=0; k<this->GetPointCoordinate(i).size(); k++ )
		{
		  if ( this->GetPointCoordinate(i)[k] != geometryTopology.GetPointCoordinate(j)[k] )
		    {
		      coordinateEqual = false;
		    } 
		}
	    }

	  if ( this->GetPointChestType(i) == geometryTopology.GetPointChestType(j) &&
	       this->GetPointChestRegion(i) == geometryTopology.GetPointChestRegion(j) &&
	       coordinateEqual && coordinateSizeEqual &&
	       geometryTopology.GetPointDescription(j).compare(this->GetPointDescription(i)) == 0)
	    {
	      found = true;
	      break;
	    }
	}
      if ( !found )
	{
	  pointsEqual = false;
	}
    }

  if ( pointsEqual && boundingBoxesEqual )
    {
      return true;
    }

  return false;
}

bool cip::GeometryTopologyData::operator!= (const GeometryTopologyData &geometryTopology) const
{
  return !(*this == geometryTopology);
}
