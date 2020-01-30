#include "cipGeometryTopologyData.h"
#include "cipExceptionObject.h"
#include <string.h>

#include <time.h>
#ifdef _WIN32
#include <windows.h>
  #include <Lmcons.h>
#else
#include <unistd.h>
#endif

cip::GeometryTopologyData::GeometryTopologyData()
{
  this->m_seedId = 1;
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
  m_seedId = geometryTopology.m_seedId;
  CoordinateSystem = geometryTopology.CoordinateSystem;
  for ( unsigned int j=0; j<geometryTopology.LPS_to_IJK_TransformationMatrix.size(); j++ ) {
    LPS_to_IJK_TransformationMatrix.push_back(geometryTopology.LPS_to_IJK_TransformationMatrix[j]);
  }

  for ( unsigned int j=0; j<geometryTopology.m_Spacing.size(); j++ ) {
    m_Spacing.push_back(geometryTopology.m_Spacing[j]);
  }
  for ( unsigned int j=0; j<geometryTopology.m_Origin.size(); j++ ) {
    m_Origin.push_back(geometryTopology.m_Origin[j]);
  }
  for ( unsigned int j=0; j<geometryTopology.m_Dimensions.size(); j++ ) {
    m_Dimensions.push_back(geometryTopology.m_Dimensions[j]);
  }

  for ( unsigned int i=0; i<geometryTopology.GetNumberOfBoundingBoxes(); i++ )
  {
    BOUNDINGBOX bb;
    bb.id = geometryTopology.GetBoundingBoxId(i);
    bb.cipRegion = geometryTopology.GetBoundingBoxChestRegion(i);
    bb.cipType = geometryTopology.GetBoundingBoxChestType(i);
    bb.cipImageFeature = geometryTopology.GetBoundingBoxImageFeature(i);
    bb.description = geometryTopology.GetBoundingBoxDescription(i);
    bb.userName = geometryTopology.GetBoundingBox(i).userName;
    bb.machineName = geometryTopology.GetBoundingBox(i).machineName;
    bb.timestamp = geometryTopology.GetBoundingBox(i).timestamp;

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
    p.id = geometryTopology.GetPointId(i);
    p.cipRegion = geometryTopology.GetPointChestRegion(i);
    p.cipType = geometryTopology.GetPointChestType(i);
    p.cipImageFeature = geometryTopology.GetPointImageFeature(i);
    p.description = geometryTopology.GetPointDescription(i);
    p.userName = geometryTopology.GetPoint(i).userName;
    p.machineName = geometryTopology.GetPoint(i).machineName;
    p.timestamp = geometryTopology.GetPoint(i).timestamp;

    for ( unsigned int j=0; j<geometryTopology.GetPointCoordinate(i).size(); j++ )
    {
      p.coordinate.push_back(geometryTopology.GetPointCoordinate(i)[j]);
    }

    m_Points.push_back( p );
  }

  // Return the existing object
  return *this;
}

cip::GeometryTopologyData::BOUNDINGBOX* cip::GeometryTopologyData::InsertBoundingBox( unsigned char cipRegion,
                                                                                      unsigned char cipType,
                                                                                      unsigned char cipImageFeature,
                                                                                      StartType start,
                                                                                      SizeType size,
                                                                                      std::string description)
{
  BOUNDINGBOX bb;
  bb.id = this->m_seedId++;
  bb.cipRegion = cipRegion;
  bb.cipType = cipType;
  bb.cipImageFeature = cipImageFeature;

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

  this->FillMetaFieldsBoundingBox(&bb);

  this->m_BoundingBoxes.push_back( bb );

  return &bb; // TODO: this returns a pointer into the stack!
}

cip::GeometryTopologyData::BOUNDINGBOX* cip::GeometryTopologyData::InsertBoundingBox( int id,
                                                                                      unsigned char cipRegion,
                                                                                      unsigned char cipType,
                                                                                      unsigned char cipImageFeature,
                                                                                      StartType start,
                                                                                      SizeType size,
                                                                                      std::string description,
                                                                                      std::string timestamp,
                                                                                      std::string userName,
                                                                                      std::string machineName)
{
  BOUNDINGBOX bb;
  bb.id = id;
  bb.cipRegion = cipRegion;
  bb.cipType = cipType;
  bb.cipImageFeature = cipImageFeature;

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
  bb.timestamp = timestamp;
  bb.userName = userName;
  bb.machineName = machineName;

  this->m_BoundingBoxes.push_back( bb );

  return &bb; // TODO: this generates a warning that we are returning a pointer to an object on the stack!
}


void cip::GeometryTopologyData::FillMetaFieldsBoundingBox(BOUNDINGBOX* bb){
  char hostname[256];
  char userName[256];
#ifdef _WIN32
  DWORD size = 256;
  GetUserName( userName, &size );
  GetComputerName(hostname, &size);
#else
  gethostname(hostname, 256);
  getlogin_r(userName, 256);
#endif

  time_t     now = time(0);
  struct tm  tstruct;
  char       buf[80];
  tstruct = *localtime(&now);
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tstruct);

  bb->machineName = hostname;
  bb->userName = userName;
  bb->timestamp = buf;
}

cip::GeometryTopologyData::POINT* cip::GeometryTopologyData::InsertPoint( unsigned char cipRegion,
                                                                          unsigned char cipType,
                                                                          unsigned char cipImageFeature,
                                                                          CoordinateType coordinate,
                                                                          std::string description
                                                                          )
{
  POINT p;
  p.id = this->m_seedId++;
  p.cipRegion = cipRegion;
  p.cipType = cipType;
  p.cipImageFeature = cipImageFeature;

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

  this->FillMetaFieldsPoint(&p);

  this->m_Points.push_back( p );
  return &p; // TODO: this returns a pointer to an object on the stack!
}

cip::GeometryTopologyData::POINT* cip::GeometryTopologyData::InsertPoint( int id,
                                              unsigned char cipRegion,
                                              unsigned char cipType,
                                              unsigned char cipImageFeature,
                                              CoordinateType coordinate,
                                              std::string description,
                                              std::string timestamp,
                                              std::string userName,
                                              std::string machineName)
{
  POINT p;
  p.id = id;
  p.cipRegion = cipRegion;
  p.cipType = cipType;
  p.cipImageFeature = cipImageFeature;

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
  p.timestamp = timestamp;
  p.userName = userName;
  p.machineName = machineName;
  this->m_Points.push_back( p );
  return &p; // TODO: this returns a pointer to an object on the stack!
}

void cip::GeometryTopologyData::FillMetaFieldsPoint(POINT* p){
  char hostname[256];
  char userName[256];
#ifdef _WIN32
  DWORD size = 256;
  GetUserName( userName, &size );
  GetComputerName(hostname, &size);
#else
  gethostname(hostname, 256);
  getlogin_r(userName, 256);
#endif

  time_t     now = time(0);
  struct tm  tstruct;
  char       buf[80];
  tstruct = *localtime(&now);
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tstruct);

  p->machineName = hostname;
  p->userName = userName;
  p->timestamp = buf;
}


cip::GeometryTopologyData::BOUNDINGBOX cip::GeometryTopologyData::GetBoundingBox(unsigned int index) const
{
  if ( index > this->m_BoundingBoxes.size() )
  {
    throw cip::ExceptionObject( __FILE__, __LINE__,
                                "cip::GeometryTopologyData::GetBoundingBox",
                                "Index out of range for m_BoundingBoxes" );
  }

  return this->m_BoundingBoxes[index];
}

unsigned int cip::GeometryTopologyData::GetBoundingBoxId(unsigned int index) const
{
  if ( index > this->m_BoundingBoxes.size() )
  {
    throw cip::ExceptionObject( __FILE__, __LINE__,
                                "cip::GeometryTopologyData::GetBoundingBox",
                                "Index out of range for m_BoundingBoxes" );
  }

  return this->m_BoundingBoxes[index].id;
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

unsigned char cip::GeometryTopologyData::GetBoundingBoxImageFeature( unsigned int index ) const
{
  if ( index > this->m_BoundingBoxes.size() )
  {
    throw cip::ExceptionObject( __FILE__, __LINE__,
                                "cip::GeometryTopologyData::GetBoundingBoxChestType",
                                "Index of range for m_BoundingBoxes" );
  }

  return this->m_BoundingBoxes[index].cipImageFeature;
}

cip::GeometryTopologyData::POINT cip::GeometryTopologyData::GetPoint(unsigned int index) const
{
  if ( index > this->m_Points.size() )
  {
    throw cip::ExceptionObject( __FILE__, __LINE__,
                                "cip::GeometryTopologyData::GetPoint",
                                "Index out of range for m_Points" );
  }

  return this->m_Points[index];
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

unsigned char cip::GeometryTopologyData::GetPointImageFeature( unsigned int index ) const
{
  if ( index > this->m_Points.size() )
  {
    throw cip::ExceptionObject( __FILE__, __LINE__,
                                "cip::GeometryTopologyData::GetPointChestType",
                                "Index of range for m_Points" );
  }

  return this->m_Points[index].cipImageFeature;
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

unsigned int cip::GeometryTopologyData::GetPointId( unsigned int index ) const
{
  if ( index > this->m_Points.size() )
  {
    throw cip::ExceptionObject( __FILE__, __LINE__,
                                "cip::GeometryTopologyData::GetPointId",
                                "Index of range for m_Points" );
  }

  return this->m_Points[index].id;
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

bool cip::GeometryTopologyData::operator== (const GeometryTopologyData &geometryTopology) const {
  if (this->GetNumberOfBoundingBoxes() != geometryTopology.GetNumberOfBoundingBoxes()) {
    return false;
  }
  if (this->GetNumberOfPoints() != geometryTopology.GetNumberOfPoints()) {
    return false;
  }

  if (this->m_Spacing.size() != geometryTopology.m_Spacing.size())
    return false;
  for (unsigned int j = 0; j < geometryTopology.m_Dimensions.size(); j++)
    if (this->m_Spacing[j] != geometryTopology.m_Spacing[j])
      return false;

  if (this->m_Origin.size() != geometryTopology.m_Origin.size())
    return false;
  for (unsigned int j = 0; j < geometryTopology.m_Dimensions.size(); j++)
    if (this->m_Origin[j] != geometryTopology.m_Origin[j])
      return false;

  if (this->m_Dimensions.size() != geometryTopology.m_Dimensions.size())
    return false;
  for (unsigned int j = 0; j < geometryTopology.m_Dimensions.size(); j++)
    if (this->m_Dimensions[j] != geometryTopology.m_Dimensions[j])
      return false;

  bool boundingBoxesEqual = true;
  for (unsigned int i = 0; i < this->GetNumberOfBoundingBoxes(); i++) {
    bool found = false;
    for (unsigned int j = 0; j < geometryTopology.GetNumberOfBoundingBoxes(); j++) {
      if (this->GetBoundingBoxId(j) != geometryTopology.GetBoundingBoxId(j))
        return false;
      bool startSizeEqual = false;
      if (this->GetBoundingBoxStart(i).size() == geometryTopology.GetBoundingBoxStart(j).size()) {
        startSizeEqual = true;
      }

      bool sizeSizeEqual = false;
      if (this->GetBoundingBoxSize(i).size() != geometryTopology.GetBoundingBoxSize(j).size()) {
        sizeSizeEqual = true;
      }

      bool sizeEqual = true;
      if (sizeSizeEqual) {
        for (unsigned int k = 0; k < this->GetBoundingBoxSize(i).size(); k++) {
          if (this->GetBoundingBoxSize(i)[k] != geometryTopology.GetBoundingBoxSize(j)[k]) {
            sizeEqual = false;
          }
        }
      }

      bool startEqual = true;
      if (startSizeEqual) {
        for (unsigned int k = 0; k < this->GetBoundingBoxStart(i).size(); k++) {
          if (this->GetBoundingBoxStart(i)[k] != geometryTopology.GetBoundingBoxStart(j)[k]) {
            startEqual = false;
          }
        }
      }

      if (this->GetBoundingBoxChestType(i) == geometryTopology.GetBoundingBoxChestType(j) &&
          this->GetBoundingBoxChestRegion(i) == geometryTopology.GetBoundingBoxChestRegion(j) &&
          sizeEqual && startEqual && startSizeEqual && sizeSizeEqual &&
          geometryTopology.GetBoundingBoxDescription(j).compare(this->GetBoundingBoxDescription(i)) == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      boundingBoxesEqual = false;
    }
  }

  bool pointsEqual = true;
  for (unsigned int i = 0; i < this->GetNumberOfPoints(); i++) {
    if (this->GetPointId(i) != geometryTopology.GetPointId(i))
      return false;
    bool found = false;
    for (unsigned int j = 0; j < geometryTopology.GetNumberOfPoints(); j++) {
      bool coordinateSizeEqual = false;
      if (this->GetPointCoordinate(i).size() == geometryTopology.GetPointCoordinate(j).size()) {
        coordinateSizeEqual = true;
      }

      bool coordinateEqual = true;
      if (coordinateSizeEqual) {
        for (unsigned int k = 0; k < this->GetPointCoordinate(i).size(); k++) {
          if (this->GetPointCoordinate(i)[k] != geometryTopology.GetPointCoordinate(j)[k]) {
            coordinateEqual = false;
          }
        }
      }

      if (this->GetPointChestType(i) == geometryTopology.GetPointChestType(j) &&
          this->GetPointChestRegion(i) == geometryTopology.GetPointChestRegion(j) &&
          coordinateEqual && coordinateSizeEqual &&
          geometryTopology.GetPointDescription(j).compare(this->GetPointDescription(i)) == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      pointsEqual = false;
    }
  }

  if (pointsEqual && boundingBoxesEqual) {
    return true;
  }

  return false;
}

bool cip::GeometryTopologyData::operator!= (const GeometryTopologyData &geometryTopology) const
{
  return !(*this == geometryTopology);
}

void cip::GeometryTopologyData::UpdateSeed(){
  int id = 0;
  for (int i=0; i<this->m_Points.size(); i++)
    if (m_Points[i].id > id)
      id = m_Points[i].id;
  for (int i=0; i<this->m_BoundingBoxes.size(); i++)
    if (m_BoundingBoxes[i].id > id)
      id = m_BoundingBoxes[i].id;
  this->m_seedId = id + 1;
}
