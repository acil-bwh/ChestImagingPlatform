#include "cipGeometryTopologyDataIO.h"
#include "cipExceptionObject.h"
#include <fstream>
#include <stdlib.h>

using namespace cip;

GeometryTopologyDataIO::GeometryTopologyDataIO()
{
  this->FileName = "NA";
  this->m_GeometryTopologyData = new cip::GeometryTopologyData();
}


GeometryTopologyDataIO::~GeometryTopologyDataIO()
{
  delete this->m_GeometryTopologyData;
}


void GeometryTopologyDataIO::SetFileName( std::string fileName )
{
  this->FileName = fileName;
}


void GeometryTopologyDataIO::SetInput( cip::GeometryTopologyData& geometryTopologyData )
{
  *this->m_GeometryTopologyData = geometryTopologyData;
}


void GeometryTopologyDataIO::Write() const
{
}

void GeometryTopologyDataIO::Read()
{
}

cip::GeometryTopologyData* GeometryTopologyDataIO::GetOutput()
{
  return this->m_GeometryTopologyData;
}
