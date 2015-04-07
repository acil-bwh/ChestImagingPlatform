#include "cipGeometryTopologyDataIO.h"
#include "cipGeometryTopologyData.h"
#include "cipExceptionObject.h"
#include <iostream>

int main( int argc, char* argv[] )
{
  cip::GeometryTopologyData::CoordinateType coordinate3D(3);
    coordinate3D[0] = 0;
    coordinate3D[1] = 1.5;
    coordinate3D[2] = -2.3;

  cip::GeometryTopologyData::StartType start3D(3);
    start3D[0] = 3;
    start3D[1] = 4;
    start3D[2] = 5;

  cip::GeometryTopologyData::SizeType size3D(3);
    size3D[0] = 6;
    size3D[1] = 7;
    size3D[2] = 8;

  cip::GeometryTopologyData::CoordinateType coordinate2D(2);
    coordinate2D[0] = 9;
    coordinate2D[1] = 10;

  cip::GeometryTopologyData::StartType start2D(2);
    start2D[0] = 11;
    start2D[1] = 12;

  cip::GeometryTopologyData::SizeType size2D(2);
    size2D[0] = 13;
    size2D[1] = 14;

  cip::GeometryTopologyData geomTop;
    geomTop.InsertPoint( coordinate3D, (unsigned char)(cip::WHOLELUNG), 
			 (unsigned char)(cip::AIRWAY), "" );
    geomTop.InsertBoundingBox( start3D, size3D, (unsigned char)(cip::LEFTLUNG), 
			       (unsigned char)(cip::VESSEL), "LeftLung-Vessel" );
    geomTop.InsertPoint( coordinate2D, (unsigned char)(cip::RIGHTLUNG), 
			 (unsigned char)(cip::NORMALPARENCHYMA), "RightLung-NormalParenchyma" );
    geomTop.InsertBoundingBox( start2D, size2D, (unsigned char)(cip::RIGHTSUPERIORLOBE), 
			       (unsigned char)(cip::UNDEFINEDTYPE), "RightSuperiorLobe-UndefinedType" );
  
  cip::GeometryTopologyDataIO geomTopWriter;
    geomTopWriter.SetInput( geomTop );
    geomTopWriter.SetFileName( argv[1] );
    geomTopWriter.Write();

  cip::GeometryTopologyDataIO geomTopReader1;
    geomTopReader1.SetFileName( argv[1] );
    geomTopReader1.Read();

  cip::GeometryTopologyDataIO geomTopReader2;
    geomTopReader2.SetFileName( argv[2] );
    geomTopReader2.Read();

  if ( *(geomTopReader1.GetOutput()) != *(geomTopReader2.GetOutput()) )
    {
      std::cout << "FAILED" << std::endl;
      return 1;
    }

    std::cout << "PASSED" << std::endl;
    return 0;
}
