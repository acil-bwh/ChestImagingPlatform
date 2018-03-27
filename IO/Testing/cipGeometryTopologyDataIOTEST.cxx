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
//
//  cip::GeometryTopologyData::StartType start3D(3);
//    start3D[0] = 3;
//    start3D[1] = 4;
//    start3D[2] = 5;
//
//  cip::GeometryTopologyData::SizeType size3D(3);
//    size3D[0] = 6;
//    size3D[1] = 7;
//    size3D[2] = 8;
//
//  cip::GeometryTopologyData::CoordinateType coordinate2D(2);
//    coordinate2D[0] = 9;
//    coordinate2D[1] = 10;
//
//  cip::GeometryTopologyData::StartType start2D(2);
//    start2D[0] = 11;
//    start2D[1] = 12;
//
//  cip::GeometryTopologyData::SizeType size2D(2);
//    size2D[0] = 13;
//    size2D[1] = 14;
//
//  cip::GeometryTopologyData geomTop;
//  geomTop.m_Spacing.push_back(0.5);
//  geomTop.m_Spacing.push_back(0.6);
//  geomTop.m_Spacing.push_back(0.7);
  geomTop.InsertPoint( coordinate3D, (unsigned char)(cip::WHOLELUNG),
			 (unsigned char)(cip::AIRWAY), (unsigned char)(cip::UNDEFINEDFEATURE), "hola", true );
//
//  geomTop.InsertPoint( coordinate3D, (unsigned char)(cip::WHOLELUNG),
//                       (unsigned char)(cip::AIRWAY), (unsigned char)(cip::UNDEFINEDFEATURE), "hola2", true );

//    geomTop.InsertBoundingBox( start3D, size3D, (unsigned char)(cip::LEFTLUNG),
//			       (unsigned char)(cip::VESSEL), (unsigned char)(cip::UNDEFINEDFEATURE), "LeftLung-Vessel" );
//    geomTop.InsertPoint( coordinate2D, (unsigned char)(cip::RIGHTLUNG),
//			 (unsigned char)(cip::NORMALPARENCHYMA), (unsigned char)(cip::UNDEFINEDFEATURE), "RightLung-NormalParenchyma" );
//    geomTop.InsertBoundingBox( start2D, size2D, (unsigned char)(cip::RIGHTSUPERIORLOBE),
//			       (unsigned char)(cip::UNDEFINEDTYPE), (unsigned char)(cip::UNDEFINEDFEATURE), "RightSuperiorLobe-UndefinedType" );
//
//  cip::GeometryTopologyDataIO geomTopWriter;
//    geomTopWriter.SetInput( geomTop );
//    geomTopWriter.SetFileName( argv[1] );
//    geomTopWriter.Write();
//
//  cip::GeometryTopologyDataIO geomTopReader1;
//    geomTopReader1.SetFileName( argv[1] );
//    geomTopReader1.Read();
//
//  cip::GeometryTopologyDataIO geomTopReader2;
//    geomTopReader2.SetFileName( argv[2] );
//    geomTopReader2.SetInput(geomTop);
//    geomTopReader2.Write();
//
//  if ( *(geomTopReader1.GetOutput()) != *(geomTopReader2.GetOutput()) )
//    {
//      std::cout << "FAILED" << std::endl;
//
//      std::cout ;
//
//      return 1;
//    }
//
//    std::cout << argv[2] << " written " << std::endl;
    // Fail test
    return 1;
}

// <?xml version="1.0" encoding="utf8"?><GeometryTopologyData>
// <NumDimensions>3</NumDimensions>
// <CoordinateSystem>RAS</CoordinateSystem>
// <LPStoIJKTransformationMatrix>
// <value>-1.900000</value>
// <value>0.000000</value>
// <value>0.000000</value>
// <value>250.000000</value>
// <value>0.000000</value>
// <value>-1.900000</value>
// <value>0.000000</value>
// <value>510.000000</value>
// <value>0.000000</value>
// <value>0.000000</value>
// <value>2.000000</value>
// <value>724.000000</value>
// <value>0.000000</value>
// <value>0.000000</value>
// <value>0.000000</value>
// <value>1.000000</value>
// </LPStoIJKTransformationMatrix>

// <Point>
// <ChestRegion>2</ChestRegion>
// <ChestType>5</ChestType>
// <ImageFeature>1</ImageFeature>
// <Description>My desc</Description>
// <Coordinate>
// <value>2.000000</value>
// <value>3.500000</value>
// <value>3.000000</value>
// </Coordinate>
// </Point>

// <Point>
// <ChestRegion>2</ChestRegion>
// <ChestType>5</ChestType>
// <ImageFeature>1</ImageFeature>
// <Coordinate><value>2</value>
// <value>3</value>
// <value>3</value>
// </Coordinate>
// </Point>

// <BoundingBox>
// <ChestRegion>2</ChestRegion>
// <ChestType>5</ChestType>
// <ImageFeature>1</ImageFeature>
// <Start><value>2</value>
// <value>3</value>
// <value>3</value>
// </Start>
// <Size><value>1</value>
// <value>1</value>
// <value>4</value>
// </Size>
// </BoundingBox>

// <BoundingBox>
// <ChestRegion>2</ChestRegion>
// <ChestType>5</ChestType>
// <ImageFeature>1</ImageFeature>
// <Start><value>2.000000</value>
// <value>3.500000</value>
// <value>3.000000</value>
// </Start>
// <Size><value>1.000000</value>
// <value>1.000000</value>
// <value>3.000000</value>
// </Size>
// </BoundingBox>

// </GeometryTopologyData>
