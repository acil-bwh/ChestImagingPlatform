#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "cipGeometryTopologyDataIO.h"
#include "cipGeometryTopologyData.h"
#include "cipExceptionObject.h"
#include "cipChestConventions.h"

std::string& trim_right_inplace(
  std::string&       s,
  const std::string& delimiters = " \f\n\r\t\v" )
{
  return s.erase( s.find_last_not_of( delimiters ) + 1 );
}

std::string& trim_left_inplace(
  std::string&       s,
  const std::string& delimiters = " \f\n\r\t\v" )
{
  return s.erase( 0, s.find_first_not_of( delimiters ) );
}

std::string& trim_inplace(
  std::string&       s,
  const std::string& delimiters = " \f\n\r\t\v" )
{
  return trim_left_inplace( trim_right_inplace( s, delimiters ), delimiters );
}


int main( int argc, char* argv[] ) {
  // Read the baseline file
  char *baseLineFilePath = argv[1];
  cip::GeometryTopologyDataIO io;
  io.SetFileName(baseLineFilePath);
  io.Read();
  cip::GeometryTopologyData *gtd = io.GetOutput();

  // Write the result to a temporary file
  cip::GeometryTopologyDataIO writer;
  writer.SetInput(*gtd);
  char *tempOutput = argv[2];
  io.SetFileName(tempOutput);
  io.Write();

  // Make sure the two files are identical
  std::ifstream baselineFile;
  std::fstream outFile;
  int i;
  std::string s1, s2;
  std::vector <std::string> v1, v2;

  baselineFile.open(baseLineFilePath, std::ifstream::in);
  while (getline(baselineFile, s1)) {
    v1.push_back(s1);
  }
  baselineFile.close();

  outFile.open(tempOutput, std::ifstream::in);
  while (getline(outFile, s2)) {
    v2.push_back(s2);
  }
  outFile.close();

  if (v1.size() != v2.size()) {
    std::cout << "The two files do not contain the same number of lines" << std::endl;
    return cip::EXITFAILURE;
  }

  for (int i = 0; i < v1.size(); i++) {
    if (trim_inplace(v1[i]) != trim_inplace(v2[i])) {
      std::cout << "Expected: " << v1[i] << std::endl;
      std::cout << "Found: " << v2[i] << std::endl;
      return cip::EXITFAILURE;
    }
  }

  // Add manually a point and a bounding box and make sure that the seed is fine
  cip::GeometryTopologyData::CoordinateType coordinate3D(3);
  coordinate3D[0] = 0;
  coordinate3D[1] = 1.5;
  coordinate3D[2] = -2.3;

  cip::GeometryTopologyData::POINT *p = gtd->InsertPoint((unsigned char) (cip::RIGHTLUNG),
                                                         (unsigned char) (cip::NORMALPARENCHYMA),
                                                         (unsigned char) (cip::UNDEFINEDFEATURE), coordinate3D, "");
  if (p->id != 5) {
    std::cout << "Expected seed: 5; Obtained: " << p->id << std::endl;
    return cip::EXITFAILURE;
  }

  cip::GeometryTopologyData::StartType start3D(3);
  start3D[0] = 3;
  start3D[1] = 4;
  start3D[2] = 5;

  cip::GeometryTopologyData::SizeType size3D(3);
  size3D[0] = 6;
  size3D[1] = 7;
  size3D[2] = 8;

  cip::GeometryTopologyData::BOUNDINGBOX *bb = gtd->InsertBoundingBox((unsigned char) (cip::RIGHTSUPERIORLOBE),
                                                                      (unsigned char) (cip::UNDEFINEDTYPE),
                                                                      (unsigned char) (cip::UNDEFINEDFEATURE), start3D,
                                                                      size3D, "RightSuperiorLobe-UndefinedType");
  if (bb->id != 6) {
    std::cout << "Expected seed: 6; Obtained: " << bb->id << std::endl;
    return cip::EXITFAILURE;
  }

  return cip::EXITSUCCESS;
}