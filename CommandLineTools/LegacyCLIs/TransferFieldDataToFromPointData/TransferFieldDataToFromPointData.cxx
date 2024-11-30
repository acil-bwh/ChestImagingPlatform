/** \file
 *  \ingroup commandLineTools
 *  \details This program can be used to transfer the contents of a VTK
 *  polydata's field data to point data and vice-versa. Generally, field
 *  data applies to a dataset as a whole and need not have a one-to-one
 *  correspondence with the points. However, this may be the case in some
 *  instances (esp. with the particles datasets). In those cases it may be
 *  helpful to have the data contained in field data arrays also stored in
 *  point data arrays (e.g. for rendering purposes). Field data will only
 *  be transferred provided that the number of tuples in the field data
 *  array is the same as the number of points.
 *
 *  USAGE:
 *
 *  TransferFieldDataToFromPointData  [--mp <bool>] [--mf <bool>] [--pf
 *                                     <bool>] [--fp <bool>] -o <string> -i
 *                                     <string> [--] [--version] [-h]
 *
 *  Where:
 *
 *  --mp <bool>
 *   Setting this to true will maintain the field data. Setting it to false
 *   will eliminate the field data from the output. Only relevant if
 *   requesting to transfer field data to point data
 *
 *  --mf <bool>
 *   Setting this to true will maintain the field data. Setting it to false
 *   will eliminate the field data from the output. Only relevant if
 *   requesting to transfer field data to point data
 *
 *  --pf <bool>
 *   Set to true to transfer point data to field data
 *
 *  --fp <bool>
 *   Set to true to transfer field data to point data
 *
 *  -o <string>,  --output <string>
 *   (required)  Output VTK polydata file name
 *
 *  -i <string>,  --input <string>
 *   (required)  Input VTK polydata file name
 *
 *  --,  --ignore_rest
 *   Ignores the rest of the labeled arguments following this flag.
 *
 *  --version
 *   Displays version information and exits.
 *
 *  -h,  --help
 *   Displays usage information and exits.
 *
 *  $Date: 2013-03-25 13:23:52 -0400 (Mon, 25 Mar 2013) $
 *  $Revision: 383 $
 *  $Author: jross $
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "cipHelper.h"
#include "vtkSmartPointer.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "vtkFieldData.h"
#include "vtkPolyData.h"
#include "vtkIndent.h"
#include "TransferFieldDataToFromPointDataCLP.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;
  
  // Read the poly data
  std::cout << "Reading VTK polydata..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > reader = vtkSmartPointer< vtkPolyDataReader >::New();
  reader->SetFileName( inFileName.c_str() );
  reader->Update();
  
  vtkSmartPointer< vtkPolyData > outPolyData = vtkSmartPointer< vtkPolyData >::New();
  cip::TransferFieldDataToFromPointData( reader->GetOutput(), outPolyData,
                                        fieldToPoint, pointToField, maintainPoint, maintainField );
  
  // Write the poly data
  std::cout << "Writing VTK polydata..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > writer = vtkSmartPointer< vtkPolyDataWriter >::New();
  writer->SetFileName( outFileName.c_str() );
  writer->SetInputData( outPolyData );
  if ( saveToBinary )
  {
    writer->SetFileTypeToBinary();
  }
  else
  {
    writer->SetFileTypeToASCII();
  }
  writer->Update();
  
  std::cout << "DONE." << std::endl;
  
  return 0;
}

#endif