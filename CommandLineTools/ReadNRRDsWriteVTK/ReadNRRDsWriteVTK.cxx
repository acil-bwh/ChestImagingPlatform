/** \file
 *  \ingroup commandLineTools
 *  \details This program reads a number of NRRD files and collects
 *  the data in those files into a single VTK polydata file for
 *  writing. The input data files typically contain particles
 *  information.
 */

#include "vtkSmartPointer.h"
#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkCellArray.h"
#include "vtkVertex.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "cipChestConventions.h"
#include "ReadNRRDsWriteVTKCLP.h"
#include "cipHelper.h"

namespace
{
  typedef itk::Image< double, 2 >                NRRDImageType;
  typedef itk::ImageFileReader< NRRDImageType >  ReaderType;
}

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  cip::ChestConventions conventions;

  if ( !conventions.IsChestRegion( cipRegionArg ) )
    {
      std::cerr << "Chest region is not valid" << std::endl;
      return cip::EXITFAILURE;
    }
  if ( !conventions.IsChestType( cipTypeArg ) )
    {
      std::cerr << "Chest type is not valid" << std::endl;
      return cip::EXITFAILURE;
    }

  unsigned char cipRegion = conventions.GetChestRegionValueFromName( cipRegionArg );
  unsigned char cipType = conventions.GetChestTypeValueFromName( cipTypeArg );

  float chestRegionChestTypeValue = conventions.GetValueFromChestRegionAndType( cipRegion, cipType );

  std::vector<std::string> arrayNameVec;
  std::vector<std::string> inFileNameVec;

  if ( inFileNameVecArg.size() != arrayNameVecArg.size() )
    {
      std::cerr << "Mismatch between input file name (specified with -i or --inFileName) and array name ";
      std::cerr << "(specified with -a or --arrayName). See help for details" << std::endl;
      return cip::ARGUMENTPARSINGERROR;
    }
  
  outFileName = outFileNameArg.getValue();
  for ( unsigned int i=0; i<inFileNameVecArg.size(); i++ )
    {
      inFileNameVec.push_back( inFileNameVecArg[i] );
    }
  for ( unsigned int i=0; i<arrayNameVecArg.size(); i++ )
    {
      arrayNameVec.push_back( arrayNameVecArg[i] );
    }
  
  vtkSmartPointer< vtkPolyData > polyData = vtkSmartPointer< vtkPolyData >::New();
  vtkSmartPointer< vtkPoints >   points   = vtkSmartPointer< vtkPoints >::New();

  NRRDImageType::IndexType index;

  for ( unsigned int i=0; i<inFileNameVec.size(); i++ )
    {
    std::cout << "Reading file for " << arrayNameVec[i] << "..." << std::endl;
    ReaderType::Pointer reader = ReaderType::New();
      reader->SetFileName( inFileNameVec[i] );
    try
      {
      reader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught reading input:";
      std::cerr << excp << std::endl;
      return cip::NRRDREADFAILURE;
      }

    unsigned int numComponents = reader->GetOutput()->GetBufferedRegion().GetSize()[0];
    unsigned int numParticles  = reader->GetOutput()->GetBufferedRegion().GetSize()[1];

    // If 4 components, this file consists of the spatial position and
    // scale
    if ( numComponents == 4 )
      {
      vtkSmartPointer< vtkFloatArray > scale = vtkSmartPointer< vtkFloatArray >::New();
        scale->SetNumberOfComponents( 1 );
        scale->SetName( "scale" );

      for ( unsigned int j=0; j<numParticles; j++ )
        {
        float* point = new float[3];

        index[0] = 0;        index[1] = j;
        point[0] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 1;        index[1] = j;
        point[1] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 2;        index[1] = j;
        point[2] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 3;        index[1] = j;
        float sc = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        points->InsertNextPoint( point );
        scale->InsertTuple( j, &sc );
        }

      polyData->GetPointData()->AddArray( scale );
      }

    //
    // If only one component
    //
    if ( numComponents == 1 )
      {
      vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
        array->SetNumberOfComponents( 1 );
        array->SetName( arrayNameVec[i].c_str() );

      for ( unsigned int j=0; j<numParticles; j++ )
        {
        index[0] = 0;
        index[1] = j;

        float value = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        array->InsertTuple( j, &value );
        }

      polyData->GetPointData()->AddArray( array );
      }

    //
    // If 3 components
    //
    if ( numComponents == 3 )
      {
      vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
        array->SetNumberOfComponents( 3 );
        array->SetName( arrayNameVec[i].c_str() );

      for ( unsigned int j=0; j<numParticles; j++ )
        {
        float* vec = new float[3];

        index[0] = 0;
        index[1] = j;
        vec[0] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 1;
        index[1] = j;
        vec[1] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 2;
        index[1] = j;
        vec[2] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        array->InsertTuple( j, vec );
        }

      polyData->GetPointData()->AddArray( array );
      }

    // If 9 components
    if ( numComponents == 9 )
      {
      vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
        array->SetNumberOfComponents( 9 );
        array->SetName( arrayNameVec[i].c_str() );

      for ( unsigned int j=0; j<numParticles; j++ )
        {
        float* vec = new float[9];

        index[0] = 0;
        index[1] = j;
        vec[0] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 1;
        index[1] = j;
        vec[1] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 2;
        index[1] = j;
        vec[2] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 3;
        index[1] = j;
        vec[3] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 4;
        index[1] = j;
        vec[4] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 5;
        index[1] = j;
        vec[5] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 6;
        index[1] = j;
        vec[6] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 7;
        index[1] = j;
        vec[7] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 8;
        index[1] = j;
        vec[8] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        array->InsertTuple( j, vec );
        }

      polyData->GetPointData()->AddArray( array );
      }

    // If 7 components
    if ( numComponents == 7 )
      {
      vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
        array->SetNumberOfComponents( 9 );
        array->SetName( arrayNameVec[i].c_str() );

      for ( unsigned int j=0; j<numParticles; j++ )
        {
        float* vec = new float[9];

        index[0] = 1;
        index[1] = j;
        vec[0] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 2;
        index[1] = j;
        vec[1] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 3;
        index[1] = j;
        vec[2] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 2;
        index[1] = j;
        vec[3] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 4;
        index[1] = j;
        vec[4] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 5;
        index[1] = j;
        vec[5] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 3;
        index[1] = j;
        vec[6] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 5;
        index[1] = j;
        vec[7] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        index[0] = 6;
        index[1] = j;
        vec[8] = static_cast< float >( reader->GetOutput()->GetPixel( index ) );

        array->InsertTuple( j, vec );
        }

      polyData->GetPointData()->AddArray( array );
      }
    }
  
  // Store metadata information in a field array
  if (irad !=0 )
    {
      vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
      array->SetNumberOfComponents( 1 );
      array->SetNumberOfTuples(1);
      array->SetName("irad");
      array->SetValue(0,irad);
      
      polyData->GetFieldData()->AddArray(array);
    }
  
  if (srad != 0 )
  {
      vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
      array->SetNumberOfComponents( 1 );
      array->SetNumberOfTuples(1);
      array->SetName("srad");
      array->SetValue(0,srad);
      polyData->GetFieldData()->AddArray(array);
  }
  
  if ((spacing[0] > 0 ||  spacing[1] >0 || spacing[2] >0)  && spacing.size() == 3 )
  {
    
    vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
    array->SetNumberOfComponents( 3 );
    array->SetNumberOfTuples(1);
    array->SetName("spacing");
    array->SetComponent(0,0,spacing[0]);
    array->SetComponent(0,1,spacing[1]);
    array->SetComponent(0,2,spacing[2]);
    polyData->GetFieldData()->AddArray(array);
  }
  
  if (liveThreshold != 0 )
  {
    vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
    array->SetNumberOfComponents( 1 );
    array->SetNumberOfTuples(1);
    array->SetName("liveth");
    array->SetValue(0,liveThreshold);
    polyData->GetFieldData()->AddArray(array);
  }
  
  if (seedThreshold != 0 )
  {
    vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
    array->SetNumberOfComponents( 1 );
    array->SetNumberOfTuples(1);
    array->SetName("seedth");
    array->SetValue(0,seedThreshold);
    polyData->GetFieldData()->AddArray(array);
  }
    
  polyData->SetPoints( points );

  // Create the 'ChestRegion' and 'ChestType' arrays, and set the region and/or type if
  // they have been specified by the user.
  cip::AssertChestRegionChestTypeArrayExistence( polyData );
  for ( unsigned int i=0; i<polyData->GetNumberOfPoints(); i++ )
    {
      polyData->GetPointData()->GetArray("ChestRegionChestType")->SetTuple( i, &chestRegionChestTypeValue );
    }

  // If not present, add Vertices to the polydata file
  if ( polyData->GetNumberOfVerts() == 0 )
    {
    vtkSmartPointer< vtkCellArray > cellArray = vtkSmartPointer< vtkCellArray >::New();
    for ( unsigned int pid = 0; pid < polyData->GetNumberOfPoints(); pid++ )
        {
        vtkSmartPointer< vtkVertex > Vertex = vtkSmartPointer< vtkVertex >::New();
        Vertex->GetPointIds()->SetId(0, pid);
        cellArray->InsertNextCell(Vertex);
      }
  polyData->SetVerts(cellArray);
  }
  
  std::cout << "Writing poly data..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > writer = vtkSmartPointer< vtkPolyDataWriter >::New();
    writer->SetFileName( outFileName.c_str() );
    writer->SetInputData( polyData );
  if ( binaryOutput )
    {
    writer->SetFileTypeToBinary();
    }
  else
    {
    writer->SetFileTypeToASCII();
    }
    writer->Write();

    std::cout << "DONE." << std::endl;
    
    return cip::EXITSUCCESS;
}

