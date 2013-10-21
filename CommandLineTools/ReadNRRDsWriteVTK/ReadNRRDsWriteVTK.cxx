/** \file
 *  \ingroup commandLineTools 
 *  \details This program reads a number of NRRD files and collects
 *  the data in those files into a single VTK polydata file for
 *  writing. The input data files typically contain particles
 *  information. 
 * 
 *  $Date: 2012-09-04 15:09:57 -0400 (Tue, 04 Sep 2012) $
 *  $Revision: 224 $
 *  $Author: jross $
 *
 *  USAGE: 
 *
 *  ReadNRRDsWriteVTK [-a \<string\>] ...  [-i \<string\>] ...  -o \<string\>
 *                    [--] [--version] [-h]
 *
 *  Where: 
 *
 *   -a \<string\>,  --arrayName \<string\>  (accepted multiple times)
 *     Array names corresponding to files immediately preceding invocation of
 *     this flag (specified with the -i or --inFileName flags). Array names
 *     follow conventinos laid out in the ACIL wiki for particles polydata
 *     point data arrays
 *
 *   -i \<string\>,  --inFileName \<string\>  (accepted multiple times)
 *     Specify an input NRRD file name followed by a string (using the -a or
 *     --arrayName flags) designating the name of the array in the output vtk
 *     file. Can specify multiple inputs. Note that a file name specified
 *     with this flag must immediately be followed by a corresponding array
 *     name using the -a or --arrayName flags. Files that are 1xN are assumed
 *     to have scalar data, 3xN are assumed to have vector data, and 9xN are
 *     assumed to have matrix data. A 4xN file is assumed to contain spatial
 *     coordinates for the first 3 components and a scale component for the
 *     4th. Note that in this case, the string value assigned to this file is
 *     just a placeholder -- the scale data will be placed in matrix with
 *     name 'scale'. 7xN files are assumed to have the following format: mask
 *     xx xy xz yy yz zz, so a matrix is constructed using the components in
 *     the following order: [1 2 3 2 4 5 3 5 6] (zero-based indexing)
 *
 *   -o \<string\>,  --outFileName \<string\>
 *     (required)  Ouput vtk file name. All particles information will be
 *     stored in this file.
 *
 *   --,  --ignore_rest
 *     Ignores the rest of the labeled arguments following this flag.
 *
 *   --version
 *     Displays version information and exits.
 *
 *   -h,  --help
 *     Displays usage information and exits.
 *
 */



#include "vtkSmartPointer.h"
#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "cipConventions.h"
#include "ReadNRRDsWriteVTKCLP.h"

namespace
{
typedef itk::Image< double, 2 >                NRRDImageType;
typedef itk::ImageFileReader< NRRDImageType >  ReaderType;
}

int main( int argc, char *argv[] )
{

  //
  // Parse the input arguments
  //
  PARSE_ARGS;
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

    //
    // If 4 components, this file consists of the spatial position and
    // scale
    //
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

    //
    // If 9 components
    //
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

    //
    // If 7 components
    //
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

  polyData->SetPoints( points );

  std::cout << "Writing poly data..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > writer = vtkSmartPointer< vtkPolyDataWriter >::New();
    writer->SetFileName( outFileName.c_str() );
    writer->SetInput( polyData );
    writer->SetFileTypeToASCII();
    writer->Write();  

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

