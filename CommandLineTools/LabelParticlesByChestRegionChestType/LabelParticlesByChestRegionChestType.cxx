/** \file
 *  \ingroup commandLineTools
 *  \details This program is used to label particles datasets by chest
 *  region and chest type. The user must specify the type of the input
 *  particles, but the chest region can either be determined by an
 *  input label map or be specified at the command line.
 */

#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkUnsignedCharArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "cipChestConventions.h"
#include "LabelParticlesByChestRegionChestTypeCLP.h"
#include "cipHelper.h"


int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  // Instantiate ChestConventions for later use
  cip::ChestConventions conventions;

  float cipRegion = float(conventions.GetChestRegionValueFromName( cipRegionArg ));
  float cipType = float(conventions.GetChestTypeValueFromName( cipTypeArg ));

  // Read the particles
  std::cout << "Reading polydata..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > particlesReader = vtkSmartPointer< vtkPolyDataReader >::New();
    particlesReader->SetFileName( inParticlesFileName.c_str() );
    particlesReader->Update();

  // Initialize chest region and chest type field data arrays. We
  // don't assume that the incoming particles have these arrays. If
  // they don't we add them. If they do, we initialize them with
  // 'UNDEFINEDREGION' and 'UNDEFINEDTYPE'.
  cip::AssertChestRegionChestTypeArrayExistence( particlesReader->GetOutput() );

  // If specified, read the input label map
  if ( labelMapFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Reading label map..." << std::endl;
    cip::LabelMapReaderType::Pointer labelMapReader = cip::LabelMapReaderType::New();
      labelMapReader->SetFileName( labelMapFileName );
    try
      {
      labelMapReader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught reading label map:";
      std::cerr << excp << std::endl;

      return cip::LABELMAPREADFAILURE;
      }

    cip::LabelMapType::PointType point;
    cip::LabelMapType::IndexType index;

    // Loop through the particles to label them
    for ( unsigned int i=0; i<particlesReader->GetOutput()->GetNumberOfPoints(); i++ )
      {
      point[0] = particlesReader->GetOutput()->GetPoint(i)[0];
      point[1] = particlesReader->GetOutput()->GetPoint(i)[1];
      point[2] = particlesReader->GetOutput()->GetPoint(i)[2];

      labelMapReader->GetOutput()->TransformPhysicalPointToIndex( point, index );

      unsigned short labelValue = labelMapReader->GetOutput()->GetPixel( index );

      cipRegion = float( conventions.GetChestRegionFromValue( labelValue ) );

      particlesReader->GetOutput()->GetPointData()->GetArray("ChestRegion")->SetTuple( i, &cipRegion );
      particlesReader->GetOutput()->GetPointData()->GetArray("ChestType")->SetTuple( i, &cipType );
      }
    }
  else
    {
    // If here, no label map was specified, and we must assign region
    // and types based on user specification. Loop through the
    // particles to label them
    for ( unsigned int i=0; i<particlesReader->GetOutput()->GetNumberOfPoints(); i++ )
      {
      particlesReader->GetOutput()->GetPointData()->GetArray("ChestRegion")->SetTuple( i, &cipRegion );
      particlesReader->GetOutput()->GetPointData()->GetArray("ChestType")->SetTuple( i, &cipType );
      }
    }

  // Write the labeled particles
  std::cout << "Writing labeled particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > particlesWriter = vtkSmartPointer< vtkPolyDataWriter >::New();
    particlesWriter->SetFileName( outParticlesFileName.c_str() );
    particlesWriter->SetInputConnection( particlesReader->GetOutputPort() );
    particlesWriter->SetFileTypeToBinary();
    particlesWriter->Update();

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

