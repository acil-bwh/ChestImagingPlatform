/** \file
 *  \ingroup commandLineTools 
 *  \details This program is used to label particles datasets by chest
 *  region and chest type. The user must specify the type of the input
 *  particles, but the chest region can either be determined by an
 *  input label map or be specified at the command line. 
 * 
 *  $Date: 2012-07-18 13:23:15 -0700 (Wed, 18 Jul 2012) $
 *  $Revision: 196 $
 *  $Author: rjosest $
 *
 *  USAGE: 
 *
 *   LabelParticlesByChestRegionChestType  [-t \<unsigned char\>] [-r
 *                                         \<unsigned char\>] -o \<string\> -i
 *                                         \<string\> [-l \<string\>] [--]
 *                                         [--version] [-h]
 *
 *  Where: 
 *   -t \<unsigned char\>,  --type \<unsigned char\>
 *     Chest type for particles labeling. UNDEFINEDTYPE by default
 *
 *   -r \<unsigned char\>,  --region \<unsigned char\>
 *     Chest region for particles labeling. UNDEFINEDREGION by default
 *
 *   -o \<string\>,  --outParticles \<string\>
 *     (required)  Output particles file name
 *
 *   -i \<string\>,  --inParticles \<string\>
 *     (required)  Input particles file name
 *
 *   -l \<string\>,  --labelMap \<string\>
 *     Input label map file name. If specified the 'ChestRegion'value will be
 *     determined from the label map
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

#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkUnsignedCharArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "cipConventions.h"
#include "LabelParticlesByChestRegionChestTypeCLP.h"

namespace
{
typedef itk::Image< unsigned short, 3 >      ImageType;
typedef itk::ImageFileReader< ImageType >    ReaderType;

    void InitializeParticleChestRegionChestTypeArrays( vtkSmartPointer< vtkPolyData > particles )
    {
        unsigned int numberPointDataArrays = particles->GetPointData()->GetNumberOfArrays();
        unsigned int numberParticles       = particles->GetNumberOfPoints();
        
        //
        // The input particles may or may not have 'ChestType' and
        // 'ChestRegion' data arrays. As we loop through the input, we will
        // check their existence
        //
        bool foundChestRegionArray = false;
        bool foundChestTypeArray   = false;
        
        for ( unsigned int i=0; i<numberPointDataArrays; i++ )
        {
            std::string name( particles->GetPointData()->GetArray(i)->GetName() );
            if ( name.compare( "ChestType" ) == 0 )
            {
                foundChestTypeArray = true;
            }
            if ( name.compare( "ChestRegion" ) == 0 )
            {
                foundChestRegionArray = true;
            }
        }
        
        //
        // The 'chestRegionArray' and 'chestTypeArray' defined here will
        // only be used if they have not already been found in 'particles'
        //
        vtkSmartPointer< vtkUnsignedCharArray > chestRegionArray = vtkSmartPointer< vtkUnsignedCharArray >::New();
        vtkSmartPointer< vtkUnsignedCharArray > chestTypeArray = vtkSmartPointer< vtkUnsignedCharArray >::New();

        if ( !foundChestRegionArray)
          {
          chestRegionArray->SetNumberOfComponents( 1 );
          chestRegionArray->SetNumberOfTuples(numberParticles);
          chestRegionArray->SetName( "ChestRegion" );
          }

        if ( !foundChestTypeArray)
          {
          chestTypeArray->SetNumberOfComponents( 1 );
          chestTypeArray->SetNumberOfTuples(numberParticles);
          chestTypeArray->SetName( "ChestType" );
          }
        //
        // Now loop through the particles to initialize the arrays
        //
        unsigned char cipRegion = static_cast< unsigned char >( cip::UNDEFINEDREGION );
        unsigned char cipType   = static_cast< unsigned char >( cip::UNDEFINEDTYPE );
        
        for ( unsigned int i=0; i<numberParticles; i++ )
        {
            if ( foundChestRegionArray )
            {
                dynamic_cast <vtkUnsignedCharArray*> (particles->GetPointData()->GetArray( "ChestRegion" ))->SetValue( i, cipRegion );
            }
            else
            {
                chestRegionArray->SetValue( i, cipRegion );
            }
            
            if ( foundChestTypeArray )
            {
                dynamic_cast <vtkUnsignedCharArray*> (particles->GetPointData()->GetArray( "ChestType" ))->SetValue( i, cipType );
            }
            else
            {
                chestTypeArray->SetValue( i, cipType );
            }
        }
        
        if ( !foundChestRegionArray )
        {
            particles->GetPointData()->AddArray( chestRegionArray );
        }
        if ( !foundChestTypeArray )
        {
            particles->GetPointData()->AddArray( chestTypeArray );
        }
    }

}
//void InitializeParticleChestRegionChestTypeArrays( vtkSmartPointer< vtkPolyData > );


int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed
  //
    
  PARSE_ARGS;
    
  unsigned char       cipRegion            = static_cast< unsigned char >( cip::UNDEFINEDREGION );
  unsigned char       cipType              = static_cast< unsigned char >( cip::UNDEFINEDTYPE );

  if (  cipRegionArg > -1 )
      cipRegion            =  static_cast< unsigned char >(cipRegionArg);
  if ( cipTypeArg > -1 )
      cipType            =  static_cast< unsigned char >(cipTypeArg);
    

  //
  // Instantiate ChestConventions for later use
  //
  cip::ChestConventions conventions;

  //
  // Read the particles
  //
  std::cout << "Reading polydata..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > particlesReader = vtkSmartPointer< vtkPolyDataReader >::New();
    particlesReader->SetFileName( inParticlesFileName.c_str() );
    particlesReader->Update();    

  //
  // Initialize chest region and chest type field data arrays. We
  // don't assume that the incoming particles have these arrays. If
  // they don't we add them. If they do, we initialize them with
  // 'UNDEFINEDREGION' and 'UNDEFINEDTYPE'.
  //
  std::cout << "Initializing chest-region chest-type arrays..." << std::endl;
  InitializeParticleChestRegionChestTypeArrays( particlesReader->GetOutput() );

  
  //
  // Get arrays from data
  //
  vtkSmartPointer< vtkUnsignedCharArray > chestRegionArray = dynamic_cast <vtkUnsignedCharArray*> (particlesReader->GetOutput()->GetPointData()->GetArray("ChestRegion"));
  vtkSmartPointer< vtkUnsignedCharArray > chestTypeArray = dynamic_cast <vtkUnsignedCharArray*> (particlesReader->GetOutput()->GetPointData()->GetArray("ChestType"));
  
  //
  // If specified, read the input label map
  //
  if ( labelMapFileName.compare( "q" ) != 0 )
    {
    std::cout << "Reading label map..." << std::endl;
    ReaderType::Pointer labelMapReader = ReaderType::New();
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

    ImageType::PointType point;
    ImageType::IndexType index;

    //
    // Loop through the particles to label them
    //
    for ( unsigned int i=0; i<particlesReader->GetOutput()->GetNumberOfPoints(); i++ )
      {
      point[0] = particlesReader->GetOutput()->GetPoint(i)[0];
      point[1] = particlesReader->GetOutput()->GetPoint(i)[1];
      point[2] = particlesReader->GetOutput()->GetPoint(i)[2];
      
      labelMapReader->GetOutput()->TransformPhysicalPointToIndex( point, index );    

      unsigned short labelValue = labelMapReader->GetOutput()->GetPixel( index );

      cipRegion  = static_cast< unsigned char >( conventions.GetChestRegionFromValue( labelValue ) );

      chestRegionArray->SetValue( i, cipRegion );
      chestTypeArray->SetValue( i, cipType );
      }
    }
  else
    {
    //
    // If here, no label map was specified, and we must assign region
    // and types based on user specification. Loop through the
    // particles to label them 
    //
    for ( unsigned int i=0; i<particlesReader->GetOutput()->GetNumberOfPoints(); i++ )
      {
      chestRegionArray->SetValue( i, cipRegion );
      chestTypeArray->SetValue( i, cipType );
      }
    }

  //
  // Write the labeled particles
  //
  std::cout << "Writing labeled particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > particlesWriter = vtkSmartPointer< vtkPolyDataWriter >::New();
    particlesWriter->SetFileName( outParticlesFileName.c_str() );
    particlesWriter->SetInput( particlesReader->GetOutput() );
    particlesWriter->SetFileTypeToBinary();
    particlesWriter->Update();    

  std::cout << "DONE." << std::endl;

  return 0;
}

