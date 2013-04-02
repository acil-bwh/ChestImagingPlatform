/** \file
 *  \ingroup interactiveTools 
 *  \details This program can be used to label particles as being
 *  either veins or arteries, and can be used to label particles
 *  according to vessel generation. The user simply needs to mouse
 *  over the particle component of interest and hit the 'a' key to
 *  designate arteries and the 'v' key to designate veins. The
 *  component will turn either blue or red, respectively, to
 *  acknowledge the change. Similarly, hitting the 0-9 keys will label
 *  particles by generation according to the pressed key. Each
 *  generation will be assigned a unique color for reference.
 *
 *  Once the user has designated all the components, simply hitting
 *  the 'e' key  will write the particles to file and exit the
 *  program.
 *  
 *  If particles are to be labeeld in groups, it's assumed that the
 *  input particles have been filtered so that connected component
 *  labels have been assigned. 
 *
 *  USAGE:
 *
 *  EditVesselParticles.exe [-c \<string\>] [-g \<string\>] [-a \<string\>] 
 *                          [-v \<string\>] -i \<string\>
 *                          [--] [--version] [-h]
 *
 *  Where:
 *
 *   -c \<string\>,  --ct \<string\>
 *     Input CT file name
 *
 *   -g \<string\>,  --generation \<string\>
 *     Output particles file name corresponding to labeled generations
 *
 *   -a \<string\>,  --artery \<string\>
 *     Output particles file name corresponding to labeled arteries
 *
 *   -v \<string\>,  --vein \<string\>
 *     Output particles file name corresponding to labeled veins
 *
 *   -i \<string\>,  --in \<string\>
 *     (required)  Input particles file name
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
 *  $Date: 2012-06-21 15:08:58 -0700 (Thu, 21 Jun 2012) $
 *  $Revision: 188 $
 *  $Author: jross $
 *
 *  TODO: There is a bug that causes the interactor to re-render
 *  unexpectedly. See TODO statement below
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS


#include <iostream>
#include <sstream>

#include <tclap/CmdLine.h>
#include "vtkPolyDataWriter.h"
#include "vtkPolyDataReader.h"
#include "vtkSmartPointer.h"
#include "cipVesselDataInteractor.h"
#include "vtkFloatArray.h"
#include "cipConventions.h"
#include "vtkPointData.h"
#include "itkImage.h"
#include "itkImageFileReader.h"

typedef itk::Image< short, 3 >             ImageType;
typedef itk::ImageFileReader< ImageType >  ReaderType;
void AddComponentsToInteractor( cipVesselDataInteractor*, vtkSmartPointer< vtkPolyData >, std::string, 
                                std::map< unsigned short, std::string >*, double );
void AssertChestRegionChestTypeArrayExistence( vtkSmartPointer< vtkPolyData > );
vtkSmartPointer< vtkPolyData > GetLabeledVesselParticles( cipVesselDataInteractor*, vtkSmartPointer< vtkPolyData >, 
                                                          std::map< unsigned short, std::string >*  );


int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed. 
  //
  std::string  inParticlesFileName     = "NA";
  std::string  veinParticlesFileName   = "NA";
  std::string  arteryParticlesFileName = "NA";
  std::string  genParticlesFileName    = "NA";
  std::string  ctFileName              = "NA";
  double       particleSize            = 1.0;

  //
  // Input descriptions for user convenience
  //
  std::string programDesc = "This program can be used to label particles as being \
either veins or arteries, and can be used to label particles \
according to vessel generation. The user simply needs to mouse \
over the particle component of interest and hit the 'a' key to \
designate arteries and the 'v' key to designate veins. The \
component will turn either blue or red, respectively, to \
acknowledge the change. Similarly, hitting the 0-9 keys will label \
particles by generation according to the pressed key. Each \
generation will be assigned a unique color for reference. \
Once the user has designated all the components, simply hitting \
the 'e' key  will write the particles to file and exit the \
program. If particles are to be labeeld in groups, it's assumed that the \
input particles have been filtered so that connected component \
labels have been assigned.";
  std::string inParticlesFileNameDesc     = "Input particles file name";
  std::string veinParticlesFileNameDesc   = "Output particles file name corresponding to labeled veins";
  std::string arteryParticlesFileNameDesc = "Output particles file name corresponding to labeled arteries";
  std::string genParticlesFileNameDesc    = "Output particles file name corresponding to labeled generations";
  std::string ctFileNameDesc              = "Input CT file name";
  std::string particleSizeDesc            = "Particle size scale factor";

  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( programDesc, ' ', "$Revision: 188 $" );

    TCLAP::ValueArg<std::string> inParticlesFileNameArg( "i", "in", inParticlesFileNameDesc, true, inParticlesFileName, "string", cl );
    TCLAP::ValueArg<std::string> veinParticlesFileNameArg( "v", "vein", veinParticlesFileNameDesc, false, veinParticlesFileName, "string", cl );
    TCLAP::ValueArg<std::string> arteryParticlesFileNameArg( "a", "artery", arteryParticlesFileNameDesc, false, arteryParticlesFileName, "string", cl );
    TCLAP::ValueArg<std::string> genParticlesFileNameArg( "g", "generation", genParticlesFileNameDesc, false, genParticlesFileName, "string", cl );
    TCLAP::ValueArg<std::string> ctFileNameArg( "c", "ct", ctFileNameDesc, false, ctFileName, "string", cl );

    cl.parse( argc, argv );

    inParticlesFileName     = inParticlesFileNameArg.getValue();
    veinParticlesFileName   = veinParticlesFileNameArg.getValue();
    arteryParticlesFileName = arteryParticlesFileNameArg.getValue();
    genParticlesFileName    = genParticlesFileNameArg.getValue();
    ctFileName              = ctFileNameArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  cipVesselDataInteractor interactor;               
  
  if ( ctFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Reading CT..." << std::endl;
    ReaderType::Pointer ctReader = ReaderType::New();
      ctReader->SetFileName( ctFileName );
    try
      {
      ctReader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught reading CT:";
      std::cerr << excp << std::endl;

      return cip::NRRDREADFAILURE;
      }

    interactor.SetGrayscaleImage( ctReader->GetOutput() );
    }    

  //
  // The nameToComponentMap will keep track of the mapping between the
  // names we assign the actors and the corresponding component
  // numbers of the original polyData
  //
  std::map< unsigned short, std::string > componentLabelToNameMap;
  
  std::cout << "Reading vessel particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > particlesReader = vtkSmartPointer< vtkPolyDataReader >::New();
    particlesReader->SetFileName( inParticlesFileName.c_str() );
    particlesReader->Update();    
  
  //
  // Assert that the input particles have 'ChestRegion' and
  // 'ChestType' arrays. If the input does not, this function will
  // add them
  //
  std::cout << "Asserting ChestRegion and ChestType array existence..." << std::endl;
  AssertChestRegionChestTypeArrayExistence( particlesReader->GetOutput() );

  std::cout << "Adding components to interactor..." << std::endl;
  AddComponentsToInteractor( &interactor, particlesReader->GetOutput(), "vesselParticles", &componentLabelToNameMap, particleSize );

  std::cout << "Rendering..." << std::endl;  
  interactor.Render();

  vtkSmartPointer< vtkPolyData > outParticles = vtkSmartPointer< vtkPolyData >::New();

  std::cout << "Retrieving labeled particles..." << std::endl;
  outParticles = GetLabeledVesselParticles( &interactor, particlesReader->GetOutput(), &componentLabelToNameMap ); 

//   std::cout << "Writing labeled particles..." << std::endl;
//   vtkSmartPointer< vtkPolyDataWriter > writer = vtkSmartPointer< vtkPolyDataWriter >::New();
//     writer->SetFileName( genParticlesFileName.c_str() );
//     writer->SetInput( outParticles );
//     writer->Write();  

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}


//
// This function is used to verify that the input particles have
// 'ChestRegion' and 'ChestType' arrays. If the particles don't have
// these arrays, they are assigned with default entries
// 'UNDEFINEDREGION' and 'UNDEFINEDTYPE'
//
void AssertChestRegionChestTypeArrayExistence( vtkSmartPointer< vtkPolyData > particles )
{
  unsigned int numberParticles         = particles->GetNumberOfPoints();
  unsigned int numberOfFieldDataArrays = particles->GetFieldData()->GetNumberOfArrays();

  bool foundChestRegionArray = false;
  bool foundChestTypeArray   = false;

  for ( unsigned int i=0; i<numberOfFieldDataArrays; i++ )
    {
    std::string name( particles->GetFieldData()->GetArray(i)->GetName() );

    if ( name.compare( "ChestRegion" ) == 0 )
      {
      foundChestRegionArray = true;
      }
    if ( name.compare( "ChestType" ) == 0 )
      {
      foundChestTypeArray = true;
      }
    }  

  if ( !foundChestRegionArray )
    {
    vtkSmartPointer< vtkFloatArray > chestRegionArray = vtkSmartPointer< vtkFloatArray >::New();
      chestRegionArray->SetNumberOfComponents( 1 );
      chestRegionArray->SetName( "ChestRegion" );

    particles->GetFieldData()->AddArray( chestRegionArray );
    }
  if ( !foundChestTypeArray )
    {
    vtkSmartPointer< vtkFloatArray > chestTypeArray = vtkSmartPointer< vtkFloatArray >::New();
      chestTypeArray->SetNumberOfComponents( 1 );
      chestTypeArray->SetName( "ChestType" );

    particles->GetFieldData()->AddArray( chestTypeArray );
    }

  float cipRegion = static_cast< float >( cip::UNDEFINEDREGION );
  float cipType   = static_cast< float >( cip::UNDEFINEDTYPE );
  if ( !foundChestRegionArray || !foundChestTypeArray )
    {
    for ( unsigned int i=0; i<numberParticles; i++ )
      {
      if ( !foundChestRegionArray )
        {
        particles->GetFieldData()->GetArray( "ChestRegion" )->InsertTuple( i, &cipRegion );
        }
      if ( !foundChestTypeArray )
        {
        particles->GetFieldData()->GetArray( "ChestType" )->InsertTuple( i, &cipType );
        }
      }
    }
}


void AddComponentsToInteractor( cipVesselDataInteractor* interactor, vtkSmartPointer< vtkPolyData > particles, std::string whichLung, 
                                std::map< unsigned short, std::string >* componentLabelToNameMap, double particleSize )  
{
  unsigned int numberParticles         = particles->GetNumberOfPoints();
  unsigned int numberOfFieldDataArrays = particles->GetFieldData()->GetNumberOfArrays();

  unsigned short component;
  std::vector< unsigned short > componentVec;

  for ( unsigned int i=0; i<numberParticles; i++ )
    {
    component = static_cast< unsigned short >( *(particles->GetFieldData()->GetArray( "unmergedComponents" )->GetTuple(i)) );

    bool addComponent = true;

    for ( unsigned int j=0; j<componentVec.size(); j++ )
      {
      if ( component == componentVec[j] )
        {
        addComponent = false;
        
        break;
        }
      }
    if ( addComponent )
      {
      componentVec.push_back( component );
      }
    }

  //
  // Now create the different poly data for the different components
  // and add them to the editor
  //
  for ( unsigned int c=0; c<componentVec.size(); c++ )
    {
    vtkPolyData* polyData = vtkPolyData::New();

    vtkPoints* points  = vtkPoints::New();
  
    std::vector< vtkFloatArray* > arrayVec;

    for ( unsigned int i=0; i<numberOfFieldDataArrays; i++ )
      {
      vtkFloatArray* array = vtkFloatArray::New();
        array->SetNumberOfComponents( particles->GetFieldData()->GetArray(i)->GetNumberOfComponents() );
        array->SetName( particles->GetFieldData()->GetArray(i)->GetName() );

      arrayVec.push_back( array );
      }

    unsigned int inc = 0;
    for ( unsigned int p=0; p<numberParticles; p++ )
      {
      component = static_cast< unsigned short >( *(particles->GetFieldData()->GetArray( "unmergedComponents" )->GetTuple(p)) );
      float floatComponent = static_cast< float >( component );

      if ( component == componentVec[c] )
        {
        points->InsertNextPoint( particles->GetPoint(p) );

        for ( unsigned int j=0; j<numberOfFieldDataArrays; j++ )
          {
          arrayVec[j]->InsertTuple( inc, particles->GetFieldData()->GetArray(j)->GetTuple(p) );
          }

        inc++;
        }
      }

    std::stringstream stream;
    stream << componentVec[c];

    std::string name = stream.str();
    name.append( whichLung );

    double r = 1;
    double g = 1;
    double b = 1;
    
    polyData->SetPoints( points );
    for ( unsigned int j=0; j<numberOfFieldDataArrays; j++ )
      {
      polyData->GetFieldData()->AddArray( arrayVec[j] );
      }

    interactor->SetVesselParticlesAsCylinders( polyData, particleSize, name ); 
    interactor->SetActorColor( name, r, g, b );
    interactor->SetActorOpacity( name, 1 );

    (*componentLabelToNameMap)[componentVec[c]] = name;
    }
}


//
// Iterate over all particles, get the particle's component, get the
// component's name, using the name get the component color, with the
// color assign the proper generation label
//
vtkSmartPointer< vtkPolyData > GetLabeledVesselParticles( cipVesselDataInteractor* interactor, vtkSmartPointer< vtkPolyData > particles, 
                                                          std::map< unsigned short, std::string >* componentLabelToNameMap )
{
  ChestConventions conventions;

  unsigned int numberParticles         = particles->GetNumberOfPoints();
  unsigned int numberOfFieldDataArrays = particles->GetFieldData()->GetNumberOfArrays();

  double* actorColor = new double[3];

  vtkSmartPointer< vtkPolyData > outPolyData = vtkSmartPointer< vtkPolyData >::New();
  vtkSmartPointer< vtkPoints >   outPoints   = vtkSmartPointer< vtkPoints >::New();
  
  std::vector< vtkFloatArray* > arrayVec;

  for ( unsigned int i=0; i<numberOfFieldDataArrays; i++ )
    {
    vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
      array->SetNumberOfComponents( particles->GetFieldData()->GetArray(i)->GetNumberOfComponents() );
      array->SetName( particles->GetFieldData()->GetArray(i)->GetName() );

      std::cout << particles->GetFieldData()->GetArray(i)->GetName() << std::endl;

    arrayVec.push_back( array );
    }

  unsigned int inc = 0;
  for ( unsigned int i=0; i<numberParticles; i++ )
    {
    unsigned short componentLabel = particles->GetFieldData()->GetArray( "unmergedComponents" )->GetTuple(i)[0];
    std::string    name           = (*componentLabelToNameMap)[componentLabel];

    if ( interactor->Exists( name ) )
      {
      interactor->GetActorColor( name, actorColor ); 
      
      float cipRegion = static_cast< float >( UNDEFINEDREGION );
      float cipType   = static_cast< float >( conventions.GetChestTypeFromColor( actorColor ) );
      
      particles->GetFieldData()->GetArray( "ChestRegion" )->SetTuple( i, &cipRegion );
      particles->GetFieldData()->GetArray( "ChestType" )->SetTuple( i, &cipType );

      //
      // TODO: Something fishy here. With this block in, the
      // interactor re-renders...
      //
      outPoints->InsertNextPoint( particles->GetPoint(i) );
      for ( unsigned int j=0; j<numberOfFieldDataArrays; j++ )
        {
        arrayVec[j]->InsertTuple( inc, particles->GetFieldData()->GetArray(j)->GetTuple(i) );
        }

      inc++;
      }
    }

  outPolyData->SetPoints( outPoints );
  for ( unsigned int j=0; j<numberOfFieldDataArrays; j++ )
    {
    outPolyData->GetFieldData()->AddArray( arrayVec[j] );
    }
  
  return outPolyData;
}

#endif
