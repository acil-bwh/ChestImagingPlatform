/** \file
 *  \ingroup interactiveTools 
 *  \details This program can be used to label particles by generation:
 *  hitting the 0-9 keys will label particles by generation according 
 *  to the pressed key. Each generation will be assigned a unique color 
 *  for reference.
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
 * EditAirwayParticles  [--rtpSc \<double\>] ...  [--rtpOp \<double\>] ... 
 *                      [--rtpBlue \<double\>] ...  [--rtpGreen \<double\>]
 *                      ...  [--rtpRed \<double\>] ...  
 *                      [-t \<unsigned char\>] ...  [-r \<unsigned char\>] ...  
 *                      [--rtp \<string\>] [-s \<double\>] [-c \<string\>] 
 *                      [-g \<string\>] -i \<string\> [--] [--version] [-h]
 *
 *  Where:
 *
 *  --rtpSc \<double\>  (accepted multiple times)
 *     Use when specifying a region-type file name to specify the red channel
 *     when rendering. Must be used with the --rtpRegions, --rtpTypes,
 *     --rtpRed, --rtpGreen, --rtpBlue, and --rtpOp flags.
 *
 *   --rtpOp \<double\>  (accepted multiple times)
 *     Use when specifying a region-type file name to specify the red channel
 *     when rendering. Must be used with the --rtpRegions, --rtpTypes,
 *     --rtpRed, --rtpGreen, --rtpBlue, and --rtpSc flags.
 *
 *   --rtpBlue \<double\>  (accepted multiple times)
 *     Use when specifying a region-type file name to specify the red channel
 *     when rendering. Must be used with the --rtpRegions, --rtpTypes,
 *     --rtpRed, --rtpGreen, --rtpOp, and --rtpSc flags.
 *
 *   --rtpGreen \<double\>  (accepted multiple times)
 *     Use when specifying a region-type file name to specify the red channel
 *     when rendering. Must be used with the --rtpRegions, --rtpTypes,
 *     --rtpRed, --rtpBlue, --rtpOp, and --rtpSc flags.
 *
 *   --rtpRed \<double\>  (accepted multiple times)
 *     Use when specifying a region-type file name to specify the red channel
 *     when rendering. Must be used with the --rtpRegions, --rtpTypes,
 *     --rtpGreen, --rtpBlue, --rtpOp, and --rtpSc flags.
 *
 *   -t \<unsigned char\>,  --rtpType \<unsigned char\>  (accepted multiple
 *      times)
 *     Use when specifying a region-type file name to specify which types
 *     should be specified. For each type specified, there must be a region
 *     specified with the --rtpRegions flag. Additionally, you must specify
 *     red, green, blue channels opacity and scale with the --rtpRed,
 *     --rtpGreen, --rtpBlue, --rtpOp, and --rtpSc flags, respectively.
 *
 *   -r \<unsigned char\>,  --rtpRegion \<unsigned char\>  (accepted multiple
 *      times)
 *     Use when specifying a region-type file name to specify which regions
 *     should be specified. For each region specified, there must be a type
 *     specified with the --rtpTypes flag. Additionally, you must specify red,
 *     green, blue channels opacity and scale with the --rtpRed, --rtpGreen,
 *     --rtpBlue, --rtpOp, and --rtpSc flags, respectively.
 *
 *   --rtp \<string\>
 *     Region and type points file name. This should be used with the -r, and
 *     -t flags to specify which objects should be rendered
 *
 *   -s \<double\>,  --pSize \<double\>
 *     Particle size scale factor
 *
 *   -c \<string\>,  --ct \<string\>
 *     Input CT file name
 *
 *   -g \<string\>,  --generation \<string\>
 *     Output particles file name corresponding to labeled generations
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
 *  $Date: 2013-02-22 18:00:01 -0500 (Fri, 22 Feb 2013) $
 *  $Revision: 370 $
 *  $Author: jross $
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <iostream>
#include <sstream>
#include <tclap/CmdLine.h>
#include "vtkPolyDataWriter.h"
#include "vtkPolyDataReader.h"
#include "vtkSmartPointer.h"
#include "cipAirwayDataInteractor.h"
#include "vtkFloatArray.h"
#include "cipChestConventions.h"
#include "vtkPointData.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "cipChestRegionChestTypeLocationsIO.h"

typedef itk::Image< short, 3 >             ImageType;
typedef itk::ImageFileReader< ImageType >  ReaderType;
void AddComponentsToInteractor( cipAirwayDataInteractor*, vtkSmartPointer< vtkPolyData >, std::string, 
                                std::map< unsigned short, std::string >*, double );
void AssertChestRegionChestTypeArrayExistence( vtkSmartPointer< vtkPolyData > );
vtkSmartPointer< vtkPolyData > GetLabeledAirwayParticles( cipAirwayDataInteractor*, vtkSmartPointer< vtkPolyData >, 
                                                          std::map< unsigned short, std::string >*  );
void AddRegionTypePointsAsSpheresToInteractor( cipAirwayDataInteractor*, std::string, std::vector< unsigned char >, std::vector< unsigned char >, 
					       std::vector< double >, std::vector< double >, std::vector< double >, std::vector< double >, 
					       std::vector< double > );


int main( int argc, char *argv[] )
{
  // Begin by defining the arguments to be passed. 
  std::string  inParticlesFileName     = "NA";
  std::string  genParticlesFileName    = "NA";
  std::string  airwayModelFileName     = "NA";
  std::string  ctFileName              = "NA";
  double       particleSize            = 1.0;
  // Arguments for optional region-type points file input
  std::string regionTypePointsFileName = "NA";
  std::vector< unsigned char > regionTypePointsRegions;
  std::vector< unsigned char > regionTypePointsTypes;
  std::vector< double > regionTypePointsRed;
  std::vector< double > regionTypePointsGreen;
  std::vector< double > regionTypePointsBlue;
  std::vector< double > regionTypePointsOpacity;
  std::vector< double > regionTypePointsScale;
  bool prune;
  bool label;

  // Input descriptions for user convenience
  std::string programDesc = "This program can be used to label airway particles \
according to generation. The user simply needs to mouse \
over the particle component of interest and hit the 0-9 keys. This will label \
particles by generation according to the pressed key. Each \
generation will be assigned a unique color for reference. \
Once the user has designated all the components, simply hitting \
the 'e' key  will write the particles to file and exit the \
program. If particles are to be labeled in groups, it's assumed that the \
input particles have been filtered so that connected component \
labels have been assigned.";
  std::string inParticlesFileNameDesc     = "Input particles file name";
  std::string genParticlesFileNameDesc    = "Output particles file name corresponding to labeled generations";
  std::string ctFileNameDesc              = "Input CT file name";
  std::string airwayModelFileNameDesc     = "Airway model to render semi-transparently over particles for reference. The \
model can be toggled on and off within the editor.";
  std::string particleSizeDesc            = "Particle size scale factor";
  // Descriptions for region-type points file input
  std::string regionTypePointsFileNameDesc = "Region and type points file name. This should be used with the -r, and -t \
flags to specify which objects should be rendered";
  std::string regionTypePointsRegionsDesc = "Use when specifying a region-type file name to specify which regions should \
be specified. For each region specified, there must be a type specified with the --rtpTypes flag. Additionally, you must \
specify red, green, blue channels opacity and scale with the --rtpRed, --rtpGreen, --rtpBlue, --rtpOp, and --rtpSc flags, \
respectively.";
  std::string regionTypePointsTypesDesc   = "Use when specifying a region-type file name to specify which types should \
be specified. For each type specified, there must be a region specified with the --rtpRegions flag. Additionally, you must \
specify red, green, blue channels opacity and scale with the --rtpRed, --rtpGreen, --rtpBlue, --rtpOp, and --rtpSc flags, \
respectively.";
  std::string regionTypePointsRedDesc     = "Use when specifying a region-type file name to specify the red channel when \
rendering. Must be used with the --rtpRegions, --rtpTypes, --rtpGreen, --rtpBlue, --rtpOp, and --rtpSc flags.";
  std::string regionTypePointsGreenDesc   = "Use when specifying a region-type file name to specify the red channel when \
rendering. Must be used with the --rtpRegions, --rtpTypes, --rtpRed, --rtpBlue, --rtpOp, and --rtpSc flags.";
  std::string regionTypePointsBlueDesc    = "Use when specifying a region-type file name to specify the red channel when \
rendering. Must be used with the --rtpRegions, --rtpTypes, --rtpRed, --rtpGreen, --rtpOp, and --rtpSc flags.";
  std::string regionTypePointsOpacityDesc = "Use when specifying a region-type file name to specify the red channel when \
rendering. Must be used with the --rtpRegions, --rtpTypes, --rtpRed, --rtpGreen, --rtpBlue, and --rtpSc flags.";
  std::string regionTypePointsScaleDesc   = "Use when specifying a region-type file name to specify the red channel when \
rendering. Must be used with the --rtpRegions, --rtpTypes, --rtpRed, --rtpGreen, --rtpBlue, and --rtpOp flags.";
  std::string pruneDesc   = "Set this flag to indicated that the editor should be used in prune mode, which allows \
the user to remove particles with the k key";
  std::string labelDesc   = "Set this flag to indicated that the editor should be used in label mode, which allows \
the user to label groups of particles according to their airway generation.";

  // Parse the input arguments
  try
    {
    TCLAP::CmdLine cl( programDesc, ' ', "$Revision: 370 $" );

    TCLAP::ValueArg<std::string> inParticlesFileNameArg( "i", "in", inParticlesFileNameDesc, true, inParticlesFileName, "string", cl );
    TCLAP::ValueArg<std::string> genParticlesFileNameArg( "g", "generation", genParticlesFileNameDesc, false, genParticlesFileName, "string", cl );
    TCLAP::ValueArg<std::string> ctFileNameArg( "c", "ct", ctFileNameDesc, false, ctFileName, "string", cl );
    TCLAP::ValueArg<std::string> airwayModelFileNameArg( "m", "model", airwayModelFileNameDesc, false, airwayModelFileName, "string", cl );
    TCLAP::ValueArg<double>      particleSizeArg( "s", "pSize", particleSizeDesc, false, particleSize, "double", cl );
    TCLAP::SwitchArg             pruneArg( "", "prune", pruneDesc, cl, false );
    TCLAP::SwitchArg             labelArg( "", "label", labelDesc, cl, false );
    // Region-type args:
    TCLAP::ValueArg<std::string>   regionTypePointsFileNameArg( "", "rtp", regionTypePointsFileNameDesc, false, regionTypePointsFileName, "string", cl );
    TCLAP::MultiArg<unsigned int>  regionTypePointsRegionsArg( "r", "rtpRegion", regionTypePointsRegionsDesc, false, "unsigned char", cl );
    TCLAP::MultiArg<unsigned int>  regionTypePointsTypesArg( "t", "rtpType", regionTypePointsTypesDesc, false, "unsigned char", cl );
    TCLAP::MultiArg<double>        regionTypePointsRedArg( "", "rtpRed", regionTypePointsRedDesc, false, "double", cl );
    TCLAP::MultiArg<double>        regionTypePointsGreenArg( "", "rtpGreen", regionTypePointsGreenDesc, false, "double", cl );
    TCLAP::MultiArg<double>        regionTypePointsBlueArg( "", "rtpBlue", regionTypePointsBlueDesc, false, "double", cl );
    TCLAP::MultiArg<double>        regionTypePointsOpacityArg( "", "rtpOp", regionTypePointsOpacityDesc, false, "double", cl );
    TCLAP::MultiArg<double>        regionTypePointsScaleArg( "", "rtpSc", regionTypePointsScaleDesc, false, "double", cl );

    cl.parse( argc, argv );

    if ( pruneArg.isSet() )
      {
      prune = true;
      }
    if ( labelArg.isSet() )
      {
      label = true;
      }

    inParticlesFileName     = inParticlesFileNameArg.getValue();
    genParticlesFileName    = genParticlesFileNameArg.getValue();
    ctFileName              = ctFileNameArg.getValue();
    airwayModelFileName     = airwayModelFileNameArg.getValue();
    particleSize            = particleSizeArg.getValue();
    // Region-type points
    regionTypePointsFileName = regionTypePointsFileNameArg.getValue();
    if ( regionTypePointsFileName.compare( "NA" ) != 0  &&
	 !(regionTypePointsRegionsArg.getValue().size() == regionTypePointsTypesArg.getValue().size() &&
	   regionTypePointsRegionsArg.getValue().size() == regionTypePointsRedArg.getValue().size() &&
	   regionTypePointsRegionsArg.getValue().size() == regionTypePointsGreenArg.getValue().size() &&
	   regionTypePointsRegionsArg.getValue().size() == regionTypePointsBlueArg.getValue().size() &&
	   regionTypePointsRegionsArg.getValue().size() == regionTypePointsScaleArg.getValue().size() &&
	   regionTypePointsRegionsArg.getValue().size() == regionTypePointsOpacityArg.getValue().size() &&
	   regionTypePointsRegionsArg.getValue().size() > 0) )
      {
	std::cerr << "Error: When specifying a region-type points file name, must specify an equal number of ";
	std::cerr << "inputs for flags --rtp, --rtpRegion, --rtpType, --rtpRed, ";
	std::cerr << "--rtpGreen, --rtpBlue, --rtpOp, and --rtpSc" << std::endl;
	return cip::ARGUMENTPARSINGERROR;
      }
    for ( unsigned int i=0; i<regionTypePointsRegionsArg.getValue().size(); i++ )
      {
	regionTypePointsRegions.push_back( static_cast< unsigned char >( regionTypePointsRegionsArg.getValue()[i]) );
      }
    for ( unsigned int i=0; i<regionTypePointsTypesArg.getValue().size(); i++ )
      {
	regionTypePointsTypes.push_back( static_cast< unsigned char >( regionTypePointsTypesArg.getValue()[i]) );
      }
    for ( unsigned int i=0; i<regionTypePointsRedArg.getValue().size(); i++ )
      {
	regionTypePointsRed.push_back( regionTypePointsRedArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<regionTypePointsGreenArg.getValue().size(); i++ )
      {
	regionTypePointsGreen.push_back( regionTypePointsGreenArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<regionTypePointsBlueArg.getValue().size(); i++ )
      {
	regionTypePointsBlue.push_back( regionTypePointsBlueArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<regionTypePointsOpacityArg.getValue().size(); i++ )
      {
	regionTypePointsOpacity.push_back( regionTypePointsOpacityArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<regionTypePointsScaleArg.getValue().size(); i++ )
      {
	regionTypePointsScale.push_back( regionTypePointsScaleArg.getValue()[i] );
      }
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  cip::ChestConventions conventions;

  cipAirwayDataInteractor interactor;               
  
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
  if ( airwayModelFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Reading airway model..." << std::endl;
    vtkSmartPointer< vtkPolyDataReader > modelReader = vtkSmartPointer< vtkPolyDataReader >::New();
      modelReader->SetFileName( airwayModelFileName.c_str() );
      modelReader->Update();    

    interactor.SetAirwayModel( modelReader->GetOutput() );
    }      
  if ( regionTypePointsFileName.compare( "NA" ) != 0 )
    {
      AddRegionTypePointsAsSpheresToInteractor( &interactor, regionTypePointsFileName, regionTypePointsRegions, regionTypePointsTypes, 
						regionTypePointsRed, regionTypePointsGreen, regionTypePointsBlue, 
						regionTypePointsScale, regionTypePointsOpacity );
    }

  //
  // The nameToComponentMap will keep track of the mapping between the
  // names we assign the actors and the corresponding component
  // numbers of the original polyData
  //
  std::map< unsigned short, std::string > componentLabelToNameMap;
  
  std::cout << "Reading airway particles..." << std::endl;
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

  if ( label )
    {
      // Now add the particles. They will be used to create a minimum
      // spanning tree, and this tree will be used in order to label
      // particles between specified root and intermediate nodes /
      // particles 
      std::cout << "Adding particles to interactor for minimum spanning tree representation..." << std::endl;
      interactor.SetAirwayParticlesAsMinimumSpanningTree( particlesReader->GetOutput(), particleSize );

      std::cout << "Rendering..." << std::endl;  
      interactor.Render();

      std::cout << "Writing labeled particles..." << std::endl;
      vtkSmartPointer< vtkPolyDataWriter > writer = vtkSmartPointer< vtkPolyDataWriter >::New();
        writer->SetFileName( genParticlesFileName.c_str() );
	writer->SetInputData( particlesReader->GetOutput() );
	//writer->SetFileTypeToASCII();
	writer->Write();  
    }
  else if ( prune )
    {
      std::cout << "Adding components to interactor..." << std::endl;
      AddComponentsToInteractor( &interactor, particlesReader->GetOutput(), "airwayParticles", &componentLabelToNameMap, particleSize );

      std::cout << "Rendering..." << std::endl;  
      interactor.Render();

      vtkSmartPointer< vtkPolyData > outParticles = vtkSmartPointer< vtkPolyData >::New();

      std::cout << "Retrieving labeled particles..." << std::endl;
      outParticles = GetLabeledAirwayParticles( &interactor, particlesReader->GetOutput(), &componentLabelToNameMap ); 

      std::cout << "Writing labeled particles..." << std::endl;
      vtkSmartPointer< vtkPolyDataWriter > writer = vtkSmartPointer< vtkPolyDataWriter >::New();
        writer->SetFileName( genParticlesFileName.c_str() );
	writer->SetInputData( outParticles );
	//writer->SetFileTypeToASCII();
	writer->Write();  
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}


// This function is used to verify that the input particles have
// 'ChestRegion' and 'ChestType' arrays. If the particles don't have
// these arrays, they are assigned with default entries
// 'UNDEFINEDREGION' and 'UNDEFINEDTYPE'
void AssertChestRegionChestTypeArrayExistence( vtkSmartPointer< vtkPolyData > particles )
{
  unsigned int numberParticles         = particles->GetNumberOfPoints();
  unsigned int numberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();

  bool foundChestRegionArray = false;
  bool foundChestTypeArray   = false;

  for ( unsigned int i=0; i<numberOfPointDataArrays; i++ )
    {
    std::string name( particles->GetPointData()->GetArray(i)->GetName() );

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

    particles->GetPointData()->AddArray( chestRegionArray );
    }
  if ( !foundChestTypeArray )
    {
    vtkSmartPointer< vtkFloatArray > chestTypeArray = vtkSmartPointer< vtkFloatArray >::New();
      chestTypeArray->SetNumberOfComponents( 1 );
      chestTypeArray->SetName( "ChestType" );

    particles->GetPointData()->AddArray( chestTypeArray );
    }

  float cipRegion = static_cast< float >( cip::UNDEFINEDREGION );
  float cipType   = static_cast< float >( cip::UNDEFINEDTYPE );
  if ( !foundChestRegionArray || !foundChestTypeArray )
    {
    for ( unsigned int i=0; i<numberParticles; i++ )
      {
      if ( !foundChestRegionArray )
        {
        particles->GetPointData()->GetArray( "ChestRegion" )->InsertTuple( i, &cipRegion );
        }
      if ( !foundChestTypeArray )
        {
        particles->GetPointData()->GetArray( "ChestType" )->InsertTuple( i, &cipType );
        }
      }
    }
}


void AddComponentsToInteractor( cipAirwayDataInteractor* interactor, vtkSmartPointer< vtkPolyData > particles, std::string whichLung, 
                                std::map< unsigned short, std::string >* componentLabelToNameMap, double particleSize )  
{
  cip::ChestConventions conventions;

  unsigned int numberParticles         = particles->GetNumberOfPoints();
  unsigned int numberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();

  unsigned short component;
  std::vector< unsigned short > componentVec;
  std::vector< unsigned char > cipTypeVec;

  for ( unsigned int i=0; i<numberParticles; i++ )
    {
    component = static_cast< unsigned short >( *(particles->GetPointData()->GetArray( "unmergedComponents" )->GetTuple(i)) );

    // The input particles may already be labeled. Get the ChestType
    // recorded for thie component. By default we will color according
    // to this type
    unsigned char cipType = static_cast< unsigned char >( *(particles->GetPointData()->GetArray( "ChestType" )->GetTuple(i)) );

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
      cipTypeVec.push_back( cipType );
      }
    }

  // Now create the different poly data for the different components
  // and add them to the editor
  for ( unsigned int c=0; c<componentVec.size(); c++ )
    {
    vtkPolyData* polyData = vtkPolyData::New();
    vtkPoints* points  = vtkPoints::New();
    std::vector< vtkFloatArray* > arrayVec;

    for ( unsigned int i=0; i<numberOfPointDataArrays; i++ )
      {
      vtkFloatArray* array = vtkFloatArray::New();
        array->SetNumberOfComponents( particles->GetPointData()->GetArray(i)->GetNumberOfComponents() );
        array->SetName( particles->GetPointData()->GetArray(i)->GetName() );

      arrayVec.push_back( array );
      }

    unsigned int inc = 0;
    for ( unsigned int p=0; p<numberParticles; p++ )
      {
      component = static_cast< unsigned short >( *(particles->GetPointData()->GetArray( "unmergedComponents" )->GetTuple(p)) );

      if ( component == componentVec[c] )
        {
        points->InsertNextPoint( particles->GetPoint(p) );

        for ( unsigned int j=0; j<numberOfPointDataArrays; j++ )
          {
          arrayVec[j]->InsertTuple( inc, particles->GetPointData()->GetArray(j)->GetTuple(p) );
          }

        inc++;
        }
      }

    std::stringstream stream;
    stream << componentVec[c];

    std::string name = stream.str();
    name.append( whichLung );

    double* color = new double[3];
    conventions.GetChestTypeColor( cipTypeVec[c], color );

    double r = color[0];
    double g = color[1];
    double b = color[2];
    
    polyData->SetPoints( points );
    for ( unsigned int j=0; j<numberOfPointDataArrays; j++ )
      {
      polyData->GetPointData()->AddArray( arrayVec[j] );
      }

    interactor->SetAirwayParticlesAsDiscs( polyData, particleSize, name ); 
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
vtkSmartPointer< vtkPolyData > GetLabeledAirwayParticles( cipAirwayDataInteractor* interactor, vtkSmartPointer< vtkPolyData > particles, 
                                                          std::map< unsigned short, std::string >* componentLabelToNameMap )
{
  cip::ChestConventions conventions;

  unsigned int numberParticles         = particles->GetNumberOfPoints();
  unsigned int numberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();

  double* actorColor = new double[3];

  vtkSmartPointer< vtkPolyData > outPolyData = vtkSmartPointer< vtkPolyData >::New();
  vtkSmartPointer< vtkPoints >   outPoints   = vtkSmartPointer< vtkPoints >::New();
 
  std::vector< vtkSmartPointer< vtkFloatArray > > arrayVec;

  for ( unsigned int i=0; i<numberOfPointDataArrays; i++ )
    {
    vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
      array->SetNumberOfComponents( particles->GetPointData()->GetArray(i)->GetNumberOfComponents() );
      array->SetName( particles->GetPointData()->GetArray(i)->GetName() );

    arrayVec.push_back( array );
    }
 
  unsigned int inc = 0;
  for ( unsigned int i=0; i<numberParticles; i++ )
    {
    unsigned short componentLabel = particles->GetPointData()->GetArray( "unmergedComponents" )->GetTuple(i)[0];
    std::string    name           = (*componentLabelToNameMap)[componentLabel];

    if ( interactor->Exists( name ) )
      {
      interactor->GetActorColor( name, actorColor ); 
      
      float cipRegion = static_cast< float >( cip::UNDEFINEDREGION );
      float cipType   = static_cast< float >( conventions.GetChestTypeFromColor( actorColor ) );
      
      particles->GetPointData()->GetArray( "ChestRegion" )->SetTuple( i, &cipRegion );
      particles->GetPointData()->GetArray( "ChestType" )->SetTuple( i, &cipType );

      //
      // TODO: Something fishy here. With this block in, the
      // interactor re-renders...
      //
      outPoints->InsertNextPoint( particles->GetPoint(i) );
      for ( unsigned int j=0; j<numberOfPointDataArrays; j++ )
        {
        arrayVec[j]->InsertTuple( inc, particles->GetPointData()->GetArray(j)->GetTuple(i) );
        }

      inc++;
      }
    }

  outPolyData->SetPoints( outPoints );
  for ( unsigned int j=0; j<numberOfPointDataArrays; j++ )
    {
    arrayVec[j];
    outPolyData->GetPointData()->AddArray( arrayVec[j] );
    }

  return outPolyData;
}


void AddRegionTypePointsAsSpheresToInteractor( cipAirwayDataInteractor* interactor, std::string regionTypePointsFileName, std::vector< unsigned char > regionTypePointsRegions, 
					       std::vector< unsigned char > regionTypePointsTypes, std::vector< double > regionTypePointsRed, 
					       std::vector< double > regionTypePointsGreen, std::vector< double > regionTypePointsBlue, 
					       std::vector< double > regionTypePointsScale, std::vector< double > regionTypePointsOpacity )
{
  cip::ChestConventions conventions;

  cipChestRegionChestTypeLocationsIO regionTypeIO;
    regionTypeIO.SetFileName( regionTypePointsFileName );
  if ( !regionTypeIO.Read() )
    {
      std::cout << "Failed to read region-type points file" << std::endl;
    }

  for ( unsigned int i=0; i<regionTypePointsRegions.size(); i++ )
    {
      std::string name = conventions.GetChestRegionName( regionTypePointsRegions[i] );
      name.append( conventions.GetChestTypeName( regionTypePointsTypes[i] ) );

      unsigned char cipRegion = regionTypePointsRegions[i];
      unsigned char cipType   = regionTypePointsTypes[i];

      vtkSmartPointer< vtkPolyData > spheresPoly = vtkSmartPointer< vtkPolyData >::New();

      regionTypeIO.GetOutput()->GetPolyDataFromChestRegionChestTypeDesignation( spheresPoly, cipRegion, cipType );

      interactor->SetPointsAsSpheres( spheresPoly, regionTypePointsScale[i], name );
      interactor->SetActorColor( name, regionTypePointsRed[i], regionTypePointsGreen[i], regionTypePointsBlue[i] );
      interactor->SetActorOpacity( name, regionTypePointsOpacity[i] );
    }
}

#endif
