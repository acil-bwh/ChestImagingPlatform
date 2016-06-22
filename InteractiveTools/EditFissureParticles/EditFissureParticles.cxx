/** \file
 *  \ingroup interactiveTools 
 *  \details This program can be used to label fissure particles. Hitting
 *  the 't' key will label a collection of connected component particles
 *  as being fissure particles.
 *
 *  Once the user has designated all the components, simply hitting
 *  the 'e' key  will write the particles to file and exit the
 *  program.
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <iostream>
#include <sstream>
#include <tclap/CmdLine.h>
#include "vtkPolyDataWriter.h"
#include "vtkPolyDataReader.h"
#include "vtkSmartPointer.h"
#include "cipFissureDataInteractor.h"
#include "vtkFloatArray.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include "vtkPointData.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "cipFissureParticleConnectedComponentFilter.h"

typedef itk::Image< short, 3 >             ImageType;
typedef itk::ImageFileReader< ImageType >  ReaderType;

void AddComponentsToInteractor( cipFissureDataInteractor*, vtkSmartPointer< vtkPolyData >, std::string, 
                                std::map< unsigned short, std::string >*, double );
vtkSmartPointer< vtkPolyData > GetLabeledFissureParticles( cipFissureDataInteractor*, vtkSmartPointer< vtkPolyData >, 
                                                          std::map< unsigned short, std::string >*  );
void AddSpecifiedParticlesToInteractor( cipFissureDataInteractor*, vtkSmartPointer< vtkPolyData >, std::string, float, std::string, double );

int main( int argc, char *argv[] )
{
  // Begin by defining the arguments to be passed. 
  std::string  inParticlesFileName     = "NA";
  std::string  outParticlesFileName    = "NA";
  std::string  ctFileName              = "NA";
  double       particleSize            = 1.0;

  // Filter parameters
  double interParticleSpacing = 1.7;
  double maxAllowableDistance = 3.0; 
  double particleAngleThreshold = 70.0;
  unsigned int componentSizeThreshold = 10;
  unsigned int maxComponentSize = 10000;

  // Input descriptions for user convenience
  std::string programDesc = "This program can be used to label fissure \
particles. Hitting the 't' key will label a collection of connected \
component particles \as being fissure particles. Once the user has \
designated all the components, simply hitting the 'e' key  will \
write the particles to file and exit the program.";

  std::string inParticlesFileNameDesc     = "Input particles file name";
  std::string outParticlesFileNameDesc    = "Output (labeled) particles file name";
  std::string ctFileNameDesc              = "Input CT file name";
  std::string particleSizeDesc            = "Particle size scale factor";

  std::string maxAllowableDistanceDesc = "Maximum inter-particle distance. Two particles must be at least this close \
together to be considered for connectivity";
  std::string particleAngleThresholdDesc = "Particle angle threshold used to test the connectivity between two particles (in degrees). \
The vector connecting two particles is computed. The angle formed between the connecting vector and the particle Hessian \
eigenvector pointing in the direction of the fissure axis is then considered. For both particles, this angle must be above \
the specified threshold for the particles to be connected";
  std::string componentSizeThresholdDesc = "Component size cardinality threshold. Only components with this many particles or more \
will be retained in the output";
  std::string maxComponentSizeDesc = "The maximum number of particles than can be in a single component";

  // Parse the input arguments
  try
    {
    TCLAP::CmdLine cl( programDesc, ' ', "$Revision: 370 $" );

    TCLAP::ValueArg<std::string> inParticlesFileNameArg( "i", "in", inParticlesFileNameDesc, true, inParticlesFileName, "string", cl );
    TCLAP::ValueArg<std::string> outParticlesFileNameArg( "", "out", outParticlesFileNameDesc, false, outParticlesFileName, "string", cl );
    TCLAP::ValueArg<std::string> ctFileNameArg( "c", "ct", ctFileNameDesc, false, ctFileName, "string", cl );
    TCLAP::ValueArg<double>      particleSizeArg( "s", "pSize", particleSizeDesc, false, particleSize, "double", cl );
    // Filter args:
    TCLAP::ValueArg<double>        maxAllowableDistanceArg( "d", "", maxAllowableDistanceDesc, false, maxAllowableDistance, "double", cl );
    TCLAP::ValueArg<double>        particleAngleThresholdArg( "", "angle", particleAngleThresholdDesc, false, particleAngleThreshold, "double", cl );
    TCLAP::ValueArg<unsigned int>  componentSizeThresholdArg( "", "cs", componentSizeThresholdDesc, false, componentSizeThreshold, "unsigned int", cl );
    TCLAP::ValueArg<unsigned int>  maxComponentSizeArg( "", "max", maxComponentSizeDesc, false, maxComponentSize, "unsigned int", cl );

    cl.parse( argc, argv );

    maxAllowableDistance   = maxAllowableDistanceArg.getValue();
    particleAngleThreshold = particleAngleThresholdArg.getValue();
    componentSizeThreshold = componentSizeThresholdArg.getValue();
    maxComponentSize       = maxComponentSizeArg.getValue();
    inParticlesFileName    = inParticlesFileNameArg.getValue();
    outParticlesFileName   = outParticlesFileNameArg.getValue();
    ctFileName             = ctFileNameArg.getValue();
    particleSize           = particleSizeArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  cip::ChestConventions conventions;
  cipFissureDataInteractor interactor;               
  
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

  // The nameToComponentMap will keep track of the mapping between the
  // names we assign the actors and the corresponding component
  // numbers of the original polyData
  std::map< unsigned short, std::string > componentLabelToNameMap;
  
  std::cout << "Reading fissure particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > particlesReader = vtkSmartPointer< vtkPolyDataReader >::New();
    particlesReader->SetFileName( inParticlesFileName.c_str() );
    particlesReader->Update();    

  // Assert that the input particles have 'ChestRegion' and
  // 'ChestType' arrays. If the input does not, this function will
  // add them
  std::cout << "Asserting ChestRegion and ChestType array existence..." << std::endl;
  cip::AssertChestRegionChestTypeArrayExistence( particlesReader->GetOutput() );

  std::cout << "Filtering particles..." << std::endl;
  cipFissureParticleConnectedComponentFilter* filter = new cipFissureParticleConnectedComponentFilter();
    filter->SetInterParticleSpacing( interParticleSpacing );
    filter->SetParticleDistanceThreshold( maxAllowableDistance );
    filter->SetInput( particlesReader->GetOutput() );
    filter->SetComponentSizeThreshold( componentSizeThreshold );
    filter->SetParticleAngleThreshold( particleAngleThreshold );
    filter->SetMaximumComponentSize( maxComponentSize );
    filter->Update();

  // Give the output file name to the interactor. This will allow the user to
  // save work as he/she goes along.
  interactor.SetFileName( outParticlesFileName.c_str() );

  std::cout << "Adding components to interactor..." << std::endl;
  AddComponentsToInteractor( &interactor, filter->GetOutput(), "fissureParticles", &componentLabelToNameMap, 
			     particleSize );
  
  std::cout << "Rendering..." << std::endl;  
  interactor.Render();
  
  vtkSmartPointer< vtkPolyData > outParticles = vtkSmartPointer< vtkPolyData >::New();
  
  std::cout << "Retrieving labeled particles..." << std::endl;
  outParticles = GetLabeledFissureParticles( &interactor, filter->GetOutput(), &componentLabelToNameMap ); 
  
  std::cout << "Writing labeled particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > writer = vtkSmartPointer< vtkPolyDataWriter >::New();
    writer->SetFileName( outParticlesFileName.c_str() );
    writer->SetInputData( outParticles );
    //writer->SetFileTypeToASCII();
    writer->Write();  

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

void AddComponentsToInteractor( cipFissureDataInteractor* interactor, vtkSmartPointer< vtkPolyData > particles, std::string whichLung, 
                                std::map< unsigned short, std::string >* componentLabelToNameMap, double particleSize )  
{
  unsigned int numberParticles         = particles->GetNumberOfPoints();
  unsigned int numberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();

  unsigned short component;
  std::vector< unsigned short > componentVec;
  //std::vector< unsigned char > cipTypeVec;

  // First get all previously labeled fissure particles
  // std::vector< unsigned int > labeledIDs;
  // for ( unsigned int i=0; i<numberParticles; i++ )
  //   {
  //     if ( *(particles->GetPointData()->GetArray( "ChestType" )->GetTuple(i)) == float(cip::FISSURE) ||
  // 	   *(particles->GetPointData()->GetArray( "ChestType" )->GetTuple(i)) == float(cip::OBLIQUEFISSURE) ||
  // 	   *(particles->GetPointData()->GetArray( "ChestType" )->GetTuple(i)) == float(cip::HORIZONTALFISSURE) )
  // 	{
  // 	  labeledIDs.push_back( i );
  // 	}
  //   }

  for ( unsigned int i=0; i<numberParticles; i++ )
    {
      component = (unsigned short)( *(particles->GetPointData()->GetArray( "unmergedComponents" )->GetTuple(i)) );
	
      // The input particles may already be labeled. Get the ChestType
      // recorded for thie component. By default we will color according
      // to this type
      //unsigned char cipType = (unsigned char)( *(particles->GetPointData()->GetArray( "ChestType" )->GetTuple(i)) );
      
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
	    //cipTypeVec.push_back( cipType );
	  }
    }
  
  // Now create the different poly data for the different components
  // and add them to the editor
  for ( unsigned int c=0; c<componentVec.size(); c++ )
    {
      std::stringstream stream;
      stream << componentVec[c];
      
      std::string actorName = stream.str();
      actorName.append( whichLung );

      (*componentLabelToNameMap)[componentVec[c]] = actorName;

      AddSpecifiedParticlesToInteractor( interactor, particles, "unmergedComponents", componentVec[c], 
					 actorName, particleSize );
    }  
}

void AddSpecifiedParticlesToInteractor( cipFissureDataInteractor* interactor, vtkSmartPointer< vtkPolyData > particles,
					std::string specifiedArrayName, float specifiedArrayVal,
					std::string interactorActorName, double particleSize )
{
  cip::ChestConventions conventions;

  unsigned char cipType; // Used to determine actor color

  unsigned int numberParticles         = particles->GetNumberOfPoints();
  unsigned int numberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();

  vtkSmartPointer< vtkPolyData > polyData = vtkSmartPointer< vtkPolyData >::New();
  vtkSmartPointer< vtkPoints > points  = vtkSmartPointer< vtkPoints >::New();
  std::vector< vtkSmartPointer< vtkFloatArray > > arrayVec;
  
  for ( unsigned int i=0; i<numberOfPointDataArrays; i++ )
    {
      vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
        array->SetNumberOfComponents( particles->GetPointData()->GetArray(i)->GetNumberOfComponents() );
	array->SetName( particles->GetPointData()->GetArray(i)->GetName() );
      
      arrayVec.push_back( array );
    }

  unsigned int inc = 0;
  for ( unsigned int p=0; p<numberParticles; p++ )
    {
      float val = *(particles->GetPointData()->GetArray( specifiedArrayName.c_str() )->GetTuple(p));

      if ( val == specifiedArrayVal )
	{
	  // Get the particle's type in order to retrieve color later. We're assuming that all particles
	  // for the specification have the same type, so we can grab the type of any one of them for the
	  // color.
	  cipType = conventions.GetChestTypeFromValue((unsigned short)(*(particles->GetPointData()->
									 GetArray( "ChestRegionChestType" )->GetTuple(p))));
	  
	  points->InsertNextPoint( particles->GetPoint(p) );

	  for ( unsigned int j=0; j<numberOfPointDataArrays; j++ )
	    {
	      arrayVec[j]->InsertTuple( inc, particles->GetPointData()->GetArray(j)->GetTuple(p) );
	    }

	  inc++;
	}
    }
  
  if ( inc > 0 )
    {
    double* color = new double[3];
    double r, g, b;
    if ( cipType == (unsigned char)(cip::UNDEFINEDTYPE) )
      {
      r = 1.0;
      g = 1.0;
      b = 1.0;
      }
    else
      {
      conventions.GetChestTypeColor( cipType, color );

      r = 1.0; //color[0];
      g = 1.0; //color[1];
      b = 1.0; //color[2];
      }

    polyData->SetPoints( points );
    for ( unsigned int j=0; j<numberOfPointDataArrays; j++ )
      {
      polyData->GetPointData()->AddArray( arrayVec[j] );
      }

    interactor->SetFissureParticlesAsDiscs( polyData, particleSize, interactorActorName ); 
    interactor->SetActorColor( interactorActorName, r, g, b );
    interactor->SetActorOpacity( interactorActorName, 1 );  
    }
}

// Iterate over all particles, get the particle's component, get the
// component's name, using the name get the component color, with the
// color assign the proper generation label
vtkSmartPointer< vtkPolyData > GetLabeledFissureParticles( cipFissureDataInteractor* interactor, vtkSmartPointer< vtkPolyData > particles, 
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
    unsigned char cipType = conventions.GetChestTypeFromValue((unsigned short)(*(particles->GetPointData()->
				GetArray( "ChestRegionChestType" )->GetTuple(i))));
    
    std::string name = (*componentLabelToNameMap)[componentLabel];
    
    if ( interactor->Exists( name ) )
      {
	interactor->GetActorColor( name, actorColor ); 	
	unsigned char cipTypeFromColor = conventions.GetChestTypeFromColor( actorColor );
	if ( cipTypeFromColor == cip::FISSURE || cipTypeFromColor == cip::OBLIQUEFISSURE || cipTypeFromColor == cip::HORIZONTALFISSURE ||
	     cipType == cip::FISSURE || cipType == cip::OBLIQUEFISSURE || cipType == cip::HORIZONTALFISSURE )
	  {
	    outPoints->InsertNextPoint( particles->GetPoint(i) );
	    unsigned char tmp;
	    if ( cipTypeFromColor != 0 )
	      {
		tmp = cipTypeFromColor;
	      }
	    else
	      {
		tmp = cipType;
	      }

	    unsigned short currentChestRegionChestTypeValue = 
	      (unsigned short)(*(particles->GetPointData()->GetArray( "ChestRegionChestType" )->GetTuple(i)));
	    unsigned char cipRegion = conventions.GetChestRegionFromValue( currentChestRegionChestTypeValue );
	    float newChestRegionChestTypeValue = float(conventions.GetValueFromChestRegionAndType( cipRegion, tmp ));
	    particles->GetPointData()->GetArray( "ChestRegionChestType" )->SetTuple( i, &newChestRegionChestTypeValue );
	    
	    for ( unsigned int j=0; j<numberOfPointDataArrays; j++ )
	      {
		arrayVec[j]->InsertTuple( inc, particles->GetPointData()->GetArray(j)->GetTuple(i) );
	      }	    
	    inc++;
	  }
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

#endif
