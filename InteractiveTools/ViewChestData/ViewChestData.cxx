/** \file
 *  \ingroup interactiveTools 
 *  \details This ...
 *  
 *  $Date: 2013-04-02 12:04:01 -0400 (Tue, 02 Apr 2013) $
 *  $Revision: 399 $
 *  $Author: jross $
 *
 *  TODO:
 *  
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include <fstream>
#include "vtkPolyDataReader.h"
#include "cipChestDataViewer.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "cipConventions.h"
#include "vtkPointData.h"
#include "itkContinuousIndex.h"
#include "vtkSmartPointer.h"
#include "vtkIndent.h"
#include "cipChestRegionChestTypeLocationsIO.h"
#include "vtkPolyData.h"


typedef itk::Image< unsigned short, 3 >                     LabelMapType;
typedef itk::ImageFileReader< LabelMapType >                LabelMapReaderType;
typedef itk::Image< short, 3 >                              CTImageType;
typedef itk::ImageFileReader< CTImageType >                 CTReaderType;
typedef itk::ImageRegionIteratorWithIndex< LabelMapType >   IteratorType;
typedef itk::ContinuousIndex< double, 3 >                   ContinuousIndexType;


struct ACTORPROPERTIES
{
  double red;
  double green;
  double blue;
  double opacity;
  std::string name;
};


struct REGIONTYPEPOINTS
{
  unsigned char lungRegion;
  unsigned char lungType;
  double x;
  double y;
  double z;
};

struct PARTICLEACTORPROPERTIES
{
  ACTORPROPERTIES properties;
  double scaleFactor;
};


//void ParseRegionTypePointsFile( char*, std::vector< REGIONTYPEPOINTS >* );
void GetPolyDataFromRegionPoints( vtkPolyData*, std::vector< REGIONTYPEPOINTS >, unsigned char );
void GetPolyDataFromTypePoints( vtkPolyData*, std::vector< REGIONTYPEPOINTS >, unsigned char );
void GetPolyDataFromRegionTypePoints( vtkPolyData*, std::vector< REGIONTYPEPOINTS >, unsigned char, unsigned char );
void AddParticlesToViewer( cipChestDataViewer*, std::vector< std::string >, std::vector< double >, std::vector< double >, 
			   std::vector< double >, std::vector< double >, std::vector< double >, std::string, std::string );
void AddRegionTypePointsAsSpheresToViewer( cipChestDataViewer*, std::string, std::vector< unsigned char >, std::vector< unsigned char >, 
					   std::vector< double >, std::vector< double >, std::vector< double >, std::vector< double >, 
					   std::vector< double > );
vtkSmartPointer< vtkPolyData > GetChestTypeParticlesPolyData( vtkSmartPointer< vtkPolyData >, unsigned char );
void AddParticlesToViewerUsingPresets( cipChestDataViewer*, std::vector< std::string >, std::string, std::string );


int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed. Because of the
  // constraints imposed by TCLAP for command line parsing, we must
  // define lots of entities for the desired viewing flexibility.
  //
  std::string ctFileName               = "NA";
  std::string regionTypePointsFileName = "NA";
  std::vector< unsigned char > regionTypePointsRegions;
  std::vector< unsigned char > regionTypePointsTypes;
  std::vector< double > regionTypePointsRed;
  std::vector< double > regionTypePointsGreen;
  std::vector< double > regionTypePointsBlue;
  std::vector< double > regionTypePointsOpacity;
  std::vector< double > regionTypePointsScale;

  // Background color:
  double bgRed           = 1.0;
  double bgGreen         = 1.0;
  double bgBlue          = 1.0;
  // Airway particles file names options:
  std::vector< std::string > airwayCylindersFileNames;
  std::vector< std::string > airwayCylindersPresetsFileNames;
  std::vector< std::string > airwaySpheresFileNames;
  std::vector< std::string > airwayParticlesFileNames;
  std::vector< std::string > airwayScaledCylindersFileNames;
  std::vector< std::string > airwayScaledSpheresFileNames;
  // Airway cylinder colors:
  std::vector< double > airwayCylindersRed;
  std::vector< double > airwayCylindersGreen;
  std::vector< double > airwayCylindersBlue;
  // Airway sphere colors:
  std::vector< double > airwaySpheresRed;
  std::vector< double > airwaySpheresGreen;
  std::vector< double > airwaySpheresBlue;
  // Airway particles colors:
  std::vector< double > airwayParticlesRed;
  std::vector< double > airwayParticlesGreen;
  std::vector< double > airwayParticlesBlue;
  // Airway scaled cylinder colors:
  std::vector< double > airwayScaledCylindersRed;
  std::vector< double > airwayScaledCylindersGreen;
  std::vector< double > airwayScaledCylindersBlue;
  // Airway scaled sphere colors:
  std::vector< double > airwayScaledSpheresRed;
  std::vector< double > airwayScaledSpheresGreen;
  std::vector< double > airwayScaledSpheresBlue;
  // Airway opacities for various glyph options
  std::vector< double > airwayCylindersOpacity;
  std::vector< double > airwaySpheresOpacity;
  std::vector< double > airwayParticlesOpacity;
  std::vector< double > airwayScaledCylindersOpacity;
  std::vector< double > airwayScaledSpheresOpacity;
  // Airway glyph size
  std::vector< double > airwayCylindersSize;
  std::vector< double > airwaySpheresSize;
  std::vector< double > airwayParticlesSize;
  std::vector< double > airwayScaledCylindersSize;
  std::vector< double > airwayScaledSpheresSize;

  // Fissure particles file names
  std::vector< std::string > fissureParticlesFileNames;
  std::vector< double >      fissureParticlesRed;
  std::vector< double >      fissureParticlesGreen;
  std::vector< double >      fissureParticlesBlue;
  std::vector< double >      fissureParticlesOpacity;
  std::vector< double >      fissureParticlesSize;

  //
  // Input descriptions for user convenience
  //
  std::string programDesc = "This program...";

  std::string ctFileNameDesc               = "CT file name (single, 3D volume)";
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

  std::string bgRedDesc      = "Background color red channel in interval [0,1]";
  std::string bgGreenDesc    = "Background color green channel in interval [0,1]";
  std::string bgBlueDesc     = "Background color blue channel in interval [0,1]";
  // Airway particles file names option descriptions 
  std::string airwayCylindersFileNamesDesc       = "Airway particles file name to be rendered as cylinders. You \
must also specify the color, opacity, and size of the glyphs with the --acr, --acg, --acb, --aco, and --acs \
flags, respectively. These flags should be invoked immediately after invoking the --aCy flag.";
  std::string airwayCylindersPresetsFileNamesDesc = "Airway particles file name to be rendered as cylinders using \
color, size, and opacity presets. Colors are chosen based on the 'ChestType' array values.";
  std::string airwaySpheresFileNamesDesc         = "Airway particles file name to be rendered as spheres. You \
must also specify the color, opacity, and size of the glyphs with the --asr, --asg, --asb, --aso, and --ass \
flags, respectively. These flags should be invoked immediately after invoking the --aSph flag.";
  std::string airwayParticlesFileNamesDesc       = "Airway particles file name to be rendered as particles. You \
must also specify the color, opacity, and size of the glyphs with the --apr, --apg, --apb, --apo, and --aps \
flags, respectively. These flags should be invoked immediately after invoking the --aPart flag.";
  std::string airwayScaledCylindersFileNamesDesc = "Airway particles file name to be rendered as scaled cylinders. You \
must also specify the color, opacity, and size of the glyphs with the --ascr, --ascg, --ascb, --asco, and --ascs \
flags, respectively. These flags should be invoked immediately after invoking the --asCy flag.";
  std::string airwayScaledSpheresFileNamesDesc   = "Airway particles file name to be rendered as scaled spheres. You \
must also specify the color, opacity, and size of the glyphs with the --assr, --assg, --assb, --asso, and --asss \
flags, respectively. These flags should be invoked immediately after invoking the --asSph flag.";
  // Airway cylinder color descriptions:
  std::string airwayCylindersRedDesc   = "Red channel for airway cylinders in interval [0,1]. See notes for --aCy flag";
  std::string airwayCylindersGreenDesc = "Green channel for airway cylinders in interval [0,1]. See notes for --aCy flag";
  std::string airwayCylindersBlueDesc  = "Blue channel for airway cylinders in interval [0,1]. See notes for --aCy flag";
  // Airway sphere color descriptions:
  std::string airwaySpheresRedDesc   = "Red channel for airway spheres in interval [0,1]. See notes for --aSph flag";
  std::string airwaySpheresGreenDesc = "Green channel for airway spheres in interval [0,1]. See notes for --aSph flag";
  std::string airwaySpheresBlueDesc  = "Blue channel for airway spheres in interval [0,1]. See notes for --aSph flag";
  // Airway particles color descriptions:
  std::string airwayParticlesRedDesc   = "Red channel for airway particles in interval [0,1]. See notes for --aPart flag";
  std::string airwayParticlesGreenDesc = "Green channel for airway particles in interval [0,1]. See notes for --aPart flag";
  std::string airwayParticlesBlueDesc  = "Blue channel for airway particles in interval [0,1]. See notes for --aPart flag";
  // Airway scaled cylinder color descriptions:
  std::string airwayScaledCylindersRedDesc   = "Red channel for airway scaled cylinders in interval [0,1]. See notes for --asCy flag";
  std::string airwayScaledCylindersGreenDesc = "Green channel for airway scaled cylinders in interval [0,1]. See notes for --asCy flag";
  std::string airwayScaledCylindersBlueDesc  = "Blue channel for airway scaled cylinders in interval [0,1]. See notes for --asCy flag";
  // Airway scaled sphere color descriptions:
  std::string airwayScaledSpheresRedDesc   = "Red channel for airway scaled spheres in interval [0,1]. See notes for --asSp flag";
  std::string airwayScaledSpheresGreenDesc = "Green channel for airway scaled spheres in interval [0,1]. See notes for --asSp flag";
  std::string airwayScaledSpheresBlueDesc  = "Blue channel for airway scaled spheres in interval [0,1]. See notes for --asSp flag";
  // Airway opacities for various glyph option descriptions:
  std::string airwayCylindersOpacityDesc       = "Airway cylinders opacity in interval [0,1]. See notes for --aCy flag";
  std::string airwaySpheresOpacityDesc         = "Airway spheres opacity in interval [0,1]. See notes for --aSp flag";
  std::string airwayParticlesOpacityDesc       = "Airway particles opacity in interval [0,1]. See notes for --aPart flag";
  std::string airwayScaledCylindersOpacityDesc = "Airway scaled cylinders opacity in interval [0,1]. See notes for --asCy flag";
  std::string airwayScaledSpheresOpacityDesc   = "Airway scaled spheres opacity in interval [0,1]. See notes for --asSp flag";
  // Airway glyph size descriptions:
  std::string airwayCylindersSizeDesc       = "Airway cylinder size. See notes for --aCy flag";
  std::string airwaySpheresSizeDesc         = "Airway spheres size. See notes for --aSp flag";
  std::string airwayParticlesSizeDesc       = "Airway particles size. See notes for --aPart flag";
  std::string airwayScaledCylindersSizeDesc = "Airway scaled cylinder size. See notes for --asCy flag";
  std::string airwayScaledSpheresSizeDesc   = "Airway scaled spheres size. See notes for --asSp flag";

  // Descs for fissure particle inputs
  std::string fissureParticlesFileNamesDesc  = "Fissure particles file name to be rendered as particles. You \
must also specify the color, opacity, and size of the glyphs with the --fpr, --fpg, --fpb, --fpo, and --fps \
flags, respectively. These flags should be invoked immediately after invoking the --fPart flag.";
  std::string fissureParticlesRedDesc        = "Red channel for fissure particles in interval [0,1]. See notes for --fPart flag";
  std::string fissureParticlesGreenDesc      = "Green channel for fissure particles in interval [0,1]. See notes for --fPart flag";
  std::string fissureParticlesBlueDesc       = "Blue channel for fissure particles in interval [0,1]. See notes for --fPart flag";
  std::string fissureParticlesOpacityDesc    = "Opacity for fissure particles in interval [0,1]. See notes for --fPart flag";
  std::string fissureParticlesSizeDesc       = "Fissure particle size. See notes for --fPart flag";

  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( programDesc, ' ', "$Revision: 399 $" );

    TCLAP::ValueArg<std::string> ctFileNameArg( "c", "ct", ctFileNameDesc, false, ctFileName, "string", cl );
    TCLAP::ValueArg<double> bgRedArg( "", "bgRed", bgRedDesc, false, bgRed, "double", cl );
    TCLAP::ValueArg<double> bgGreenArg( "", "bgGreen", bgGreenDesc, false, bgGreen, "double", cl );
    TCLAP::ValueArg<double> bgBlueArg( "", "bgBlue", bgBlueDesc, false, bgBlue, "double", cl );
    // Airway particles file names:
    TCLAP::MultiArg<std::string> airwayCylindersFileNamesArg( "", "aCy", airwayCylindersFileNamesDesc, false, "string", cl );
    TCLAP::MultiArg<std::string> airwaySpheresFileNamesArg( "", "aSp", airwaySpheresFileNamesDesc, false, "string", cl );
    TCLAP::MultiArg<std::string> airwayParticlesFileNamesArg( "", "aPart", airwayParticlesFileNamesDesc, false, "string", cl );
    TCLAP::MultiArg<std::string> airwayScaledCylindersFileNamesArg( "", "asCy", airwayScaledCylindersFileNamesDesc, false, "string", cl );
    TCLAP::MultiArg<std::string> airwayScaledSpheresFileNamesArg( "", "asSp", airwayScaledSpheresFileNamesDesc, false, "string", cl );
    TCLAP::MultiArg<std::string> airwayCylindersPresetsFileNamesArg( "", "aCyPre", airwayCylindersPresetsFileNamesDesc, false, "string", cl );
    // Airway particles colors, opacity and size
    TCLAP::MultiArg<double> airwayParticlesRedArg( "", "apr", airwayParticlesRedDesc, false, "double", cl );
    TCLAP::MultiArg<double> airwayParticlesGreenArg( "", "apg", airwayParticlesGreenDesc, false, "double", cl );
    TCLAP::MultiArg<double> airwayParticlesBlueArg( "", "apb", airwayParticlesBlueDesc, false, "double", cl );
    TCLAP::MultiArg<double> airwayParticlesOpacityArg( "", "apo", airwayParticlesOpacityDesc, false, "double", cl );
    TCLAP::MultiArg<double> airwayParticlesSizeArg( "", "aps", airwayParticlesSizeDesc, false, "double", cl );    

    TCLAP::MultiArg<double> airwayCylindersRedArg( "", "acr", airwayCylindersRedDesc, false, "double", cl );
    TCLAP::MultiArg<double> airwayCylindersGreenArg( "", "acg", airwayCylindersGreenDesc, false, "double", cl );
    TCLAP::MultiArg<double> airwayCylindersBlueArg( "", "acb", airwayCylindersBlueDesc, false, "double", cl );
    TCLAP::MultiArg<double> airwayCylindersOpacityArg( "", "aco", airwayCylindersOpacityDesc, false, "double", cl );
    TCLAP::MultiArg<double> airwayCylindersSizeArg( "", "acs", airwayCylindersSizeDesc, false, "double", cl );    
    // Fissure particles inputs
    TCLAP::MultiArg<std::string> fissureParticlesFileNamesArg( "", "fPart", fissureParticlesFileNamesDesc, false, "string", cl );
    TCLAP::MultiArg<double>      fissureParticlesRedArg( "", "fpr", fissureParticlesRedDesc, false, "double", cl );
    TCLAP::MultiArg<double>      fissureParticlesGreenArg( "", "fpg", fissureParticlesGreenDesc, false, "double", cl );
    TCLAP::MultiArg<double>      fissureParticlesBlueArg( "", "fpb", fissureParticlesBlueDesc, false, "double", cl );
    TCLAP::MultiArg<double>      fissureParticlesOpacityArg( "", "fpo", fissureParticlesOpacityDesc, false, "double", cl );
    TCLAP::MultiArg<double>      fissureParticlesSizeArg( "", "fps", fissureParticlesSizeDesc, false, "double", cl );
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

    ctFileName = ctFileNameArg.getValue();
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
	regionTypePointsRegions.push_back( static_cast< unsigned char >(regionTypePointsRegionsArg.getValue()[i]) );
      }
    for ( unsigned int i=0; i<regionTypePointsTypesArg.getValue().size(); i++ )
      {
	regionTypePointsTypes.push_back( static_cast< unsigned char >(regionTypePointsTypesArg.getValue()[i]) );
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
    // Background colors
    bgRed   = bgRedArg.getValue();
    bgGreen = bgGreenArg.getValue();
    bgBlue  = bgBlueArg.getValue();
    // Airway particles file names:
    for ( unsigned int i=0; i<airwayCylindersFileNamesArg.getValue().size(); i++ )
      {
      airwayCylindersFileNames.push_back( airwayCylindersFileNamesArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<airwayCylindersPresetsFileNamesArg.getValue().size(); i++ )
      {
      airwayCylindersPresetsFileNames.push_back( airwayCylindersPresetsFileNamesArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<airwaySpheresFileNamesArg.getValue().size(); i++ )
      {
      airwaySpheresFileNames.push_back( airwaySpheresFileNamesArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<airwayParticlesFileNamesArg.getValue().size(); i++ )
      {
      airwayParticlesFileNames.push_back( airwayParticlesFileNamesArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<airwayScaledCylindersFileNamesArg.getValue().size(); i++ )
      {
      airwayScaledCylindersFileNames.push_back( airwayScaledCylindersFileNamesArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<airwayScaledSpheresFileNamesArg.getValue().size(); i++ )
      {
      airwayScaledSpheresFileNames.push_back( airwayScaledSpheresFileNamesArg.getValue()[i] );
      }

    // Airway particles color, opacity and size
    for ( unsigned int i=0; i<airwayParticlesRedArg.getValue().size(); i++ )
      {
      airwayParticlesRed.push_back( airwayParticlesRedArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<airwayParticlesGreenArg.getValue().size(); i++ )
      {
      airwayParticlesGreen.push_back( airwayParticlesGreenArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<airwayParticlesBlueArg.getValue().size(); i++ )
      {
      airwayParticlesBlue.push_back( airwayParticlesBlueArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<airwayParticlesOpacityArg.getValue().size(); i++ )
      {
      airwayParticlesOpacity.push_back( airwayParticlesOpacityArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<airwayParticlesSizeArg.getValue().size(); i++ )
      {
      airwayParticlesSize.push_back( airwayParticlesSizeArg.getValue()[i] );
      }

    // Airway cylinders color, opacity and size
    for ( unsigned int i=0; i<airwayCylindersRedArg.getValue().size(); i++ )
      {
      airwayCylindersRed.push_back( airwayCylindersRedArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<airwayCylindersGreenArg.getValue().size(); i++ )
      {
      airwayCylindersGreen.push_back( airwayCylindersGreenArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<airwayCylindersBlueArg.getValue().size(); i++ )
      {
      airwayCylindersBlue.push_back( airwayCylindersBlueArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<airwayCylindersOpacityArg.getValue().size(); i++ )
      {
      airwayCylindersOpacity.push_back( airwayCylindersOpacityArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<airwayCylindersSizeArg.getValue().size(); i++ )
      {
      airwayCylindersSize.push_back( airwayCylindersSizeArg.getValue()[i] );
      }

    // Fissure particles inputs    
    for ( unsigned int i=0; i<fissureParticlesFileNamesArg.getValue().size(); i++ )
      {
      fissureParticlesFileNames.push_back( fissureParticlesFileNamesArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<fissureParticlesRedArg.getValue().size(); i++ )
      {
      fissureParticlesRed.push_back( fissureParticlesRedArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<fissureParticlesGreenArg.getValue().size(); i++ )
      {
      fissureParticlesGreen.push_back( fissureParticlesGreenArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<fissureParticlesBlueArg.getValue().size(); i++ )
      {
      fissureParticlesBlue.push_back( fissureParticlesBlueArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<fissureParticlesOpacityArg.getValue().size(); i++ )
      {
      fissureParticlesOpacity.push_back( fissureParticlesOpacityArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<fissureParticlesSizeArg.getValue().size(); i++ )
      {
      fissureParticlesSize.push_back( fissureParticlesSizeArg.getValue()[i] );
      }

    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  ChestConventions conventions;
  
  cipChestDataViewer* viewer = new cipChestDataViewer();
//     viewer.SetParticleGlyphThetaResolution( particleGlyphThetaResolution );
//     viewer.SetParticlePhiThetaResolution( particleGlyphPhiResolution );
    viewer->SetBackgroundColor( bgRed, bgGreen, bgBlue );

  if ( ctFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Reading CT..." << std::endl;
    CTReaderType::Pointer ctReader = CTReaderType::New();
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

    viewer->SetGrayscaleImage( ctReader->GetOutput() );
    }    

  if ( airwayCylindersPresetsFileNames.size() > 0 )
    {
      AddParticlesToViewerUsingPresets( viewer, airwayCylindersPresetsFileNames, "airwayCylinders", "cylinder" );
    }
  if ( airwayCylindersFileNames.size() > 0 )
    {
    AddParticlesToViewer( viewer, airwayCylindersFileNames, airwayCylindersRed, airwayCylindersGreen, airwayCylindersBlue,
			  airwayCylindersOpacity, airwayCylindersSize, "airwayCylinders", "cylinder" );
    }
  if ( fissureParticlesFileNames.size() > 0 )
    {
    AddParticlesToViewer( viewer, fissureParticlesFileNames, fissureParticlesRed, fissureParticlesGreen, fissureParticlesBlue,
			  fissureParticlesOpacity, fissureParticlesSize, "fissureParticles", "particle" );
    }
  if ( airwayParticlesFileNames.size() > 0 )
    {
    AddParticlesToViewer( viewer, airwayParticlesFileNames, airwayParticlesRed, airwayParticlesGreen, airwayParticlesBlue,
			  airwayParticlesOpacity, airwayParticlesSize, "airwayParticles", "particle" );
    }
  if ( regionTypePointsFileName.compare( "NA" ) != 0 )
    {
      AddRegionTypePointsAsSpheresToViewer( viewer, regionTypePointsFileName, regionTypePointsRegions, regionTypePointsTypes, 
					    regionTypePointsRed, regionTypePointsGreen, regionTypePointsBlue, 
					    regionTypePointsScale, regionTypePointsOpacity );
    }

  std::cout << "Rendering..." << std::endl;  
  viewer->Render();

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}


void GetPolyDataFromRegionPoints( vtkPolyData* polyData, std::vector< REGIONTYPEPOINTS > regionPoints, unsigned char lungRegion )
{
  vtkPoints* points = vtkPoints::New();

  for ( unsigned int i=0; i<regionPoints.size(); i++ )
    {
    if ( regionPoints[i].lungRegion == lungRegion )
      {
      points->InsertNextPoint( regionPoints[i].x, regionPoints[i].y, regionPoints[i].z );
      }
    }

  polyData->SetPoints( points );
}


void GetPolyDataFromTypePoints( vtkPolyData* polyData, std::vector< REGIONTYPEPOINTS > typePoints, unsigned char lungType )
{
  vtkPoints* points = vtkPoints::New();

  for ( unsigned int i=0; i<typePoints.size(); i++ )
    {
    if ( typePoints[i].lungType == lungType )
      {
      points->InsertNextPoint( typePoints[i].x, typePoints[i].y, typePoints[i].z );
      }
    }

  polyData->SetPoints( points );
}


void GetPolyDataFromRegionTypePoints( vtkPolyData* polyData, std::vector< REGIONTYPEPOINTS > regionTypePoints, 
                                      unsigned char lungRegion, unsigned char lungType )
{
  vtkPoints* points = vtkPoints::New();

  for ( unsigned int i=0; i<regionTypePoints.size(); i++ )
    {
    if ( regionTypePoints[i].lungRegion == lungRegion && regionTypePoints[i].lungType == lungType )
      {
      points->InsertNextPoint( regionTypePoints[i].x, regionTypePoints[i].y, regionTypePoints[i].z );
      }
    }

  polyData->SetPoints( points );
}


void AddParticlesToViewerUsingPresets( cipChestDataViewer* viewer, std::vector< std::string > fileNames, 
				       std::string particlesType, std::string glyphType )
{
  ChestConventions conventions;

  double scale   = 2.0;
  double opacity = 1.0;

  unsigned char cipType;

  for ( unsigned int i=0; i<fileNames.size(); i++ )
    {
      std::cout << "Reading particles..." << std::endl;
      vtkSmartPointer< vtkPolyDataReader > reader = vtkSmartPointer< vtkPolyDataReader >::New();
      reader->SetFileName( fileNames[i].c_str() );
      reader->Update(); 

      std::list< unsigned char > cipTypeList;
      for ( unsigned int j=0; j<reader->GetOutput()->GetNumberOfPoints(); j++ )
	{
	  cipTypeList.push_back( static_cast< unsigned char >( reader->GetOutput()->GetFieldData()->GetArray( "ChestType" )->GetTuple(j)[0] ) );
	}
      cipTypeList.unique();
      cipTypeList.sort();
      cipTypeList.unique();

      std::list< unsigned char >::iterator listIt = cipTypeList.begin();

      while ( listIt != cipTypeList.end() )
	{
	  std::stringstream suffix;
	  suffix << i;

	  std::string name = particlesType;
       	    name.append( suffix.str() );
	    name.append( conventions.GetChestTypeName( *listIt ) );

	  vtkSmartPointer< vtkPolyData > tmpParticles = 
	    GetChestTypeParticlesPolyData( reader->GetOutput(), *listIt );   
	  
	  double* color = new double[3];
	  conventions.GetChestTypeColor( *listIt, color );

	  viewer->SetAirwayParticlesAsCylinders( tmpParticles, scale, name );
	  viewer->SetActorColor( name, color[0], color[1], color[2] );
	  viewer->SetActorOpacity( name, opacity );

	  listIt++;
	}
    }
}


vtkSmartPointer< vtkPolyData > GetChestTypeParticlesPolyData( vtkSmartPointer< vtkPolyData > inParticles, unsigned char cipType )
{
  std::vector< vtkSmartPointer< vtkFloatArray > > arrayVec;

  for ( unsigned int i=0; i<inParticles->GetFieldData()->GetNumberOfArrays(); i++ )
    {
      vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
        array->SetNumberOfComponents( inParticles->GetFieldData()->GetArray(i)->GetNumberOfComponents() );
	array->SetName( inParticles->GetFieldData()->GetArray(i)->GetName() );
      
      arrayVec.push_back( array );
    }

  vtkSmartPointer< vtkPolyData > outParticles = vtkSmartPointer< vtkPolyData >::New();
  vtkSmartPointer< vtkPoints >   outPoints    = vtkSmartPointer< vtkPoints >::New();

  unsigned int inc = 0;
  for ( unsigned int i=0; i<inParticles->GetNumberOfPoints(); i++ )
    {
      unsigned char tmpType = 
	static_cast< unsigned char >( inParticles->GetFieldData()->GetArray( "ChestType" )->GetTuple(i)[0] );

      if ( tmpType == cipType )
	{
	  outPoints->InsertNextPoint( inParticles->GetPoint(i) );

	  for ( unsigned int k=0; k<arrayVec.size(); k++ )
	    {	  
	      arrayVec[k]->InsertTuple( inc, inParticles->GetFieldData()->GetArray(k)->GetTuple(i) );
	    }     

	  inc++;
	}
    }

  outParticles->SetPoints( outPoints );
  for ( unsigned int j=0; j<arrayVec.size(); j++ )
    {
      outParticles->GetFieldData()->AddArray( arrayVec[j] );
    }

  return outParticles;
}


void AddParticlesToViewer( cipChestDataViewer* viewer, std::vector< std::string > fileNames, std::vector< double > red, 
			   std::vector< double > green, std::vector< double > blue, std::vector< double > opacity, 
			   std::vector< double > scale, std::string particlesType, std::string glyphType )
{
  for ( unsigned int i=0; i<fileNames.size(); i++ )
    {
    std::stringstream suffix;
    suffix << i;

    std::string name = particlesType;
    name.append( suffix.str() );

    std::cout << "Reading particles..." << std::endl;
    vtkSmartPointer< vtkPolyDataReader > reader = vtkSmartPointer< vtkPolyDataReader >::New();
      reader->SetFileName( fileNames[i].c_str() );
      reader->Update(); 

    if ( particlesType.compare( "fissureParticles" ) == 0 )
      {
	viewer->SetFissureParticles( reader->GetOutput(), scale[i], name );
      }
    if ( particlesType.compare( "airwayCylinders" ) == 0 )
      {
	viewer->SetAirwayParticlesAsCylinders( reader->GetOutput(), scale[i], name );
      }
    if ( particlesType.compare( "airwayParticles" ) == 0 )
      {
	if ( glyphType.compare( "cylinder" ) == 0 )
	  {
	    viewer->SetAirwayParticlesAsCylinders( reader->GetOutput(), scale[i], name );
	  }
	else
	  {
	    viewer->SetAirwayParticles( reader->GetOutput(), scale[i], name );
	  }
      }

    viewer->SetActorColor( name, red[i], green[i], blue[i] );
    viewer->SetActorOpacity( name, opacity[i] );
    }
}


void AddRegionTypePointsAsSpheresToViewer( cipChestDataViewer* viewer, std::string regionTypePointsFileName, std::vector< unsigned char > regionTypePointsRegions, 
					   std::vector< unsigned char > regionTypePointsTypes, std::vector< double > regionTypePointsRed, 
					   std::vector< double > regionTypePointsGreen, std::vector< double > regionTypePointsBlue, 
					   std::vector< double > regionTypePointsScale, std::vector< double >regionTypePointsOpacity )
{
  ChestConventions conventions;

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

      viewer->SetPointsAsSpheres( spheresPoly, regionTypePointsScale[i], name );
      viewer->SetActorColor( name, regionTypePointsRed[i], regionTypePointsBlue[i], regionTypePointsBlue[i] );
      viewer->SetActorOpacity( name, regionTypePointsOpacity[i] );
    }
}


#endif

