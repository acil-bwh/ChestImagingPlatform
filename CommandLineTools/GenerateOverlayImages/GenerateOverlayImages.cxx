/** \file
 *  \ingroup commandLineTools 
 *  \details This program can be used to 
 * 
 *  USAGE: 
 *
 *  GenerateOverlayImages  [-x] [-y] [-z] [--type <unsigned char>]
 *                           [--region <unsigned char>] [--opacity <double>]
 *                           [--level <short>] [--window <short>] -c
 *                           <string> -l <string> [-o <string>] ...  [--]
 *                           [--version] [-h]
 *
 *  Where: 
 *
 *   -x,  --sagittal
 *     Set to 1 if sagittal overlay images are desired (0 by default)
 *
 *   -y,  --coronal
 *     Set to 1 if coronal overlay images are desired (0 by default)
 *
 *   -z,  --axial
 *     Set to 1 if axial overlay images are desired (1 by default)
 *
 *   --type <unsigned char>
 *     The chest type over which to compute the bounding box which in turn
 *     defines where to take the slice planes from for the overlays. By
 *     default this value is set to UNDEFINEDTYPE. If both the chest region
 *     and chest type are left undefined, the entire foreground region will
 *     we considered when computing the bounding box.
 *
 *   --region <unsigned char>
 *     The chest region over which to compute the bounding box which in turn
 *     defines where to take the slice planes from for the overlays. By
 *     default this value is set to UNDEFINEDREGION. If both the chest region
 *     and chest type are left undefined, the entire foreground region will
 *     we considered when computing the bounding box
 *
 *   --opacity <double>
 *     A real number between 0 and 1 indicating the opacity of the overlay
 *     (default is 0.5)
 *
 *   --level <short>
 *     The level setting in Hounsfield units for window-leveling
 *
 *   --window <short>
 *     The window width setting in Hounsfield units for window-leveling
 *
 *   -c <string>,  --ct <string>
 *     (required)  Input CT image file name. Only needs to be specified if
 *     you intend to createlung lobe QC images
 *
 *   -l <string>,  --labelMap <string>
 *     (required)  Input label map file name
 *
 *   -o <string>,  --overlays <string>  (accepted multiple times)
 *     Names of the overlay images to produce. The images will be spaced
 *     evenly across the bounding box in the direction orthogonal to the
 *     plane of interest.
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
 *   This program produces RGB overlay images corresponding to the input CT
 *   image and its label map. The overlay images will be spaced evenly across
 *   the bounding box in the direction orthogonal to the plane of interest
 *   (axial, coronal, or sagittal). The user has control over the
 *   window-level settings as well as the opacity of the overlay. The colors
 *   used in the overlay are established in the CIP conventions
 *
 *  $Date$
 *  $Revision$
 *  $Author$
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "cipHelper.h"
#include "cipConventions.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkRGBPixel.h"

typedef itk::RGBPixel<unsigned char>       RGBPixelType;
typedef itk::Image<RGBPixelType, 2>        OverlayType;
typedef itk::ImageFileWriter<OverlayType>  OverlayWriterType;

double GetWindowLeveledValue(short, short, short);
RGBPixelType GetOverlayPixelValue(double, unsigned short, double);
void GetOverlayImages(cip::LabelMapType::Pointer, cip::CTType::Pointer, unsigned int, std::vector<OverlayType::Pointer>*, 
		      double, std::string, short, short, unsigned char, unsigned char);

int main( int argc, char *argv[] )
{
  // Begin by defining the arguments to be passed
  std::string labelMapFileName = "NA";
  std::string ctFileName       = "NA";
  bool axial    = true;
  bool coronal  = false;
  bool sagittal = false;
  short window = 1100;
  short level  = -650;
  double opacity = 0.5;
  unsigned char cipRegion = (unsigned char)(cip::UNDEFINEDREGION);
  unsigned char cipType = (unsigned char)(cip::UNDEFINEDTYPE);
  std::vector<std::string> overlayFileNameVec;

  // Descriptions of the program and inputs for user help 
  std::string programDesc = "This program produces RGB overlay images corresponding to the input CT image and \
its label map. The overlay images will be spaced evenly across the bounding box in the direction orthogonal \
to the plane of interest (axial, coronal, or sagittal). The user has control over the window-level settings \
as well as the opacity of the overlay. The colors used in the overlay are established in the CIP conventions";

  std::string labelMapDesc = "Input label map file name";
  std::string ctFileNameDesc = "Input CT image file name. Only needs to be specified if you intend to create\
lung lobe QC images";
  std::string overlayFileNameVecDesc = "Names of the overlay images to produce. The images will be spaced \
evenly across the bounding box in the direction orthogonal to the plane of interest.";
  std::string windowDesc = "The window width setting in Hounsfield units for window-leveling";
  std::string levelDesc = "The level setting in Hounsfield units for window-leveling";
  std::string opacityDesc = "A real number between 0 and 1 indicating the opacity of the overlay (default is 0.5)";
  std::string axialDesc = "Set to 1 if axial overlay images are desired (1 by default)";
  std::string coronalDesc = "Set to 1 if coronal overlay images are desired (0 by default)";
  std::string sagittalDesc = "Set to 1 if sagittal overlay images are desired (0 by default)";
  std::string cipRegionDesc = "The chest region over which to compute the bounding box which in turn defines \
where to take the slice planes from for the overlays. By default this value is set to UNDEFINEDREGION. If both \
the chest region and chest type are left undefined, the entire foreground region will we considered when \
computing the bounding box";
  std::string cipTypeDesc = "The chest type over which to compute the bounding box which in turn defines \
where to take the slice planes from for the overlays. By default this value is set to UNDEFINEDTYPE. If both \
the chest region and chest type are left undefined, the entire foreground region will we considered when \
computing the bounding box.";

  // Parse the input arguments
  try
    {
    TCLAP::CmdLine cl(programDesc, ' ', "$Revision$");

    TCLAP::MultiArg<std::string> overlayFileNameVecArg ( "o", "overlays", overlayFileNameVecDesc, false, "string", cl );
    TCLAP::ValueArg<std::string> labelMapFileNameArg ("l", "labelMap", labelMapDesc, true, labelMapFileName, "string", cl);
    TCLAP::ValueArg<std::string> ctFileNameArg ("c", "ct", ctFileNameDesc, true, ctFileName, "string", cl);
    TCLAP::ValueArg<short> windowArg ("", "window", windowDesc, false, window, "short", cl);
    TCLAP::ValueArg<short> levelArg ("", "level", levelDesc, false, level, "short", cl);
    TCLAP::ValueArg<double> opacityArg ("", "opacity", opacityDesc, false, opacity, "double", cl);
    TCLAP::ValueArg<unsigned int> cipRegionArg ("", "region", cipRegionDesc, false, cipRegion, "unsigned char", cl);
    TCLAP::ValueArg<unsigned int> cipTypeArg ("", "type", cipTypeDesc, false, cipType, "unsigned char", cl);
    TCLAP::SwitchArg axialArg("z", "axial", axialDesc, cl, false);
    TCLAP::SwitchArg coronalArg("y", "coronal", coronalDesc, cl, false);
    TCLAP::SwitchArg sagittalArg("x", "sagittal", sagittalDesc, cl, false);

    cl.parse( argc, argv );

    labelMapFileName = labelMapFileNameArg.getValue();
    ctFileName       = ctFileNameArg.getValue();
    window           = windowArg.getValue();
    level            = levelArg.getValue();
    axial            = axialArg.getValue();
    sagittal         = sagittalArg.getValue();
    coronal          = coronalArg.getValue();
    opacity          = opacityArg.getValue();
    cipRegion        = (unsigned char)(cipRegionArg.getValue());
    cipType          = (unsigned char)(cipTypeArg.getValue());

    for (unsigned int i=0; i<overlayFileNameVecArg.getValue().size(); i++)
      {
      overlayFileNameVec.push_back(overlayFileNameVecArg.getValue()[i]);
      }
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  // Determine which slice plane in which the user wants the overlays
  std::string slicePlane;
  if (axial)
    {
    slicePlane = "axial";
    }
  else if (coronal)
    {
    slicePlane = "coronal";
    }
  else
    {
    slicePlane = "sagittal";
    }

  // Read the label map
  std::cout << "Reading label map image..." << std::endl;
  cip::LabelMapReaderType::Pointer labelMapReader = cip::LabelMapReaderType::New();
    labelMapReader->SetFileName( labelMapFileName );
  try
    {
    labelMapReader->Update();
    }
  catch (itk::ExceptionObject &excp)
    {
    std::cerr << "Exception caught while reading label map:";
    std::cerr << excp << std::endl;

    return cip::LABELMAPREADFAILURE;
    }

  // Read the CT image
  std::cout << "Reading CT image..." << std::endl;
  cip::CTReaderType::Pointer ctReader = cip::CTReaderType::New();
    ctReader->SetFileName( ctFileName );
  try
    {
    ctReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading CT image:";
    std::cerr << excp << std::endl;

    return cip::NRRDREADFAILURE;
    }

  // Now get the overlay images
  std::vector<OverlayType::Pointer> overlays;

  std::cout << "Getting overlay images..." << std::endl;
  GetOverlayImages(labelMapReader->GetOutput(), ctReader->GetOutput(), overlayFileNameVec.size(),
		   &overlays, opacity, slicePlane, window, level, cipRegion, cipType);

  // Finally, write the overlays to file
  for (unsigned int i=0; i<overlayFileNameVec.size(); i++)
    {
    std::cout << "Writing overlay..." << std::endl;
    OverlayWriterType::Pointer writer = OverlayWriterType::New();
      writer->SetFileName(overlayFileNameVec[i]);
      writer->SetInput(overlays[i]);
      writer->UseCompressionOn();
    try
      {
      writer->Update();
      }
    catch (itk::ExceptionObject &excp)
      {
      std::cerr << "Exception caught writing overlay image:";
      std::cerr << excp << std::endl;

      return cip::EXITFAILURE;
      }
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

void GetOverlayImages(cip::LabelMapType::Pointer labelMap, cip::CTType::Pointer ctImage, unsigned int numImages,
		      std::vector<OverlayType::Pointer>* overlayVec, double opacity, std::string slicePlane, 
		      short window, short level, unsigned char cipRegion, unsigned char cipType)
{
  ChestConventions conventions;

  cip::LabelMapType::RegionType boundingBox;
  if (cipRegion == (unsigned char)(cip::UNDEFINEDREGION) && cipType == (unsigned char)(cip::UNDEFINEDTYPE))
    {
    boundingBox = cip::GetLabelMapChestRegionChestTypeBoundingBoxRegion(labelMap);
    }
  else
    {
    boundingBox = cip::GetLabelMapChestRegionChestTypeBoundingBoxRegion(labelMap, cipRegion, cipType);
    }

  cip::LabelMapType::SizeType size = labelMap->GetBufferedRegion().GetSize();
  cip::LabelMapType::SpacingType spacing = labelMap->GetSpacing();

  OverlayType::SizeType overlaySize;
  OverlayType::SpacingType overlaySpacing;

  unsigned int iIndex;
  unsigned int jIndex;
  unsigned int kIndex;
  unsigned int sliceMin;
  unsigned int sliceMax;

  if (slicePlane.compare("axial") == 0)
    {
    overlaySize[0] = size[0];
    overlaySize[1] = size[1];
      
    overlaySpacing[0] = spacing[0];
    overlaySpacing[1] = spacing[1];

    sliceMin = boundingBox.GetIndex()[2];
    sliceMax = boundingBox.GetIndex()[2] + boundingBox.GetSize()[2] - 1;

    iIndex = 0;
    jIndex = 1;
    kIndex = 2;
    }
  else if (slicePlane.compare("coronal") == 0)
    {      
    overlaySize[0] = size[0];
    overlaySize[1] = size[2];
      
    overlaySpacing[0] = spacing[0];
    overlaySpacing[1] = spacing[2];

    sliceMin = boundingBox.GetIndex()[1];
    sliceMax = boundingBox.GetIndex()[1] + boundingBox.GetSize()[1] - 1;

    iIndex = 0;
    jIndex = 2;
    kIndex = 1;
    }
  else if (slicePlane.compare("sagittal") == 0)
    {      
    overlaySize[0] = size[1];
    overlaySize[1] = size[2];
      
    overlaySpacing[0] = spacing[1];
    overlaySpacing[1] = spacing[2];

    sliceMin = boundingBox.GetIndex()[0];
    sliceMax = boundingBox.GetIndex()[0] + boundingBox.GetSize()[0] - 1;

    iIndex = 1;
    jIndex = 2;
    kIndex = 0;
    }

  cip::LabelMapType::IndexType index;
  OverlayType::IndexType  overlayIndex;
  
  RGBPixelType overlayValue;
    
  double         windowLeveledValue;
  unsigned short labelValue;

  for (unsigned int n=1; n<=numImages; n++ )
    {
    RGBPixelType rgbDefault;
      rgbDefault[0] = 0;
      rgbDefault[1] = 0;
      rgbDefault[2] = 0;

    OverlayType::Pointer overlay = OverlayType::New();
      overlay->SetSpacing(overlaySpacing);
      overlay->SetRegions(overlaySize);
      overlay->Allocate();
      overlay->FillBuffer(rgbDefault);

    unsigned int slice = sliceMin + n*(sliceMax - sliceMin)/(numImages + 1);
    index[kIndex] = slice;

    for ( unsigned int i=0; i<overlaySize[0]; i++ )
      {
      index[iIndex] = i;
      overlayIndex[0] = i;

      for ( unsigned int j=0; j<overlaySize[1]; j++ )
        {
        index[jIndex] = j;

	// We assume by default that the scan is head-first and supine. This requires
	// us to flip the coronal and sagittal images so they are upright.
	if (slicePlane.compare("coronal") == 0 || slicePlane.compare("sagittal") == 0)
	  {
	  overlayIndex[1] = size[2] - 1 - j;
	  }
	else
	  {
	  overlayIndex[1] = j;
	  }

        windowLeveledValue = GetWindowLeveledValue(ctImage->GetPixel(index), window, level);
        labelValue = labelMap->GetPixel(index);
        
        if (opacity == 0.0)
          {
          overlayValue[0] = windowLeveledValue;
          overlayValue[1] = windowLeveledValue;
          overlayValue[2] = windowLeveledValue;
          }
        else
          {
          overlayValue = GetOverlayPixelValue(windowLeveledValue, labelValue, opacity);
          }        

        overlay->SetPixel(overlayIndex, overlayValue);
        }
      }

    overlayVec->push_back( overlay );
    }
}

double GetWindowLeveledValue(short ctValue, short window, short level)
{
  double minHU = double(level) - 0.5*double(window);
  double maxHU = double(level) + 0.5*double(window);

  double slope     = 255.0/window;
  double intercept = -slope*minHU;

  double windowLeveledValue = double(ctValue)*slope + intercept;

  if (windowLeveledValue < 0)
    {
    windowLeveledValue = 0.0;
    }
  if (windowLeveledValue > 255)
    {
    windowLeveledValue = 255.0;
    }

  return windowLeveledValue;
}

//
// Assumes the labelValue is the full label map value (i.e. not an
// extracted region or type). The region is extracted from this value
// from within the function
//
RGBPixelType GetOverlayPixelValue(double windowLeveledValue, unsigned short labelValue, double opacity)
{
  ChestConventions conventions;

  unsigned char cipRegion = conventions.GetChestRegionFromValue(labelValue);
  unsigned char cipType = conventions.GetChestTypeFromValue(labelValue);

  double* color = new double[3];
  conventions.GetColorFromChestRegionChestType(cipRegion, cipType, color);

  if (cipRegion == (unsigned char)(UNDEFINEDREGION) && cipType == (unsigned char)(UNDEFINEDTYPE))
    {
    opacity = 0.0;
    }

  RGBPixelType rgb;
    rgb[0] = (unsigned char)((1.0 - opacity)*windowLeveledValue + opacity*255.0*color[0]);
    rgb[1] = (unsigned char)((1.0 - opacity)*windowLeveledValue + opacity*255.0*color[1]);
    rgb[2] = (unsigned char)((1.0 - opacity)*windowLeveledValue + opacity*255.0*color[2]);
  
  return rgb;
}

#endif
