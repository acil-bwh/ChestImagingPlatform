/** \file
 *  \ingroup commandLineTools 
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <string>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "cipHelper.h"
#include "cipChestConventions.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkRGBPixel.h"
#include "GenerateOverlayImagesCLP.h"

typedef itk::RGBPixel< unsigned char >                 RGBPixelType;
typedef itk::Image< RGBPixelType, 2 >                  OverlayType;
typedef itk::ImageFileWriter< OverlayType >            OverlayWriterType;
typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType >  IteratorType;

double GetWindowLeveledValue(short, short, short);
RGBPixelType GetOverlayPixelValue(double, unsigned short, double);

/** Gets the overlay images in the label map. If 'allImages' is set
 *  to true, then every slice with a foreground region in it will be
 *  used to produce an overlay. It thus trumps whatever is specified
 *  for the 'numImages' parameter */
void GetOverlayImages(cip::LabelMapType::Pointer labelMap, cip::CTType::Pointer ctImage, unsigned int numImages,
		      std::vector<OverlayType::Pointer>* overlayVec, double opacity, std::string slicePlane, 
		      short window, short level, unsigned char cipRegion, unsigned char cipType, bool bookEnds,
		      bool allImages);
bool GetSliceHasForeground(cip::LabelMapType::Pointer, unsigned int, std::string );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  cip::ChestConventions conventions;

  unsigned char cipRegion = conventions.GetChestRegionValueFromName( cipRegionName );
  unsigned char cipType   = conventions.GetChestTypeValueFromName( cipTypeName );

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
		   &overlays, opacity, slicePlane, window, level, cipRegion, cipType, bookEnds, 
		   allImages);
  
  // Finally, write the overlays to file
  for (unsigned int i=0; i<overlays.size(); i++)
    {
      char buff[4];
      std::sprintf(buff, "%04u", i);
      std::string whichOverlay( buff );
      
      std::cout << "Writing overlay..." << std::endl;
      OverlayWriterType::Pointer writer = OverlayWriterType::New();
      if ( allImages )
	{
	writer->SetFileName(prefix + whichOverlay + ".png");
	}
      else
	{
        writer->SetFileName(overlayFileNameVec[i]);	  
	}
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
		      short window, short level, unsigned char cipRegion, unsigned char cipType, bool bookEnds,
		      bool allImages)
{
  cip::ChestConventions conventions;

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

  if ( allImages )
    {
      numImages = sliceMax - sliceMin + 1;
    }

  cip::LabelMapType::IndexType index;
  OverlayType::IndexType  overlayIndex;
  
  RGBPixelType overlayValue;
    
  double         windowLeveledValue;
  unsigned short labelValue;

  for ( unsigned int n=1; n<=numImages; n++ )
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

    unsigned int slice;
    if ( allImages )
      {
	slice = sliceMin + n - 1;
      }
    else if ( bookEnds )
      {
	slice = sliceMin + (n - 1)*(sliceMax - sliceMin)/numImages;
      }
    else
      {
	slice = sliceMin + n*(sliceMax - sliceMin)/(numImages + 1);
      }

    if ( !allImages || (allImages && GetSliceHasForeground(labelMap, slice, slicePlane)) )
      {      
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

bool GetSliceHasForeground(cip::LabelMapType::Pointer labelMap, unsigned int whichSlice, std::string slicePlane )
{
  cip::LabelMapType::SizeType size = labelMap->GetBufferedRegion().GetSize();

  cip::LabelMapType::RegionType region;
  cip::LabelMapType::IndexType  start;
  cip::LabelMapType::SizeType   regionSize;

  if (slicePlane.compare("axial") == 0)
    {
      start[0] = 0;
      start[1] = 0;
      start[2] = whichSlice;
      
      regionSize[0] = size[0];
      regionSize[1] = size[1];
      regionSize[2] = 1;
    }
  else if (slicePlane.compare("coronal") == 0)
    {      
      start[0] = 0;
      start[1] = whichSlice;
      start[2] = 0;
      
      regionSize[0] = size[0];
      regionSize[1] = 1;
      regionSize[2] = size[2];
    }
  else if (slicePlane.compare("sagittal") == 0)
    {      
      start[0] = whichSlice;
      start[1] = 0;
      start[2] = 0;
      
      regionSize[0] = 1;
      regionSize[1] = size[1];
      regionSize[2] = size[2];
    }

  region.SetSize( regionSize );
  region.SetIndex( start );

  IteratorType it( labelMap, region );

  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
      if ( it.Get() > 0 )
	{
	  return true;
	}

      ++it;
    }

  return false;
}

//
// Assumes the labelValue is the full label map value (i.e. not an
// extracted region or type). The region is extracted from this value
// from within the function
//
RGBPixelType GetOverlayPixelValue(double windowLeveledValue, unsigned short labelValue, double opacity)
{
  cip::ChestConventions conventions;

  unsigned char cipRegion = conventions.GetChestRegionFromValue(labelValue);
  unsigned char cipType = conventions.GetChestTypeFromValue(labelValue);

  double* color = new double[3];
  conventions.GetColorFromChestRegionChestType(cipRegion, cipType, color);

  if (cipRegion == (unsigned char)(cip::UNDEFINEDREGION) && cipType == (unsigned char)(cip::UNDEFINEDTYPE))
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
