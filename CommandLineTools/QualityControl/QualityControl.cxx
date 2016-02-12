/** \file
 *  \ingroup commandLineTools 
 *  \details This program can be used to produce quality control
 *  projection (2D) images for multiple forms of input label map
 *  images. Currently supported use cases include label map images
 *  designating lung labelings by thirds, whole lung labelings, airway
 *  labelings, and lung lobe labelings.
 */

//#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <fstream>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkRGBPixel.h"
#include "QualityControlCLP.h"

namespace
{
  typedef itk::Image< unsigned char, 2 >                         ProjectionImageType;
  typedef itk::ImageFileWriter< ProjectionImageType >            ProjectionWriterType;
  typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType > LabelMapIteratorType;
  typedef itk::ImageRegionIteratorWithIndex< cip::CTType >       CTIteratorType;
  typedef itk::RGBPixel< unsigned char >                         RGBPixelType;
  typedef itk::Image< RGBPixelType, 2 >                          OverlayType;
  typedef itk::ImageFileWriter< OverlayType >                    OverlayWriterType;

  /*
    double GetWindowLeveledValue( short );
    RGBPixelType GetOverlayPixelValue( double, unsigned short, double );
    void GenerateLungLobeOverlayImages( cip::LabelMapType::Pointer, cip::CTType::Pointer, unsigned int, std::vector< OverlayType::Pointer >*,
    std::vector< unsigned int >, double );
    void GetLungProjectionImage( cip::LabelMapType::Pointer, ProjectionImageType::Pointer );
    void GetAirwayProjectionImage( cip::LabelMapType::Pointer, ProjectionImageType::Pointer );
  */
        
  double GetWindowLeveledValue( short ctValue )
  {
    double slope     = 255.0/1024.0;
    double intercept = 255.0;
        
    double windowLeveledValue;
        
    if ( ctValue < 0 )
      {
	windowLeveledValue = slope*static_cast< double >( ctValue ) + intercept;
            
	if ( windowLeveledValue < 0.0 )
	  {
	    windowLeveledValue = 0.0;
	  }
      }
    else
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
  RGBPixelType GetOverlayPixelValue( double windowLeveledValue, unsigned short labelValue, double opacity )
  {
    cip::ChestConventions conventions;
        
    unsigned char lungRegion = conventions.GetChestRegionFromValue( labelValue );
        
    unsigned char redChannel, greenChannel, blueChannel;
        
    if ( lungRegion == static_cast< unsigned char >( cip::LEFTSUPERIORLOBE ) )
      {
	redChannel   = 255;
	greenChannel = 0;
	blueChannel  = 0;
      }
    else if ( lungRegion == static_cast< unsigned char >( cip::LEFTINFERIORLOBE ) )
      {
	redChannel   = 0;
	greenChannel = 255;
	blueChannel  = 0;
      }
    else if ( lungRegion == static_cast< unsigned char >( cip::RIGHTSUPERIORLOBE ) )
      {
	redChannel   = 0;
	greenChannel = 255;
	blueChannel  = 255;
      }
    else if ( lungRegion == static_cast< unsigned char >( cip::RIGHTMIDDLELOBE ) )
      {
	redChannel   = 255;
	greenChannel = 0;
	blueChannel  = 255;
      }
    else if ( lungRegion == static_cast< unsigned char >( cip::RIGHTINFERIORLOBE ) )
      {
	redChannel   = 0;
	greenChannel = 0;
	blueChannel  = 255;
      }
    else
      {
	redChannel   = 0;
	greenChannel = 0;
	blueChannel  = 0;
            
	opacity = 0.0;
      }
        
    RGBPixelType rgb;
    rgb[0] = static_cast< unsigned char >( (1.0 - opacity)*windowLeveledValue + opacity*redChannel );
    rgb[1] = static_cast< unsigned char >( (1.0 - opacity)*windowLeveledValue + opacity*greenChannel );
    rgb[2] = static_cast< unsigned char >( (1.0 - opacity)*windowLeveledValue + opacity*blueChannel );
        
    return rgb;
  }
    
    void GenerateLungLobeOverlayImages( cip::LabelMapType::Pointer labelMap, cip::CTType::Pointer ctImage, unsigned int numImages,
				      std::vector< OverlayType::Pointer >* overlayVec, std::vector< unsigned int > lungRegions,
				      double opacity )
  {
    cip::LabelMapType::SizeType size = labelMap->GetBufferedRegion().GetSize();
        
    unsigned int xMin   = size[0];
    unsigned int xMax   = 0;
        
    cip::ChestConventions conventions;
    bool checkMinMax;
        
    //
    // Get the left lung and right lung bounding regions in the
    // x-direction
    //
    LabelMapIteratorType lIt( labelMap, labelMap->GetBufferedRegion() );
        
    lIt.GoToBegin();
    while ( !lIt.IsAtEnd() )
      {
	if ( lIt.Get() != 0 )
	  {
	    unsigned char lungRegion = conventions.GetChestRegionFromValue( lIt.Get() );
                
	    for ( unsigned int i=0; i<lungRegions.size(); i++ )
	      {
		checkMinMax = false;
                    
		if ( lungRegion == static_cast< unsigned char >( lungRegions[i] ) )
		  {
		    checkMinMax = true;
		    break;
		  }
	      }
	    if ( checkMinMax )
	      {
		if ( lIt.GetIndex()[0] < xMin )
		  {
		    xMin = lIt.GetIndex()[0];
		  }
		if ( lIt.GetIndex()[0] > xMax )
		  {
		    xMax = lIt.GetIndex()[0];
		  }
	      }
	  }
            
	++lIt;
      }
        
    cip::LabelMapType::IndexType index;
    OverlayType::IndexType  overlayIndex;
        
    RGBPixelType overlayValue;
        
    double         windowLeveledValue;
    unsigned short labelValue;
        
    for ( unsigned int i=1; i<=numImages; i++ )
      {
	OverlayType::Pointer overlay = OverlayType::New();
            
	OverlayType::SizeType overlaySize;
	overlaySize[0] = size[1];
	overlaySize[1] = size[2];
            
	RGBPixelType rgbDefault;
	rgbDefault[0] = 0;
	rgbDefault[1] = 0;
	rgbDefault[2] = 0;
            
	overlay->SetRegions( overlaySize );
	overlay->Allocate();
	overlay->FillBuffer( rgbDefault );
            
	unsigned int xValue  = xMin + i*(xMax - xMin)/(numImages+1);
            
	for ( unsigned int y=0; y<size[1]; y++ )
	  {
	    index[1] = y;
	    overlayIndex[0] = y;
                
	    for ( unsigned int z=0; z<size[2]; z++ )
	      {
		index[2] = z;
		overlayIndex[1] = size[2] - z - 1;
                    
		//
		// First get and assign the left value
		//
		index[0] = xValue;
                    
		windowLeveledValue = GetWindowLeveledValue( ctImage->GetPixel( index ) );
		labelValue = labelMap->GetPixel( index );
                    
		if ( opacity == 0.0 )
		  {
		    overlayValue[0] = windowLeveledValue;
		    overlayValue[1] = windowLeveledValue;
		    overlayValue[2] = windowLeveledValue;
		  }
		else
		  {
		    overlayValue = GetOverlayPixelValue( windowLeveledValue, labelValue, opacity );
		  }
                    
		overlay->SetPixel( overlayIndex, overlayValue );
	      }
	  }
            
	overlayVec->push_back( overlay );
      }
  }
  

    
    
  void GetAirwayProjectionImage( cip::LabelMapType::Pointer labelMap, ProjectionImageType::Pointer projectionImage )
  {
    ProjectionImageType::IndexType projectionIndex;
        
    cip::LabelMapType::SizeType labelMapSize = labelMap->GetBufferedRegion().GetSize();
        
    ProjectionImageType::SizeType projectionSize;
    projectionSize[0] = labelMapSize[0];
    projectionSize[1] = labelMapSize[2];
        
    cip::ChestConventions conventions;
        
    unsigned char region, type;
        
    LabelMapIteratorType it( labelMap, labelMap->GetBufferedRegion() );
        
    it.GoToBegin();
    while ( !it.IsAtEnd() )
      {
	if ( it.Get() > 511 )
	  {
	    projectionIndex[0] = projectionSize[0] - it.GetIndex()[0] - 1;
	    projectionIndex[1] = projectionSize[1] - it.GetIndex()[2] - 1;
                
	    if ( projectionImage->GetPixel( projectionIndex ) == 0 )
	      {
		type = conventions.GetChestTypeFromValue( it.Get() );
                    
		if ( type == cip::AIRWAY )
		  {
		    region = conventions.GetChestRegionFromValue( it.Get() );
                        
		    if ( region == cip::UNDEFINEDREGION )
		      {
			projectionImage->SetPixel( projectionIndex, 255 );
		      }
                        
		  }
	      }
	  }
            
	++it;
      }
  }
    
    
  void GetLungProjectionImage( cip::LabelMapType::Pointer labelMap, ProjectionImageType::Pointer projectionImage  )
  {
    ProjectionImageType::IndexType projectionIndex;
        
    cip::ChestConventions conventions;
        
    cip::LabelMapType::SizeType labelMapSize = labelMap->GetBufferedRegion().GetSize();
        
    ProjectionImageType::SizeType projectionSize;
    projectionSize[0] = labelMapSize[0];
    projectionSize[1] = labelMapSize[2];
        
    unsigned char region;
        
    LabelMapIteratorType it( labelMap, labelMap->GetBufferedRegion() );
        
    it.GoToBegin();
    while ( !it.IsAtEnd() )
      {
	if ( it.Get() > 0 )
	  {
	    projectionIndex[0] = projectionSize[0] - it.GetIndex()[0] - 1;
	    projectionIndex[1] = projectionSize[1] - it.GetIndex()[2] - 1;
                
	    if ( projectionImage->GetPixel( projectionIndex ) == 0 )
	      {
		region = conventions.GetChestRegionFromValue( it.Get() );
                    
		if ( region == cip::LEFTUPPERTHIRD || region == cip::WHOLELUNG || 
		     region == cip::LEFTLUNG )
		  { 
		    projectionImage->SetPixel( projectionIndex, 1*36 );
		  }
		else if ( region == cip::LEFTMIDDLETHIRD )
		  { 
		    projectionImage->SetPixel( projectionIndex, 2*36 );
		  }
		else if ( region == cip::LEFTLOWERTHIRD )
		  { 
		    projectionImage->SetPixel( projectionIndex, 3*36 );
		  }
		else if ( region == cip::RIGHTUPPERTHIRD )
		  { 
		    projectionImage->SetPixel( projectionIndex, 4*36 );
		  }
		else if ( region == cip::RIGHTMIDDLETHIRD )
		  { 
		    projectionImage->SetPixel( projectionIndex, 5*36 );
		  }
		else if ( region == cip::RIGHTLOWERTHIRD || region == cip::RIGHTLUNG )
		  { 
		    projectionImage->SetPixel( projectionIndex, 6*36 );
		  }
		else if ( region == cip::LOWERTHIRD )
		  {
		    projectionImage->SetPixel( projectionIndex, 1*64 );
		  }
		else if ( region == cip::MIDDLETHIRD )
		  {
		    projectionImage->SetPixel( projectionIndex, 2*64 );
		  }
		else if ( region == cip::UPPERTHIRD )
		  {
		    projectionImage->SetPixel( projectionIndex, 3*64 );
		  }
	      }
	  }
            
	++it;
      }
  }
    
} //end namespace




int main( int argc, char *argv[] )
{
  // Begin by defining the arguments to be passed
  //std::string labelMapFileName              = "NA";
  //std::string ctFileName                    = "NA";
  //std::string lungProjectionImageFileName   = "NA";
  //std::string airwayProjectionImageFileName = "NA";


  PARSE_ARGS; 

  std::vector< std::string > leftLungLobeFileNameVec;   
  std::vector< std::string > rightLungLobeFileNameVec;
  std::vector< std::string > leftLungCTFileNameVec;
  std::vector< std::string > rightLungCTFileNameVec; 

  for ( unsigned int i=0; i<leftLungLobeFileNameVecArg.size(); i++ )
    {
      leftLungLobeFileNameVec.push_back( leftLungLobeFileNameVecArg[i] );
    }

  for ( unsigned int i=0; i<rightLungLobeFileNameVecArg.size(); i++ )
    {
      rightLungLobeFileNameVec.push_back( rightLungLobeFileNameVecArg[i] );
    }

  for ( unsigned int i=0; i<rightLungCTFileNameVecArg.size(); i++ )
    {
      rightLungCTFileNameVec.push_back( rightLungCTFileNameVecArg[i] );
    }

  for ( unsigned int i=0; i<leftLungCTFileNameVecArg.size(); i++ )
    {
      leftLungCTFileNameVec.push_back( leftLungCTFileNameVecArg[i] );
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

  //
  // Get sizing and spacing info for creation of projection image 
  //
  cip::LabelMapType::SizeType    labelMapSize    = labelMapReader->GetOutput()->GetBufferedRegion().GetSize();
  cip::LabelMapType::SpacingType labelMapSpacing = labelMapReader->GetOutput()->GetSpacing(); 

  ProjectionImageType::SizeType projectionSize;
  projectionSize[0] = labelMapSize[0];
  projectionSize[1] = labelMapSize[2];

  ProjectionImageType::SpacingType projectionSpacing;
  projectionSpacing[0] = labelMapSpacing[0];
  projectionSpacing[1] = labelMapSpacing[2];

  //  
  // Create the lung projection image if requested
  //
  ProjectionImageType::Pointer lungProjectionImage   = ProjectionImageType::New();
  ProjectionImageType::Pointer airwayProjectionImage = ProjectionImageType::New();

  if ( lungProjectionImageFileName.compare( "NA" ) != 0 )
    {
      lungProjectionImage->SetRegions( projectionSize );
      lungProjectionImage->Allocate();
      lungProjectionImage->FillBuffer( 0 );
      lungProjectionImage->SetSpacing( projectionSpacing );

      std::cout << "Getting lung projection image..." << std::endl;
      GetLungProjectionImage( labelMapReader->GetOutput(), lungProjectionImage );
    }

  //
  // Create the airway projection image if requested
  //
  if ( airwayProjectionImageFileName.compare( "NA" ) != 0 )
    {
      airwayProjectionImage->SetRegions( projectionSize );
      airwayProjectionImage->Allocate();
      airwayProjectionImage->FillBuffer( 0 );
      airwayProjectionImage->SetSpacing( projectionSpacing );

      std::cout << "Getting airway projection image..." << std::endl;
      GetAirwayProjectionImage( labelMapReader->GetOutput(), airwayProjectionImage );
    }

  cip::CTType::Pointer ctImage = cip::CTType::New();

  //
  // Read the CT image if requested
  //
  if ( ctFileName.compare( "NA" ) != 0 )
    {
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

      ctImage = ctReader->GetOutput();
    }

  //
  // Generate the left lung lobe overlays and CT comparison images if
  // requested 
  //
  std::vector< OverlayType::Pointer > leftOverlayVec;
  std::vector< OverlayType::Pointer > leftCTVec;

  std::vector< unsigned int > leftLungRegions;
  leftLungRegions.push_back( cip::LEFTSUPERIORLOBE );
  leftLungRegions.push_back( cip::LEFTINFERIORLOBE );

  if ( leftLungLobeFileNameVec.size() > 0 && ctFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Generating left lung overlay images..." << std::endl;
      GenerateLungLobeOverlayImages( labelMapReader->GetOutput(), ctImage, leftLungLobeFileNameVec.size(), &leftOverlayVec, 
				     leftLungRegions, 0.3 );
    }

  if ( leftLungCTFileNameVec.size() > 0 && ctFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Generating left lung CT images..." << std::endl;
      GenerateLungLobeOverlayImages( labelMapReader->GetOutput(), ctImage, leftLungCTFileNameVec.size(), &leftCTVec, 
				     leftLungRegions, 0.0 );
    }

  //
  // Generate the right lung lobe overlays and CT comparison images if
  // requested 
  //
  std::vector< OverlayType::Pointer > rightOverlayVec;
  std::vector< OverlayType::Pointer > rightCTVec;

  std::vector< unsigned int > rightLungRegions;
  rightLungRegions.push_back( cip::RIGHTSUPERIORLOBE );
  rightLungRegions.push_back( cip::RIGHTMIDDLELOBE );
  rightLungRegions.push_back( cip::RIGHTINFERIORLOBE );

  if ( rightLungLobeFileNameVec.size() > 0 && ctFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Generating right lung overlay images..." << std::endl;
      GenerateLungLobeOverlayImages( labelMapReader->GetOutput(), ctImage, rightLungLobeFileNameVec.size(), &rightOverlayVec,
				     rightLungRegions, 0.3 );
    }

  if ( rightLungCTFileNameVec.size() > 0 && ctFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Generating right lung CT images..." << std::endl;
      GenerateLungLobeOverlayImages( labelMapReader->GetOutput(), ctImage, rightLungCTFileNameVec.size(), &rightCTVec,
				     rightLungRegions, 0.0 );
    }

  //
  // Write the lung projection image if requested
  //
  if ( lungProjectionImageFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Writing lung projection image..." << std::endl;
      ProjectionWriterType::Pointer lungProjectionWriter = ProjectionWriterType::New();
      lungProjectionWriter->SetFileName( lungProjectionImageFileName );
      lungProjectionWriter->SetInput( lungProjectionImage );
      try
	{
	  lungProjectionWriter->Update();
	}
      catch ( itk::ExceptionObject &excp )
	{
	  std::cerr << "Exception caught writing lung projection image:";
	  std::cerr << excp << std::endl;

	  return cip::QUALITYCONTROLIMAGEWRITEFAILURE;
	}
    }

  //
  // Write the airway projection image if requested
  //
  if ( airwayProjectionImageFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Writing airway projection image..." << std::endl;
      ProjectionWriterType::Pointer airwayProjectionWriter = ProjectionWriterType::New();
      airwayProjectionWriter->SetFileName( airwayProjectionImageFileName );
      airwayProjectionWriter->SetInput( airwayProjectionImage );
      try
	{
	  airwayProjectionWriter->Update();
	}
      catch ( itk::ExceptionObject &excp )
	{
	  std::cerr << "Exception caught writing airway projection image:";
	  std::cerr << excp << std::endl;

	  return cip::QUALITYCONTROLIMAGEWRITEFAILURE;
	}
    }
  
  //
  // Write the left lung lobe overlay images if requested 
  //
  if ( leftLungLobeFileNameVec.size() > 0 )
    {
      for ( unsigned int i=0; i<leftLungLobeFileNameVec.size(); i++ )
	{
	  std::cout << "Writing left overlay image..." << std::endl;
	  OverlayWriterType::Pointer leftOverlayWriter = OverlayWriterType::New();
	  leftOverlayWriter->SetInput( leftOverlayVec[i] );
	  leftOverlayWriter->SetFileName( leftLungLobeFileNameVec[i] );
	  try
	    {
	      leftOverlayWriter->Update();
	    }
	  catch ( itk::ExceptionObject &excp )
	    {
	      std::cerr << "Exception caught writing left overlay image:";
	      std::cerr << excp << std::endl;

	      return cip::QUALITYCONTROLIMAGEWRITEFAILURE;
	    }
	}
    }

  //
  // Write the left lung CT images if requested 
  //
  if ( leftLungCTFileNameVec.size() > 0 )
    {
      for ( unsigned int i=0; i<leftLungCTFileNameVec.size(); i++ )
	{
	  std::cout << "Writing left CT image..." << std::endl;
	  OverlayWriterType::Pointer leftOverlayWriter = OverlayWriterType::New();
	  leftOverlayWriter->SetInput( leftCTVec[i] );
	  leftOverlayWriter->SetFileName( leftLungCTFileNameVec[i] );
	  try
	    {
	      leftOverlayWriter->Update();
	    }
	  catch ( itk::ExceptionObject &excp )
	    {
	      std::cerr << "Exception caught writing left overlay image:";
	      std::cerr << excp << std::endl;

	      return cip::QUALITYCONTROLIMAGEWRITEFAILURE;
	    }
	}
    }

  //
  // Write the right lung lobe overlay images if requested 
  //
  if ( rightLungLobeFileNameVec.size() > 0 )
    {
      for ( unsigned int i=0; i<rightLungLobeFileNameVec.size(); i++ )
	{
	  std::cout << "Writing right overlay image..." << std::endl;
	  OverlayWriterType::Pointer rightOverlayWriter = OverlayWriterType::New();
	  rightOverlayWriter->SetInput( rightOverlayVec[i] );
	  rightOverlayWriter->SetFileName( rightLungLobeFileNameVec[i] );
	  try
	    {
	      rightOverlayWriter->Update();
	    }
	  catch ( itk::ExceptionObject &excp )
	    {
	      std::cerr << "Exception caught writing right overlay image:";
	      std::cerr << excp << std::endl;

	      return cip::QUALITYCONTROLIMAGEWRITEFAILURE;
	    }
	}
    }

  //
  // Write the right lung CT images if requested 
  //
  if ( rightLungCTFileNameVec.size() > 0 )
    {
      for ( unsigned int i=0; i<rightLungCTFileNameVec.size(); i++ )
	{
	  std::cout << "Writing right CT image..." << std::endl;
	  OverlayWriterType::Pointer rightOverlayWriter = OverlayWriterType::New();
	  rightOverlayWriter->SetInput( rightCTVec[i] );
	  rightOverlayWriter->SetFileName( rightLungCTFileNameVec[i] );
	  try
	    {
	      rightOverlayWriter->Update();
	    }
	  catch ( itk::ExceptionObject &excp )
	    {
	      std::cerr << "Exception caught writing right overlay image:";
	      std::cerr << excp << std::endl;

	      return cip::QUALITYCONTROLIMAGEWRITEFAILURE;
	    }
	}
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}


