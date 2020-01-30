#ifndef _itkCIPPartialLungLabelMapImageFilter_txx
#define _itkCIPPartialLungLabelMapImageFilter_txx

#include "itkCIPPartialLungLabelMapImageFilter.h"
#include "cipHelper.h"
#include "cipChestConventions.h"
#include "itkImageFileWriter.h" //DEB

namespace itk
{

template < class TInputImage >
CIPPartialLungLabelMapImageFilter< TInputImage >
::CIPPartialLungLabelMapImageFilter()
{
  this->m_ClosingNeighborhood[0]      = 7;
  this->m_ClosingNeighborhood[1]      = 7;
  this->m_ClosingNeighborhood[2]      = 7;
  this->m_ExponentialCoefficient      = 200;
  this->m_ExponentialTimeConstant     = -700;
  this->m_LeftRightLungSplitRadius    = 1;
  this->m_HeadFirst                   = true;
  this->m_Supine                      = true;

  this->m_AirwayMinIntensityThresholdSet = false;
  this->m_AirwayMaxIntensityThresholdSet = false;

  this->m_AirwayLabelMap = LabelMapType::New();

  // We manually segmented the airway trees from twenty five 
  // inspiratory CT scans acquired from healthy individuals. 
  // 30,000 mm^3 is approximately the smallest of the airway
  // volumes we computed
  this->m_MaxAirwayVolume = 30000;
}


template< class TInputImage >
void
CIPPartialLungLabelMapImageFilter< TInputImage >
::GenerateData()
{  
  if ( !this->m_AirwayMaxIntensityThresholdSet )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, "CIPPartialLungLabelMapImageFilter::GenerateData()", 
				  "Max airway intensity threshold not set" );
    }
  if ( !this->m_AirwayMinIntensityThresholdSet )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, "CIPPartialLungLabelMapImageFilter::GenerateData()", 
				  "Min airway intensity threshold not set" );
    }

  cip::ChestConventions conventions;

  LabelMapType::SpacingType spacing = this->GetInput()->GetSpacing();

  typedef itk::ImageFileWriter< LabelMapType > WriterType;

  // Allocate space for the output image
  typename Superclass::InputImageConstPointer inputPtr  = this->GetInput();
  typename Superclass::OutputImagePointer     outputPtr = this->GetOutput(0);
    outputPtr->SetRequestedRegion( inputPtr->GetRequestedRegion() );
    outputPtr->SetBufferedRegion( inputPtr->GetBufferedRegion() );
    outputPtr->SetLargestPossibleRegion( inputPtr->GetLargestPossibleRegion() );
    outputPtr->Allocate();
    outputPtr->FillBuffer( 0 );

  LabelMapIteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() ); 

  if ( this->m_HelperMask.IsNull() )
    {
      typename OtsuCastType::Pointer otsuCast = OtsuCastType::New();
        otsuCast->SetInput( this->GetInput() );
	otsuCast->Update();

      UCharIteratorType otIt( otsuCast->GetOutput(), otsuCast->GetOutput()->GetBufferedRegion() );
      LabelMapIteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() ); 
      
      otIt.GoToBegin();
      oIt.GoToBegin();
      while ( !oIt.IsAtEnd() )
	{
	  oIt.Set( (unsigned short)(otIt.Get()) );

	  ++otIt;
	  ++oIt;
	}
    }
  else
    {
    // Apply the helper mask
    this->ApplyHelperMask();
    }

  std::vector< OutputImageType::IndexType > airwayIndices;
  {
    // Identify airways
    std::vector< OutputImageType::IndexType > airwaySeedVec = this->GetAirwaySeeds( this->GetOutput() );

    // Now segment the airway tree
    typename AirwaySegmentationType::Pointer airwaySegmenter = AirwaySegmentationType::New();
      airwaySegmenter->SetInput( this->GetInput() );
      airwaySegmenter->SetMaxAirwayVolume( this->m_MaxAirwayVolume );
      airwaySegmenter->SetMinIntensityThreshold( this->m_AirwayMinIntensityThreshold );
      airwaySegmenter->SetMaxIntensityThreshold( this->m_AirwayMaxIntensityThreshold );
    for ( unsigned int i=0; i<airwaySeedVec.size(); i++ )
      {
      airwaySegmenter->AddSeed( airwaySeedVec[i] );
      }
      airwaySegmenter->Update();

    // Dilate the segmented airway tree. We need to do this because the airway
    // segmentation just performed gets the lumen, but not the walls. We want
    // to remove both the lumen and the walls from the Otsu cast / helper before
    // attempting to split the left and right lungs.
    UCharElementType structuringElement;
      structuringElement.SetRadius( 2 );
      structuringElement.CreateStructuringElement();
    
    typename UCharDilateType::Pointer dilater = UCharDilateType::New();
      dilater->SetInput( airwaySegmenter->GetOutput() );
      dilater->SetKernel( structuringElement );
      dilater->SetDilateValue( (unsigned char)(cip::AIRWAY) );
      dilater->Update();

    // Remove the airways from the output label map
    UCharIteratorType dIt( dilater->GetOutput(), dilater->GetOutput()->GetBufferedRegion() );

    oIt.GoToBegin();
    dIt.GoToBegin();
    while ( !dIt.IsAtEnd() )
      {
	if ( dIt.Get() != 0 )
	  {
	    airwayIndices.push_back( oIt.GetIndex() );
	    oIt.Set( 0 );
	  }
	
	++oIt;
	++dIt;
      }
  }

  {
    // It's possible that there are some small disconnected regions that remain
    // after the airways have been removed. Perform connected components analysis
    // and remove all components that collectively make up less than ten percent
    // of the label map region.
    ConnectedComponent3DType::Pointer connectedComponent = ConnectedComponent3DType::New();
      connectedComponent->SetInput( this->GetOutput() );

    Relabel3DType::Pointer relabelComponent = Relabel3DType::New();
      relabelComponent->SetInput( connectedComponent->GetOutput() );
      relabelComponent->Update();

    unsigned int totalSize = 0;
    for ( unsigned int i=0; i<relabelComponent->GetNumberOfObjects(); i++ )
      {
	totalSize += relabelComponent->GetSizeOfObjectsInPixels()[i];
      }
    
    unsigned long componentsToRemoveThreshold;
    for ( unsigned int i=0; i<relabelComponent->GetNumberOfObjects(); i++ )
      {
	if ( static_cast< double >( relabelComponent->GetSizeOfObjectsInPixels()[i] )/static_cast< double >( totalSize ) < 0.20 )
	  {
	    componentsToRemoveThreshold = i+1;
	    break;
	  }
      }

    ComponentIteratorType rIt( relabelComponent->GetOutput(), relabelComponent->GetOutput()->GetBufferedRegion() );
    oIt.GoToBegin();
    rIt.GoToBegin();
    while ( !oIt.IsAtEnd() )
      {
	if ( rIt.Get() >= componentsToRemoveThreshold )
	  {
	    oIt.Set( 0 );
	  }
	
	++oIt;
	++rIt;
      }
  }

  {
    // Now split label map so that the left and right lungs can be labeled
    typename SplitterType::Pointer splitter = SplitterType::New();
      splitter->SetInput( this->GetInput() );
      splitter->SetLeftRightLungSplitRadius( this->m_LeftRightLungSplitRadius );
      splitter->SetLungLabelMap( this->GetOutput() );
      splitter->Update();

    LabelMapIteratorType lIt( splitter->GetOutput(), splitter->GetOutput()->GetBufferedRegion() );
    oIt.GoToBegin();
    lIt.GoToBegin();
    while ( !oIt.IsAtEnd() )
      {
	oIt.Set( lIt.Get() );	
	
	++oIt;
	++lIt;
      }
  }

  bool labelingSucess = false;
  {
    LungRegionLabelerType::Pointer leftRightLabeler = LungRegionLabelerType::New();
      leftRightLabeler->SetInput( this->GetOutput() );
      leftRightLabeler->SetLabelLeftAndRightLungs( true );
      leftRightLabeler->SetHeadFirst( this->m_HeadFirst );
      leftRightLabeler->SetSupine( this->m_Supine );
      leftRightLabeler->Update();

    labelingSucess = leftRightLabeler->GetLabelingSuccess();

    UCharIteratorType lIt( leftRightLabeler->GetOutput(), leftRightLabeler->GetOutput()->GetBufferedRegion() );
    
    lIt.GoToBegin();
    oIt.GoToBegin();
    while ( !oIt.IsAtEnd() )
      {
	oIt.Set( (unsigned short)(lIt.Get()) );
	
	++lIt;
	++oIt;
      }
  }

  {
    // Perform morphological closing on the left and right lungs
    if ( labelingSucess )
      {
	this->CloseLabelMap( (unsigned short)( cip::LEFTLUNG ) );
	this->CloseLabelMap( (unsigned short)( cip::RIGHTLUNG ) );
      }
    else
      {
	this->CloseLabelMap( (unsigned short)( cip::WHOLELUNG ) );
      }
  }

  // Now that the closing has been performed, we can label by thirds
  LungRegionLabelerType::Pointer thirdsLabeler = LungRegionLabelerType::New();
    thirdsLabeler->SetInput( this->GetOutput() );
    thirdsLabeler->SetLabelLungThirds( true );
    thirdsLabeler->SetHeadFirst( this->m_HeadFirst );
    thirdsLabeler->SetSupine( this->m_Supine );
    thirdsLabeler->Update();

  // Finally, set the output to be the result of the thirds labeling
  // plus the airways added back in
  UCharIteratorType tIt( thirdsLabeler->GetOutput(), thirdsLabeler->GetOutput()->GetBufferedRegion() );

  tIt.GoToBegin();
  oIt.GoToBegin();
  while ( !oIt.IsAtEnd() )
    {
      oIt.Set( (unsigned short)(tIt.Get()) );

      ++tIt;
      ++oIt;
    }

  unsigned short labelValue;
  for ( unsigned int i=0; i<airwayIndices.size(); i++ )
    {
      labelValue = conventions.GetValueFromChestRegionAndType( (unsigned char)(this->GetOutput()->GetPixel( airwayIndices[i] )), 
							       (unsigned char)(cip::AIRWAY) );
      this->GetOutput()->SetPixel( airwayIndices[i], labelValue );
    }
}


template < class TInputImage >
void
CIPPartialLungLabelMapImageFilter< TInputImage >
::CloseLabelMap( unsigned short closeLabel )
{
  // Perform morphological closing on the mask by dilating and then
  // eroding.  We assume that at this point in the pipeline, the
  // output image only has WHOLELUNG as a foreground value.  (The
  // airways and vessels should be stored in the index vec member
  // variables). 
  LabelMapElementType structuringElement;
    structuringElement.SetRadius( this->m_ClosingNeighborhood );
    structuringElement.CreateStructuringElement();

  typename LabelMapDilateType::Pointer dilater = LabelMapDilateType::New();
    dilater->SetInput( this->GetOutput() );
    dilater->SetKernel( structuringElement );
    dilater->SetDilateValue( closeLabel );
  try
    {
    dilater->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught dilating:";
    std::cerr << excp << std::endl;
    }

  // Occasionally, dilation will extend the mask to the end slices. If
  // this occurs, the erosion step below won't be able to hit these
  // regions. To deal with this, extract the end slices from the
  // dilater and current output image.  Then set the dilater end
  // slices to be zero (provided that the output image is also zero at
  // those locations).
  OutputImageType::IndexType index;
  OutputImageType::SizeType  size = this->GetOutput()->GetBufferedRegion().GetSize();

  for ( unsigned int x=0; x<size[0]; x++ )
    {
    index[0] = x;

    for ( unsigned int y=0; y<size[1]; y++ )
      {
      index[1] = y;
      
      index[2] = 0;
      if ( this->GetOutput()->GetPixel( index ) == 0 )
        {
        dilater->GetOutput()->SetPixel( index, 0 );
        }

      index[2] = size[2]-1;
      if ( this->GetOutput()->GetPixel( index ) == 0 )
        {
        dilater->GetOutput()->SetPixel( index, 0 );
        }
      }
    }
  
  // Now erode
  typename ErodeType::Pointer eroder = ErodeType::New();
    eroder->SetInput( dilater->GetOutput() );
    eroder->SetKernel( structuringElement );
    eroder->SetErodeValue( closeLabel );
  try
    {
    eroder->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught eroding:";
    std::cerr << excp << std::endl;
    }

  UCharIteratorType eIt( eroder->GetOutput(), eroder->GetOutput()->GetBufferedRegion() );
  LabelMapIteratorType mIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

  eIt.GoToBegin();
  mIt.GoToBegin();
  while ( !mIt.IsAtEnd() )
    {
      if ( eIt.Get() != 0 )
	{
	  mIt.Set( (unsigned short)(eIt.Get()) );
	}
      
      ++eIt;
      ++mIt;
    }
}


template < class TInputImage >
void
CIPPartialLungLabelMapImageFilter< TInputImage >
::SetHelperMask( OutputImageType::Pointer helperMask )
{
  this->m_HelperMask = helperMask;
}


template < class TInputImage >
void
CIPPartialLungLabelMapImageFilter< TInputImage >
::ApplyHelperMask()
{
  // Set the output to the helper mask 
  LabelMapIteratorType mIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );
  LabelMapIteratorType hIt( this->m_HelperMask, this->m_HelperMask->GetBufferedRegion() );

  mIt.GoToBegin();
  hIt.GoToBegin();
  while ( !mIt.IsAtEnd() )
    {
    mIt.Set( hIt.Get() );

    ++mIt;
    ++hIt;
    }
}


/**
 * Extract a slice from the input label map image
 */
template < class TInputImage >
void
CIPPartialLungLabelMapImageFilter< TInputImage >
::ExtractLabelMapSlice( LabelMapType::Pointer image, LabelMapSliceType::Pointer sliceImage, int whichSlice )
{
  LabelMapType::SizeType size = image->GetBufferedRegion().GetSize();

  LabelMapSliceType::SizeType sliceSize;
    sliceSize[0] = size[0];
    sliceSize[1] = size[1];

  sliceImage->SetRegions( sliceSize );
  sliceImage->Allocate();

  LabelMapType::SizeType sliceExtractorSize;
    sliceExtractorSize[0] = size[0];
    sliceExtractorSize[1] = size[1];
    sliceExtractorSize[2] = 0;

  LabelMapType::IndexType sliceStartIndex;
    sliceStartIndex[0] = 0;
    sliceStartIndex[1] = 0;
    sliceStartIndex[2] = whichSlice;

  LabelMapType::RegionType sliceExtractorRegion;
    sliceExtractorRegion.SetSize( sliceExtractorSize );
    sliceExtractorRegion.SetIndex( sliceStartIndex );

  SliceExtractorType::Pointer sliceExtractor = SliceExtractorType::New();
    sliceExtractor->SetInput( image );
    sliceExtractor->SetDirectionCollapseToIdentity();
    sliceExtractor->SetExtractionRegion( sliceExtractorRegion );
  try
    {   
    sliceExtractor->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught extracting slice:";
    std::cerr << excp << std::endl;
    }   

  LabelMapSliceIteratorType eIt( sliceExtractor->GetOutput(), sliceExtractor->GetOutput()->GetBufferedRegion() );
  LabelMapSliceIteratorType sIt( sliceImage, sliceImage->GetBufferedRegion() );

  sIt.GoToBegin();
  eIt.GoToBegin();
  while ( !sIt.IsAtEnd() )
    {
    sIt.Set( eIt.Get() );

    ++sIt;
    ++eIt;
    }
}


/**
 * This method gets seeds for subsequent airway segmentation using
 * region growing. For qualified slices, it gets seeds from whatever
 * object is closest to the centerline (line running parallel to the
 * y-direction).
 */
template < class TInputImage >
std::vector< itk::Image< unsigned short, 3 >::IndexType >
CIPPartialLungLabelMapImageFilter< TInputImage >
::GetAirwaySeeds( LabelMapType::Pointer labelMap )
{
  std::vector< OutputImageType::IndexType > seedVec;

  LabelMapType::SizeType    size    = labelMap->GetBufferedRegion().GetSize();
  LabelMapType::SpacingType spacing = labelMap->GetSpacing();
  
  // Get seeds from 15 slices. We don't consider the slice unless the
  // total area is above a threshold
  // (foregroundSliceSizeThresholdForSeedSelection). The value is set
  // to 2000.0 to ensure that the lungs have come into the field of
  // view before we begin our seed search. If we simply attempt to
  // find seeds in the first slices that show a foreground region, we
  // can wind up with a poor airway segmentation due to pinch points
  // near the apex of the scan. This in turn can produce failure
  // modes. The number of slices that we consider (15) is somewhat
  // arbitrary. 
  unsigned int slicesProcessed                              = 0;
  unsigned int currentSliceOffset                           = 0;
  int          numberSlicesForSeedSearch                    = 15;
  double       foregroundSliceSizeThresholdForSeedSelection = 2000.0;

  while ( slicesProcessed < numberSlicesForSeedSearch && currentSliceOffset < size[2] )
    {
    // Extract a 2D slice from the mask
    int whichSlice;
    if ( this->m_HeadFirst )
      {
      whichSlice = size[2] - 1 - currentSliceOffset;
      }
    else
      {
      whichSlice = currentSliceOffset;
      }

    LabelMapSliceType::Pointer slice = LabelMapSliceType::New();    
    this->ExtractLabelMapSlice( labelMap, slice, whichSlice );

    // Perform connected component labeling    
    ConnectedComponent2DType::Pointer connectedComponent = ConnectedComponent2DType::New();
      connectedComponent->SetInput( slice );
      connectedComponent->FullyConnectedOn();
    try
      {
      connectedComponent->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught while updating connected component filter:";
      std::cerr << excp << std::endl;
      }    

    // Relabel the components    
    Relabel2DType::Pointer relabeler = Relabel2DType::New();
      relabeler->SetInput( connectedComponent->GetOutput() );
    try
      {
      relabeler->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught while relabeling:";
      std::cerr << excp << std::endl;
      }

    // If the foreground area of the slice is larger than the
    // threshold, consider the slice for seed selection. The number of
    // objects present in the slice must also be equal to three
    // (trachea, left and right lungs)
    int numVoxels = 0;

    LabelMapSliceIteratorType rIt( relabeler->GetOutput(), relabeler->GetOutput()->GetBufferedRegion() );

    rIt.GoToBegin();
    while ( !rIt.IsAtEnd() )
      {
      if ( rIt.Get() > 0 )
        {
        numVoxels++;
        }

      ++rIt;
      }

    double foregroundArea = static_cast< double >( numVoxels )*spacing[0]*spacing[1];

    if ( foregroundArea > foregroundSliceSizeThresholdForSeedSelection && relabeler->GetNumberOfObjects() >= 3 )
      {
      slicesProcessed++;

      // Identify the object who's centroid (x coordinate) is in the
      // middle. First get the centroids, then find out which one is
      // in the middle 
      std::map< unsigned int, unsigned short > xCentroidMap;

      unsigned short bestCenterLabel = 0;

      unsigned int leftmostCentroid  = size[1];
      unsigned int rightmostCentroid = 0;
      
      // Get the centroids and labels for each object      
      for ( unsigned int i=1; i<=relabeler->GetNumberOfObjects(); i++ )
        {
        int x = 0;
        int count = 0;

        rIt.GoToBegin();
        while ( !rIt.IsAtEnd() )
          {
          if ( rIt.Get() == i )
            {
            x += (rIt.GetIndex())[0];
            count++;
            }

          ++rIt;
          }
        
        unsigned int centroidLocation = (unsigned int)( double(x)/double(count) );

        if ( centroidLocation > rightmostCentroid )
          {
          rightmostCentroid = centroidLocation;
          }
        if ( centroidLocation < leftmostCentroid )
          {
          leftmostCentroid = centroidLocation;
          }

        xCentroidMap[centroidLocation] = i;
        }

      unsigned int middleLocation = (rightmostCentroid + leftmostCentroid)/2;

      // Now find the label that corresponds to the centroid that is
      // closest to the middle location
      unsigned int minDiff = size[1];

      std::map< unsigned int, unsigned short >::iterator mapIt = xCentroidMap.begin();
      while ( mapIt != xCentroidMap.end() )
        {
        int diff = std::abs( int((*mapIt).first) - int(middleLocation) );

        if ( diff < minDiff )
          {                                     
          minDiff = diff;
          bestCenterLabel = (*mapIt).second;
          }

        mapIt++;
        }

      LabelMapType::IndexType index;

      rIt.GoToBegin();
      while ( !rIt.IsAtEnd() )
        {
        if ( rIt.Get() == bestCenterLabel )
          {
          index[0] = (rIt.GetIndex())[0];
          index[1] = (rIt.GetIndex())[1];
          if ( this->m_HeadFirst )
            {
            index[2] = size[2] - 1 - currentSliceOffset;
            }
          else
            {
            index[2] = currentSliceOffset;
            }

          seedVec.push_back( index );
          }

        ++rIt;
        }
      }

    currentSliceOffset++;
    }

  return seedVec;
}

 
template < class TInputImage >
void CIPPartialLungLabelMapImageFilter< TInputImage >
::SetAirwayMinIntensityThreshold( InputPixelType threshold )
{
  this->m_AirwayMinIntensityThreshold = threshold;
  this->m_AirwayMinIntensityThresholdSet = true;
}


template < class TInputImage >
void CIPPartialLungLabelMapImageFilter< TInputImage >
::SetAirwayMaxIntensityThreshold( InputPixelType threshold )
{
  this->m_AirwayMaxIntensityThreshold = threshold;
  this->m_AirwayMaxIntensityThresholdSet = true;
}

 
/**
 * Standard "PrintSelf" method
 */
template < class TInputImage >
void
CIPPartialLungLabelMapImageFilter< TInputImage >
::PrintSelf(
  std::ostream& os, 
  Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Printing itkCIPPartialLungLabelMapImageFilter: " << std::endl;
  os << indent << "ClosingNeighborhood: " << this->m_ClosingNeighborhood[0] << "\t" << this->m_ClosingNeighborhood[1] << "\t" << this->m_ClosingNeighborhood[2] << std::endl;
  os << indent << "ExponentialCoefficient: " << this->m_ExponentialCoefficient << std::endl;
  os << indent << "ExponentialTimeConstant: " << this->m_ExponentialTimeConstant << std::endl;
  os << indent << "LeftRightLungSplitRadius: " << this->m_LeftRightLungSplitRadius << std::endl;
  os << indent << "HeadFirst: " << this->m_HeadFirst << std::endl;
  os << indent << "Supine: " << this->m_Supine << std::endl;       
}

} // end namespace itk

#endif

