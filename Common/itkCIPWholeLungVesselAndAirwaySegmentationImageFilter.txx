#ifndef _itkCIPWholeLungVesselAndAirwaySegmentationImageFilter_txx
#define _itkCIPWholeLungVesselAndAirwaySegmentationImageFilter_txx

#include "itkCIPWholeLungVesselAndAirwaySegmentationImageFilter.h"


namespace itk
{

template < class TInputImage >
CIPWholeLungVesselAndAirwaySegmentationImageFilter< TInputImage >
::CIPWholeLungVesselAndAirwaySegmentationImageFilter()

{  
  this->m_ClosingNeighborhood[0]      = 7;
  this->m_ClosingNeighborhood[1]      = 7;
  this->m_ClosingNeighborhood[2]      = 7;
  this->m_HeadFirst                   = true;
  this->m_MinAirwayVolume             = 10.0;
  this->m_MaxAirwayVolumeIncreaseRate = 2.0; 
}


template< class TInputImage >
void
CIPWholeLungVesselAndAirwaySegmentationImageFilter< TInputImage >
::GenerateData()
{
  this->ApplyOtsuThreshold();

  //
  // Get or set the airway segmentation
  //
  if ( this->m_AirwayLabelMap.IsNull() )
    {
    typename AirwaySegmentationType::Pointer airwaySegmenter = AirwaySegmentationType::New();
      airwaySegmenter->SetInput( this->GetInput() );    
      airwaySegmenter->SetMaxAirwayVolumeIncreaseRate( this->m_MaxAirwayVolumeIncreaseRate );
      airwaySegmenter->SetMinAirwayVolume( this->m_MinAirwayVolume );
    if ( this->m_AirwaySegmentationSeedVec.size() == 0 )
      {
      std::vector< OutputImageType::IndexType > airwaySeedVec = this->GetAirwaySeeds();

      for ( unsigned int i=0; i<airwaySeedVec.size(); i++ )
        {
        airwaySegmenter->AddSeed( airwaySeedVec[i] );
        }
      }
    else
      {
      for ( unsigned int i=0; i<this->m_AirwaySegmentationSeedVec.size(); i++ )
        {
        airwaySegmenter->AddSeed( this->m_AirwaySegmentationSeedVec[i] );
        }      
      }
    airwaySegmenter->Update();

    LabelMapIteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );
    LabelMapIteratorType aIt( airwaySegmenter->GetOutput(), airwaySegmenter->GetOutput()->GetBufferedRegion() );

    oIt.GoToBegin();
    aIt.GoToBegin();
    while ( !oIt.IsAtEnd() )
      {
      if ( aIt.Get() != 0 )
        {
        this->SetLungType( oIt.GetIndex(), static_cast< unsigned char >( cip::AIRWAY ) );

        this->m_AirwayIndexVec.push_back( oIt.GetIndex() );
        }

      ++oIt;
      ++aIt;
      }
    }
  else
    {
    LabelMapIteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );
    LabelMapIteratorType aIt( this->m_AirwayLabelMap, this->m_AirwayLabelMap->GetBufferedRegion() );

    oIt.GoToBegin();
    aIt.GoToBegin();
    while ( !oIt.IsAtEnd() )
      {
      if ( aIt.Get() != 0 )
        {
        this->SetLungType( oIt.GetIndex(), static_cast< unsigned char >( cip::AIRWAY ) );

        this->m_AirwayIndexVec.push_back( oIt.GetIndex() );
        }

      ++oIt;
      ++aIt;
      }
    }

  this->SetNonLungAirwayRegion();
  this->FillAndRecordVessels();
}


template < class TInputImage >
void
CIPWholeLungVesselAndAirwaySegmentationImageFilter< TInputImage >
::SetClosingNeighborhood( unsigned long* neighborhood )
{
  this->m_ClosingNeighborhood[0] = neighborhood[0];
  this->m_ClosingNeighborhood[1] = neighborhood[1];
  this->m_ClosingNeighborhood[2] = neighborhood[2];
}


/**
 * This method will apply Otsu thresholding to the input image and
 * store the result in the 'output' image.  Border objects will be
 * removed.  Also, this method makes the assumption that the lungs and
 * airways are connected to each other and that this single 3D object
 * is the largest foreground object in the image.  Any foreground
 * object not connected to this large object will be set to
 * background. 
 */
template < class TInputImage >
void
CIPWholeLungVesselAndAirwaySegmentationImageFilter< TInputImage >
::ApplyOtsuThreshold()
{
  //
  // The first step is to run Otsu threshold on the input data.  This
  // classifies each voxel as either "body" or "air"
  //
  typename OtsuThresholdType::Pointer otsuThreshold = OtsuThresholdType::New();
    otsuThreshold->SetInput( this->GetInput() );
    otsuThreshold->Update();

  int ctXDim = (this->GetInput()->GetBufferedRegion().GetSize())[0];
  int ctYDim = (this->GetInput()->GetBufferedRegion().GetSize())[1];

  this->GraftOutput( otsuThreshold->GetOutput() );

  //
  // The next step is to identify all connected components in the
  // thresholded image
  //
  ConnectedComponent3DType::Pointer connectedComponent = ConnectedComponent3DType::New();
    connectedComponent->SetInput( otsuThreshold->GetOutput() );

  //
  // Relabel the connected components
  //
  Relabel3DType::Pointer relabelComponent = Relabel3DType::New();
    relabelComponent->SetInput( connectedComponent->GetOutput() );
    relabelComponent->Update();  

  //
  // Now we want to identify the component labels that correspond to
  // the left lung and the right lung. In some cases, they might not
  // be connected, that's why we need to do a separate search for
  // each.
  //
  std::vector< int >  lungHalf1ComponentCounter;
  std::vector< int >  lungHalf2ComponentCounter;
  for ( unsigned int i=0; i<=relabelComponent->GetNumberOfObjects(); i++ )
    {
    lungHalf1ComponentCounter.push_back( 0 );
    lungHalf2ComponentCounter.push_back( 0 );
    }

  ComponentIteratorType rIt( relabelComponent->GetOutput(), relabelComponent->GetOutput()->GetBufferedRegion() );

  int lowerYBound = static_cast< int >( 0.45*static_cast< double >( ctYDim ) );
  int upperYBound = static_cast< int >( 0.55*static_cast< double >( ctYDim ) );

  int lowerXBound = static_cast< int >( 0.20*static_cast< double >( ctXDim ) );
  int upperXBound = static_cast< int >( 0.80*static_cast< double >( ctXDim ) );

  int middleX =  static_cast< int >( 0.5*static_cast< double >( ctXDim ) );

  rIt.GoToBegin();
  while ( !rIt.IsAtEnd() )
    {
    if ( rIt.Get() != 0 )
      {
      LabelMapType::IndexType index = rIt.GetIndex();

      if ( index[1] >= lowerYBound && index[1] <= upperYBound )
        {
        int whichComponent = static_cast< int >( rIt.Get() );

        if ( index[0] >= lowerXBound && index[0] <= middleX )
          {
          lungHalf1ComponentCounter[whichComponent] = lungHalf1ComponentCounter[whichComponent]+1;
          }
        else if ( index[0] < upperXBound && index[0] > middleX )
          {
          lungHalf2ComponentCounter[whichComponent] = lungHalf2ComponentCounter[whichComponent]+1;
          }
        }
      }

    ++rIt;
    }

  unsigned short lungHalf1Label;
  unsigned short lungHalf2Label;
  int maxLungHalf1Count = 0;
  int maxLungHalf2Count = 0;
  for ( unsigned int i=0; i<=relabelComponent->GetNumberOfObjects(); i++ )
    {
    if ( lungHalf1ComponentCounter[i] > maxLungHalf1Count )
      {
      maxLungHalf1Count = lungHalf1ComponentCounter[i];

      lungHalf1Label = static_cast< unsigned short >( i );
      }
    if ( lungHalf2ComponentCounter[i] > maxLungHalf2Count )
      {
      maxLungHalf2Count = lungHalf2ComponentCounter[i];

      lungHalf2Label = static_cast< unsigned short >( i );
      }
    }

  LabelMapIteratorType mIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

  mIt.GoToBegin();
  rIt.GoToBegin();
  while ( !mIt.IsAtEnd() )
    {
    if ( rIt.Get() == lungHalf1Label || rIt.Get() == lungHalf2Label )
      {
      mIt.Set( cip::WHOLELUNG );
      }
    else 
      {
      mIt.Set( 0 );
      }

    ++mIt;
    ++rIt;
    }
}


/**
 * This function will fill in holes (identified in 2D) in the lung
 * region. The assumption is made that these holes correspond to
 * vessels (bright objects in the lung region).  This is a very
 * simplistic way to extract the vessels, and vessel labels in the
 * resulting output image should be interpreted as being approximate.
 * 
 * The indices that are filled in are also recorded.  During the final
 * labeling of the output image, they will be set with the proper
 * label (once the left and right lung locations are known).
 */
template < class TInputImage >
void
CIPWholeLungVesselAndAirwaySegmentationImageFilter< TInputImage >
::FillAndRecordVessels()
{
  //
  // We need to make the mask binary for hole filling.  Set all
  // airways to be UNDEFINEDTYPE.  We will add them back later.
  //
  for ( unsigned int i=0; i<this->m_AirwayIndexVec.size(); i++ )
    {
    this->SetLungType( this->m_AirwayIndexVec[i], cip::UNDEFINEDTYPE );
    }

  //
  // Perform morphological closing on the mask by dilating and then
  // eroding.  We assume that at this point in the pipeline, the
  // output image only has WHOLELUNG as a foreground value.  (The
  // airways and vessels should be stored in the index vec member
  // variables). 
  //
  Element3DType structuringElement;
    structuringElement.SetRadius( this->m_ClosingNeighborhood );
    structuringElement.CreateStructuringElement();

  typename Dilate3DType::Pointer dilater = Dilate3DType::New();
    dilater->SetInput( this->GetOutput() );
    dilater->SetKernel( structuringElement );
    dilater->SetDilateValue( cip::WHOLELUNG );
  try
    {
    dilater->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught dilating:";
    std::cerr << excp << std::endl;
    }

  //
  // Occasionally, dilation will extend the mask to the end slices. If
  // this occurs, the erosion step below won't be able to hit these
  // regions. To deal with this, extract the end slices from the
  // dilater and current output image.  Then set the dilater end
  // slices to be zero (provided that the output image is also zero at
  // those locations).
  //
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
  
  //
  // Now erode
  //
  typename Erode3DType::Pointer eroder = Erode3DType::New();
    eroder->SetInput( dilater->GetOutput() );
    eroder->SetKernel( structuringElement );
    eroder->SetErodeValue( cip::WHOLELUNG );
  try
    {
    eroder->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught eroding:";
    std::cerr << excp << std::endl;
    }

  LabelMapIteratorType eIt( eroder->GetOutput(), eroder->GetOutput()->GetBufferedRegion() );
  LabelMapIteratorType mIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

  eIt.GoToBegin();
  mIt.GoToBegin();
  while ( !mIt.IsAtEnd() )
    {
    if ( eIt.Get() == cip::WHOLELUNG && mIt.Get() == 0 )
      {
      this->m_VesselIndexVec.push_back( eIt.GetIndex() );
      this->SetLungType( eIt.GetIndex(), cip::VESSEL );
      this->SetLungRegion( eIt.GetIndex(), cip::WHOLELUNG );
      }

    ++eIt;
    ++mIt;
    }
  
  //
  // Now add the airways back
  //
  for ( unsigned int i=0; i<this->m_AirwayIndexVec.size(); i++ )
    {
    this->SetLungType( this->m_AirwayIndexVec[i], cip::AIRWAY );
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
CIPWholeLungVesselAndAirwaySegmentationImageFilter< TInputImage >
::GetAirwaySeeds()
{
  std::vector< OutputImageType::IndexType > seedVec;

  LabelMapType::SizeType    size    = this->GetOutput()->GetBufferedRegion().GetSize();
  LabelMapType::SpacingType spacing = this->GetOutput()->GetSpacing();

  //
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
  //
  unsigned int slicesProcessed                              = 0;
  unsigned int currentSliceOffset                           = 0;
  int          numberSlicesForSeedSearch                    = 15;
  double       foregroundSliceSizeThresholdForSeedSelection = 2000.0;

  while ( slicesProcessed < numberSlicesForSeedSearch && currentSliceOffset < size[2] )
    {
    //
    // Extract a 2D slice from the mask
    //
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

    this->ExtractLabelMapSlice( this->GetOutput(), slice, whichSlice );

    //
    // Perform connected component labeling
    //
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
    
    //
    // Relabel the components
    //
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

    //
    // If the foreground area of the slice is larger than the
    // threshold, consider the slice for seed selection. The number of
    // objects present in the slice must also be equal to three
    // (trachea, left and right lungs)
    //
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

      //
      // Identify the object who's centroid (x coordinate) is in the
      // middle. First get the centroids, then find out which one is
      // in the middle 
      //
      std::map< unsigned int, unsigned short > xCentroidMap;

      unsigned short bestCenterLabel = 0;

      unsigned int leftmostCentroid  = size[1];
      unsigned int rightmostCentroid = 0;

      //
      // Get the centroids and labels for each object
      //
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
        
        unsigned int centroidLocation = static_cast< unsigned int >( static_cast< double >( x )/static_cast< double >( count ) );

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

      //
      // Now find the label that corresponds to the centroid that is
      // closest to the middle location
      //
      unsigned int minDiff = size[1];

      std::map< unsigned int, unsigned short >::iterator mapIt = xCentroidMap.begin();
      while ( mapIt != xCentroidMap.end() )
        {
        int diff = std::abs( static_cast< int >( (*mapIt).first ) - static_cast< int >( middleLocation ) );

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
void
CIPWholeLungVesselAndAirwaySegmentationImageFilter< TInputImage >
::SetLungType( OutputImageType::IndexType index, unsigned char lungType )
{
  //
  // Get the curent value at the specified location
  //
  unsigned short currentValue = this->GetOutput()->GetPixel( index );

  //
  // Determine the region corresponding to this value
  //
  unsigned char lungRegion = this->m_LungConventions.GetLungRegionFromValue( currentValue );

  //
  // Determine the new value given the type to set and the current
  // region 
  //
  unsigned short newValue = this->m_LungConventions.GetValueFromLungRegionAndType( lungRegion, lungType );

  //
  // Set the new value at the specified index
  //
  this->GetOutput()->SetPixel( index, newValue );
}


template < class TInputImage >
void
CIPWholeLungVesselAndAirwaySegmentationImageFilter< TInputImage >
::SetLungRegion( OutputImageType::IndexType index, unsigned char lungRegion )
{
  //
  // Get the current value at the specified location
  //
  unsigned short currentValue = this->GetOutput()->GetPixel( index );

  //
  // Determine the type corresponding to this value
  //
  unsigned char lungType = this->m_LungConventions.GetLungTypeFromValue( currentValue );

  //
  // Determine the new value given the region to set and the current
  // type
  //
  unsigned short newValue = this->m_LungConventions.GetValueFromLungRegionAndType( lungRegion, lungType );

  //
  // Set the new value at the specified index
  //
  this->GetOutput()->SetPixel( index, newValue );
}


/**
 * In this method, we'll temporarily zero-out all the airways that
 * aren't interior to the lung.  All the indices that correspond to
 * airway have been previously recorded, so we'll be able to recover
 * them later.  We need to eliminate the trachea and main bronchi
 * temporarily for subsequent morphological operations.
 */
template < class TInputImage >
void
CIPWholeLungVesselAndAirwaySegmentationImageFilter< TInputImage >
::SetNonLungAirwayRegion()
{
  LabelMapType::SizeType size = this->GetOutput()->GetBufferedRegion().GetSize();
  
  int elementRadius = 3;
  unsigned short wholeLungAirwayValue = this->m_LungConventions.GetValueFromLungRegionAndType( cip::WHOLELUNG, cip::AIRWAY );

  for ( unsigned int whichSlice = 0; whichSlice < size[2]; whichSlice++ )
    {
    //
    // Extract a slice
    //
    LabelMapSliceType::Pointer slice = LabelMapSliceType::New();

    this->ExtractLabelMapSlice( this->GetOutput(), slice, whichSlice );

    //
    // Compute connected components for the entire slice (all
    // foreground regions)
    //
    ConnectedComponent2DType::Pointer totalConnectedComponent = ConnectedComponent2DType::New();    
      totalConnectedComponent->SetInput( slice );
      totalConnectedComponent->FullyConnectedOn();

    Relabel2DType::Pointer totalRelabeler = Relabel2DType::New();
      totalRelabeler->SetInput( totalConnectedComponent->GetOutput() );
      totalRelabeler->Update();      

    //
    // We'll also keep a list of relabeled values that need to be
    // set to undefined region based on a criterion described below
    //
    std::list< unsigned short > undefinedRegionList;

    //
    // Dilate the airways.  The segmented airways will typically have a
    // ring of 'WHOLELUNG' surrounding them.  We'll dilate in order to
    // engulf most of this.  
    //
    Element2DType structuringElement;
      structuringElement.SetRadius( elementRadius );
      structuringElement.CreateStructuringElement();

    Dilate2DType::Pointer sliceDilater = Dilate2DType::New();
      sliceDilater->SetInput( slice );
      sliceDilater->SetKernel( structuringElement );
      sliceDilater->SetDilateValue( wholeLungAirwayValue );

    Threshold2DType::Pointer thresholder = Threshold2DType::New();
      thresholder->SetInput( sliceDilater->GetOutput() );
      thresholder->SetInsideValue( 1 );
      thresholder->SetOutsideValue( itk::NumericTraits< OutputPixelType >::Zero );
      thresholder->SetLowerThreshold( wholeLungAirwayValue );
      thresholder->SetUpperThreshold( wholeLungAirwayValue );

    //
    // Run connected components and then relabel
    //
    ConnectedComponent2DType::Pointer connectedComponent = ConnectedComponent2DType::New();    
      connectedComponent->SetInput( thresholder->GetOutput() );
      connectedComponent->FullyConnectedOn();

    Relabel2DType::Pointer relabeler = Relabel2DType::New();
      relabeler->SetInput( connectedComponent->GetOutput() );
      relabeler->Update();

    std::vector< int > perimeterCountVec;
    std::vector< int > touchingBackgroundCountVec;

    for ( unsigned int i=0; i<relabeler->GetNumberOfObjects(); i++ )
      {
      perimeterCountVec.push_back( 0 );
      touchingBackgroundCountVec.push_back( 0 );
      }

    //
    // Find all labels for which 50% of the perimeter pixels touch
    // background.  These will be removed from the slice.  The 50%
    // value is somewhat arbitrary.  We're basically looking for those
    // airway components that are largely exposed to the background
    // (i.e. not within the lung interior). In some cases, with the
    // current airway segmentation algorithm, explosions are
    // impossible to avoid, and in extreme cases, these explosions can
    // fill the majority of the lung region. To avoid removing regions
    // that should not be removed (based on the 50% criterion just
    // described), we'll "protect" a region if it is the left-most or
    // right-most region. So first determine what the left-most and
    // right-most labels are for this slice. Note also that if a
    // region is the left-most AND the right-most, it's assumed to be
    // trachea, and is thus not protected.
    //
    int xMin = size[0];
    int xMax = 0;

    unsigned short leftMostLabel  = 0;
    unsigned short rightMostLabel = 0;

    LabelMapSliceIteratorType rIt( relabeler->GetOutput(), relabeler->GetOutput()->GetBufferedRegion() );
    LabelMapSliceIteratorType trIt( totalRelabeler->GetOutput(), totalRelabeler->GetOutput()->GetBufferedRegion() );

    trIt.GoToBegin();
    while ( !trIt.IsAtEnd() )
      {
      if ( trIt.Get() > 0 )
        {
        if ( trIt.GetIndex()[0] < xMin )
          {
          xMin = trIt.GetIndex()[0];
          
          leftMostLabel = trIt.Get();
          }

        if ( trIt.GetIndex()[0] > xMax )
          {
          xMax = trIt.GetIndex()[0];

          rightMostLabel = trIt.Get();
          }
        }

      ++trIt;
      }

    rIt.GoToBegin();
    trIt.GoToBegin();
    while ( !rIt.IsAtEnd() )
      {
      if ( rIt.Get() != 0 )
        {
        if ( trIt.Get() != 0 )
          {
          //
          // Get size of both the airway relabeled component and the
          // total relabeled component. If the ratio of the two is
          // greater than 50% (arbitrary), then add the total
          // relabeled value to the list of values to set to undefined
          // region
          //
          double numAirwayRelabeled = static_cast< double >( relabeler->GetSizeOfObjectsInPixels()[rIt.Get()-1] );
          double numTotalRelabeled  = static_cast< double >( totalRelabeler->GetSizeOfObjectsInPixels()[trIt.Get()-1] );

          if ( (numAirwayRelabeled/numTotalRelabeled) >= 0.5 )
            {
            if ( trIt.Get() == leftMostLabel && trIt.Get() == rightMostLabel )
              {
              undefinedRegionList.push_back( trIt.Get() );
              }
            else if ( trIt.Get() != leftMostLabel && trIt.Get() != rightMostLabel )
              {
              undefinedRegionList.push_back( trIt.Get() );
              }
            }
          }

        unsigned short label = rIt.Get();
        
        LabelMapSliceType::IndexType index = rIt.GetIndex();
        LabelMapType::IndexType tempIndex;
        tempIndex[2] = whichSlice;

        bool isPerimeter = false;
        bool touchingBackground = false;

        for ( int x=int(index[0])-1; x<=(index[0]+1); x++ )
          {
          tempIndex[0] = x;
          
          if ( x >= 0 && x < size[0] )
            {
            for ( int y=int(index[1])-1; y<=(index[1]+1); y++ )
              {
              tempIndex[1] = y;
              
              if ( y >= 0 && y < size[1] )
                {
                if ( this->GetOutput()->GetPixel( tempIndex ) == itk::NumericTraits< OutputPixelType >::Zero )
                  {
                  touchingBackground = true;
                  isPerimeter = true;
                  }
                if ( this->GetOutput()->GetPixel( tempIndex ) == static_cast< OutputPixelType >( cip::WHOLELUNG ) )
                  {
                  isPerimeter = true;
                  }
                }
              }
            }
          }

        if ( isPerimeter )
          {
          perimeterCountVec[label-1]++;
          }
        if ( touchingBackground )
          {
          touchingBackgroundCountVec[label-1]++;
          }
        }
      
      ++trIt;
      ++rIt;
      }

    //
    // Now figure out the labels corresponding to UNDEFINEDREGION
    //
    std::vector< int > undefinedRegionLabelsVec;

    for ( unsigned int i=0; i<relabeler->GetNumberOfObjects(); i++ )
      {
      if ( static_cast< double >( touchingBackgroundCountVec[i] )/static_cast< double >( perimeterCountVec[i] ) >= 0.50 )
        {
        undefinedRegionLabelsVec.push_back( i+1 );
        }
      }

    //
    // Now iterate over the relabeler image and set all pixels in the
    // output image to have UNDEFINEDREGION if the relabeled value is
    // in the vector we just computed.
    //
    rIt.GoToBegin();
    while ( !rIt.IsAtEnd() )
      {
      if ( rIt.Get() != 0 )
        {
        for ( unsigned int i=0; i<undefinedRegionLabelsVec.size(); i++ )
          {
          if ( rIt.Get() == undefinedRegionLabelsVec[i] )
            {
            LabelMapType::IndexType tempIndex;
              tempIndex[0] = rIt.GetIndex()[0];
              tempIndex[1] = rIt.GetIndex()[1];
              tempIndex[2] = whichSlice;

            this->GetOutput()->SetPixel( tempIndex, 0 );
            }
          }
        }
      
      ++rIt;
      }

    //
    // Set voxels to be UNDEFINEDREGION that correspond to those
    // connected components for which at least 50% of the region is
    // airway 
    //
    undefinedRegionList.unique();
    undefinedRegionList.sort();
    undefinedRegionList.unique();

    std::list< unsigned short >::iterator listIt;

    trIt.GoToBegin();
    while ( !trIt.IsAtEnd() )
      {
      if ( trIt.Get() > 0 )
        {
        listIt = undefinedRegionList.begin();

        for ( unsigned int u=0; u<undefinedRegionList.size(); u++, listIt++ )
          {
          if ( trIt.Get() == *listIt )
            {
            LabelMapType::IndexType tempIndex;
              tempIndex[0] = trIt.GetIndex()[0];
              tempIndex[1] = trIt.GetIndex()[1];
              tempIndex[2] = whichSlice;            

            this->GetOutput()->SetPixel( tempIndex, 0 );

            this->m_AirwayIndexVec.push_back( tempIndex );
            }
          }
        }

      ++trIt;
      }
    }

  //
  // In some rare cases it is possible that a few trachea regions
  // persist to this point, though they should have been removed. We
  // assume that they are disconnected from the rest of the
  // lung. We'll first extract the WHOLELUNG region, perform connected
  // components, and then identify the regions that make up a small
  // percentage of the foreground. These will be set to zero.
  //
  ExtractLabelMapType::Pointer labelMapExtractor = ExtractLabelMapType::New();
    labelMapExtractor->SetInput( this->GetOutput() );
    labelMapExtractor->SetChestRegion( (unsigned char)( cip::WHOLELUNG ) );
    labelMapExtractor->Update();

  ConnectedComponent3DType::Pointer connectedComponent = ConnectedComponent3DType::New();
    connectedComponent->SetInput( labelMapExtractor->GetOutput() );

  Relabel3DType::Pointer relabeler = Relabel3DType::New();
    relabeler->SetInput( connectedComponent->GetOutput() );
    relabeler->Update();
  
  unsigned int totalNumberForegroundVoxels = 0;

  ComponentIteratorType rIt( relabeler->GetOutput(), relabeler->GetOutput()->GetBufferedRegion() );

  rIt.GoToBegin();
  while ( !rIt.IsAtEnd() )
    {
    if ( rIt.Get() != 0 )
      {
      totalNumberForegroundVoxels++;
      }

    ++rIt;
    }

  int cutoffLabel = itk::NumericTraits< int >::max();

  for ( unsigned int i=0; i<relabeler->GetNumberOfObjects(); i++ )
    {
    double percentage = static_cast< double >( relabeler->GetSizeOfObjectsInPixels()[i] )/static_cast< double >( totalNumberForegroundVoxels );

    //
    // Note that the 0.15 value has been somewhat arbitrarily chosen
    //
    if ( percentage <= 0.15 )
      {
      cutoffLabel = i+1;

      break;
      }
    }

  LabelMapIteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

  rIt.GoToBegin();
  oIt.GoToBegin();
  while ( !rIt.IsAtEnd() )
    {
    if ( rIt.Get() >= cutoffLabel )
      {
      oIt.Set( 0 );

      this->m_AirwayIndexVec.push_back( oIt.GetIndex() );
      }

    ++rIt;
    ++oIt;
    }
}


/**
 * Extract a slice from the input label map image
 */
template < class TInputImage >
void
CIPWholeLungVesselAndAirwaySegmentationImageFilter< TInputImage >
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
  
  LabelMapExtractorType::Pointer sliceExtractor = LabelMapExtractorType::New();
    sliceExtractor->SetInput( image );
    sliceExtractor->SetExtractionRegion( sliceExtractorRegion );
    sliceExtractor->Update();

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


template < class TInputImage >
void
CIPWholeLungVesselAndAirwaySegmentationImageFilter< TInputImage >
::SetAirwayLabelMap( OutputImageType::Pointer airwayLabelMap )
{
  this->m_AirwayLabelMap = LabelMapType::New();
  this->m_AirwayLabelMap = airwayLabelMap;
}


template < class TInputImage >
void
CIPWholeLungVesselAndAirwaySegmentationImageFilter< TInputImage >
::AddAirwaySegmentationSeed( OutputImageType::IndexType seed )
{
  this->m_AirwaySegmentationSeedVec.push_back( seed );
}
  

/**
 * Standard "PrintSelf" method
 */
template < class TInputImage >
void
CIPWholeLungVesselAndAirwaySegmentationImageFilter< TInputImage >
::PrintSelf(
  std::ostream& os, 
  Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Printing itkCIPWholeLungVesselAndAirwaySegmentationImageFilter: " << std::endl;
}

} // end namespace itk

#endif
