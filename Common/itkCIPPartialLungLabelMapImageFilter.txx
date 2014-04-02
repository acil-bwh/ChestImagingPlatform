#ifndef _itkCIPPartialLungLabelMapImageFilter_txx
#define _itkCIPPartialLungLabelMapImageFilter_txx

#include "itkCIPPartialLungLabelMapImageFilter.h"
#include "itkImageFileWriter.h"

namespace itk
{

template < class TInputImage >
CIPPartialLungLabelMapImageFilter< TInputImage >
::CIPPartialLungLabelMapImageFilter()
{
  this->m_ClosingNeighborhood[0]      = 7;
  this->m_ClosingNeighborhood[1]      = 7;
  this->m_ClosingNeighborhood[2]      = 7;
  this->m_MinAirwayVolume             = 10.0;
  this->m_MaxAirwayVolume             = 500.0;
  this->m_MaxAirwayVolumeIncreaseRate = 2.0; 
  this->m_ExponentialCoefficient      = 200;
  this->m_ExponentialTimeConstant     = -700;
  this->m_LeftRightLungSplitRadius    = 2;
  this->m_OtsuThreshold               = -1024;  
  this->m_AggressiveLeftRightSplitter = false;
  this->m_HeadFirst                   = true;
  this->m_Supine                      = true;

  this->m_AirwayLabelMap = LabelMapType::New();
}


template< class TInputImage >
void
CIPPartialLungLabelMapImageFilter< TInputImage >
::GenerateData()
{  
  cip::ChestConventions conventions;

  LabelMapType::SpacingType spacing = this->GetInput()->GetSpacing();

  typedef itk::ImageFileWriter< LabelMapType > WriterType;

  //
  // Allocate space for the output image
  //
  typename Superclass::InputImageConstPointer inputPtr  = this->GetInput();
  typename Superclass::OutputImagePointer     outputPtr = this->GetOutput(0);
    outputPtr->SetRequestedRegion( inputPtr->GetRequestedRegion() );
    outputPtr->SetBufferedRegion( inputPtr->GetBufferedRegion() );
    outputPtr->SetLargestPossibleRegion( inputPtr->GetLargestPossibleRegion() );
    outputPtr->Allocate();
    outputPtr->FillBuffer( 0 );

  if ( this->m_HelperMask.IsNull() )
    {
    //
    // Apply Otsu threshold
    //
    this->ApplyOtsuThreshold();
    }
  else
    {
    //
    // Apply the helper mask
    //
    this->ApplyHelperMask();
    }

  //
  // We want to set the max airway volume to be a percentage of the
  // lung volume (lungs plus airways). Compute the volume of the lungs
  // and airways
  //
  unsigned int counter = 0;

  LabelMapIteratorType  mIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );  

  mIt.GoToBegin();
  while ( !mIt.IsAtEnd() )
    {
    if ( mIt.Get() != 0 )
      {
      counter++;
      }

    ++mIt;
    }

  std::cout << "---Total volume:\t" << static_cast< double >( counter )*spacing[0]*spacing[1]*spacing[2] << std::endl;
  std::cout << "---Percentage:\t" << this->m_MaxAirwayVolume/(static_cast< double >( counter )*spacing[0]*spacing[1]*spacing[2]) << std::endl;

  this->m_MaxAirwayVolume = 0.0085*static_cast< double >( counter )*spacing[0]*spacing[1]*spacing[2];
  this->m_MinAirwayVolume = 0.005*static_cast< double >( counter )*spacing[0]*spacing[1]*spacing[2];
  std::cout << "---Max airway volume:\t" << this->m_MaxAirwayVolume << std::endl;
  std::cout << "---Min airway volume:\t" << this->m_MinAirwayVolume << std::endl;

  //
  // Identify airways
  //
  std::vector< OutputImageType::IndexType > airwaySeedVec = this->GetAirwaySeeds();

  std::cout << "---Segmenting airways..." << std::endl;
  typename AirwaySegmentationType::Pointer airwaySegmenter = AirwaySegmentationType::New();
    airwaySegmenter->SetInput( this->GetInput() );    
    airwaySegmenter->SetMaxIntensityThreshold( -800 );
  for ( unsigned int i=0; i<airwaySeedVec.size(); i++ )
    {
    airwaySegmenter->AddSeed( airwaySeedVec[i] );
    }          
    airwaySegmenter->Update();

//   std::cout << "---Removing trachea and main bronchi..." << std::endl;
//   this->RemoveTracheaAndMainBronchi();

  // Collect / remove airway indices
  std::cout << "---Recorind and removing airways..." << std::endl;
  this->RecordAndRemoveAirways( airwaySegmenter->GetOutput() );  
  
  // There may still be small foreground regions
  // within the trachea / main bronchi that were not picked up via the
  // airway segmentation routine. We'll zero out all components that
  // don't accomodate for a significant portion of the overall
  // foreground region
  ConnectedComponent3DType::Pointer connectedComponent = ConnectedComponent3DType::New();
    connectedComponent->SetInput( this->GetOutput() );

  // Relabel the connected components
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
//    std::cout << "Component label:\t" << i+1 << "\t Percentage:\t" << static_cast< double >( relabelComponent->GetSizeOfObjectsInPixels()[i] )/static_cast< double >( totalSize ) << std::endl;
    if ( static_cast< double >( relabelComponent->GetSizeOfObjectsInPixels()[i] )/static_cast< double >( totalSize ) < 0.20 )
      {
      componentsToRemoveThreshold = i+1;
      break;
      }
    }  

  ComponentIteratorType rIt( relabelComponent->GetOutput(), relabelComponent->GetOutput()->GetBufferedRegion() );

  mIt.GoToBegin();
  rIt.GoToBegin();
  while ( !mIt.IsAtEnd() )
    {
    if ( rIt.Get() >= componentsToRemoveThreshold )
      {
      mIt.Set( 0 );
      }

    ++mIt;
    ++rIt;
    }

  //
  // The erosion step below has been added specifically for use with
  // helper input images. It's assumed that the helper image has the
  // left and right lungs split "pretty well", but some small
  // connections may persist. Therefore, we'll erode and then attemp
  // to label the left and right, after which we'll refill the output
  // image.
  //
  LungRegionLabelerType::Pointer leftRightLabeler = LungRegionLabelerType::New();

  if ( this->m_HelperMask.IsNotNull() )
    {
    Element3DType structuringElement;
      structuringElement.SetRadius( 1 );
      structuringElement.CreateStructuringElement();

//    std::cout << "---Eroding helper image..." << std::endl;
    Erode3DType::Pointer eroder = Erode3DType::New();
      eroder->SetInput( this->GetOutput() );
      eroder->SetKernel( structuringElement );
      eroder->SetErodeValue( 1 );
    try
      {
      eroder->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught eroding:";
      std::cerr << excp << std::endl;
      }

//    std::cout << "---Labeling helper image..." << std::endl;
    leftRightLabeler->SetInput( eroder->GetOutput() );
    leftRightLabeler->LabelLeftAndRightLungsOn();
    leftRightLabeler->SetHeadFirst( this->m_HeadFirst );
    leftRightLabeler->SetSupine( this->m_Supine );
    leftRightLabeler->Update();

    LabelMapIteratorType lrIt( leftRightLabeler->GetOutput(), leftRightLabeler->GetOutput()->GetBufferedRegion() );
    LabelMapIteratorType hIt( this->m_HelperMask, this->m_HelperMask->GetBufferedRegion() ); 

    LabelMapType::IndexType index;
    unsigned short labelValue;

//    std::cout << "---Filling output image will left / right labeled helper..." << std::endl;

    mIt.GoToBegin();
    lrIt.GoToBegin();
    hIt.GoToBegin();
    while ( !hIt.IsAtEnd() )
      {
      if ( hIt.Get() != 0 )
        {
        labelValue = 0;

        for ( int x=-1; x<=1; x++ )
          {
          index[0] = hIt.GetIndex()[0] + x;

          for ( int y=-1; y<=1; y++ )
            {
            index[1] = hIt.GetIndex()[1] + y;

            for ( int z=-1; z<=1; z++ )
              {
              index[2] = hIt.GetIndex()[2] + z;
              
              if ( this->GetInput()->GetBufferedRegion().IsInside( index ) )
                {
                if ( leftRightLabeler->GetOutput()->GetPixel( index ) != 0 )
                  {
                  labelValue = leftRightLabeler->GetOutput()->GetPixel( index );
                  }
                }
              }
            }
          }

        mIt.Set( labelValue );
        }

      ++mIt;
      ++lrIt;
      ++hIt;
      }
    }

  //if ( this->m_HelperMask.IsNull() )
  if ( false ) //DEB
    {
    // Attempt to label left and right. Further processing may not be necessary.
    leftRightLabeler->SetInput( this->GetOutput() );
    leftRightLabeler->LabelLeftAndRightLungsOn();
    leftRightLabeler->SetHeadFirst( this->m_HeadFirst );
    leftRightLabeler->SetSupine( this->m_Supine );
    leftRightLabeler->Update();

//    LabelMapIteratorType mIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );
    LabelMapIteratorType lrIt( leftRightLabeler->GetOutput(), leftRightLabeler->GetOutput()->GetBufferedRegion() );

    mIt.GoToBegin();
    lrIt.GoToBegin();
    while ( !mIt.IsAtEnd() )
      {
      mIt.Set( lrIt.Get() );
      
      ++mIt;
      ++lrIt;
      }
    
    if ( !leftRightLabeler->GetLabelingSuccess() )
      {
      //
      // Threshold the input with a more conservative upper threshold value
      // 
      typename BinaryThresholdType::Pointer thresholder = BinaryThresholdType::New();
        thresholder->SetInput( this->GetInput() );
        thresholder->SetOutsideValue( 0 );
        thresholder->SetInsideValue( static_cast< unsigned short >( cip::WHOLELUNG ) );
        thresholder->SetLowerThreshold( itk::NumericTraits< short >::min() );
//      thresholder->SetUpperThreshold( -700 );
        thresholder->SetUpperThreshold( -800 );
        thresholder->Update();
      
      LabelMapIteratorType tIt( thresholder->GetOutput(), thresholder->GetOutput()->GetBufferedRegion() );

      mIt.GoToBegin();
      tIt.GoToBegin();
      while ( !tIt.IsAtEnd() )
        {
        if ( mIt.Get() == 0 )
          {                                         
          tIt.Set( 0 );
          }
      
        ++mIt;
        ++tIt;
        }

      //
      // Attempt to label left and right. Splitting may not be necessary
      //
      leftRightLabeler->SetInput( thresholder->GetOutput() );
      leftRightLabeler->LabelLeftAndRightLungsOn();
      leftRightLabeler->SetHeadFirst( this->m_HeadFirst );
      leftRightLabeler->SetSupine( this->m_Supine );
      leftRightLabeler->Update();

      if ( !leftRightLabeler->GetLabelingSuccess() )
        {
        //
        // Split left and right lungs
        //
        typename SplitterType::Pointer splitter = SplitterType::New();
          splitter->SetInput( this->GetInput() );
          splitter->SetLungLabelMap( thresholder->GetOutput() );
          splitter->SetExponentialCoefficient( this->m_ExponentialCoefficient );
          splitter->SetExponentialTimeConstant( this->m_ExponentialTimeConstant );
          splitter->SetLeftRightLungSplitRadius( this->m_LeftRightLungSplitRadius );
          splitter->SetAggressiveLeftRightSplitter( this->m_AggressiveLeftRightSplitter );    
          splitter->Update();    
      
        //
        // Label left and right lungs
        //
        leftRightLabeler->SetInput( splitter->GetOutput() );
        leftRightLabeler->LabelLeftAndRightLungsOn();
        leftRightLabeler->SetHeadFirst( this->m_HeadFirst );
        leftRightLabeler->SetSupine( this->m_Supine );
        leftRightLabeler->Update();
        }

      lrIt.GoToBegin();
      mIt.GoToBegin();
      while ( !lrIt.IsAtEnd() )
        {
        mIt.Set( lrIt.Get() );
        
        ++mIt;
        ++lrIt;
        }

      //
      // Perform conditional dilation
      //
      this->ConditionalDilation( this->m_OtsuThreshold );
      }
    }

  // Perform morphological closing on the left and right lungs
//  if ( leftRightLabeler->GetLabelingSuccess() )
  if ( false ) //DEB
    {
    this->CloseLabelMap( static_cast< unsigned short >( cip::LEFTLUNG ) );
    this->CloseLabelMap( static_cast< unsigned short >( cip::RIGHTLUNG ) );
    }
  else
    {
    this->CloseLabelMap( static_cast< unsigned short >( cip::WHOLELUNG ) );
    }

  std::cout << "---Labeling regions..." << std::endl;
  LungRegionLabelerType::Pointer lungRegionLabeler = LungRegionLabelerType::New();
    lungRegionLabeler->SetInput( this->GetOutput() );
    lungRegionLabeler->LabelLungThirdsOn();
    lungRegionLabeler->SetHeadFirst( this->m_HeadFirst );
    lungRegionLabeler->SetSupine( this->m_Supine );
    lungRegionLabeler->Update();

  this->GraftOutput( lungRegionLabeler->GetOutput() );

  //
  // Add back the airways
  //
  unsigned char  lungRegion;
  unsigned short labelValue;

  LabelMapIteratorType m2It( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );
  LabelMapIteratorType aIt( this->m_AirwayLabelMap, this->m_AirwayLabelMap->GetBufferedRegion() );

//  LabelMapIteratorType aIt( airwaySegmenter->GetOutput(), airwaySegmenter->GetOutput()->GetBufferedRegion() );

  aIt.GoToBegin();
  m2It.GoToBegin();
  while ( !m2It.IsAtEnd() )
    {
    if ( aIt.Get() != 0 )
      {
      lungRegion = conventions.GetChestRegionFromValue( m2It.Get() );
      labelValue = conventions.GetValueFromChestRegionAndType( lungRegion, static_cast< unsigned char >( cip::AIRWAY ) );

      m2It.Set( labelValue );
      }

    ++aIt;
    ++m2It;
    }
}

template < class TInputImage >
void
CIPPartialLungLabelMapImageFilter< TInputImage >
::ConditionalDilation( short threshold )
{
  typedef itk::Image< unsigned char, 3 >                       UCharImageType;
  typedef itk::ImageRegionIteratorWithIndex< UCharImageType >  UCharIteratorType;

  UCharImageType::Pointer tracker = UCharImageType::New();
    tracker->SetRegions( this->GetOutput()->GetBufferedRegion().GetSize() );
    tracker->Allocate();
    tracker->FillBuffer( 0 );

  std::vector< LabelMapType::IndexType > prevLeftIndicesVec;
  std::vector< LabelMapType::IndexType > currLeftIndicesVec;
  std::vector< LabelMapType::IndexType > prevRightIndicesVec;
  std::vector< LabelMapType::IndexType > currRightIndicesVec;
  std::vector< LabelMapType::IndexType > prevWholeIndicesVec;
  std::vector< LabelMapType::IndexType > currWholeIndicesVec;

  LabelMapType::IndexType tempIndex;

  LabelMapIteratorType it( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
    if ( it.Get() != 0 )
      {
      for ( int x=-1; x<=1; x++ )
        {
        tempIndex[0] = it.GetIndex()[0] + x;
      
        for ( int y=-1; y<=1; y++ )
          {
          tempIndex[1] = it.GetIndex()[1] + y;
          
          for ( int z=-1; z<=1; z++ )
            {
            tempIndex[2] = it.GetIndex()[2] + z;

            if ( this->GetOutput()->GetBufferedRegion().IsInside( tempIndex ) )
              {
              if ( this->GetOutput()->GetPixel( tempIndex ) == 0 && this->GetInput()->GetPixel( tempIndex ) <= threshold 
                   && this->m_AirwayLabelMap->GetPixel( tempIndex ) == 0 )
                {
                if ( it.Get() == static_cast< unsigned short >( cip::LEFTLUNG ) )
                  {
                  prevLeftIndicesVec.push_back( tempIndex );
                  }
                if ( it.Get() == static_cast< unsigned short >( cip::RIGHTLUNG ) )
                  {
                  prevRightIndicesVec.push_back( tempIndex );
                  }
                if ( it.Get() == static_cast< unsigned short >( cip::WHOLELUNG ) )
                  {
                  prevWholeIndicesVec.push_back( tempIndex );
                  }
                }
              }
            }
          }
        }
      }

    ++it;
    }

  while ( prevLeftIndicesVec.size() > 0 || prevRightIndicesVec.size() > 0 || prevWholeIndicesVec.size() > 0 )
    {
    tracker->FillBuffer( 0 );

    for ( unsigned int i=0; i<prevRightIndicesVec.size(); i++ )
      {
      this->GetOutput()->SetPixel( prevRightIndicesVec[i], static_cast< unsigned short >( cip::RIGHTLUNG ) );

      for ( int x=-1; x<=1; x++ )
        {
        tempIndex[0] = prevRightIndicesVec[i][0] + x;
        
        for ( int y=-1; y<=1; y++ )
          {
          tempIndex[1] = prevRightIndicesVec[i][1] + y;
          
          for ( int z=-1; z<=1; z++ )
            {
            tempIndex[2] = prevRightIndicesVec[i][2] + z;
            
            if ( this->GetOutput()->GetBufferedRegion().IsInside( tempIndex ) )
              {
              if ( (this->GetOutput()->GetPixel( tempIndex ) == 0 || this->GetOutput()->GetPixel( tempIndex ) == static_cast< unsigned short >( cip::WHOLELUNG ))
                   && this->GetInput()->GetPixel( tempIndex ) <= threshold && this->m_AirwayLabelMap->GetPixel( tempIndex ) == 0 )
                {
                if ( this->GetOutput()->GetPixel( prevRightIndicesVec[i] ) == static_cast< unsigned short >( cip::RIGHTLUNG ) && 
                     tracker->GetPixel( tempIndex ) == 0 )
                  {
                  currRightIndicesVec.push_back( tempIndex );
                  tracker->SetPixel( tempIndex, 1 );
                  }
                }
              }
            }
          }
        }
      }
    prevRightIndicesVec.clear();
    for ( unsigned int i=0; i<currRightIndicesVec.size(); i++ )
      {
      prevRightIndicesVec.push_back( currRightIndicesVec[i] );
      }
    currRightIndicesVec.clear();
    
    for ( unsigned int i=0; i<prevLeftIndicesVec.size(); i++ )
      {
      this->GetOutput()->SetPixel( prevLeftIndicesVec[i], static_cast< unsigned short >( cip::LEFTLUNG ) );
      
      for ( int x=-1; x<=1; x++ )
        {
        tempIndex[0] = prevLeftIndicesVec[i][0] + x;
        
        for ( int y=-1; y<=1; y++ )
          {
          tempIndex[1] = prevLeftIndicesVec[i][1] + y;
          
          for ( int z=-1; z<=1; z++ )
            {
            tempIndex[2] = prevLeftIndicesVec[i][2] + z;
            
            if ( this->GetOutput()->GetBufferedRegion().IsInside( tempIndex ) )
              {
              if ( (this->GetOutput()->GetPixel( tempIndex ) == 0 || this->GetOutput()->GetPixel( tempIndex ) == static_cast< unsigned short >( cip::WHOLELUNG ))
                   && this->GetInput()->GetPixel( tempIndex ) <= threshold && this->m_AirwayLabelMap->GetPixel( tempIndex ) == 0 )
                {
                if ( this->GetOutput()->GetPixel( prevLeftIndicesVec[i] ) == static_cast< unsigned short >( cip::LEFTLUNG ) &&
                     tracker->GetPixel( tempIndex ) == 0 )
                  {
                  currLeftIndicesVec.push_back( tempIndex );
                  tracker->SetPixel( tempIndex, 1 );
                  }
                }
              }
            }
          }
        }
      }
    prevLeftIndicesVec.clear();
    for ( unsigned int i=0; i<currLeftIndicesVec.size(); i++ )
      {
      prevLeftIndicesVec.push_back( currLeftIndicesVec[i] );
      }
    currLeftIndicesVec.clear();
    
    for ( unsigned int i=0; i<prevWholeIndicesVec.size(); i++ )
      {
      this->GetOutput()->SetPixel( prevWholeIndicesVec[i], static_cast< unsigned short >( cip::WHOLELUNG ) );
      
      for ( int x=-1; x<=1; x++ )
        {
        tempIndex[0] = prevWholeIndicesVec[i][0] + x;
        
        for ( int y=-1; y<=1; y++ )
          {
          tempIndex[1] = prevWholeIndicesVec[i][1] + y;
          
          for ( int z=-1; z<=1; z++ )
            {
            tempIndex[2] = prevWholeIndicesVec[i][2] + z;
            
            if ( this->GetOutput()->GetBufferedRegion().IsInside( tempIndex ) )
              {
              if ( this->GetOutput()->GetPixel( tempIndex ) == 0 && this->GetInput()->GetPixel( tempIndex ) <= threshold 
                   && this->m_AirwayLabelMap->GetPixel( tempIndex ) == 0 )
                {
                if ( this->GetOutput()->GetPixel( prevWholeIndicesVec[i] ) == static_cast< unsigned short >( cip::WHOLELUNG ) &&
                     tracker->GetPixel( tempIndex ) == 0 )
                  {
                  currWholeIndicesVec.push_back( tempIndex );
                  tracker->SetPixel( tempIndex, 1 );
                  }
                }
              }
            }
          }
        }
      }
    prevWholeIndicesVec.clear();
    for ( unsigned int i=0; i<currWholeIndicesVec.size(); i++ )
      {
      prevWholeIndicesVec.push_back( currWholeIndicesVec[i] );
      }
    currWholeIndicesVec.clear();
    }
}


template < class TInputImage >
void
CIPPartialLungLabelMapImageFilter< TInputImage >
::CloseLabelMap( unsigned short closeLabel )
{
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

  LabelMapIteratorType eIt( eroder->GetOutput(), eroder->GetOutput()->GetBufferedRegion() );
  LabelMapIteratorType mIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

  eIt.GoToBegin();
  mIt.GoToBegin();
  while ( !mIt.IsAtEnd() )
    {
    if ( eIt.Get() != 0 )
      {
      mIt.Set( eIt.Get() );
      }

    ++eIt;
    ++mIt;
    }
}


template < class TInputImage >
void
CIPPartialLungLabelMapImageFilter< TInputImage >
::RecordAndRemoveAirways( LabelMapType::Pointer airwayLabelMap )
{
  cip::ChestConventions conventions;
  
//  this->m_AirwayLabelMap = airwayLabelMap;

 unsigned short airwayLabel = conventions.GetValueFromChestRegionAndType( static_cast< unsigned char >( cip::UNDEFINEDREGION ), static_cast< unsigned char >( cip::AIRWAY ) );

  this->m_AirwayLabelMap->SetRegions( this->GetInput()->GetBufferedRegion().GetSize() );
  this->m_AirwayLabelMap->Allocate();
  this->m_AirwayLabelMap->FillBuffer( 0 );

  LabelMapType::IndexType index;

  LabelMapIteratorType aIt( airwayLabelMap, airwayLabelMap->GetBufferedRegion() );
  LabelMapIteratorType maIt( this->m_AirwayLabelMap, this->m_AirwayLabelMap->GetBufferedRegion() );

  aIt.GoToBegin();
  maIt.GoToBegin();
  while ( !aIt.IsAtEnd() )
    {
    if ( aIt.Get() != 0 )
      {
      this->m_AirwayLabelMap->SetPixel( aIt.GetIndex(), airwayLabel );
      this->GetOutput()->SetPixel( aIt.GetIndex(), 0 );
      }

    ++maIt;
    ++aIt;
    }

  //
  // The airway segmentation produced by the region growing
  // algorithm may not get all the airway voxels present in our
  // current mask. To make sure the airways are cleaned up, zero
  // out all the voxels around the airway segmentation. A 7x7x7
  // neighborhood is reasonable
  //
  for ( unsigned int i=0; i<1; i++ )
    {
    aIt.GoToBegin();
    while ( !aIt.IsAtEnd() )
      {
      if ( aIt.Get() != 0 )
        {       
        for ( int x=-1; x<=1; x++ )
          {
          index[0] = aIt.GetIndex()[0] + x;
          
          for ( int y=-1; y<=1; y++ )
            {
            index[1] = aIt.GetIndex()[1] + y;
            
            for ( int z=-1; z<=1; z++ )
              {
              index[2] = aIt.GetIndex()[2] + z;
              
              if ( this->GetOutput()->GetBufferedRegion().IsInside( index ) )
                {               
                if ( this->GetInput()->GetPixel( index ) <= this->m_OtsuThreshold )
                  {
                  this->GetOutput()->SetPixel( index, 0 );
                  this->m_AirwayLabelMap->SetPixel( index, airwayLabel );
                  }             
                }
              }
            }
          }
        }
      
      ++aIt;
      }

    aIt.GoToBegin();
    maIt.GoToBegin();
    while ( !maIt.IsAtEnd() )
      {
      aIt.Set( maIt.Get() );

      ++aIt;
      ++maIt;
      }
    }
}


template < class TInputImage >
void
CIPPartialLungLabelMapImageFilter< TInputImage >
::SetClosingNeighborhood( unsigned long* neighborhood )
{
  this->m_ClosingNeighborhood[0] = neighborhood[0];
  this->m_ClosingNeighborhood[1] = neighborhood[1];
  this->m_ClosingNeighborhood[2] = neighborhood[2];
}


template < class TInputImage >
void
CIPPartialLungLabelMapImageFilter< TInputImage >
::AddAirwaySegmentationSeed( OutputImageType::IndexType index )
{
  this->m_AirwaySegmentationSeedVec.push_back( index );
}


template < class TInputImage >
void
CIPPartialLungLabelMapImageFilter< TInputImage >
::SetAirwayLabelMap( OutputImageType::Pointer airwayLabelMap )
{
  this->m_AirwayLabelMap = airwayLabelMap;
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
  //
  // Set the output to the helper mask 
  //
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

  //
  // The 'ApplyOtsuThreshold' routine typically sets the otsu
  // threshold. Because it is not called in favor of calling this
  // method, we have to set the threshold to a reasonable level
  //
  this->m_OtsuThreshold = -400;  
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
CIPPartialLungLabelMapImageFilter< TInputImage >
::ApplyOtsuThreshold()
{
  //
  // The first step is to run Otsu threshold on the input data.  This
  // classifies each voxel as either "body" or "air"
  //
  typename OtsuThresholdType::Pointer otsuThreshold = OtsuThresholdType::New();
    otsuThreshold->SetInput( this->GetInput() );
    otsuThreshold->Update();

  this->m_OtsuThreshold = otsuThreshold->GetThreshold();

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

  LabelMapExtractorType::Pointer sliceExtractor = LabelMapExtractorType::New();
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


template < class TInputImage >
void
CIPPartialLungLabelMapImageFilter< TInputImage >
::RemoveTracheaAndMainBronchi()
{
  typedef itk::ImageFileWriter< LabelMapType > WriterType;

  cip::ChestConventions conventions;
  
  LabelMapType::IndexType index;

  LabelMapType::SizeType    size    = this->GetOutput()->GetBufferedRegion().GetSize();
  LabelMapType::SpacingType spacing = this->GetInput()->GetSpacing();

  this->m_AirwayLabelMap->SetRegions( this->GetInput()->GetBufferedRegion().GetSize() );
  this->m_AirwayLabelMap->Allocate();
  this->m_AirwayLabelMap->FillBuffer( 0 );

  double area = spacing[0]*spacing[1];
  double areaThreshold = 500; // Set heuristically (units of mm)

  unsigned short airwayLabel = conventions.GetValueFromChestRegionAndType( static_cast< unsigned char >( cip::UNDEFINEDREGION ), 
                                                                           static_cast< unsigned char >( cip::AIRWAY ) );

  //
  // DEBUG: Applying otsu threshold here for helper assisted segmentation.
  //
  typename OtsuThresholdType::Pointer otsuThreshold = OtsuThresholdType::New();
    otsuThreshold->SetInput( this->GetInput() );
    otsuThreshold->Update();

  //
  // Erode the current mask with a 5x5x5 structuring element. This
  // will serve to break some of the attachments of the trachea / main
  // bronchi structures to the lung field region.
  //
  Element3DType structuringElement3D;
    structuringElement3D.SetRadius( 2 );
    structuringElement3D.CreateStructuringElement();

  Erode3DType::Pointer erode3D = Erode3DType::New();
//    erode3D->SetInput( this->GetOutput() );
    erode3D->SetInput( otsuThreshold->GetOutput() );
    erode3D->SetKernel( structuringElement3D );
    erode3D->SetErodeValue( static_cast< unsigned short >( cip::WHOLELUNG ) );
    erode3D->Update();

  //
  // Go slice by slice, run connected components, relabel, and then
  // retain all components below a size threshold. These are assumed
  // to belong to the trachea / main bronchi.
  //
  LabelMapSliceType::Pointer slice = LabelMapSliceType::New();    

  for ( unsigned int i=0; i<size[2]; i++ )
    {
    this->ExtractLabelMapSlice( erode3D->GetOutput(), slice, i );

    ConnectedComponent2DType::Pointer connectedComponent2D = ConnectedComponent2DType::New();
      connectedComponent2D->SetInput( slice );
      connectedComponent2D->FullyConnectedOn();
    try
      {
      connectedComponent2D->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught while updating connected component filter:";
      std::cerr << excp << std::endl;
      }    
    
    Relabel2DType::Pointer relabeler2D = Relabel2DType::New();
      relabeler2D->SetInput( connectedComponent2D->GetOutput() );
    try
      {
      relabeler2D->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught while relabeling:";
      std::cerr << excp << std::endl;
      }

    LabelMapSliceIteratorType sIt2D( relabeler2D->GetOutput(), relabeler2D->GetOutput()->GetBufferedRegion() );

    sIt2D.GoToBegin();
    while ( !sIt2D.IsAtEnd() )
      {
      if ( sIt2D.Get() != 0 )
        {
        if ( static_cast< double >( relabeler2D->GetSizeOfObjectsInPixels()[sIt2D.Get()-1] )*area < areaThreshold )
          {
          index[0] = sIt2D.GetIndex()[0];
          index[1] = sIt2D.GetIndex()[1];
          index[2] = i;
          
          this->GetOutput()->SetPixel( index, 0 );
          this->m_AirwayLabelMap->SetPixel( index, airwayLabel );
          }
        }

      ++sIt2D;
      }
    }

  //
  // Dilate in 3D the structures we accumlated in the last step. This
  // serves to recover the erosion operation we did initially.
  //
  Dilate3DType::Pointer dilate3D = Dilate3DType::New();
    dilate3D->SetInput( this->m_AirwayLabelMap );
    dilate3D->SetKernel( structuringElement3D );
    dilate3D->SetDilateValue( airwayLabel );
    dilate3D->Update();

  //
  // Now perform conditional dilation to connect regions of the
  // trachea / main bronchi that may be disconnected
  //
  LabelMapIteratorType dIt( dilate3D->GetOutput(), dilate3D->GetOutput()->GetBufferedRegion() );
  std::vector< LabelMapType::IndexType > indicesVec;

//  std::cout << "---Performing conditional dilation..." << std::endl;
  for ( unsigned int i=0; i<2; i++ )
    {

    dIt.GoToBegin();
    while ( !dIt.IsAtEnd() )
      {
      if ( dIt.Get() !=0 )
        {
        for ( int x=-1; x<=1; x++ )
          {
          index[0] = dIt.GetIndex()[0] + x;

          for ( int y=-1; y<=1; y++ )
            {
            index[1] = dIt.GetIndex()[1] + y;

            for ( int z=-1; z<=1; z++ )
              {
              index[2] = dIt.GetIndex()[2] + z;
              
              if ( dilate3D->GetOutput()->GetBufferedRegion().IsInside( index ) )
                {
                if ( this->GetInput()->GetPixel( index ) < -800 )
                  {
                  indicesVec.push_back( index );         
                  }
                }
              }
            }
          }
        }

      ++dIt;
      }

    for ( unsigned int j=0; j<indicesVec.size(); j++ )
      {
      dilate3D->GetOutput()->SetPixel( indicesVec[j], airwayLabel );
      }
    indicesVec.clear();
    }

  // 
  // Run connected components followed by relabeling on the dilated
  // structure we just produced. The largest component is assumed to
  // be the trachea and main bronchi. It is these voxels we'll remove
  // from the lung mask image
  //
  ConnectedComponent3DType::Pointer connectedComponent3D = ConnectedComponent3DType::New();
    connectedComponent3D->SetInput( dilate3D->GetOutput() );
    connectedComponent3D->FullyConnectedOn();
  try
    {
    connectedComponent3D->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught while updating connected component filter:";
    std::cerr << excp << std::endl;
    }    
    
  Relabel3DType::Pointer relabeler3D = Relabel3DType::New();
    relabeler3D->SetInput( connectedComponent3D->GetOutput() );
  try
    {
    relabeler3D->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught while relabeling:";
    std::cerr << excp << std::endl;
    }

  ComponentIteratorType rIt( relabeler3D->GetOutput(), relabeler3D->GetOutput()->GetBufferedRegion() );
  LabelMapIteratorType  oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );
  LabelMapIteratorType  aIt( this->m_AirwayLabelMap, this->m_AirwayLabelMap->GetBufferedRegion() );

  rIt.GoToBegin();
  oIt.GoToBegin();
  aIt.GoToBegin();
  while ( !oIt.IsAtEnd() )
    {
    if ( rIt.Get() == 1 )
      {
      oIt.Set( 0 );
      }
    else
      {
      aIt.Set( 0 );
      }
    
    ++aIt;
    ++rIt;
    ++oIt;
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

    this->ExtractLabelMapSlice( this->GetOutput(), slice, whichSlice );

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
        int diff = vcl_abs( static_cast< int >( (*mapIt).first ) - static_cast< int >( middleLocation ) );

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
  os << indent << "MinAirwayVolume: " << this->m_MinAirwayVolume << std::endl;
  os << indent << "MaxAirwayVolumeIncreaseRate: " << this->m_MaxAirwayVolumeIncreaseRate << std::endl;
  os << indent << "ExponentialCoefficient: " << this->m_ExponentialCoefficient << std::endl;
  os << indent << "ExponentialTimeConstant: " << this->m_ExponentialTimeConstant << std::endl;
  os << indent << "LeftRightLungSplitRadius: " << this->m_LeftRightLungSplitRadius << std::endl;
  os << indent << "AggressiveLeftRightSplitter: " << this->m_AggressiveLeftRightSplitter << std::endl;
  os << indent << "HeadFirst: " << this->m_HeadFirst << std::endl;
  os << indent << "Supine: " << this->m_Supine << std::endl;       
}

} // end namespace itk

#endif

