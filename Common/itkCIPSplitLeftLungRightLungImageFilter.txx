#ifndef _itkCIPSplitLeftLungRightLungImageFilter_txx
#define _itkCIPSplitLeftLungRightLungImageFilter_txx

#include "itkCIPSplitLeftLungRightLungImageFilter.h"
#include "itkRGBPixel.h"
#include "itkLabelToRGBImageFilter.h"
#include "cipExceptionObject.h"
#include "itkRescaleIntensityImageFilter.h"

namespace itk
{

template< class TInputImage >
CIPSplitLeftLungRightLungImageFilter< TInputImage >
::CIPSplitLeftLungRightLungImageFilter()
{
  this->m_ExponentialCoefficient      = 200;
  this->m_ExponentialTimeConstant     = -700;
  this->m_LeftRightLungSplitRadius    = 2;
  this->m_AggressiveLeftRightSplitter = false;
  this->m_UseLocalGraphROI            = true;
}


template< class TInputImage >
void
CIPSplitLeftLungRightLungImageFilter< TInputImage >
::GenerateData()
{
  if ( this->m_LungLabelMap.IsNull() )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, "CIPSplitLeftLungRightLungImageFilter::GenerateData()",
      				  "Lung label map not set" );
    }

  typename Superclass::InputImageConstPointer inputPtr  = this->GetInput();
  typename Superclass::OutputImagePointer     outputPtr = this->GetOutput(0);
    outputPtr->SetRequestedRegion( inputPtr->GetRequestedRegion() );
    outputPtr->SetBufferedRegion( inputPtr->GetBufferedRegion() );
    outputPtr->SetLargestPossibleRegion( inputPtr->GetLargestPossibleRegion() );
    outputPtr->Allocate();

  // Fill the output image with the contents of the input image. We will also
  // get the max intensity of the input image beneath the mask. We'll use
  // this to threshold the input image -- all bright voxels should be treated the 
  // same in order to prevent paths from taking odd detours.
  LabelMapIteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );
  LabelMapIteratorType lIt( this->m_LungLabelMap, this->m_LungLabelMap->GetBufferedRegion() );
  InputIteratorType    iIt( this->GetInput(), this->GetInput()->GetBufferedRegion() );

  InputPixelType maxIntensity = itk::NumericTraits< InputPixelType >::min();

  oIt.GoToBegin();
  lIt.GoToBegin();
  iIt.GoToBegin();
  while ( !lIt.IsAtEnd() )
    {
    oIt.Set( lIt.Get() );

    if ( iIt.Get() > maxIntensity && lIt.Get() > 0 )
      {
	maxIntensity = iIt.Get();
      }

    ++oIt;
    ++lIt;
    ++iIt;
    }

  LabelMapType::SizeType size = this->GetOutput()->GetBufferedRegion().GetSize();

  int minX = size[0]/3;
  int maxX = size[0]-size[0]/3;
  int minY = 0;
  int maxY = size[1]-1;

  // We will look through each slice, test whether or not there appears to be
  // a connection and, if so, we will split. The region to consider for a given
  // split will depend on the region used in the previous slice (provided the
  // previous slice needed a split). For the first slice determined to need
  // a split, we will initiate the min cost path search at the top middle of 
  // the ROI and designate as the endpoint of the path the lower middle of
  // the ROI.
  typename InputImageSliceType::IndexType searchStartIndex;
    searchStartIndex[0] = (maxX + minX)/2;
    searchStartIndex[1] = minY;

  typename InputImageSliceType::IndexType searchEndIndex;        
    searchEndIndex[0] = (maxX + minX)/2;
    searchEndIndex[1] = maxY;

  LabelMapType::IndexType index3D;

  // We will keep track of the path indices used to split the
  // previous slice. To insure that the left and right lungs are
  // split in 3D, we will zero-out all label map points falling
  // within the region between the path in the current slice and the
  // path in the previous slice.
  std::map< short, short > previousPathMap;

  int previousMinY = size[1];
  int previousMaxY = 0;

  for ( unsigned int i=0; i<size[2]; i++ )
    {
    bool merged = this->GetLungsMergedInSliceRegion( size[0]/3, 0, size[0]/3, size[1], i ); 

    // We will only use the local search region provided that the 
    // previous slice was originally merged and then successfully
    // split. If the current slice is already split, and if we're 
    // currently using a local search, we want to indicate that 
    // the full, default search ROI should be used for the next slice
    if ( !merged and this->m_UseLocalGraphROI )
      {
	this->m_UseLocalGraphROI = false;
      }

    // If the current slice is merged and we're not supposed to use the
    // local search region, then set the default search region for
    // this slice.
    if ( merged and !this->m_UseLocalGraphROI ) 
      {
	this->SetDefaultGraphROIAndSearchIndices( i );
      }

    bool attemptSplit = true;
    while ( merged && attemptSplit )
      {
	std::cout << "i:\t" << i << "\t merged" << std::endl;
	std::cout << "ROI start:\t" << this->m_GraphROIStartIndex << std::endl;
	std::cout << "ROI size:\t" << this->m_GraphROISize << std::endl;
	std::cout << "Start search:\t" << this->m_StartSearchIndex << std::endl;
	std::cout << "End search:\t" << this->m_EndSearchIndex << std::endl;
	this->FindMinCostPath();
	std::cout << "a" << std::endl;
	this->EraseConnection( i );
	std::cout << "b" << std::endl;
	merged = this->GetLungsMergedInSliceRegion( size[0]/3, 0, size[0]/3, size[1], i );
	std::cout << "c" << std::endl;
	if ( !merged )
	  {
	    // Set the local search region for the next slice
	    this->SetLocalGraphROIAndSearchIndices( i+1 );
	  }
	else if ( merged and this->m_UseLocalGraphROI )
	  {
	    this->SetDefaultGraphROIAndSearchIndices( i );
	  }
	else
	  {
	    // If we're here, we've tried everything and the slice is
	    // still merged. Admit defeat and move on to the next
	    // slice. The overall result will not be split, but
	    // hopefully we've improved things
	    attemptSplit = false;
	  }
	std::cout << "d" << std::endl;

    /* 	std::cout << "Merged" << std::endl; */
    /*   int numSplitAttempts = 0; */
      
    /*   while ( merged && numSplitAttempts < 3 ) */
    /*     { */
    /*     numSplitAttempts++; */
        
    /*     typename InputImageType::SizeType roiSize; */
    /*       roiSize[0] = maxX - minX + 20; */

    /*     if ( maxX - minX + 20 < 0 ) */
    /*       { */
    /*       roiSize[0] = 0; */
    /*       } */
    /*     if ( roiSize[0] > size[0] ) */
    /*       { */
    /*       roiSize[0] = size[0]; */
    /*       } */
        
    /*     roiSize[1] = maxY - minY + 20; */
    /*     if ( maxY - minY + 20 < 0 ) */
    /*       { */
    /*       roiSize[1] = 0; */
    /*       } */
    /*     if ( roiSize[1] > size[1] ) */
    /*       { */
    /*       roiSize[1] = size[1]; */
    /*       } */

    /*     roiSize[2] = 0; */
        
    /*     typename InputImageType::IndexType roiStartIndex; */
    /*       roiStartIndex[0] = minX - 10; */
        
    /*     if ( roiStartIndex[0] < 0 ) */
    /*       { */
    /*       roiStartIndex[0] = 0; */
    /*       } */
        
    /*     roiStartIndex[1] = minY - 10; */
    /*     if ( roiStartIndex[1] < 0 ) */
    /*       { */
    /*       roiStartIndex[1] = 0; */
    /*       } */
        
    /*     roiStartIndex[2] = i; */
        
    /*     typename InputImageType::RegionType roiRegion; */
    /*       roiRegion.SetSize( roiSize ); */
    /*       roiRegion.SetIndex( roiStartIndex ); */
        
    /*     typename InputExtractorType::Pointer roiExtractor = InputExtractorType::New(); */
    /*  	  roiExtractor->SetInput( thresholder->GetOutput() ); */
    /*       roiExtractor->SetExtractionRegion( roiRegion ); */
    /*       roiExtractor->Update(); */

    	  if ( i == 307 )
    	    {
    	      typedef itk::Image< unsigned char, 2 > UCharSliceType;
    	      typedef itk::RescaleIntensityImageFilter< InputImageSliceType, UCharSliceType > RescaleType;
    	      typedef itk::ImageFileWriter< UCharSliceType > UCharSliceWriterType;

	      typename InputImageType::Pointer dummyImage = InputImageType::New();
	      dummyImage->SetRegions( this->GetInput()->GetBufferedRegion().GetSize() );
	      dummyImage->Allocate();
	      dummyImage->FillBuffer( 0 );

	      typedef itk::ImageRegionIterator< InputImageType > DummyIteratorType;

	      InputIteratorType iIt( this->GetInput(), this->GetInput()->GetBufferedRegion() );
	      DummyIteratorType dIt( dummyImage, dummyImage->GetBufferedRegion() );

	      iIt.GoToBegin();
	      dIt.GoToBegin();
	      while ( !dIt.IsAtEnd() )
	      	{
	      	  dIt.Set( iIt.Get() );
		  
	      	  ++dIt;
	      	  ++iIt;
	      	}

	      typename InputImageType::IndexType sliceIndex;
	      for ( unsigned int k=0; k<this->m_MinCostPathIndices.size(); k++ )
	      	{
		  typename InputImageType::IndexType blahIndex;
  		    blahIndex[0] = this->m_MinCostPathIndices[k][0];
		    blahIndex[1] = this->m_MinCostPathIndices[k][1];
		    blahIndex[2] = i;

	      	  dummyImage->SetPixel( blahIndex, 500 );
	      	}

	      typename InputImageType::RegionType roiRegion;
	      roiRegion.SetSize( this->m_GraphROISize );
	      roiRegion.SetIndex( this->m_GraphROIStartIndex );

    	      typename InputExtractorType::Pointer roiExtractor2 = InputExtractorType::New();
    	      roiExtractor2->SetInput( dummyImage );
    	      roiExtractor2->SetExtractionRegion( roiRegion );
    	      roiExtractor2->Update();

    	      typename RescaleType::Pointer rescaler = RescaleType::New();
    	      rescaler->SetInput( roiExtractor2->GetOutput() );
    	      rescaler->SetOutputMinimum( 0 );
    	      rescaler->SetOutputMaximum( 255 );
	      
    	      std::cout << "Writing slice..." << std::endl;
    	      typename UCharSliceWriterType::Pointer sliceWriter = UCharSliceWriterType::New();
    	      sliceWriter->SetInput( rescaler->GetOutput() );
    	      sliceWriter->SetFileName( "/Users/jross/tmp/fooROI.png" );
    	      sliceWriter->Write();
    	      std::cout << "done" << std::endl;
    	    }

    /* 	//searchStartIndex[0] = roiStartIndex[0] + roiSize[0]/2; */
    /*     searchStartIndex[1] = roiStartIndex[1]; */
        
    /*     //searchEndIndex[0] = roiStartIndex[0] + roiSize[0]/2; */
    /*     searchEndIndex[1] = roiStartIndex[1] + roiSize[1] - 1; */

    /*     // Set the startIndex to the the top-center of the ROI and the */
    /*     // endIndex to be the bottom-center of the ROI */
    /*     std::vector< LabelMapSliceType::IndexType > pathIndices = this->GetMinCostPath( roiExtractor->GetOutput(), searchStartIndex, searchEndIndex ); */
        
    /*     minX = size[0]; */
    /*     maxX = 0; */
    /*     minY = size[1]; */
    /*     maxY = 0; */
        
    /*     bool foundMinMax = false; */
    /*     for ( unsigned int j=0; j<pathIndices.size(); j++ ) */
    /*       { */
    /*       index3D[0] = (pathIndices[j])[0]; */
    /*       index3D[1] = (pathIndices[j])[1]; */
          
    /*       if ( this->GetOutput()->GetPixel( index3D ) !=0 ) */
    /*         { */
    /*         foundMinMax = true; */
            
    /*         if ( index3D[0] < minX ) */
    /*           { */
    /* 		minX = index3D[0]; */
    /*           } */
    /*         if ( index3D[0] > maxX ) */
    /*           { */
    /* 		maxX = index3D[0]; */
    /*           } */
    /*         if ( index3D[1] < minY ) */
    /*           { */
    /* 		minY = index3D[1]; */

    /* 		// This is the uppermost path location found so  */
    /* 		// far. Initiate the search on the next slice */
    /* 		// at this location	        */
    /* 		searchStartIndex[0] = index3D[1]; */
    /*           } */
    /*         if ( index3D[1] > maxY ) */
    /*           { */
    /* 		maxY = index3D[1];  */

    /* 		// This is the lowermost path location found so  */
    /* 		// far. Terminate the search on the next slice */
    /* 		// at this location	        */
    /* 		searchEndIndex[0] = index3D[1]; */
    /*           } */
    /*         }           */
    /*       } */
        
    /*     if ( !foundMinMax || this->m_AggressiveLeftRightSplitter ) */
    /*       { */
    /*       minX = size[0]/3; */
    /*       maxX = size[0]-size[0]/3; */
    /*       minY = 0; */
    /*       maxY = size[1]-1; */
    /*       } */
        
    /*     for ( unsigned int j=0; j<pathIndices.size(); j++ ) */
    /*       { */
    /*       LabelMapType::IndexType tempIndex; */
    /*         tempIndex[2] = i; */

    /*       int currentX  = (pathIndices[j])[0]; */
    /*       int currentY  = (pathIndices[j])[1]; */

    /*       int startX = currentX - this->m_LeftRightLungSplitRadius; */
    /*       int endX   = currentX + this->m_LeftRightLungSplitRadius; */

    /*       if ( previousPathMap.size() > 0 ) */
    /*         { */
    /*         if ( currentY >= previousMinY && currentY <= previousMaxY ) */
    /*           { */
    /*           // Determine the extent in the x-direction to zero-out */
    /*           int previousX = previousPathMap[(pathIndices[j])[1]]; */
              
    /*           if ( previousX - currentX < 0 ) */
    /*             { */
    /*             startX = previousX - this->m_LeftRightLungSplitRadius; */
    /*             endX   = currentX  + this->m_LeftRightLungSplitRadius; */
    /*             } */
    /*           else  */
    /*             { */
    /*             startX = currentX  - this->m_LeftRightLungSplitRadius; */
    /*             endX   = previousX + this->m_LeftRightLungSplitRadius; */
    /*             }                 */
    /*           } */
    /*         } */

    /*       tempIndex[1] = (pathIndices[j])[1]; */
    /*       for ( int x=startX; x<=endX; x++ ) */
    /*         { */
    /*         tempIndex[0] = x; */

    /*         if ( this->GetOutput()->GetBufferedRegion().IsInside( tempIndex ) ) */
    /*           { */
    /*           if ( this->GetOutput()->GetPixel( tempIndex ) != 0 ) */
    /*             { */
    /*             this->m_RemovedIndices.push_back( tempIndex ); */
    /*             } */
    /*           this->GetOutput()->SetPixel( tempIndex, 0 ); */
    /*           } */
    /*         } */

    /*       for ( int y=-this->m_LeftRightLungSplitRadius; y<=this->m_LeftRightLungSplitRadius; y++ ) */
    /*         { */
    /*         tempIndex[1] = (pathIndices[j])[1] + y;             */

    /*         for ( int x=-this->m_LeftRightLungSplitRadius; x<=this->m_LeftRightLungSplitRadius; x++ ) */
    /*           { */
    /*           tempIndex[0] = (pathIndices[j])[0] + x; */
              
    /*           if ( this->GetOutput()->GetBufferedRegion().IsInside( tempIndex ) ) */
    /*             { */
    /*             if ( x==this->m_LeftRightLungSplitRadius || x==-this->m_LeftRightLungSplitRadius ||  */
    /*                  y==this->m_LeftRightLungSplitRadius || y==-this->m_LeftRightLungSplitRadius ) */
    /*               { */
    /*               if ( this->GetType( tempIndex ) == static_cast< unsigned char >( cip::VESSEL ) ) */
    /*                 { */
    /*                 if ( this->GetOutput()->GetPixel( tempIndex ) != 0 ) */
    /*                   { */
    /*                   this->m_RemovedIndices.push_back( tempIndex ); */
    /*                   } */
    /*                 this->GetOutput()->SetPixel( tempIndex, 0 ); */
    /*                 } */
    /*               } */
    /*             else */
    /*               { */
    /*               if ( this->GetOutput()->GetPixel( tempIndex ) != 0 ) */
    /*                 { */
    /*                 this->m_RemovedIndices.push_back( tempIndex ); */
    /*                 } */
    /*               this->GetOutput()->SetPixel( tempIndex, 0 ); */
    /*               } */
    /*             } */
    /*           } */
    /*         }           */
    /*       } */
        
    /*     merged = this->GetLungsMergedInSliceRegion( size[0]/3, 0, size[0]/3, size[1], i );  */
        
    /*     if ( merged ) */
    /*       { */
    /*       minX = size[0]/3; */
    /*       maxX = size[0]-size[0]/3; */
    /*       minY = 0; */
    /*       maxY = size[1]-1; */
    /*       } */
    /*     else */
    /*       { */
    /*       // Assign the map values to use while splitting the next */
    /*       // slice  */
    /*       previousPathMap.clear(); */

    /*       previousMinY = size[1]; */
    /*       previousMaxY = 0; */

    /*       for ( unsigned int i=0; i<pathIndices.size(); i++ ) */
    /*         { */
    /*         previousPathMap[(pathIndices[i])[1]] = (pathIndices[i])[0]; */

    /*         if ( (pathIndices[i])[1] < previousMinY ) */
    /*           { */
    /*           previousMinY = (pathIndices[i])[1]; */
    /*           } */
    /*         if ( (pathIndices[i])[1] > previousMaxY ) */
    /*           { */
    /*           previousMaxY = (pathIndices[i])[1]; */
    /*           } */
    /*         } */
    /*       } */
    /*     } */
    /*   } */
    /* else */
    /*   { */
    /*   minX = size[0]/3; */
    /*   maxX = size[0]-size[0]/3; */
    /*   minY = 0; */
    /*   maxY = size[1]-1; */

    /*   previousPathMap.clear(); */

    /*   previousMinY = size[1]; */
    /*   previousMaxY = 0; */
      }
    }
}



/**
 * 
 */
template< class TInputImage >
unsigned char 
CIPSplitLeftLungRightLungImageFilter< TInputImage >
::GetType( OutputImageType::IndexType index )
{
  unsigned short currentValue = this->GetOutput()->GetPixel( index );

  unsigned char typeValue = 0;

  for ( int i=15; i>=0; i-- )
    {
    int power = static_cast< int >( pow( 2, i ) );

    if ( power <= currentValue )
      {
      if ( i >= 8 )
        {
        typeValue += static_cast< unsigned char >( pow( 2, i-8 ) );
        }
      
      currentValue = currentValue % power;
      }
    }

  return typeValue;
}



/**
 * 
 */
template< class TInputImage >
bool
CIPSplitLeftLungRightLungImageFilter< TInputImage >
::GetLungsMergedInSliceRegion( int startX, int startY, int sizeX, int sizeY, int whichSlice )
{
  std::cout << "whichSlice:\t" << whichSlice << std::endl;

  LabelMapType::SizeType sliceSize;
    sliceSize[0] = sizeX;
    sliceSize[1] = sizeY;
    sliceSize[2] = 0;

  LabelMapType::IndexType sliceStartIndex;
    sliceStartIndex[0] = startX;
    sliceStartIndex[1] = startY;
    sliceStartIndex[2] = whichSlice;
  
  LabelMapType::RegionType sliceRegion;
    sliceRegion.SetSize( sliceSize );
    sliceRegion.SetIndex( sliceStartIndex );

  LabelMapExtractorType::Pointer sliceROI = LabelMapExtractorType::New();
    sliceROI->SetInput( this->GetOutput() );
    sliceROI->SetExtractionRegion( sliceRegion );
    sliceROI->Update();

  LabelMapSliceType::SizeType roiSize = sliceROI->GetOutput()->GetBufferedRegion().GetSize();

  LabelMapSliceIteratorType sIt( sliceROI->GetOutput(), sliceROI->GetOutput()->GetBufferedRegion() );

  sIt.GoToBegin();
  LabelMapSliceType::IndexType startIndex = sIt.GetIndex();

  // Perform connected components on the region
  ConnectedComponent2DType::Pointer connectedComponent = ConnectedComponent2DType::New();
    connectedComponent->SetInput( sliceROI->GetOutput() );
    connectedComponent->FullyConnectedOn();
    connectedComponent->Update();

  if ( whichSlice == 310 )
    {
      typedef itk::RGBPixel<unsigned char> RGBPixelType;
      typedef itk::Image<RGBPixelType, 2>  RGBImageType;

      typedef itk::LabelToRGBImageFilter<LabelMapSliceType, RGBImageType> RGBFilterType;
      RGBFilterType::Pointer rgb = RGBFilterType::New();
      rgb->SetInput( sliceROI->GetOutput() );
      rgb->Update();

      std::cout << "writing" << std::endl;
      typedef itk::ImageFileWriter< RGBImageType > WriterType;
      WriterType::Pointer writer = WriterType::New();
      writer->SetFileName( "/Users/jross/tmp/foo4.png" );
      writer->SetInput( rgb->GetOutput() );
      writer->Update();
    }

  // If there is an object that touches both the left border and the
  // right border, then the lungs are merged in this slice.  Test this
  // condition
  std::list< unsigned short >  lefthandLabelList;
  std::list< unsigned short >  righthandLabelList;

  LabelMapSliceType::IndexType index;

  for ( unsigned int i=0; i<roiSize[1]; i++ )
    {
    index[1] = i;
    index[0] = startIndex[0];

    unsigned short value = connectedComponent->GetOutput()->GetPixel( index );

    if ( value != (unsigned short)( 0 ) )
      {
      lefthandLabelList.push_back( value );
      }

     index[0] = startIndex[0] + roiSize[0] - 1;

     value = connectedComponent->GetOutput()->GetPixel( index );

    if ( value != 0 )
      {
      righthandLabelList.push_back( value );
      }
    }

  lefthandLabelList.unique();
  lefthandLabelList.sort();
  lefthandLabelList.unique();

  righthandLabelList.unique();
  righthandLabelList.sort();
  righthandLabelList.unique();

  std::list< unsigned short >::iterator leftIt   = lefthandLabelList.begin();
  std::list< unsigned short >::iterator rightIt  = righthandLabelList.begin();

  for ( unsigned int i=0; i<lefthandLabelList.size(); i++, leftIt++ )
    {
    rightIt  = righthandLabelList.begin();
    for ( unsigned int j=0; j<righthandLabelList.size(); j++, rightIt++ )
      {
      if ( *leftIt == *rightIt )
        {
        return true;
        }
      }
    }
    
  return false;
}


template< class TInputImage >
void CIPSplitLeftLungRightLungImageFilter< TInputImage >
::SetDefaultGraphROIAndSearchIndices( unsigned int z )
{
  this->m_UseLocalGraphROI = false;

  LabelMapType::SizeType size = this->GetOutput()->GetBufferedRegion().GetSize();

  int minX = size[0]/3;
  int maxX = size[0]-size[0]/3;
  int minY = 0;
  int maxY = size[1]-1;

  // Set the start and end search indices
  this->m_StartSearchIndex[0] = (maxX + minX)/2;
  this->m_StartSearchIndex[1] = minY;

  this->m_EndSearchIndex[0] = (maxX + minX)/2;
  this->m_EndSearchIndex[1] = maxY;

  // Set the start index of the graph ROI
  this->m_GraphROIStartIndex[0] = minX - 10;        
  if ( this->m_GraphROIStartIndex[0] < 0 )
    {
      this->m_GraphROIStartIndex[0] = 0;
    }        
  this->m_GraphROIStartIndex[1] = minY - 10;
  if ( this->m_GraphROIStartIndex[1] < 0 )
    {
      this->m_GraphROIStartIndex[1] = 0;
    }
  this->m_GraphROIStartIndex[2] = z;

  // Now set the size of the graph ROI
  this->m_GraphROISize[0] = maxX - minX + 20;
  if ( maxX - minX + 20 < 0 )
    {
      this->m_GraphROISize[0] = 0;
    }
  if ( this->m_GraphROISize[0] > size[0] )
    {
      this->m_GraphROISize[0] = size[0];
    }
  
  this->m_GraphROISize[1] = maxY - minY + 20;
  if ( maxY - minY + 20 < 0 )
    {
      this->m_GraphROISize[1] = 0;
    }
  if ( this->m_GraphROISize[1] > size[1] )
    {
      this->m_GraphROISize[1] = size[1];
    }   
  
  this->m_GraphROISize[2] = 0;        
}


template< class TInputImage >
void CIPSplitLeftLungRightLungImageFilter< TInputImage >
::SetLocalGraphROIAndSearchIndices( unsigned int z )
{
  this->m_UseLocalGraphROI = true;

  LabelMapType::SizeType size = this->GetOutput()->GetBufferedRegion().GetSize();

  int minX = size[0];
  int maxX = 0;
  int minY = size[1];
  int maxY = 0;

  for ( unsigned int i=0; i<this->m_ErasedSliceIndices.size(); i++ )
    {
      if ( this->m_ErasedSliceIndices[i][0] > maxX )
	{
	  maxX = this->m_ErasedSliceIndices[i][0];
	}
      if ( this->m_ErasedSliceIndices[i][0] < minX )
	{
	  minX = this->m_ErasedSliceIndices[i][0];
	}
      if ( this->m_ErasedSliceIndices[i][1] > maxY )
	{
	  maxY = this->m_ErasedSliceIndices[i][1];
	  this->m_EndSearchIndex[0] = this->m_ErasedSliceIndices[i][0];
	}
      if ( this->m_ErasedSliceIndices[i][1] < minY )
	{
	  minY = this->m_ErasedSliceIndices[i][1];
	  this->m_StartSearchIndex[0] = this->m_ErasedSliceIndices[i][0];
	}
    }

  // Set the start index of the graph ROI
  this->m_GraphROIStartIndex[0] = minX - 10;        
  if ( this->m_GraphROIStartIndex[0] < 0 )
    {
      this->m_GraphROIStartIndex[0] = 0;
    }        
  this->m_GraphROIStartIndex[1] = minY - 10;
  if ( this->m_GraphROIStartIndex[1] < 0 )
    {
      this->m_GraphROIStartIndex[1] = 0;
    }
  this->m_GraphROIStartIndex[2] = z;
  this->m_StartSearchIndex[1] = this->m_GraphROIStartIndex[1];

  // Now set the size of the graph ROI
  this->m_GraphROISize[0] = maxX - minX + 20;
  if ( maxX - minX + 20 < 0 )
    {
      this->m_GraphROISize[0] = 0;
    }
  if ( this->m_GraphROISize[0] > size[0] )
    {
      this->m_GraphROISize[0] = size[0];
    }
  
  this->m_GraphROISize[1] = maxY - minY + 20;
  if ( maxY - minY + 20 < 0 )
    {
      this->m_GraphROISize[1] = 0;
    }
  if ( this->m_GraphROISize[1] > size[1] )
    {
      this->m_GraphROISize[1] = size[1];
    }   
  this->m_EndSearchIndex[1] = this->m_GraphROIStartIndex[1] + this->m_GraphROISize[1] - 1;

  this->m_GraphROISize[2] = 0;
}


template< class TInputImage >
void CIPSplitLeftLungRightLungImageFilter< TInputImage >
::EraseConnection( unsigned int z )
{
  if ( this->m_ErasedSliceIndices.size() > 0 )
    {
      this->m_ErasedSliceIndices.clear();
    }

  LabelMapSliceType::IndexType erasedIndex;  

  typename OutputImageType::IndexType tmpIndex;
    tmpIndex[2] = z;

  for ( unsigned int i=0; i<this->m_MinCostPathIndices.size(); i++ )
    {
      for ( int y=-this->m_LeftRightLungSplitRadius; y<=this->m_LeftRightLungSplitRadius; y++ )
	{
	  tmpIndex[1] = this->m_MinCostPathIndices[i][1] + y;
	  erasedIndex[1] = this->m_MinCostPathIndices[i][1];
	  
	  for ( int x=-this->m_LeftRightLungSplitRadius; x<=this->m_LeftRightLungSplitRadius; x++ )
	    {
	      tmpIndex[0] = this->m_MinCostPathIndices[i][0] + x;
	      erasedIndex[0] = this->m_MinCostPathIndices[i][0];
	      
	      if ( this->GetOutput()->GetBufferedRegion().IsInside( tmpIndex ) )
		{
		  if ( this->GetOutput()->GetPixel( tmpIndex ) != 0 )
		    {
		      this->GetOutput()->SetPixel( tmpIndex, 0 );
		      this->m_ErasedSliceIndices.push_back( erasedIndex );	  
		    }
		}
	    }
	}
    }
}



/**
 * 
 */
template< class TInputImage >
void CIPSplitLeftLungRightLungImageFilter< TInputImage >
::FindMinCostPath()
{
  if ( this->m_MinCostPathIndices.size() > 0 )
    {
      this->m_MinCostPathIndices.clear();
    }  

  typename InputImageType::RegionType roiRegion;
    roiRegion.SetSize( this->m_GraphROISize );
    roiRegion.SetIndex( this->m_GraphROIStartIndex );

  typename InputExtractorType::Pointer roiExtractor = InputExtractorType::New();
    roiExtractor->SetInput( this->GetInput() );
    roiExtractor->SetExtractionRegion( roiRegion );
    roiExtractor->Update();
  
  InputPixelType lowerThreshold = itk::NumericTraits< InputPixelType >::NonpositiveMin();
  InputPixelType upperThreshold = itk::NumericTraits< InputPixelType >::max();

  typename FunctorType::Pointer graphFunctor = FunctorType::New();
    graphFunctor->SetRadius( 1 );
    graphFunctor->SetLowerThreshold( lowerThreshold );
    graphFunctor->SetUpperThreshold( upperThreshold );
    graphFunctor->SetExponentialCoefficient( this->m_ExponentialCoefficient );
    graphFunctor->SetExponentialTimeConstant( this->m_ExponentialTimeConstant );
    graphFunctor->ActivateAllNeighbors();

  typename GraphFilterType::Pointer graphFilter = GraphFilterType::New();
    graphFilter->SetInput( roiExtractor->GetOutput() );
    graphFilter->SetImageToGraphFunctor( graphFunctor );
    graphFilter->Update();

  LabelMapSliceType::IndexType index;

  GraphType::NodeIdentifierType startNodeID;
  GraphType::NodeIdentifierType endNodeID;

  GraphType::NodeIterator nIt( graphFilter->GetOutput() );
  
  nIt.GoToBegin();
  while ( !nIt.IsAtEnd() )
    {
    index = nIt.Get().ImageIndex;

    if ( index[0] == this->m_StartSearchIndex[0] && index[1] == this->m_StartSearchIndex[1] )
      {
      startNodeID = nIt.Get().Identifier;
      }
    if ( index[0] == this->m_EndSearchIndex[0] && index[1] == this->m_EndSearchIndex[1] )
      {
      endNodeID = nIt.Get().Identifier;
      }

    ++nIt;
    }

  MinPathType::Pointer minPathFilter = MinPathType::New();
    minPathFilter->SetInput( graphFilter->GetOutput() );
    minPathFilter->SetStartNode( startNodeID );
    minPathFilter->SetEndNode( endNodeID );
    minPathFilter->Update();

  GraphType::NodeIterator onIt( minPathFilter->GetOutput() );
  
  onIt.GoToBegin();
  while ( !onIt.IsAtEnd() )
    {
    this->m_MinCostPathIndices.push_back( onIt.Get().ImageIndex );

    ++onIt;
    }
}


/**
 * Extract a slice from the input label map image
 */
template< class TInputImage >
void
CIPSplitLeftLungRightLungImageFilter< TInputImage >
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
CIPSplitLeftLungRightLungImageFilter< TInputImage >
::GetRemovedIndices( std::vector< LabelMapType::IndexType >* removedIndicesVec )
{
  for ( unsigned int i=0; i<this->m_RemovedIndices.size(); i++ )
    {
    removedIndicesVec->push_back( this->m_RemovedIndices[i] );
    }
}


template < class TInputImage >
void
CIPSplitLeftLungRightLungImageFilter< TInputImage >
::SetLungLabelMap( OutputImageType::Pointer labelMap )
{
  this->m_LungLabelMap = LabelMapType::New();
  this->m_LungLabelMap = labelMap;
}

  
/**
 * Standard "PrintSelf" method
 */
template< class TInputImage >
void
CIPSplitLeftLungRightLungImageFilter< TInputImage >
::PrintSelf(
  std::ostream& os, 
  Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Printing itkCIPSplitLeftLungRightLungImageFilter: " << std::endl;
  os << indent << "ExponentialCoefficient:\t" << this->m_ExponentialCoefficient << std::endl;
  os << indent << "ExponentialTimeConstant:\t" << this->m_ExponentialTimeConstant << std::endl;
  os << indent << "LeftRightLungSplitRadius:\t" << this->m_LeftRightLungSplitRadius << std::endl;
  os << indent << "AggressiveLeftRightSplitter:\t" << this->m_AggressiveLeftRightSplitter << std::endl;
}

} // end namespace itk

#endif
