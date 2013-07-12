/**
 *  $Date: 2012-04-24 16:29:25 -0700 (Tue, 24 Apr 2012) $
 *  $Revision: 89 $
 *  $Author: jross $
 */
#ifndef _itkCIPSplitLeftLungRightLungImageFilter_txx
#define _itkCIPSplitLeftLungRightLungImageFilter_txx

#include "itkCIPSplitLeftLungRightLungImageFilter.h"


namespace itk
{

template< class TInputImage >
CIPSplitLeftLungRightLungImageFilter< TInputImage >
::CIPSplitLeftLungRightLungImageFilter()
{
  this->m_MinForegroundSlice          = UINT_MAX;
  this->m_MaxForegroundSlice          = 0;
  this->m_ExponentialCoefficient      = 200;
  this->m_ExponentialTimeConstant     = -700;
  this->m_LeftRightLungSplitRadius    = 2;
  this->m_AggressiveLeftRightSplitter = false;
  this->m_ChestLabelMap               = LabelMapType::New();
}


template< class TInputImage >
void
CIPSplitLeftLungRightLungImageFilter< TInputImage >
::GenerateData()
{
  ChestConventions conventions;

  typename Superclass::InputImageConstPointer inputPtr  = this->GetInput();
  typename Superclass::OutputImagePointer     outputPtr = this->GetOutput(0);
    outputPtr->SetRequestedRegion( inputPtr->GetRequestedRegion() );
    outputPtr->SetBufferedRegion( inputPtr->GetBufferedRegion() );
    outputPtr->SetLargestPossibleRegion( inputPtr->GetLargestPossibleRegion() );
    outputPtr->Allocate();

  //
  // The input may already be split in which case we'll want to do
  // nothing and just return the input.
  //
  LabelMapIteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );
  LabelMapIteratorType lIt( this->m_ChestLabelMap, this->m_ChestLabelMap->GetBufferedRegion() );

  if ( this->IsVolumeSplit( this->m_ChestLabelMap ) )
    {
    oIt.GoToBegin();
    lIt.GoToBegin();
    while ( !lIt.IsAtEnd() )
      {
      oIt.Set( lIt.Get() );

      ++oIt;
      ++lIt;
      }
    }
  else
    {
    //
    // If we're here, the input label map is not split, so we have to
    // split it. As we'll be looping over the image, determine the min
    // and max foreground slices.
    //
    unsigned char cipRegion;

    oIt.GoToBegin();
    lIt.GoToBegin();
    while ( !lIt.IsAtEnd() )
      {
      cipRegion = conventions.GetChestRegionFromValue( lIt.Get() );

      if ( conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, static_cast< unsigned char >( cip::WHOLELUNG ) ) )
        {
        if ( lIt.GetIndex()[2] < this->m_MinForegroundSlice )
          {
          this->m_MinForegroundSlice = lIt.GetIndex()[2];
          }
        if ( lIt.GetIndex()[2] > this->m_MaxForegroundSlice )
          {
          this->m_MaxForegroundSlice = lIt.GetIndex()[2];
          }

        oIt.Set( cip::WHOLELUNG );
        }

      ++oIt;
      ++lIt;
      }

    //
    // If a connection exists, it tends to occur in the middle of the
    // lung. The top portion and the bottom portion are split. We next
    // want to determine the middle subvolume to focus on. Begin by
    // splitting the volume into thirds. The bottom third extends from
    // m_MinForegroundSlice to lowerMiddleForeground slice. The middle
    // third extends from lowerMiddleForegroundSlice to
    // upperMiddleForegroundSlice. The upper third extends from
    // upperMiddleForegroundSlice to m_MaxForegroundSlice. We do a
    // binary search on both [upper,lower]MiddleForegroundSlices to
    // identify the slices that yield upper and lower regions that are
    // split
    //
    unsigned int temp = (this->m_MaxForegroundSlice - this->m_MinForegroundSlice)/3;
    unsigned int lowerMiddleForegroundSlice = this->m_MinForegroundSlice + temp;
    unsigned int upperMiddleForegroundSlice = this->m_MaxForegroundSlice - temp;

    //
    // Now adjust lowerMiddleForegroundSlice until we find a split
    // lower third
    //
    







//   LabelMapType::SizeType size = this->GetOutput()->GetBufferedRegion().GetSize();

//   int minX = size[0]/3;
//   int maxX = size[0]-size[0]/3;
//   int minY = 0;
//   int maxY = size[1]-1;

//   typename InputImageSliceType::IndexType searchStartIndex;
//   typename InputImageSliceType::IndexType searchEndIndex;

//   LabelMapType::IndexType index3D;

//   //
//   // We will keep track of the path indices used to split the
//   // previous slice. To insure that the left and right lungs are
//   // split in 3D, we will zero-out all label map points falling
//   // within the region between the path in the current slice and the
//   // path in the previous slice.
//   //
//   std::map< short, short > previousPathMap;

//   int previousMinY = size[1];
//   int previousMaxY = 0;

//   for ( unsigned int i=0; i<size[2]; i++ )
//     {
//     bool merged = this->GetLungsMergedInSliceRegion( size[0]/3, 0, size[0]/3, size[1], i ); 

//     index3D[2] = i;

//     if ( merged )
//       {
//       int numSplitAttempts = 0;
      
//       while ( merged && numSplitAttempts < 3 )
//         {
//         numSplitAttempts++;
        
//         typename InputImageType::SizeType roiSize;
//           roiSize[0] = maxX - minX + 20;

//         if ( roiSize[0] < 0 )
//           {
//           roiSize[0] = 0;
//           }
//         if ( roiSize[0] > size[0] )
//           {
//           roiSize[0] = size[0];
//           }
        
//         roiSize[1] = maxY - minY + 20;
//         if ( roiSize[1] < 0 )
//           {
//           roiSize[1] = 0;
//           }
//         if ( roiSize[1] > size[1] )
//           {
//           roiSize[1] = size[1];
//           }

//         roiSize[2] = 0;
        
//         typename InputImageType::IndexType roiStartIndex;
//           roiStartIndex[0] = minX - 10;
        
//         if ( roiStartIndex[0] < 0 )
//           {
//           roiStartIndex[0] = 0;
//           }
        
//         roiStartIndex[1] = minY - 10;
//         if ( roiStartIndex[1] < 0 )
//           {
//           roiStartIndex[1] = 0;
//           }
        
//         roiStartIndex[2] = i;
        
//         typename InputImageType::RegionType roiRegion;
//           roiRegion.SetSize( roiSize );
//           roiRegion.SetIndex( roiStartIndex );
        
//         typename InputExtractorType::Pointer roiExtractor = InputExtractorType::New();
//           roiExtractor->SetInput( this->GetInput() );
//           roiExtractor->SetExtractionRegion( roiRegion );
//           roiExtractor->Update();

//         searchStartIndex[0] = roiStartIndex[0] + roiSize[0]/2;
//         searchStartIndex[1] = roiStartIndex[1];
        
//         searchEndIndex[0] = roiStartIndex[0] + roiSize[0]/2;
//         searchEndIndex[1] = roiStartIndex[1] + roiSize[1] - 1;

//         //
//         // Set the startIndex to the the top-center of the ROI and the
//         // endIndex to be the bottom-center of the ROI
//         //
//         std::vector< LabelMapSliceType::IndexType > pathIndices = this->GetMinCostPath( roiExtractor->GetOutput(), searchStartIndex, searchEndIndex );
        
//         minX = size[0];
//         maxX = 0;
//         minY = size[1];
//         maxY = 0;
        
//         bool foundMinMax = false;
//         for ( unsigned int j=0; j<pathIndices.size(); j++ )
//           {
//           index3D[0] = (pathIndices[j])[0];
//           index3D[1] = (pathIndices[j])[1];
          
//           if ( this->GetOutput()->GetPixel( index3D ) !=0 )
//             {
//             foundMinMax = true;
            
//             if ( index3D[0] < minX )
//               {
//               minX = index3D[0];
//               }
//             if ( index3D[0] > maxX )
//               {
//               maxX = index3D[0];
//               }
//             if ( index3D[1] < minY )
//               {
//               minY = index3D[1];
//               }
//             if ( index3D[1] > maxY )
//               {
//               maxY = index3D[1];
//               }
//             }          
//           }
        
//         if ( !foundMinMax || this->m_AggressiveLeftRightSplitter )
//           {
//           minX = size[0]/3;
//           maxX = size[0]-size[0]/3;
//           minY = 0;
//           maxY = size[1]-1;
//           }
        
//         for ( unsigned int j=0; j<pathIndices.size(); j++ )
//           {
//           LabelMapType::IndexType tempIndex;
//             tempIndex[2] = i;

//           int currentX  = (pathIndices[j])[0];
//           int currentY  = (pathIndices[j])[1];

//           int startX = currentX - this->m_LeftRightLungSplitRadius;
//           int endX   = currentX + this->m_LeftRightLungSplitRadius;

//           if ( previousPathMap.size() > 0 )
//             {
//             if ( currentY >= previousMinY && currentY <= previousMaxY )
//               {
//               //
//               // Determine the extent in the x-direction to zero-out
//               //
//               int previousX = previousPathMap[(pathIndices[j])[1]];
              
//               if ( previousX - currentX < 0 )
//                 {
//                 startX = previousX - this->m_LeftRightLungSplitRadius;
//                 endX   = currentX  + this->m_LeftRightLungSplitRadius;
//                 }
//               else 
//                 {
//                 startX = currentX  - this->m_LeftRightLungSplitRadius;
//                 endX   = previousX + this->m_LeftRightLungSplitRadius;
//                 }                
//               }
//             }

//           tempIndex[1] = (pathIndices[j])[1];
//           for ( int x=startX; x<=endX; x++ )
//             {
//             tempIndex[0] = x;

//             if ( this->GetOutput()->GetBufferedRegion().IsInside( tempIndex ) )
//               {
//               if ( this->GetOutput()->GetPixel( tempIndex ) != 0 )
//                 {
//                 this->m_RemovedIndices.push_back( tempIndex );
//                 }
//               this->GetOutput()->SetPixel( tempIndex, 0 );
//               }
//             }

//           for ( int y=-this->m_LeftRightLungSplitRadius; y<=this->m_LeftRightLungSplitRadius; y++ )
//             {
//             tempIndex[1] = (pathIndices[j])[1] + y;            

//             for ( int x=-this->m_LeftRightLungSplitRadius; x<=this->m_LeftRightLungSplitRadius; x++ )
//               {
//               tempIndex[0] = (pathIndices[j])[0] + x;
              
//               if ( this->GetOutput()->GetBufferedRegion().IsInside( tempIndex ) )
//                 {
//                 if ( x==this->m_LeftRightLungSplitRadius || x==-this->m_LeftRightLungSplitRadius || 
//                      y==this->m_LeftRightLungSplitRadius || y==-this->m_LeftRightLungSplitRadius )
//                   {
//                   if ( this->GetType( tempIndex ) == static_cast< unsigned char >( VESSEL ) )
//                     {
//                     if ( this->GetOutput()->GetPixel( tempIndex ) != 0 )
//                       {
//                       this->m_RemovedIndices.push_back( tempIndex );
//                       }
//                     this->GetOutput()->SetPixel( tempIndex, 0 );
//                     }
//                   }
//                 else
//                   {
//                   if ( this->GetOutput()->GetPixel( tempIndex ) != 0 )
//                     {
//                     this->m_RemovedIndices.push_back( tempIndex );
//                     }
//                   this->GetOutput()->SetPixel( tempIndex, 0 );
//                   }
//                 }
//               }
//             }          
//           }
        
//         merged = this->GetLungsMergedInSliceRegion( size[0]/3, 0, size[0]/3, size[1], i ); 
        
//         if ( merged )
//           {
//           minX = size[0]/3;
//           maxX = size[0]-size[0]/3;
//           minY = 0;
//           maxY = size[1]-1;
//           }
//         else
//           {
//           //
//           // Assign the map values to use while splitting the next
//           // slice 
//           //
//           previousPathMap.clear();

//           previousMinY = size[1];
//           previousMaxY = 0;

//           for ( unsigned int i=0; i<pathIndices.size(); i++ )
//             {
//             previousPathMap[(pathIndices[i])[1]] = (pathIndices[i])[0];

//             if ( (pathIndices[i])[1] < previousMinY )
//               {
//               previousMinY = (pathIndices[i])[1];
//               }
//             if ( (pathIndices[i])[1] > previousMaxY )
//               {
//               previousMaxY = (pathIndices[i])[1];
//               }
//             }
//           }
//         }
//       }
//     else
//       {
//       minX = size[0]/3;
//       maxX = size[0]-size[0]/3;
//       minY = 0;
//       maxY = size[1]-1;

//       previousPathMap.clear();

//       previousMinY = size[1];
//       previousMaxY = 0;
//       }
//     }
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

  //
  // Perform connected components on the region
  //
  ConnectedComponent2DType::Pointer connectedComponent = ConnectedComponent2DType::New();
    connectedComponent->SetInput( sliceROI->GetOutput() );
    connectedComponent->FullyConnectedOn();
    connectedComponent->Update();

  //
  // If there is an object that touches both the left border and the
  // right border, then the lungs are merged in this slice.  Test this
  // condition
  //
  std::list< unsigned short >  lefthandLabelList;
  std::list< unsigned short >  righthandLabelList;

  LabelMapSliceType::IndexType index;

  for ( unsigned int i=0; i<roiSize[1]; i++ )
    {
    index[1] = i;
    index[0] = startIndex[0];

    unsigned short value = connectedComponent->GetOutput()->GetPixel( index );

    if ( value != static_cast< unsigned short >( 0 ) )
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


/**
 * 
 */
template< class TInputImage >
std::vector< itk::Image< unsigned short, 2 >::IndexType >
CIPSplitLeftLungRightLungImageFilter< TInputImage >
::GetMinCostPath( InputSlicePointerType imageROI, LabelMapSliceIndexType startIndex, LabelMapSliceIndexType endIndex )
{
  std::vector< LabelMapSliceIndexType > minCostPathIndices;

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
    graphFilter->SetInput( imageROI );
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

    if ( index[0] == startIndex[0] && index[1] == startIndex[1] )
      {
      startNodeID = nIt.Get().Identifier;
      }
    if ( index[0] == endIndex[0] && index[1] == endIndex[1] )
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
    minCostPathIndices.push_back( onIt.Get().ImageIndex );

    ++onIt;
    }

  return minCostPathIndices;
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
::SetLungLabelMap( OutputImageType::Pointer airwayLabelMap )
{
  this->m_ChestLabelMap = airwayLabelMap;
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
