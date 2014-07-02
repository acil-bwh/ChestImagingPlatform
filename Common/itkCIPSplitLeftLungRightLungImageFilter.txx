#ifndef _itkCIPSplitLeftLungRightLungImageFilter_txx
#define _itkCIPSplitLeftLungRightLungImageFilter_txx

#include "itkCIPSplitLeftLungRightLungImageFilter.h"
#include "cipExceptionObject.h"

namespace itk
{

template< class TInputImage >
CIPSplitLeftLungRightLungImageFilter< TInputImage >
::CIPSplitLeftLungRightLungImageFilter()
{
  this->m_ExponentialCoefficient      = 200;
  this->m_ExponentialTimeConstant     = -700;
  this->m_LeftRightLungSplitRadius    = 1;
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

  // Fill the output image with the contents of the input image. 
  LabelMapIteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );
  LabelMapIteratorType lIt( this->m_LungLabelMap, this->m_LungLabelMap->GetBufferedRegion() );

  oIt.GoToBegin();
  lIt.GoToBegin();
  while ( !lIt.IsAtEnd() )
    {
    oIt.Set( lIt.Get() );

    ++oIt;
    ++lIt;
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

  unsigned int slicesSinceLastSplit = size[2];

  for ( unsigned int i=0; i<size[2]; i++ )
    {
    bool merged = this->GetLungsMergedInSliceRegion( size[0]/3, 0, size[0]/3, size[1], i ); 

    // We will only use the local search region provided that 
    // we're relatively near the slice where we had our last
    // split
    if ( !merged )
      {
	if ( slicesSinceLastSplit > 10 )
	  {
	    this->m_UseLocalGraphROI = false;
	  }
	else
	  {
	    slicesSinceLastSplit++;
	  }
      }

    // If the current slice is merged and we're not supposed to use the
    // local search region, then set the default search region for
    // this slice.
    if ( merged && !this->m_UseLocalGraphROI ) 
      {
	this->SetDefaultGraphROIAndSearchIndices( i );
      }

    bool attemptSplit = true;
    while ( merged && attemptSplit )
      {
	this->FindMinCostPath();
	this->EraseConnection( i );
	merged = this->GetLungsMergedInSliceRegion( size[0]/3, 0, size[0]/3, size[1], i );

	if ( !merged )
	  {
	    // Set the local search region for the next slice
	    this->SetLocalGraphROIAndSearchIndices( i+1 );
	    slicesSinceLastSplit = 0;
	  }
	else if ( merged && this->m_UseLocalGraphROI )
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
      }
    }
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
    sliceROI->SetDirectionCollapseToIdentity();
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
  this->m_StartSearchIndex[2] = 0; // Value not used except for assert statements below

  this->m_EndSearchIndex[0] = (maxX + minX)/2;
  this->m_EndSearchIndex[1] = maxY;
  this->m_EndSearchIndex[2] = 0; // Value not used except for assert statements below

  // Set the start index of the graph ROI
  int tmp;
  tmp = minX - 10;
  if ( tmp < 0 )
    {
      this->m_GraphROIStartIndex[0] = 0;
    }
  else
    {
      this->m_GraphROIStartIndex[0] = tmp;
    }
  tmp  = minY - 10;
  if ( tmp < 0 )
    {
      this->m_GraphROIStartIndex[1] = 0;
    }
  else
    {
      this->m_GraphROIStartIndex[1] = tmp;
    }
  this->m_GraphROIStartIndex[2] = z;

  assert( this->GetOutput()->GetBufferedRegion().IsInside( this->m_GraphROIStartIndex ) );

  // Now set the size of the graph ROI
  tmp  = maxX - minX + 20;
  if ( tmp < 0 )
    {
      this->m_GraphROISize[0] = 0;
    }	  
  else if ( tmp > size[0] )
    {
      this->m_GraphROISize[0] = size[0];
    }
  else
    {
      this->m_GraphROISize[0] = tmp;
    }

  tmp  = maxY - minY + 20;
  if ( tmp < 0 )
    {
      this->m_GraphROISize[1] = 0;
    }
  else if ( tmp > size[1] )
    {
      this->m_GraphROISize[1] = size[1];
    }
  else
    {
      this->m_GraphROISize[1] = tmp;
    }
  
  assert( this->GetOutput()->GetBufferedRegion().IsInside( this->m_StartSearchIndex ) );
  assert( this->GetOutput()->GetBufferedRegion().IsInside( this->m_EndSearchIndex ) );
  assert( this->m_GraphROIStartIndex[0] + this->m_GraphROISize[0] <= size[0] );
  assert( this->m_GraphROIStartIndex[1] + this->m_GraphROISize[1] <= size[1] );

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
  int tmp;
  tmp = minX - 10;
  if ( tmp < 0 )
    {
      this->m_GraphROIStartIndex[0] = 0;
    }
  else
    {
      this->m_GraphROIStartIndex[0] = tmp;
    }
  tmp = minY - 10;
  if ( tmp < 0 )
    {
      this->m_GraphROIStartIndex[1] = 0;
    }
  else
    {
      this->m_GraphROIStartIndex[1] = tmp;
    }
  this->m_GraphROIStartIndex[2] = z;

  assert( this->GetOutput()->GetBufferedRegion().IsInside( this->m_GraphROIStartIndex ) );

  this->m_StartSearchIndex[1] = this->m_GraphROIStartIndex[1];

  // Now set the size of the graph ROI
  this->m_GraphROISize[0] = maxX - minX + 20;
  if ( maxX - minX + 20 < 0 )
    {
      this->m_GraphROISize[0] = 0;
    }
  if ( this->m_GraphROISize[0] >= size[0] )
    {
      this->m_GraphROISize[0] = size[0] - 1;
    }
  
  this->m_GraphROISize[1] = maxY - minY + 20;
  if ( maxY - minY + 20 < 0 )
    {
      this->m_GraphROISize[1] = 0;
    }
  if ( this->m_GraphROISize[1] >= size[1] )
    {
      this->m_GraphROISize[1] = size[1] - 1;
    }   
  this->m_EndSearchIndex[1] = this->m_GraphROIStartIndex[1] + this->m_GraphROISize[1] - 1;

  assert( this->GetOutput()->GetBufferedRegion().IsInside( this->m_StartSearchIndex ) );
  assert( this->GetOutput()->GetBufferedRegion().IsInside( this->m_EndSearchIndex ) );
  assert( this->m_GraphROIStartIndex[0] + this->m_GraphROISize[0] < size[0] );
  assert( this->m_GraphROIStartIndex[1] + this->m_GraphROISize[1] < size[1] );

  this->m_GraphROISize[2] = 0;
}


template< class TInputImage >
void CIPSplitLeftLungRightLungImageFilter< TInputImage >
::EraseConnection( unsigned int slice )
{
  if ( this->m_ErasedSliceIndices.size() > 0 )
    {
      this->m_ErasedSliceIndices.clear();
    }

  LabelMapSliceType::IndexType erasedIndex;  

  typename OutputImageType::IndexType tmpIndex;
    tmpIndex[2] = slice;

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
	      
	      for ( int z=-1; z<=1; z++ )
		{
		  tmpIndex[2] = slice + z;
 
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
    roiExtractor->SetDirectionCollapseToIdentity();
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
}

} // end namespace itk

#endif
