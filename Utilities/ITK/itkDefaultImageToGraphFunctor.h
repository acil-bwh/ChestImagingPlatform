/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkDefaultImageToGraphFunctor.h,v $
  Language:  C++
  Date:      $Date: 2009/02/09 21:38:19 $
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  Portions of this code are covered under the VTK copyright.
  See VTKCopyright.txt or http://www.kitware.com/VTKCopyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkDefaultImageToGraphFunctor_h
#define __itkDefaultImageToGraphFunctor_h

#include "itkProcessObject.h"
#include "itkShapedNeighborhoodIterator.h"
#include "itkArray.h"

#include <list>

namespace itk
{

/** \class ImageToGraphFunctor
 * \brief Abstract base class which defines node/edge weighting in constructing a
 *        graph from an image.
 **/
template<typename TInputImage, typename TOutputGraph>
class ImageToGraphFunctor : public ProcessObject
{
public:
  /** Standard class typedefs. */
  typedef ImageToGraphFunctor       Self;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageToGraphFunctor, ProcessObject);


  /** Extract dimension from input image. */
  itkStaticConstMacro(ImageDimension,
    unsigned int, TInputImage::ImageDimension);

  /** Declare image-related types */
  typedef TInputImage                        InputImageType;
  typedef typename InputImageType::IndexType IndexType;
  typedef typename InputImageType::PixelType PixelType;

  /** Some other image related types */
  typedef ShapedNeighborhoodIterator<InputImageType> NeighborhoodIteratorType;
  
  typedef typename NeighborhoodIteratorType::IndexListType IndexListType;
  typedef typename NeighborhoodIteratorType::OffsetType    OffsetType;
  typedef typename NeighborhoodIteratorType::RadiusType    RadiusType;

  /** Set/Get the image input of this process object.  */
  virtual void SetInput( const InputImageType *);
  virtual void SetInput( unsigned int, const TInputImage *);
  const InputImageType * GetInput(void);
  const InputImageType * GetInput(unsigned int);

  /** Declare graph-related types */
  typedef TOutputGraph                              OutputGraphType;
  typedef typename OutputGraphType::GraphTraitsType GraphTraitsType;
  typedef typename OutputGraphType::NodeType        NodeType;
  typedef typename OutputGraphType::EdgeType        EdgeType;
  typedef typename OutputGraphType::NodeIterator    NodeIteratorType;
  typedef typename OutputGraphType::EdgeIterator    EdgeIteratorType;
  typedef typename GraphTraitsType::NodeWeightType  NodeWeightType;
  typedef typename GraphTraitsType::EdgeWeightType  EdgeWeightType;
  typedef typename GraphTraitsType::NodePointerType NodePointerType;
  typedef typename GraphTraitsType::EdgePointerType EdgePointerType;
  typedef typename GraphTraitsType::EdgeIdentifierContainerType
                                                    EdgeIdentifierContainerType;

  typedef Image<typename OutputGraphType::NodeIdentifierType,
                itkGetStaticConstMacro( ImageDimension )> NodeImageType;

  /** virtual functions */
  virtual EdgeWeightType GetEdgeWeight( IndexType, IndexType ) = 0;
  virtual NodeWeightType GetNodeWeight( IndexType ) = 0;
  virtual bool IsPixelANode( IndexType ) = 0;
  virtual void NormalizeGraph( NodeImageType *, OutputGraphType * ) = 0;

  /** ExcludeBackground - specify as true if the output graph
    * is to be constructed on a subregion.  This subregion is
    * defined by the presence of non-'BackgroundValue' pixels.
    */
  itkGetMacro( ExcludeBackground, bool );
  itkSetMacro( ExcludeBackground, bool );
  itkGetMacro( BackgroundValue, PixelType );
  itkSetMacro( BackgroundValue, PixelType );

  /** Macros for specifying the neighborhood of an image */
  void SetRadius( unsigned int r )
    {
    this->m_Radius.Fill( r );
    this->m_NumberOfPixelsInNeighborhood = 1;
    for ( unsigned int d = 0; d < this->m_Radius.GetSizeDimension(); d++ )
      {
      this->m_NumberOfPixelsInNeighborhood *= ( 2*this->m_Radius[d] + 1 );
      }
    this->ComputeNeighborhoodStrideTable();
    this->Modified();
    }
  void SetRadius( RadiusType R )
    {
    this->m_Radius = R;
    this->m_NumberOfPixelsInNeighborhood = 1;
    for ( unsigned int d = 0; d < this->m_Radius.GetSizeDimension(); d++ )
      {
      this->m_NumberOfPixelsInNeighborhood *= ( 2*this->m_Radius[d] + 1 );
      }
    this->ComputeNeighborhoodStrideTable();
    this->Modified();
    }

  itkGetMacro( Radius, RadiusType );
  itkGetMacro( NumberOfPixelsInNeighborhood, unsigned int );

  /** Add/Remove a neighborhood offset (from the center of the neighborhood)
   *  to/from the active list.  Active list offsets are the only locations
   *  updated and accessible through the iterator.  */
  virtual void ActivateOffset( const OffsetType& off )
    { this->ActivateIndex( this->GetNeighborhoodIndex( off ) ); }
  virtual void DeactivateOffset( const OffsetType& off )
    { this->DeactivateIndex( this->GetNeighborhoodIndex( off ) ); }
  virtual void ActivateIndex( const unsigned int );
  virtual void DeactivateIndex( const unsigned int );

  void ActivateAllNeighbors()
    {
    for( unsigned int n = 0; n < this->m_NumberOfPixelsInNeighborhood; n++ )
      {
      this->ActivateIndex( n );
      }
    this->Modified();
    }
  void DeactivateAllNeighbors()
    {
    this->ClearActiveList();
    }

  /** Removes all active pixels from this neighborhood. */
  virtual void ClearActiveList()
    { m_ActiveIndexList.clear(); }

  /** Returns the list of active indicies in the neighborhood */
  const IndexListType &GetActiveIndexList() const
    { return m_ActiveIndexList; }

  /** Returns the size of the list of active neighborhood indicies. */
  typename IndexListType::size_type GetActiveIndexListSize() const
    { return m_ActiveIndexList.size(); }


protected:
  ImageToGraphFunctor();
  ~ImageToGraphFunctor() {}
  void PrintSelf( std::ostream& os, Indent indent ) const override;

  unsigned int GetNeighborhoodIndex( const OffsetType &o ) const;
  void ComputeNeighborhoodStrideTable();


  ImageToGraphFunctor( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented

  PixelType                m_BackgroundValue;
  IndexListType            m_ActiveIndexList;
  bool                     m_ExcludeBackground;
  RadiusType               m_Radius;
  unsigned int             m_NumberOfPixelsInNeighborhood;
  Array<unsigned int>      m_StrideTable;

};

/** \class DefaultImageToGraphFunctor
 * \brief Default class which defines node/edge weighting in constructing a
 *        graph from an image.
 **/
template<typename TInputImage, typename TOutputGraph>
class DefaultImageToGraphFunctor
: public ImageToGraphFunctor<TInputImage, TOutputGraph>
{
public:
  /** Standard class typedefs. */
  typedef DefaultImageToGraphFunctor                     Self;
  typedef ImageToGraphFunctor<TInputImage, TOutputGraph> Superclass;
  typedef SmartPointer<Self>                             Pointer;
  typedef SmartPointer<const Self>                       ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DefaultImageToGraphFunctor, ImageToGraphFunctor);

  typedef TOutputGraph OutputGraphType;

  typedef typename Superclass::IndexType            IndexType;
  typedef typename Superclass::NodeType             NodeType;
  typedef typename Superclass::EdgeType             EdgeType;
  typedef typename Superclass::NodeIteratorType     NodeIteratorType;
  typedef typename Superclass::EdgeIteratorType     EdgeIteratorType;
  typedef typename Superclass::NodeWeightType       NodeWeightType;
  typedef typename Superclass::EdgeWeightType       EdgeWeightType;
  typedef typename Superclass::NodeImageType        NodeImageType;
  typedef typename Superclass::NodePointerType      NodePointerType;
  typedef typename Superclass::EdgePointerType      EdgePointerType;
  typedef typename Superclass::EdgeIdentifierContainerType
                                                    EdgeIdentifierContainerType;

  virtual bool IsPixelANode(IndexType idx) override
    { return ( !this->m_ExcludeBackground ||
      ( this->GetInput()->GetPixel( idx ) != this->m_BackgroundValue ) ); }
  virtual EdgeWeightType GetEdgeWeight(IndexType idx1, IndexType idx2 ) override
      { return ( static_cast<EdgeWeightType>( 1 ) ); }
  virtual NodeWeightType GetNodeWeight( IndexType idx ) override
      { return ( static_cast<NodeWeightType>( 1 ) ); }
  virtual void NormalizeGraph( NodeImageType *im, OutputGraphType *g ) override {}

protected:
  DefaultImageToGraphFunctor() {}
  ~DefaultImageToGraphFunctor() {}
  void PrintSelf(std::ostream& os, Indent indent) const override
     { Superclass::PrintSelf( os, indent ); }

private:
  DefaultImageToGraphFunctor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDefaultImageToGraphFunctor.txx"
#endif

#endif
