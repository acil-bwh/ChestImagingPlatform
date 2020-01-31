/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkImageToGraphFilter.h,v $
  Language:  C++
  Date:      $Date: 2009/02/09 21:38:19 $
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkImageToGraphFilter_h
#define __itkImageToGraphFilter_h

#include "itkGraphSource.h"
#include "itkObjectFactory.h"
#include "itkDefaultImageToGraphFunctor.h"


namespace itk
{

/** \class ImageToGraphFilter
 * \brief 
 *
 * ImageToGraphFilter is the base class for all process objects that output
 * Graph data and require image data as input. Specifically, this class
 * defines the SetInput() method for defining the input to a filter.
 *
 * \ingroup ImageFilters
 */
template <class TInputImage, class TOutputGraph >
class ITK_EXPORT ImageToGraphFilter : public GraphSource<TOutputGraph>
{
public:
  /** Standard class typedefs. */
  typedef ImageToGraphFilter            Self;
  typedef GraphSource<TOutputGraph>     Superclass;
  typedef SmartPointer<Self>            Pointer;
  typedef SmartPointer<const Self>      ConstPointer;
  
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( ImageToGraphFilter, GraphSource );

  /** Create a valid output. */
  DataObject::Pointer  MakeOutput( unsigned int idx ) override;

  /** Some Image related typedefs. */
  typedef   TInputImage                        ImageType;
  typedef   typename ImageType::Pointer        ImagePointer;
  typedef   typename ImageType::ConstPointer   ImageConstPointer;
  typedef   typename ImageType::RegionType     RegionType; 
  typedef   typename ImageType::PixelType      PixelType; 
  typedef   typename ImageType::IndexType      IndexType; 

  /** Some Graph related typedefs. */
  typedef TOutputGraph                              GraphType;
  typedef typename GraphType::GraphTraitsType       GraphTraitsType; 
  typedef typename GraphType::Pointer               GraphPointer;
  typedef typename GraphType::NodeType              NodeType; 
  typedef typename GraphType::NodePointerType       NodePointerType; 
  typedef typename GraphType::NodeIdentifierType    NodeIdentifierType; 
  typedef typename GraphTraitsType::NodeWeightType  NodeWeightType;
  typedef typename GraphType::EdgeType              EdgeType; 
  typedef typename GraphType::EdgePointerType       EdgePointerType; 
  typedef typename GraphType::EdgeIdentifierType    EdgeIdentifierType; 
  typedef typename GraphTraitsType::EdgeWeightType  EdgeWeightType;
  
  /** Abstract ImageToGraphFunctorType */
  typedef ImageToGraphFunctor<ImageType, GraphType>  ImageToGraphFunctorType;
  //typedef TFunctorType< ImageType, GraphType >      ImageToGraphFunctorType;
  typedef typename ImageToGraphFunctorType::Pointer ImageToGraphFunctorPointer;
  
  /** Set the input image of this process object.  */
  void SetInput( unsigned int idx, const ImageType *input );
  void SetInput( const ImageType * );   

  /** Get the input image of this process object.  */
  const ImageType * GetInput( unsigned int idx );

  /** Get the output Graph of this process object.  */
  GraphType * GetOutput( void );
  
  /** Set/Get ImageToGraphFunctor */
  itkGetObjectMacro( ImageToGraphFunctor, ImageToGraphFunctorType );
  itkSetObjectMacro( ImageToGraphFunctor, ImageToGraphFunctorType );

  /** Prepare the output */
  void GenerateOutputInformation( void ) override;

protected:
  ImageToGraphFilter();
  ~ImageToGraphFilter();
  void PrintSelf( std::ostream& os, Indent indent ) const override;
  
  void GenerateData() override;
 
private:
  ImageToGraphFilter( const ImageToGraphFilter& ); //purposely not implemented
  void operator=( const ImageToGraphFilter& ); //purposely not implemented

  ImageToGraphFunctorPointer m_ImageToGraphFunctor;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageToGraphFilter.txx"
#endif

#endif
