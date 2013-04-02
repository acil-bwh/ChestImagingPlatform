/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkGraphToGraphFilter.h,v $
  Language:  C++
  Date:      $$
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  Portions of this code are covered under the VTK copyright.
  See VTKCopyright.txt or http://www.kitware.com/VTKCopyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkGraphToGraphFilter_h
#define __itkGraphToGraphFilter_h

#include "itkGraphSource.h"

namespace itk
{

/** \class GraphToGraphFilter
 * \brief Base class for filters that take a graph as an input and output
 * another graph.
 *
 * GraphToGraphFilter is the base class for all process objects that output
 * graph data, and require graph data as input. Specifically, this class
 * defines the SetInput() method for defining the input to a filter.
 *
 * \ingroup GraphFilters
 *
 */
template <class TInputGraph, class TOutputGraph>
class ITK_EXPORT GraphToGraphFilter : public GraphSource<TOutputGraph>
{
public:
  /** Standard class typedefs. */
  typedef GraphToGraphFilter            Self;
  typedef GraphSource<TOutputGraph>     Superclass;
  typedef SmartPointer<Self>            Pointer;
  typedef SmartPointer<const Self>      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GraphToGraphFilter, GraphSource );

  /** Some convenient typedefs. */
  typedef TInputGraph                        InputGraphType;
  typedef typename InputGraphType::Pointer   InputGraphPointer;

  /** Set the graph input of this process object.  */
  void SetInput(InputGraphType *input);

  /** Get the graph input of this process object.  */
  InputGraphType * GetInput( void );
  InputGraphType * GetInput( unsigned int idx );

protected:
  GraphToGraphFilter();
  ~GraphToGraphFilter() {};

private:
  GraphToGraphFilter( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGraphToGraphFilter.txx"
#endif

#endif
