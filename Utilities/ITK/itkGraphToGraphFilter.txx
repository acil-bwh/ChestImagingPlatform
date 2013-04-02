/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkGraphToGraphFilter.txx,v $
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
#ifndef __itkGraphToGraphFilter_txx
#define __itkGraphToGraphFilter_txx

#include "itkGraphToGraphFilter.h"


namespace itk
{

/**
 *
 */
template <class TInputGraph, class TOutputGraph>
GraphToGraphFilter<TInputGraph,TOutputGraph>
::GraphToGraphFilter()
{
  // Modify superclass default values, can be overridden by subclasses
  this->SetNumberOfRequiredInputs( 1 );
  this->ProcessObject::SetNumberOfRequiredOutputs( 1 );
}


/**
 *
 */
template <class TInputGraph, class TOutputGraph>
void
GraphToGraphFilter<TInputGraph,TOutputGraph>
::SetInput( TInputGraph *input )
{
  this->ProcessObject::SetNthInput( 0, input );
}


/**
 *
 */
template <class TInputGraph, class TOutputGraph>
typename GraphToGraphFilter<TInputGraph,TOutputGraph>::InputGraphType *
GraphToGraphFilter<TInputGraph,TOutputGraph>
::GetInput()
{
  if ( this->GetNumberOfInputs() < 1 )
    {
    return 0;
    }

  return static_cast<TInputGraph*>
    ( this->ProcessObject::GetInput( 0 ) );
}


/**
 *
 */
template <class TInputGraph, class TOutputGraph>
typename GraphToGraphFilter<TInputGraph,TOutputGraph>::InputGraphType *
GraphToGraphFilter<TInputGraph,TOutputGraph>
::GetInput( unsigned int idx )
{
  return static_cast<TInputGraph*>
    ( this->ProcessObject::GetInput( idx ) );
}


} // end namespace itk

#endif
