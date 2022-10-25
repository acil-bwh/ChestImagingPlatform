/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkDefaultGraphTraits.h,v $
  Language:  C++
  Date:      $Date: 2009/12/01 20:25:46 $
  Version:   $Revision: 1.2 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  Portions of this code are covered under the VTK copyright.
  See VTKCopyright.txt or http://www.kitware.com/VTKCopyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkDefaultGraphTraits_h
#define __itkDefaultGraphTraits_h

#include <vector>

#include <itkConfigure.h>

namespace itk
{

/** \class DefaultGraphTraits
 *  \brief Base class for all graph traits.
 *
 *  Defines the node and edge structures.  Templated over
 *  the edge and node weight types.
 *
 */


template <typename TNodeWeight = short, typename TEdgeWeight = short>
class DefaultGraphTraits
{
public:
  typedef DefaultGraphTraits Self;

  struct  EdgeType;
  struct  NodeType;
  
  typedef NodeType*                       NodePointerType;
  typedef EdgeType*                       EdgePointerType;
  typedef unsigned long                   NodeIdentifierType;
  typedef unsigned long                   EdgeIdentifierType;
  typedef TNodeWeight                     NodeWeightType;
  typedef TEdgeWeight                     EdgeWeightType;
  typedef std::vector<EdgeIdentifierType> EdgeIdentifierContainerType;

  struct NodeType
    {
    NodeIdentifierType          Identifier;
    EdgeIdentifierContainerType IncomingEdges;
    EdgeIdentifierContainerType OutgoingEdges;
    NodeWeightType              Weight;
    };

  struct EdgeType
    {
    EdgeIdentifierType Identifier;
    NodeIdentifierType SourceIdentifier;
    NodeIdentifierType TargetIdentifier;
    EdgeIdentifierType ReverseEdgeIdentifier;
    EdgeWeightType     Weight;
    };
};


} // end namespace itk

#endif
