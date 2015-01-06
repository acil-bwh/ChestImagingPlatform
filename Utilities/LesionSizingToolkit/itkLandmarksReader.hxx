/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkLandmarksReader.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkLandmarksReader_hxx
#define __itkLandmarksReader_hxx

#include "itkLandmarksReader.h"
#include "itkSpatialObjectReader.h"


namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
LandmarksReader<NDimension>
::LandmarksReader()
{
  this->SetNumberOfRequiredOutputs( 1 );

  this->m_SpatialObjectReader = SpatialObjectReaderType::New();

  typename SpatialObjectType::Pointer outputObject = SpatialObjectType::New();

  this->ProcessObject::SetNthOutput( 0, outputObject );
}


/**
 * Destructor
 */
template <unsigned int NDimension>
LandmarksReader<NDimension>
::~LandmarksReader()
{
}

template <unsigned int NDimension>
const typename LandmarksReader<NDimension>::SpatialObjectType *
LandmarksReader<NDimension>
::GetOutput() const
{
  return static_cast<const SpatialObjectType*>(this->ProcessObject::GetOutput(0));
}

template <unsigned int NDimension>
void
LandmarksReader<NDimension>
::GenerateData()
{
  this->m_SpatialObjectReader->SetFileName( this->GetFileName() );

  this->m_SpatialObjectReader->Update();

  typename SpatialObjectReaderType::ScenePointer scene = this->m_SpatialObjectReader->GetScene();

  if( !scene )
    {
    itkExceptionMacro("Couldn't fine a scene in file" << this->GetFileName() );
    }

  ObjectListType * sceneChildren = scene->GetObjects(999999);

  typename ObjectListType::const_iterator spatialObjectItr = sceneChildren->begin();

  typename SpatialObjectType::Pointer landmarkSpatialObject;

  while( spatialObjectItr != sceneChildren->end() ) 
    {
    std::string objectName = (*spatialObjectItr)->GetTypeName();
    if( objectName == "LandmarkSpatialObject" )
      {
      const SpatialObjectType * landmarkSpatialObjectRawPointer = 
        dynamic_cast< const SpatialObjectType * >( spatialObjectItr->GetPointer() );
      landmarkSpatialObject =
        const_cast< SpatialObjectType * >( landmarkSpatialObjectRawPointer );
      break;
      }
    spatialObjectItr++;
    }

  if( landmarkSpatialObject.IsNull() )
    {
    itkExceptionMacro("Input file does not contain landmarks");
    }
 
  landmarkSpatialObject->DisconnectPipeline();

  SpatialObjectType * outputObject =
    dynamic_cast< SpatialObjectType * >(this->ProcessObject::GetOutput(0));

  outputObject->SetPoints( landmarkSpatialObject->GetPoints() );

  delete sceneChildren;
}

/*
 * PrintSelf
 */
template <unsigned int NDimension>
void
LandmarksReader<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << this->m_FileName << std::endl;
}


} // end namespace itk

#endif
