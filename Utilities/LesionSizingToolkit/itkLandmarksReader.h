/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkLandmarksReader.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkLandmarksReader_h
#define __itkLandmarksReader_h

#include "itkProcessObject.h"
#include "itkImage.h"
#include "itkLandmarkSpatialObject.h"
#include "itkSpatialObjectReader.h"

namespace itk
{

/** \class LandmarksReader
 * \brief Class that reads a file containing spatial object landmarks.
 *
 * A LandmarkSpatialObject is produced as output.
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT LandmarksReader : public ProcessObject
{
public:
  /** Standard class typedefs. */
  typedef LandmarksReader              Self;
  typedef ProcessObject                Superclass;
  typedef SmartPointer<Self>           Pointer;
  typedef SmartPointer<const Self>     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(LandmarksReader, ProcessObject);

  /** Dimension of the space */
  itkStaticConstMacro(Dimension, unsigned int, NDimension);

  /** Type of spatialObject that will be passed as input and output of this
   * segmentation method. */
  typedef LandmarkSpatialObject< NDimension >   SpatialObjectType;
  typedef typename SpatialObjectType::Pointer   SpatialObjectPointer;

  /** Output data that carries the feature in the form of a
   * SpatialObject. */
  const SpatialObjectType * GetOutput() const;

  /** Set / Get the input filename */
  itkSetStringMacro( FileName );
  itkGetStringMacro( FileName );

protected:
  LandmarksReader();
  virtual ~LandmarksReader();
  void PrintSelf(std::ostream& os, Indent indent) const;

  void GenerateData();

private:
  LandmarksReader(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented


  typedef SpatialObjectReader< NDimension, unsigned short >   SpatialObjectReaderType;
  typedef typename SpatialObjectReaderType::Pointer           SpatialObjectReaderPointer;
  typedef typename SpatialObjectReaderType::SceneType         SceneType;
  typedef typename SceneType::ObjectListType                  ObjectListType;

  std::string                     m_FileName;
  SpatialObjectReaderPointer      m_SpatialObjectReader;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkLandmarksReader.hxx"
#endif

#endif
