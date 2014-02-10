/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __itkUnaryFunctorImageFilter2_h
#define __itkUnaryFunctorImageFilter2_h

#include "itkInPlaceImageFilter.h"
#include "itkUnaryFunctorBase.h"


namespace itk
{
/** \class UnaryFunctorImageFilter2
 * \brief Implements pixel-wise generic operation on one image.
 *
 * This class is parameterized over the type of the input image and
 * the type of the output image.  It is also parameterized by the
 * operation to be applied, using a Functor style.
 *
 * UnaryFunctorImageFilter2 allows the output dimension of the filter
 * to be larger than the input dimension. Thus subclasses of the
 * UnaryFunctorImageFilter (like the CastImageFilter) can be used
 * to promote a 2D image to a 3D image, etc.
 *
 * Compared to the original UnaryFunctorImageFilter this class is not
 * templated over the Functor type, but allows run-time setting of the
 * Functor, as long as it derives from a UnaryFunctorBase class.
 *
 * \sa BinaryFunctorImageFilter2 UnaryFunctorBase
 *
 * \ingroup   IntensityImageFilters     MultiThreaded
 * \ingroup ITKCommon
 *
 * \wiki
 * \wikiexample{ImageProcessing/UnaryFunctorImageFilter,Apply a custom operation to each pixel in an image}
 * \endwiki
 */
template< class TInputImage, class TOutputImage >
class ITK_EXPORT UnaryFunctorImageFilter2
  : public InPlaceImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef UnaryFunctorImageFilter2                        Self;
  typedef InPlaceImageFilter< TInputImage, TOutputImage > Superclass;
  typedef SmartPointer< Self >                            Pointer;
  typedef SmartPointer< const Self >                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( UnaryFunctorImageFilter2, InPlaceImageFilter );

  /** Some typedefs. */
  typedef TInputImage                               InputImageType;
  typedef typename    InputImageType::ConstPointer  InputImagePointer;
  typedef typename    InputImageType::RegionType    InputImageRegionType;
  typedef typename    InputImageType::PixelType     InputImagePixelType;
  typedef TOutputImage                              OutputImageType;
  typedef typename     OutputImageType::Pointer     OutputImagePointer;
  typedef typename     OutputImageType::RegionType  OutputImageRegionType;
  typedef typename     OutputImageType::PixelType   OutputImagePixelType;

  /** Set a functor deriving from UnaryFunctorBase. */
  typedef Functor::UnaryFunctorBase<
    InputImagePixelType, OutputImagePixelType >     FunctorType;
  itkSetObjectMacro( Functor, FunctorType );

protected:
  UnaryFunctorImageFilter2();
  virtual ~UnaryFunctorImageFilter2() {}

  /** UnaryFunctorImageFilter can produce an image which is a different
   * resolution than its input image.  As such, UnaryFunctorImageFilter
   * needs to provide an implementation for
   * GenerateOutputInformation() in order to inform the pipeline
   * execution model.  The original documentation of this method is
   * below.
   *
   * \sa ProcessObject::GenerateOutputInformaton()  */
  virtual void GenerateOutputInformation( void );

  /** UnaryFunctorImageFilter can be implemented as a multithreaded filter.
   * Therefore, this implementation provides a ThreadedGenerateData() routine
   * which is called for each processing thread. The output image data is
   * allocated automatically by the superclass prior to calling
   * ThreadedGenerateData().  ThreadedGenerateData can only write to the
   * portion of the output image specified by the parameter
   * "outputRegionForThread"
   *
   * \sa ImageToImageFilter::ThreadedGenerateData(),
   *     ImageToImageFilter::GenerateData()  */
  virtual void ThreadedGenerateData(
    const OutputImageRegionType & outputRegionForThread,
    ThreadIdType threadId );

private:
  UnaryFunctorImageFilter2(const Self &); // purposely not implemented
  void operator=(const Self &);           // purposely not implemented

  typename FunctorType::Pointer m_Functor;
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkUnaryFunctorImageFilter2.hxx"
#endif

#endif
