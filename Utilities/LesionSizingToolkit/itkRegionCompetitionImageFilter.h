/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkRegionCompetitionImageFilter.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkRegionCompetitionImageFilter_h
#define __itkRegionCompetitionImageFilter_h

#include "itkImage.h"
#include "itkImageToImageFilter.h"

#include <vector>

namespace itk
{

/** \class RegionCompetitionImageFilter 
 *
 * \brief Perform front-propagation from different starting labeled regions.
 * 
 * The filter expects two inputs: One gray-scale image and a labeled image.
 * The labels will be used as initial regions from which the fronts will be
 * propagated until they collide with other labeled regions. Each labeled front
 * will compete for pixels against other labels.
 *
 * \ingroup RegionGrowingSegmentation 
 * \ingroup LesionSizingToolkit
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT RegionCompetitionImageFilter:
    public ImageToImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef RegionCompetitionImageFilter                    Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>    Superclass;
  typedef SmartPointer<Self>                              Pointer;
  typedef SmartPointer<const Self>                        ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods).  */
  itkTypeMacro(RegionCompetitionImageFilter, ImageToImageFilter);

  typedef TInputImage             InputImageType;
  typedef typename InputImageType::Pointer                InputImagePointer;
  typedef typename InputImageType::ConstPointer           InputImageConstPointer;
  typedef typename InputImageType::RegionType             InputImageRegionType; 
  typedef typename InputImageType::SizeType               InputSizeType;
  typedef typename InputImageType::PixelType              InputImagePixelType; 
  typedef typename InputImageType::IndexType              IndexType;
  typedef typename InputImageType::OffsetValueType        OffsetValueType;
  
  typedef TOutputImage            OutputImageType;
  typedef typename OutputImageType::Pointer               OutputImagePointer;
  typedef typename OutputImageType::RegionType            OutputImageRegionType; 
  typedef typename OutputImageType::PixelType             OutputImagePixelType; 
  
  
  /** Image dimension constants */
  itkStaticConstMacro(InputImageDimension,  unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int, TOutputImage::ImageDimension);

  /** Set/Get the maximum number of iterations that will be applied to the
   * propagating front */
  itkSetMacro( MaximumNumberOfIterations, unsigned int );
  itkGetMacro( MaximumNumberOfIterations, unsigned int );

  /** Returned the number of iterations used so far. */
  itkGetMacro( CurrentIterationNumber, unsigned int );

  /** Returned the number of pixels changed in total. */
  itkGetMacro( TotalNumberOfPixelsChanged, unsigned int );

  /** Input Labels */
  void SetInputLabels( const TOutputImage * inputLabelImage );

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(OutputEqualityComparableCheck, (Concept::EqualityComparable<OutputImagePixelType>));
  itkConceptMacro(InputEqualityComparableCheck, (Concept::EqualityComparable<InputImagePixelType>));
  itkConceptMacro(InputConvertibleToOutputCheck, (Concept::Convertible<InputImagePixelType, OutputImagePixelType>));
  itkConceptMacro(SameDimensionCheck, (Concept::SameDimension<InputImageDimension, OutputImageDimension>));
  itkConceptMacro(IntConvertibleToInputCheck, (Concept::Convertible<int, InputImagePixelType>));
  itkConceptMacro(OutputOStreamWritableCheck, (Concept::OStreamWritable<OutputImagePixelType>));
  /** End concept checking */
#endif

protected:
  RegionCompetitionImageFilter();
  ~RegionCompetitionImageFilter();

  void GenerateData();
  
  void PrintSelf ( std::ostream& os, Indent indent ) const;

private:
  RegionCompetitionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented


  void AllocateOutputImageWorkingMemory();

  void AllocateFrontsWorkingMemory();

  void ComputeNumberOfInputLabels();

  void InitializeNeighborhood();

  void FindAllPixelsInTheBoundaryAndAddThemAsSeeds();

  void IterateFrontPropagations();

  void VisitAllSeedsAndTransitionTheirState();

  void PasteNewSeedValuesToOutputImage();

  void SwapSeedArrays();

  void ClearSecondSeedArray();

  bool TestForAvailabilityAtCurrentPixel() const;
 
  void PutCurrentPixelNeighborsIntoSeedArray();

  void ComputeArrayOfNeighborhoodBufferOffsets();

  void ComputeBirthThreshold();

  itkSetMacro( CurrentPixelIndex, IndexType );
  itkGetConstReferenceMacro( CurrentPixelIndex, IndexType );

  typedef std::vector<IndexType>    SeedArrayType;

  SeedArrayType *                   m_SeedArray1;
  SeedArrayType *                   m_SeedArray2;

  InputImageRegionType              m_InternalRegion;
  
  typedef std::vector<OutputImagePixelType> SeedNewValuesArrayType;

  SeedNewValuesArrayType *          m_SeedsNewValues;

  unsigned int                      m_CurrentIterationNumber;
  unsigned int                      m_MaximumNumberOfIterations;
  unsigned int                      m_NumberOfPixelsChangedInLastIteration;
  unsigned int                      m_TotalNumberOfPixelsChanged;
  
  IndexType                         m_CurrentPixelIndex;

  //
  // Variables used for addressing the Neighbors.
  // This could be factorized into a helper class.
  //
  OffsetValueType                   m_OffsetTable[ InputImageDimension + 1 ]; 
  
  typedef std::vector< OffsetValueType >   NeighborOffsetArrayType;

  NeighborOffsetArrayType           m_NeighborBufferOffset;


  //
  // Helper cache variables 
  //
  const InputImageType *            m_InputImage;
  const OutputImageType*       m_inputLabelsImage; 
  OutputImageType *                 m_OutputImage;

  typedef itk::Image< unsigned char, InputImageDimension >  SeedMaskImageType;
  typedef typename SeedMaskImageType::Pointer               SeedMaskImagePointer;

  SeedMaskImagePointer              m_SeedsMask;

  typedef itk::Neighborhood< InputImagePixelType, InputImageDimension >  NeighborhoodType;

  NeighborhoodType                  m_Neighborhood;

  mutable unsigned int              m_NumberOfLabels;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRegionCompetitionImageFilter.hxx"
#endif

#endif
