#ifndef __itkSignedMaurerDistanceMapImageFilter_txx
#define __itkSignedMaurerDistanceMapImageFilter_txx

#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "itkSignedDanielssonDistanceMapImageFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkFastIncrementalBinaryDilateImageFilter.h"
#include "itkUnaryFunctorImageFilter.h"
#include "itkSubtractImageFilter.h"

#include "vnl/vnl_vector.h"


namespace itk
{

template<class TInputImage, class TOutputImage>
void
SignedMaurerDistanceMapImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  this->GetOutput()->SetRegions(this->GetInput()->GetRequestedRegion());
  this->GetOutput()->Allocate();
  
  m_Spacing = this->GetOutput()->GetSpacing();
  m_MaximumValue = vnl_huge_val(m_MaximumValue);

  m_BinaryImage = InputImageType::New();
  m_BinaryImage->SetRegions(this->GetInput()->GetRequestedRegion());
  m_BinaryImage->Allocate();

  typedef BinaryThresholdImageFilter<InputImageType, 
                                     InputImageType> BinaryFilterType;
  typename BinaryFilterType::Pointer binaryFilter = BinaryFilterType::New();
  binaryFilter->SetLowerThreshold(m_BackgroundValue);
  binaryFilter->SetUpperThreshold(m_BackgroundValue);
  binaryFilter->SetInsideValue(0);
  binaryFilter->SetOutsideValue(1);
  binaryFilter->SetInput(this->GetInput());
  binaryFilter->Update();
  m_BinaryImage = binaryFilter->GetOutput();

  typedef Functor::InvertIntensityFunctor<InputPixelType>  FunctorType;
  typedef UnaryFunctorImageFilter< InputImageType, 
                                   InputImageType,
                                   FunctorType >    InverterType;

  typename InverterType::Pointer inverter1 = InverterType::New();
  typename InverterType::Pointer inverter2 = InverterType::New();

  inverter1->SetInput(m_BinaryImage);

  //Dilate the inverted image by 1 pixel to give it the same boundary
  //as the univerted this->GetInput().
  
  typedef BinaryBallStructuringElement< 
                     InputPixelType, 
                     InputImageDimension  > StructuringElementType;  

  typedef FastIncrementalBinaryDilateImageFilter< 
                         InputImageType, 
                         InputImageType, 
                         StructuringElementType >     DilatorType; 

  typename DilatorType::Pointer dilator = DilatorType::New();

  StructuringElementType structuringElement;
  structuringElement.SetRadius(1);
  structuringElement.CreateStructuringElement();
  dilator->SetKernel(structuringElement);
  dilator->SetDilateValue(1);
  dilator->SetInput(inverter1->GetOutput());
  inverter2->SetInput(dilator->GetOutput());

  typedef SubtractImageFilter<InputImageType, 
                              InputImageType, 
                              InputImageType > SubtracterType;
  typename SubtracterType::Pointer subtracter = SubtracterType::New();
  subtracter->SetInput1(m_BinaryImage);
  subtracter->SetInput2(inverter2->GetOutput());
  subtracter->Update();

  ImageRegionConstIterator<InputImageType> 
         inIterator(subtracter->GetOutput(), subtracter->GetOutput()->GetRequestedRegion());
  ImageRegionIterator<OutputImageType> outIterator
        (this->GetOutput(), this->GetOutput()->GetRequestedRegion());

  for (inIterator.GoToBegin(), outIterator.GoToBegin(); !inIterator.IsAtEnd(); ++inIterator, ++outIterator)
  {
    outIterator.Set(inIterator.Get() ? 0 : m_MaximumValue);
  }     

  vnl_vector<unsigned int> k(InputImageDimension-1);
  for (unsigned int i = 0; i < InputImageDimension; i++)
  {    
    OutputIndexType idx;
    unsigned int NumberOfRows = 1;
    for (unsigned int d = 0; d < InputImageDimension; d++)
    { 
      idx[d] = 0;   
      if (d != i)
      {
        NumberOfRows *= this->GetInput()->GetRequestedRegion().GetSize()[d];
      }  	  
    }  

    k[0] = 1;  
    unsigned int count = 1;
    for (unsigned int d = i+2; d < i+InputImageDimension; d++)
    { 
      k[count] = k[count-1]*this->GetInput()->GetRequestedRegion().GetSize()[d % InputImageDimension];    
      count++;
    }  
    k.flip();

    unsigned int index;      
    for (unsigned int n = 0; n < NumberOfRows;n++)
    { 
      index = n;
      count = 0;
      for (unsigned int d = i+1; d < i+InputImageDimension; d++)      
      {
        idx[d % InputImageDimension] = static_cast<unsigned int>(static_cast<double>(index)/static_cast<double>(k[count]));
        index %= k[count];
   	    count++;
      }	
      this->VoronoiEDT(i, idx);
    }  
  }

  if (!m_SquaredDistance)
  { 
    ImageRegionIteratorWithIndex<OutputImageType> It(this->GetOutput(), this->GetOutput()->GetRequestedRegion());

    for (It.GoToBegin(); !It.IsAtEnd(); ++It)
    {
      (m_BinaryImage->GetPixel(It.GetIndex()) && m_InsideIsPositive) 
        ? It.Set( static_cast<OutputPixelType>(sqrt(double(std::fabs(It.Get())))))
        : It.Set(-static_cast<OutputPixelType>(sqrt(double(std::fabs(It.Get())))));
    }
  }  
}

template<class TInputImage, class TOutputImage>
void
SignedMaurerDistanceMapImageFilter<TInputImage, TOutputImage>
::VoronoiEDT(unsigned int d, OutputIndexType idx)
{
  typename OutputImageType::Pointer output(this->GetOutput());
  unsigned int nd = output->GetRequestedRegion().GetSize()[d];

  vnl_vector<OutputPixelType> g(nd);  g = 0;
  vnl_vector<OutputPixelType> h(nd);  h = 0;
  OutputPixelType di;

  int l = -1;
  for (unsigned int i = 0; i < nd; i++)
  {
    idx[d] = i;

    di = output->GetPixel(idx);
    
    OutputPixelType iw = (m_UseImageSpacing) ? static_cast<OutputPixelType>(i*m_Spacing[d])
                                             : static_cast<OutputPixelType>(i);
    
    if (di != m_MaximumValue)
    {
      if (l < 1)
      {
        l++;
        g(l) = di;
        h(l) = iw;
      }  	
      else
      {         
        while ((l >= 1) && this->RemoveEDT(g(l-1), g(l), di, h(l-1), h(l), iw))
	{
          l--;
	}  
        l++;
        g(l) = di;
        h(l) = iw;
      }
    }    	
  }

  if (l == -1)  return;

  int ns = l;
  l = 0;
  for (unsigned int i = 0; i < nd; i++)
  {
    OutputPixelType iw = (m_UseImageSpacing) ? static_cast<OutputPixelType>(i*m_Spacing[d])
                                             : static_cast<OutputPixelType>(i);

    OutputPixelType d1 = std::fabs(g(l  )) + (h(l  )-iw)*(h(l  )-iw);
    OutputPixelType d2 = std::fabs(g(l+1)) + (h(l+1)-iw)*(h(l+1)-iw);
    while ((l < ns) && (d1 > d2))
    {
      l++;
      d1 = d2;
      d2 = std::fabs(g(l+1)) + (h(l+1)-iw)*(h(l+1)-iw);      
    }      
    idx[d] = i;
    (m_BinaryImage->GetPixel(idx) && m_InsideIsPositive) 
            ? output->SetPixel(idx,  d1) 
            : output->SetPixel(idx, -d1);
  }  
}

template<class TInputImage, class TOutputImage>
bool
SignedMaurerDistanceMapImageFilter<TInputImage, TOutputImage>
::RemoveEDT(OutputPixelType d1, OutputPixelType d2, OutputPixelType df, 
            OutputPixelType x1, OutputPixelType x2, OutputPixelType xf)
{
  OutputPixelType a = x2 - x1;
  OutputPixelType b = xf - x2;
  OutputPixelType c = xf - x1;

  return ((c*std::fabs(d2) - b*std::fabs(d1) - a*std::fabs(df) - a*b*c) > 0);
}

/**
 * Standard "PrintSelf" method
 */
template <class TInputImage, class TOutputImage>
void
SignedMaurerDistanceMapImageFilter<TInputImage, TOutputImage>
::PrintSelf(
  std::ostream& os, 
  Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Background Value: " << this->m_BackgroundValue << std::endl;
}

} // end namespace itk

#endif
