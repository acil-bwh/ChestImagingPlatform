#ifndef __GenerateLobeSurfaceModelsHelper_h
#define __GenerateLobeSurfaceModelsHelper_h

#include "itkImage.h"
#include "cipHelper.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "cipChestConventions.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkExtractImageFilter.h"

struct REGIONTYPEPAIR
{
  unsigned char region;
  unsigned char type;
};

typedef unsigned short                               LabelMapPixelType;
typedef itk::Image< LabelMapPixelType, 2 >  LabelMapSliceType;
typedef LabelMapSliceType::IndexType        LabelMapSliceIndexType;
typedef itk::CIPExtractChestLabelMapImageFilter                                                   LabelMapChestExtractorType;
typedef itk::ExtractImageFilter< cip::LabelMapType, LabelMapSliceType >                             LabelMapSliceExtractorType;

typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType >                                    LabelMapIteratorType;
typedef itk::ImageRegionIteratorWithIndex< LabelMapSliceType >                                    LabelMapIterator2DType;

typedef itk::ConnectedComponentImageFilter< cip::LabelMapType, cip::LabelMapType >                 ConnectedComponent3DType;
  typedef itk::ConnectedComponentImageFilter< LabelMapSliceType, LabelMapSliceType >             ConnectedComponent2DType;

typedef itk::RelabelComponentImageFilter <cip::LabelMapType, cip::LabelMapType>    RelabelFilter3DType;
typedef itk::RelabelComponentImageFilter <LabelMapSliceType, LabelMapSliceType>   RelabelFilter2DType;
typedef itk::ImageRegionIteratorWithIndex< LabelMapSliceType >                                 LabelMapSliceIteratorType;

cip::LabelMapType::Pointer ReadLabelMapFromFile( std::string);
void ExtractLabelMapSlice( cip::LabelMapType::Pointer , LabelMapSliceType::Pointer , int , std::string  );
cip::LabelMapType::Pointer FilterConnectedComponents(cip::LabelMapType::Pointer , int , std::vector< unsigned char> , std::vector< unsigned char> , std::vector<REGIONTYPEPAIR> , std::string  , bool , bool );




#endif
