/** \file
 *  \ingroup commandLineTools 
 */

#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageSeriesReader.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include <fstream>
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "CropLungCLP.h"

#define MAXBB 10000
#define MINBB -10000

extern "C" int CheckPointEntryPoint()
{
  std::cout << "This is a dummy checkpoint test for demo purposes" << std::endl;
  return cip::EXITSUCCESS;
}

namespace
{
  typedef itk::GDCMImageIO                                                          ImageIOType;
  typedef itk::GDCMSeriesFileNames                                                  NamesGeneratorType;
  typedef itk::ImageSeriesReader< cip::CTType >                                     CTSeriesReaderType;
  typedef itk::ImageFileReader< cip::CTType >                                       CTFileReaderType;
  typedef itk::CIPExtractChestLabelMapImageFilter< 3 >                              LabelMapExtractorType;
  typedef itk::RegionOfInterestImageFilter< cip::CTType, cip::CTType >              RegionOfInterestFilterType;
  typedef itk::RegionOfInterestImageFilter< cip::LabelMapType, cip::LabelMapType >  RegionOfInterestLabelMapFilterType;
  typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType >                    LabelMapIteratorType;
  typedef itk::ImageRegionIterator< cip::CTType >                                   CTImageIteratorType;
  typedef itk::ImageRegionIterator< cip::LabelMapType >                             LabelMapIteratorType2;

  struct REGIONTYPEPAIR
  {
    unsigned char region;
    unsigned char type;
  };

  cip::CTType::Pointer ReadCTFromFile( std::string  );
  cip::LabelMapType::Pointer ReadLabelMapFromFile( std::string );
 
  cip::CTType::Pointer ReadCTFromFile( std::string fileName )
  {
    CTFileReaderType::Pointer reader = CTFileReaderType::New();
    reader->SetFileName( fileName );
    try
      {
      reader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught reading CT image:";
      std::cerr << excp << std::endl;
      return cip::CTType::Pointer(nullptr);
      }
    
    return reader->GetOutput();
  }
    
  cip::LabelMapType::Pointer ReadLabelMapFromFile( std::string labelMapFileName )
  {
    std::cout << "Reading label map..." << std::endl;
    cip::LabelMapReaderType::Pointer reader = cip::LabelMapReaderType::New();
    reader->SetFileName( labelMapFileName );
    try
      {
	reader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
	std::cerr << "Exception caught reading label map:";
	std::cerr << excp << std::endl;
      }
    
    return reader->GetOutput();
  }
  
} //end namespace

int main( int argc, char *argv[] )
{
  //
  // old Definiton of command line argument variables
  //
  //std::string  ctDir              = "q";
  //std::string ctDirArgDescription = "Input dicom directory with CT scan";
  //TCLAP::ValueArg<std::string> ctDirArg ( "", "dicom", ctDirArgDescription, false, ctDir, "directory", cl );
  
  //cl.xorAdd( ctDirArg, ctFileNameArg );
  
  std::vector< unsigned char >  regionVec;
  std::vector< unsigned char >  typeVec;
  std::vector< REGIONTYPEPAIR > regionTypePairVec;
  cip::ChestConventions conventions;
  
  
  PARSE_ARGS;
  
  // Param error checking
  if (paddingVecArg.size()!=3)
    {
      std::cout<<" Padding needs three input params"<<std::endl;
      return cip::ARGUMENTPARSINGERROR;
    }
  
  
  short maskValue = (short)(maskValueTemp);
  
  /*
    if (ctDirArg.isSet() ) 
    {
    ctDir = ctDirArg.getValue();
    }
    else if (ctFileNameArg.isSet())
    {
    ctFileName = ctFileNameArg.getValue();  
    }
    else
    {
    TCLAP::ArgException("Either CT Dicom directory or CT filename is necessary");
    }
  */
  
  for ( unsigned int i=0; i<regionVecArg.size(); i++ )
    {
      regionVec.push_back( (unsigned char) conventions.GetChestRegionValueFromName(regionVecArg[i]) );
    }
  for ( unsigned int i=0; i<typeVecArg.size(); i++ )
    { 
      typeVec.push_back( (unsigned char) conventions.GetChestTypeValueFromName(typeVecArg[i]) );
    }
  if (regionPairVecArg.size() == typePairVecArg.size())
    {
      for ( unsigned int i=0; i<regionPairVecArg.size(); i++ )
	{
	  REGIONTYPEPAIR regionTypePairTemp;
	  
	  regionTypePairTemp.region = (unsigned char)conventions.GetChestRegionValueFromName(regionPairVecArg[i]);
	  //argc--; argv++;
	  regionTypePairTemp.type   = (unsigned char)conventions.GetChestTypeValueFromName(typePairVecArg[i]);
	  
	  regionTypePairVec.push_back( regionTypePairTemp );
	} 
    }
  
  
  //
  // First get the CT image, either from a directory or from a single
  // file 
  //
  cip::CTType::Pointer ctImage;
  
  if (strcmp( ctFileName.c_str(), "q") != 0 )
    {
      std::cout << "Reading CT from file..." << std::endl;
      ctImage = ReadCTFromFile( ctFileName );
      
      if (ctImage.GetPointer() == nullptr)
        {
          return cip::NRRDREADFAILURE;
        }
    }
  else
    {
      std::cerr << "ERROR: No CT image specified" << std::endl;
      return cip::EXITFAILURE;
    }
  
  //
  // Now get the label map. Get it from an input file or
  // compute it if an inpute file has not been specified
  //
  cip::LabelMapType::Pointer labelMap = cip::LabelMapType::New();
  
  if ( strcmp( plInputFileName.c_str(), "q") != 0 )
    {
      std::cout << "Reading label map from file..." << std::endl;
      labelMap = ReadLabelMapFromFile( plInputFileName );
      
      if (labelMap.GetPointer() == nullptr)
        {
          return cip::LABELMAPREADFAILURE;
        }
    }
  else
    {
      std::cerr <<"Error: No lung label map specified"<< std::endl;
      return cip::EXITFAILURE;
    }
  
  //
  // Extract region that we want to crop
  //
  std::cout << "Extracting region and type..." << std::endl;
  LabelMapExtractorType::Pointer extractor = LabelMapExtractorType::New();
  extractor->SetInput( labelMap );
  for ( unsigned int i=0; i<regionVec.size(); i++ )
    {
      extractor->SetChestRegion( regionVec[i] );
    }
  for ( unsigned int i=0; i<typeVec.size(); i++ )
    {
      extractor->SetChestType( typeVec[i] );
    }
  for ( unsigned int i=0; i<regionTypePairVec.size(); i++ )
    {
      std::cout<<"Region: "<<(int) regionTypePairVec[i].region<<" Type: "<<(int) regionTypePairVec[i].type<<std::endl;
      extractor->SetRegionAndType( regionTypePairVec[i].region, regionTypePairVec[i].type );
    }  
  extractor->Update();
  
  //
  // Loop through extracted region and find bounding box
  //
  std::cout << "Computing Bounding Box..." << std::endl;
  CTImageIteratorType   ctIt ( ctImage, ctImage->GetBufferedRegion() );
  LabelMapIteratorType  lIt( extractor->GetOutput(), extractor->GetOutput()->GetBufferedRegion() );
  LabelMapIteratorType2 l2It (labelMap, labelMap->GetBufferedRegion() );
  
  cip::LabelMapType::IndexType index;
  
  lIt.GoToBegin();
  l2It.GoToBegin();
  ctIt.GoToBegin();
  int bbox[6];
  bbox[0] = MAXBB;
  bbox[1] = MINBB;
  bbox[2] = MAXBB;
  bbox[3] = MINBB;
  bbox[4] = MAXBB;
  bbox[5] = MINBB;
  int bboxset =0;
  while ( !lIt.IsAtEnd() )
    {
      if (lIt.Get() != 0)
        {
	  index = lIt.GetIndex();
          for (int i=0;i<cip::CTType::ImageDimension;i++)
	    {
	      if (index[i] < bbox[i*2])
		bbox[i*2] = static_cast<int>( index[i] );
	      if (index[i] > bbox[i*2+1])
		bbox[i*2+1] = static_cast<int>( index[i] );
	    }
          bboxset=1;
	}
      if (maskOutputFlag && l2It.Get() == 0)
	{
	  ctIt.Set(maskValue);
	}
      ++lIt;
      ++l2It;
      ++ctIt;
    }
  
  if (bboxset == 0 )
    {
      std::cout<<"Region/type not present"<<std::endl;
      return cip::EXITFAILURE;
    }
  
  for (int i=0;i<2*cip::CTType::ImageDimension;i++)
    std::cout<<i<<": "<<bbox[i]<<std::endl;
  
  // Add padding values
  for (int i=0; i<cip::CTType::ImageDimension;i++)
    {
      bbox[i*2] -= paddingVecArg[i];
      bbox[i*2+1] += paddingVecArg[i];
    }
  
  // Check bounding box limits
  cip::CTType::RegionType region = ctImage->GetLargestPossibleRegion();
  cip::CTType::SizeType sizeCT = region.GetSize();
  cip::CTType::IndexType indexCT = region.GetIndex();
  
  for (int i=0; i<cip::CTType::ImageDimension; i++)
    {
      if (bbox[i*2] < indexCT[i])
	{
	  bbox[i*2]=indexCT[i];
	}
      if (bbox[i*2+1] > indexCT[i]+sizeCT[i]-1)
	{
	  bbox[i*2+1]=indexCT[i]+sizeCT[i]-1;
	}
    }
  for (int i=0;i<5;i++)
    std::cout<<i<<": "<<bbox[i]<<std::endl;
  
  
  if (strcmp(ctOutputFileName.c_str(),"q") != 0) 
    {
      cip::CTType::RegionType roi;
      cip::CTType::IndexType startIndex;
      cip::CTType::SizeType  regionSize;
      startIndex[0] = bbox[0];
      startIndex[1] = bbox[2];
      startIndex[2] = bbox[4];
      regionSize[0] = bbox[1]-bbox[0]+1;
      regionSize[1] = bbox[3]-bbox[2]+1;
      regionSize[2] = bbox[5]-bbox[4]+1; 
      roi.SetSize( regionSize );
      roi.SetIndex( startIndex );
      
      RegionOfInterestFilterType::Pointer roiFilter = RegionOfInterestFilterType::New();
      roiFilter->SetInput( ctImage );
      try
	{
	  roiFilter->SetRegionOfInterest(roi);
	}
      catch ( itk::ExceptionObject &excp )
	{
          std::cerr << "Exception caught extracting ROI...";
          std::cerr << excp << std::endl;
          return cip::EXITFAILURE;
	}
      roiFilter->Update();
      
      
      std::cout<< "Writing CT cropped image..." << std::endl;
      cip::CTWriterType::Pointer writer = cip::CTWriterType::New();
      writer->SetInput ( roiFilter->GetOutput() );
      writer->SetFileName( ctOutputFileName );
      writer->UseCompressionOn();
      try 
        {
	  writer->Update();
        }
      catch (itk::ExceptionObject &excp )
        {
          std::cerr << "Exception caught writing output image";
          std::cerr << excp << std::endl;
          return cip::NRRDWRITEFAILURE;
        }
      
    }
  
  //Check if we have to produce the labelmap output  
  if (strcmp(plOutputFileName.c_str(),"q") != 0) 
    {
      cip::LabelMapType::RegionType roi2;
      cip::LabelMapType::IndexType startIndex2;
      cip::LabelMapType::SizeType  regionSize2;
      startIndex2[0] = bbox[0];
      startIndex2[1] = bbox[2];
      startIndex2[2] = bbox[4];
      regionSize2[0] = bbox[1]-bbox[0]+1;
      regionSize2[1] = bbox[3]-bbox[2]+1;
      regionSize2[2] = bbox[5]-bbox[4]+1; 
      roi2.SetSize( regionSize2 );
      roi2.SetIndex( startIndex2 );
      
      RegionOfInterestLabelMapFilterType::Pointer roiFilter = RegionOfInterestLabelMapFilterType::New();
      roiFilter->SetInput( labelMap );
      try
	{
          roiFilter->SetRegionOfInterest(roi2);
          roiFilter->Update();
	}
      catch ( itk::ExceptionObject &excp )
	{
          std::cerr << "Exception caught extracting ROI:";
          std::cerr << excp << std::endl;
          return cip::EXITFAILURE;
	}
      
      std::cout<< "Writing cropped label map..." << std::endl;
      cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
      writer->SetInput ( roiFilter->GetOutput() );
      writer->SetFileName( plOutputFileName );
      writer->UseCompressionOn();
      try
        {
	  writer->Update();
        } 
      catch ( itk::ExceptionObject &excp )
        {
	  std::cerr << "Exception caught extracting ROI:";
	  std::cerr << excp << std::endl;
	  return cip::LABELMAPWRITEFAILURE;
        }
    }
  
  std::cout<< "DONE." << std::endl;
  
  return cip::EXITSUCCESS;
}


