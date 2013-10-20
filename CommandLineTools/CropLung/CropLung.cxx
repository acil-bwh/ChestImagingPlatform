/** \file
 *  \ingroup commandLineTools 
 *  \details This program takes a CT volume and a Lung label map and
 *  crops the input volume and/or label map to the specified
 *  region/type
 * 
 *  USAGE: 
 *
 *   ./CropLung  [--opl \<filename\>] -o \<filename\> 
 *               [--typePair \<unsigned char\>] ...  
 *               [--regionPair \<unsigned char\>] ...  
 *               [-t \<unsigned char\>] ...  [-r \<unsigned char\>]
 *               ...  
 *               [-m \<boolean\>] [-v \<integer\>] --plf \<string\> -i
 *               \<filename\> 
 *               [--] [--version] [-h]
 *
 *  Where: 
 *
 *   --opl \<filename\>
 *     Ouput label map volume
 *
 *   -o \<filename\>,  --outFileName \<filename\>
 *     (required)  Output Cropped CT volume
 *
 *   --typePair \<unsigned char\>  (accepted multiple times)
 *     Specify a type in a region-type pair you want to crop. This flag 
 *     should be used together with the -regionPair flag
 *
 *   --regionPair \<unsigned char\>  (accepted multiple times)
 *     Specify a region in a region-type pair you want to crop. This flag 
 *     should be used together with the -typePair flag
 *
 *   -t \<unsigned char\>,  --type \<unsigned char\>  (accepted multiple times)
 *     Specify a type you want to crop
 *
 *   -r \<unsigned char\>,  --region \<unsigned char\>  (accepted multiple times)
 *     Specify a region you want to crop
 *
 *   -m \<boolean\>,  --maskFlag \<boolean\>
 *     Set to 0 if you don't want the voxels outside the defined region-type
 *     to be set to a fixed value. Set to 1 otherwise (default=1)
 *
 *   -v \<integer\>,  --value \<integer\>
 *     Value to set voxels outside the region that is cropped. (default=0)
 *
 *   --plf \<string\>
 *     (required)  Label map file name
 *
 *   -i \<filename\>,  --inFileName \<filename\>
 *     (required)  Input CT file
 *
 *   --,  --ignore_rest
 *     Ignores the rest of the labeled arguments following this flag.
 *
 *   --version
 *     Displays version information and exits.
 *
 *   -h,  --help
 *     Displays usage information and exits.
 *
 * 
 *  $Date: 2012-10-24 17:51:27 -0400 (Wed, 24 Oct 2012) $
 *  $Revision: 305 $
 *  $Author: jross $
 *
 */

#include <tclap/CmdLine.h>
#include "cipConventions.h"
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

namespace
{
typedef itk::GDCMImageIO                                                          ImageIOType;
typedef itk::GDCMSeriesFileNames                                                  NamesGeneratorType;
typedef itk::ImageSeriesReader< cip::CTType >                                     CTSeriesReaderType;
typedef itk::ImageFileReader< cip::CTType >                                       CTFileReaderType;
typedef itk::CIPExtractChestLabelMapImageFilter                                   LabelMapExtractorType;
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


cip::CTType::Pointer ReadCTFromDirectory( std::string );
cip::CTType::Pointer ReadCTFromFile( std::string  );
cip::LabelMapType::Pointer ReadLabelMapFromFile( std::string );

    cip::CTType::Pointer ReadCTFromDirectory( std::string ctDir )
    {
        ImageIOType::Pointer gdcmIO = ImageIOType::New();
        
        std::cout << "---Getting file names..." << std::endl;
        NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
        namesGenerator->SetInputDirectory( ctDir );
        
        const CTSeriesReaderType::FileNamesContainer & filenames = namesGenerator->GetInputFileNames();
        
        std::cout << "---Reading DICOM image..." << std::endl;
        CTSeriesReaderType::Pointer dicomReader = CTSeriesReaderType::New();
        dicomReader->SetImageIO( gdcmIO );
        dicomReader->SetFileNames( filenames );
        try
        {
            dicomReader->Update();
        }
        catch (itk::ExceptionObject &excp)
        {
            std::cerr << "Exception caught while reading dicom:";
            std::cerr << excp << std::endl;
            return NULL;
        }
        
        return dicomReader->GetOutput();
    }
    
    
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
            return NULL;
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
  
        
    PARSE_ARGS;
        
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
	  regionVec.push_back( (unsigned char) regionVecArg[i] );
	}
    for ( unsigned int i=0; i<typeVecArg.size(); i++ )
	{
	  typeVec.push_back( (unsigned char)typeVecArg[i] );
	}
    if (regionPairVecArg.size() == typePairVecArg.size())
	{
	  for ( unsigned int i=0; i<regionPairVecArg.size(); i++ )
	    {
	      REGIONTYPEPAIR regionTypePairTemp;
	      
	      regionTypePairTemp.region = (unsigned char)regionPairVecArg[i];
	      argc--; argv++;
	      regionTypePairTemp.type   = (unsigned char)typePairVecArg[i];
	      
	      regionTypePairVec.push_back( regionTypePairTemp );
	    } 
	}

  
  //
  // First get the CT image, either from a directory or from a single
  // file 
  //
  cip::CTType::Pointer ctImage;

  /*
  if ( strcmp( ctDir.c_str(), "q") != 0 )
    {
    std::cout << "Reading CT from directory..." << std::endl;
    ctImage = ReadCTFromDirectory( ctDir );
      if (ctImage.GetPointer() == NULL)
        {
        return cip::DICOMREADFAILURE;
        }
    }
   */
  
  if (strcmp( ctFileName.c_str(), "q") != 0 )
    {
      std::cout << "Reading CT from file..." << std::endl;
      ctImage = ReadCTFromFile( ctFileName );

      if (ctImage.GetPointer() == NULL)
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

      if (labelMap.GetPointer() == NULL)
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
  bbox[0] = 10000;
  bbox[1] = -10000;
  bbox[2] = 10000;
  bbox[3] = -10000;
  bbox[4] = 10000;
  bbox[5] = -10000; 
  while ( !lIt.IsAtEnd() )
    {
      if (lIt.Get() != 0)
        {
         index = lIt.GetIndex();
         for (int i=0;i<3;i++) 
           {
           if (index[i] < bbox[i*2])
             bbox[i*2] = static_cast<int>( index[i] );
           if (index[i] > bbox[i*2+1])
             bbox[i*2+1] = static_cast<int>( index[i] );
           }
        }
       if (maskOutputFlag && l2It.Get() == 0)
         {
         ctIt.Set(maskValue);
         }
     ++lIt;
     ++l2It;
     ++ctIt;
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


