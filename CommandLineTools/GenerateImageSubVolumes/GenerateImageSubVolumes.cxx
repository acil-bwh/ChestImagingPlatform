/** \file
 *  \ingroup commandLineTools 
 *  \details This program accepts as input a CT image and a corresponding label 
 *  map. It produces a collection of subvolumes of the CT image and (optionally)
 *  of the label map image. The user specifies the size of the sub-volume
 *  to extract and can optionally supply an overlap value (the amount of
 *  overlap between sub-volumes). Sub-volumes over a given region of the image
 *  will only be extracted provided that there is at least one foreground
 *  label map voxel in that region. The user only specifies an output prefix;
 *  each of the subvolumes written will have a numerical suffix attached to
 *  it.
 *
 *  USAGE: 
 * 
 *  GenerateImageSubVolumes  [--wls] [-o \<unsigned int\>] 
 *                           [-r \<unsigned int\>] [--lmPre \<string\>] 
 *                            --ctPre \<string\> -l \<string\> 
 *                           -c \<string\> [--] [--version] [-h]
 * 
 *  Where: 
 * 
 *  --wls
 *    Boolean flag to indicate whether label map sub-volumes 
 *    written in addition to the CT sub-volumes
 * 
 *  -o \<unsigned int\>,  --overlap \<unsigned int\>
 *    Length in voxels of overlap between sub-volume regions
 * 
 *  -r \<unsigned int\>,  --roi \<unsigned int\>
 *    Length in voxels of sub-volume edge to extract
 * 
 *  --lmPre \<string\>
 *    Label map sub-volume file name prefix. This is an optional argument
 *    and will have no effect unless the --wls flag is set to 1. Each
 *    sub-volume extracted will be written to file. The file name used will
 *    be this prefix plus a numerical identifier followed by the .nhdr file
 *    extension.
 * 
 *  --ctPre \<string\>
 *    (required)  CT sub-volume file name prefix. Each sub-volume extracted
 *    will be written to file. The file name used will be this prefix plus a
 *    numerical identifier followed by the .nhdr file extension.
 * 
 *  -l \<string\>,  --lm \<string\>
 *    (required)  Label map file name
 * 
 *  -c \<string\>,  --ct \<string\>
 *    (required)  CT file name
 * 
 *  --,  --ignore_rest
 *    Ignores the rest of the labeled arguments following this flag.
 * 
 *  --version
 *    Displays version information and exits.
 * 
 *  -h,  --help
 *    Displays usage information and exits.
 * 
 *  $Date: 2013-03-25 13:23:52 -0400 (Mon, 25 Mar 2013) $
 *  $Revision: 383 $
 *  $Author: jross $
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageSeriesReader.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkExtractImageFilter.h"
#include <sstream>
#include "GenerateImageSubVolumesCLP.h"

typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType >          LabelMapIteratorType;
typedef itk::ExtractImageFilter< cip::CTType, cip::CTType >             CTExtractorType; 
typedef itk::ExtractImageFilter< cip::LabelMapType, cip::LabelMapType > LabelMapExtractorType; 

void WriteSubVolume( cip::CTType::Pointer, cip::LabelMapType::Pointer, cip::CTType::RegionType,
		     std::string, std::string, bool, unsigned int* );


int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  unsigned int  roiLength  = (unsigned int) roiLengthInt; 
  unsigned int  overlap    = (unsigned int) overlapInt;
  
  // Read the CT image
  std::cout << "Reading CT from file..." << std::endl;
  cip::CTReaderType::Pointer ctReader = cip::CTReaderType::New();
    ctReader->SetFileName( ctFileName );
  try
    {
    ctReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading CT image:";
    std::cerr << excp << std::endl;
      
    return cip::NRRDREADFAILURE;
    }

  // Read the label map (if required)
  std::cout << "Reading label map image..." << std::endl;
  cip::LabelMapReaderType::Pointer labelMapReader = cip::LabelMapReaderType::New();
    labelMapReader->SetFileName( labelMapFileName );
  try
    {
    labelMapReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading label map:";
    std::cerr << excp << std::endl;
      
    return cip::LABELMAPREADFAILURE;
    }

  cip::CTType::SizeType size = ctReader->GetOutput()->GetBufferedRegion().GetSize();

  // Now iterate over the label map image to determine
  // the bounding box. Anything in the foreground will
  // be considered for computing the bounding box
  unsigned int xMax = 0;
  unsigned int yMax = 0;
  unsigned int zMax = 0;
  unsigned int xMin = ctReader->GetOutput()->GetBufferedRegion().GetSize()[0];
  unsigned int yMin = ctReader->GetOutput()->GetBufferedRegion().GetSize()[1];
  unsigned int zMin = ctReader->GetOutput()->GetBufferedRegion().GetSize()[2];

  LabelMapIteratorType lIt( labelMapReader->GetOutput(), labelMapReader->GetOutput()->GetBufferedRegion() );
  
  lIt.GoToBegin();
  while ( !lIt.IsAtEnd() )
    {
      if ( lIt.Get() != 0 )
	{
	  if ( lIt.GetIndex()[0] > xMax )
	    {
	      xMax = lIt.GetIndex()[0];
	    }
	  if ( lIt.GetIndex()[1] > yMax )
	    {
	      yMax = lIt.GetIndex()[1];
	    }
	  if ( lIt.GetIndex()[2] > zMax )
	    {
	      zMax = lIt.GetIndex()[2];
	    }

	  if ( lIt.GetIndex()[0] < xMin )
	    {
	      xMin = lIt.GetIndex()[0];
	    }
	  if ( lIt.GetIndex()[1] < yMin )
	    {
	      yMin = lIt.GetIndex()[1];
	    }
	  if ( lIt.GetIndex()[2] < zMin )
	    {
	      zMin = lIt.GetIndex()[2];
	    }
	}

      ++lIt;
    }

  // Now we're ready to extract the sub-volumes
  cip::CTType::SizeType roiSize;
    roiSize[0] = roiLength;
    roiSize[1] = roiLength;
    roiSize[2] = roiLength;

  cip::CTType::RegionType roiRegion;
  cip::CTType::RegionType::IndexType roiStart;

  int radius = (roiLength-1)/2;

  unsigned int fileNameIncrement = 0;

  int i, j, k;
  for ( int x=xMin; x<=xMax; x=x+roiSize[0]-overlap )
    {
      roiRegion.SetSize( roiSize );

      if ( x-radius >= 0 )
  	{
  	  roiStart[0] = x-radius;
  	}
      else
  	{
  	  roiStart[0] = 0;
  	}

      for ( int y=yMin; y<=yMax; y=y+roiSize[1]-overlap )
  	{
  	  if ( y-radius >= 0 )
  	    {
  	      roiStart[1] = y-radius;
  	    }
  	  else
  	    {
  	      roiStart[1] = 0;
  	    }

  	  for ( int z=zMin; z<=zMax; z=z+roiSize[2]-overlap )
  	    {
  	      if ( z-radius >= 0 )
  		{
  		  roiStart[2] = z-radius;
  		}
  	      else
  		{
  		  roiStart[2] = 0;
  		}
  	      roiRegion.SetIndex( roiStart );

	      cip::CTType::SizeType tmpSize;

	      if ( roiStart[0] + roiSize[0] > size[0] )
		{
		  tmpSize[0] = size[0] - roiStart[0];
		}
	      else
		{
		  tmpSize[0] = roiSize[0];
		}
	      if ( roiStart[1] + roiSize[1] > size[1] )
		{
		  tmpSize[1] = size[1] - roiStart[1];
		}
	      else
		{
		  tmpSize[1] = roiSize[1];
		}
	      if ( roiStart[2] + roiSize[2] > size[2] )
		{
		  tmpSize[2] = size[2] - roiStart[2];
		}
	      else
		{
		  tmpSize[2] = roiSize[2];
		}
	      
	      roiRegion.SetSize( tmpSize );       
  	      WriteSubVolume( ctReader->GetOutput(), labelMapReader->GetOutput(), roiRegion, ctSubVolumeFileNamePrefix,
  			      labelMapSubVolumeFileNamePrefix, writeLabelMapSubVolumes, &fileNameIncrement );
  	    }
  	}
    }
  
  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

void WriteSubVolume( cip::CTType::Pointer ctImage, cip::LabelMapType::Pointer labelMap, cip::CTType::RegionType roiRegion, 
		     std::string ctSubVolumeFileNamePrefix, std::string labelMapSubVolumeFileNamePrefix,
		     bool writeLabelMapSubVolumes, unsigned int* fileNameIncrement )
{
  bool writeSubVolume = false;

  LabelMapIteratorType lIt( labelMap, roiRegion );
  
  lIt.GoToBegin();
  while ( !lIt.IsAtEnd() )
    {
      if ( lIt.Get() != 0 )
	{
	  writeSubVolume = true;
	  break;
	}

      ++lIt;
    }

  //
  // Generate the complete file name
  //
  std::string ctFileName       = ctSubVolumeFileNamePrefix;
  std::string labelMapFileName = labelMapSubVolumeFileNamePrefix;

  std::stringstream stream;
  stream << *fileNameIncrement;

  std::string number = stream.str();

  if ( *fileNameIncrement < 10 )
    {
      ctFileName.append( "000" );
      ctFileName.append( number );

      labelMapFileName.append( "000" );
      labelMapFileName.append( number );
    }
  else if ( *fileNameIncrement < 100 )
    {
      ctFileName.append( "00" );
      ctFileName.append( number );

      labelMapFileName.append( "00" );
      labelMapFileName.append( number );
    }
  else if ( *fileNameIncrement < 1000 )
    {
      ctFileName.append( "0" );
      ctFileName.append( number );

      labelMapFileName.append( "0" );
      labelMapFileName.append( number );
    }
  else
    {
      ctFileName.append( number );
      labelMapFileName.append( number );
    }
  ctFileName.append( ".nhdr" );  
  labelMapFileName.append( ".nhdr" );

  //
  // Now extract and write
  // 
  if ( writeSubVolume )
    {
      CTExtractorType::Pointer ctExtractor = CTExtractorType::New();
        ctExtractor->SetInput( ctImage );
	ctExtractor->SetExtractionRegion( roiRegion );
	ctExtractor->Update();
      
      std::cout << "---Writing CT sub-volume..." << std::endl;
      cip::CTWriterType::Pointer ctWriter = cip::CTWriterType::New();
      ctWriter->SetFileName( ctFileName );
	    ctWriter->SetInput( ctExtractor->GetOutput() );
      ctWriter->UseCompressionOn();
      try
	{
	ctWriter->Update();
	}
      catch ( itk::ExceptionObject &excp )
	{
        std::cerr << "Exception caught writing CT sub-volume:";
        std::cerr << excp << std::endl;
	}  
      
      if ( writeLabelMapSubVolumes )
	{
	  LabelMapExtractorType::Pointer labelMapExtractor = LabelMapExtractorType::New();
	    labelMapExtractor->SetInput( labelMap );
	    labelMapExtractor->SetExtractionRegion( roiRegion );
	    labelMapExtractor->Update();
	  
	  std::cout << "---Writing label map sub-volume..." << std::endl;
	  cip::LabelMapWriterType::Pointer labelMapWriter = cip::LabelMapWriterType::New();
  	    labelMapWriter->SetFileName( labelMapFileName );
	    labelMapWriter->SetInput( labelMapExtractor->GetOutput() );
	    labelMapWriter->UseCompressionOn();
	  try
	    {
	    labelMapWriter->Update();
	    }
	  catch ( itk::ExceptionObject &excp )
	    {
	    std::cerr << "Exception caught writing label map sub-volume:";
	    std::cerr << excp << std::endl;
	    }  
	}
      
      (*fileNameIncrement)++;
    }
}

#endif

