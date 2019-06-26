/** \file
 *  \ingroup commandLineTools 
 *  \details This ... 
 *
 *  $Date$
 *  $Revision$
 *  $Author$
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImageRegionIterator.h"
#include "PerformMorphologicalCLP.h"

typedef itk::ImageRegionIterator<cip::LabelMapType> IteratorType;

int main(int argc, char *argv[])
{
  PARSE_ARGS;

  cip::ChestConventions conventions;

  unsigned int kernelRadiusX = (unsigned int)kernelRadiusXint;
  unsigned int kernelRadiusY = (unsigned int)kernelRadiusYint;
  unsigned int kernelRadiusZ = (unsigned int)kernelRadiusZint;

  std::vector< unsigned char > typesVec;
  std::vector< unsigned char > regionsVec;

  // Parse the regions and type pairs
  try
    {      
      if (regionsVecArg.size() != typesVecArg.size() && !allRegionTypePairs)
	{
	  std::cerr << "Error: must specify same number of chest regions and chest types" << std::endl;
	  return cip::ARGUMENTPARSINGERROR;
	}
      
      if (!allRegionTypePairs)
	{
	  for (unsigned int i=0; i<regionsVecArg.size(); i++)
	    {
	      regionsVec.push_back( conventions.GetChestRegionValueFromName(regionsVecArg[i]) );
	    }
	  for (unsigned int i=0; i<typesVecArg.size(); i++)
	    {
	      typesVec.push_back( conventions.GetChestTypeValueFromName(typesVecArg[i]) );
	    }
	}
    }
  catch ( TCLAP::ArgException excp )
    {
      std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
      return cip::ARGUMENTPARSINGERROR;
    }
  
  std::cout << "Reading label map..." << std::endl;
  cip::LabelMapReaderType::Pointer reader = cip::LabelMapReaderType::New();
    reader->SetFileName(inFileName);
  try
    {
    reader->Update();
    }
  catch (itk::ExceptionObject &excp)
    {
    std::cerr << "Exception caught reading label map:";
    std::cerr << excp << std::endl;
    }

  // If morphological operations are to be performed over all chest-region chest-type pairs present
  // in the label map, collect them now
  if (allRegionTypePairs)
    {
      std::list<unsigned short> labelsList;
      
      IteratorType it(reader->GetOutput(), reader->GetOutput()->GetBufferedRegion());
      
      it.GoToBegin();
      while (!it.IsAtEnd())
	{
	  if (it.Get() != 0)
	    {
	      labelsList.push_back(it.Get());
	    }
	  
	  ++it;
	}
      labelsList.unique();
      labelsList.sort();
      labelsList.unique();
      
      std::list<unsigned short>::iterator listIt = labelsList.begin();
      
      while (listIt != labelsList.end())
	{
	  regionsVec.push_back(conventions.GetChestRegionFromValue(*listIt));
	  typesVec.push_back(conventions.GetChestTypeFromValue(*listIt));
	  
	  listIt++;
	}
    }
  
  if (dilate)
    {
    for (unsigned int i=0; i<regionsVec.size(); i++)
      {      
      std::cout << "Dilating..." << std::endl;
      cip::DilateLabelMap(reader->GetOutput(), regionsVec[i], typesVec[i], 
			  kernelRadiusX, kernelRadiusY, kernelRadiusZ );
      }
    }
  else if (erode)
    {
    for (unsigned int i=0; i<regionsVec.size(); i++)
      {
      std::cout << "Eroding..." << std::endl;      
      cip::ErodeLabelMap(reader->GetOutput(), regionsVec[i], typesVec[i],
  			 kernelRadiusX*2, kernelRadiusY*2, kernelRadiusZ*2 );
      }
    }
  else if (open)
    {
    for (unsigned int i=0; i<regionsVec.size(); i++)
      {      
      std::cout << "Opening..." << std::endl;
      cip::OpenLabelMap(reader->GetOutput(), regionsVec[i], typesVec[i],
  			kernelRadiusX, kernelRadiusY, kernelRadiusZ );
      }
    }
  else if (close)
    {
    for (unsigned int i=0; i<regionsVec.size(); i++)
      {      
      std::cout << "Closing..." << std::endl;
      cip::CloseLabelMap(reader->GetOutput(), regionsVec[i], typesVec[i], 
  			 kernelRadiusX, kernelRadiusY, kernelRadiusZ );
      }
    }

  std::cout << "Writing label map..." << std::endl;
  cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
    writer->SetFileName(outFileName);
    writer->SetInput(reader->GetOutput());
    writer->UseCompressionOn();
  try
    {
    writer->Update();
    }
  catch (itk::ExceptionObject &excp)
    {
    std::cerr << "Exception caught writing label map:";
    std::cerr << excp << std::endl;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
