/**  This program performs connected components and removes components 
 *  that are smaller than a specified size. If a set of inclusions is specified, 
 *  then, the connected components and removal will only be on these included  
 *  types, regions, and region and type pairs. Types are processed first, where 
 *  a binarised labelmap for each type is extracted, connected components is 
 *  performed. The type of each voxel from small components is then set to 
 *  UNDEFINEDTYPE and the region is preserved. Regions are then processed. 
 *  Finally, region/type pairare processed. For each region/type pair component 
 *  that is too small, if the region has been specified in the region vector, 
 *  then do not delete that region value. Otherwise set the region ot undefined. 
 *  Same thing applies for type.
 *
 *  If no inclusions or exclusions are specified, then the whole labelmap is 
 *  binarized and a connected components is performed on the entire image. If 
 *  the exclusions options are specified, then removal will first be done 
 *  using the overall binarised labelmap, then voxels that are part of the 
 *  exclusions are added back in, using a precedence rule. Where we first 
 *  add only the type values, then the region values, then the region/type pairs. 
 */

#include "FilterConnectedComponentsHelper.h"
#include "FilterConnectedComponentsCLP.h"
#include "itkCIPMergeChestLabelMapsImageFilter.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;    

  std::vector< unsigned char >  regionVec;
  std::vector< unsigned char >  typeVec;
  std::vector< unsigned char >  regionPairVec;
  std::vector< unsigned char >  typePairVec;
  std::vector< REGIONTYPEPAIR > regionTypePairVec;

  std::string evalMethod;
  bool is_include = false;
  bool is_exclude = false;
  //Read in region and type pair

  cip::LabelMapType::Pointer inputLabelMap;
  cip::LabelMapType::Pointer outputLabelMap;

  //should not have inputed both inclusion and exclusion criteria

  //Include
  for ( unsigned int i=0; i<regionVecArgInclude.size(); i++ )
    {
      regionVec.push_back(regionVecArgInclude[i]);
      is_include = true;
    }
  for ( unsigned int i=0; i<typeVecArgInclude.size(); i++ )
    {
      typeVec.push_back( typeVecArgInclude[i] );
      is_include = true;
    }
  if (regionPairVecArgInclude.size() == typePairVecArgInclude.size())
    {
      for ( unsigned int i=0; i<regionPairVecArgInclude.size(); i++ )
	{
	  REGIONTYPEPAIR regionTypePairTemp;

	  regionTypePairTemp.region = regionPairVecArgInclude[i];
	  argc--; argv++;
	  regionTypePairTemp.type   = typePairVecArgInclude[i];

	  regionTypePairVec.push_back( regionTypePairTemp );
	  std::cout<<"region and type: "<<regionPairVecArgInclude[i]<<" "<<typePairVecArgInclude[i]<<regionTypePairTemp.region<<" "<<regionTypePairTemp.type<<std::endl;
	  is_include = true;
	}
    }
  else
    {
      std::cerr <<"region and type pair should be same size"<< std::endl;
      return cip::EXITFAILURE;
    }

  //Exclude
  for ( unsigned int i=0; i<regionVecArgExclude.size(); i++ )
    {
      regionVec.push_back(regionVecArgExclude[i]);
      is_exclude = true;
    }
  for ( unsigned int i=0; i<typeVecArgExclude.size(); i++ )
    {
      typeVec.push_back( typeVecArgExclude[i] );
      is_exclude = true;
    }
  if (regionPairVecArgExclude.size() == typePairVecArgExclude.size())
    {
      for ( unsigned int i=0; i<regionPairVecArgExclude.size(); i++ )
	{
	  REGIONTYPEPAIR regionTypePairTemp;

	  regionTypePairTemp.region = regionPairVecArgExclude[i];
	  argc--; argv++;
	  regionTypePairTemp.type   = typePairVecArgExclude[i];

	  regionTypePairVec.push_back( regionTypePairTemp );


	    
	  is_exclude = true;
	}
    }
  else
    {
      std::cerr <<"region and type pair should be same size"<< std::endl;
      return cip::EXITFAILURE;
    }
    
  if( (is_include == true) && (is_exclude == true))
    {
      std::cerr <<"Cannot specify inclusion and exclusion criteria"<< std::endl;
      return cip::EXITFAILURE;
    }

  // Read the input labelmap from file
 try
    {
      std::cout << "Reading the input label map from file..." << std::endl;

      inputLabelMap  = ReadLabelMapFromFile( inputFileName );
      if (inputLabelMap.GetPointer() == NULL)
	{
	  return cip::LABELMAPREADFAILURE;
	}	
    }
 catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading  label map:";
    std::cerr << excp << std::endl;
    }
 //check how the volume will be evaluated, if more than 1 value is set, throw error
 evalMethod = "";


 if(isVol)
   {
     if(evalMethod.compare("") == 0)
       evalMethod = "vol";
   else
     {
      std::cerr <<"more than one evaluation method specified"<< std::endl;
      return cip::EXITFAILURE;
     }
   }
 if(isAxial)
   {   
     if(evalMethod.compare("") == 0)
       evalMethod = "axial";
   else
     {
      std::cerr <<"more than one evaluation method specified"<< std::endl;
      return cip::EXITFAILURE;
     }
   }
 if(isCoronal) 
   {   
      if(evalMethod.compare("") == 0)
       evalMethod = "coronal";
   else
     {
      std::cerr <<"more than one evaluation method specified"<< std::endl;
      return cip::EXITFAILURE;
     }
   }
 if (isSaggital)
   {   
     if(evalMethod.compare("") == 0)
       evalMethod = "sagittal";
   else
     {
      std::cerr <<"more than one evaluation method specified"<< std::endl;
      return cip::EXITFAILURE;
     }
   }
 if(evalMethod.compare("") == 0)
     {
      std::cerr <<"No evaluation method specified"<< std::endl;
      return cip::EXITFAILURE;
     }

 DuplicatorType::Pointer duplicator = DuplicatorType::New();
 duplicator->SetInputImage(inputLabelMap);
 duplicator->Update();
 outputLabelMap = duplicator->GetOutput();
 FilterConnectedComponents(inputLabelMap,outputLabelMap,sizeThreshold, regionVec, typeVec, regionTypePairVec, evalMethod, is_include, is_exclude);


  // Write the resulting label map to file
  std::cout << "Writing resampled label map..." << std::endl;
  cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
    writer->SetFileName( outputFileName.c_str());
    writer->UseCompressionOn();
    writer->SetInput(outputLabelMap );
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught writing label map:";
    std::cerr << excp << std::endl;
    
    return cip::LABELMAPWRITEFAILURE;
    }

  std::cout << "DONE." << std::endl;
  
  return 0;

}
