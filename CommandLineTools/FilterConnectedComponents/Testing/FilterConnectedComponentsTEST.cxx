#include "FilterConnectedComponentsHelper.h"
#include "cipHelper.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "cipChestConventions.h"
//#include "itkRelabelComponentImageFilter.h"

typedef itk::CIPExtractChestLabelMapImageFilter                                                   LabelMapExtractorType;
typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType >                                    LabelMapIteratorType;
//typedef itk::RelabelComponentImageFilter <cip::LabelMapType, cip::LabelMapType>
//    RelabelFilterType;

  typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType >                                    LabelMapIteratorType;


int main( int argc, char* argv[] )
{
  cip::LabelMapType::Pointer inputLabelMap= cip::LabelMapType::New();
  cip::LabelMapType::Pointer outputLabelMap= cip::LabelMapType::New();
  cip::LabelMapType::Pointer expectedOutputLabelMap= cip::LabelMapType::New();


  // read the test image,  
 try
    {
      std::cout << "Reading label map from file..." << std::endl;

      inputLabelMap  = ReadLabelMapFromFile( argv[1] );
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


  //call filter with paramrs
  std::vector< unsigned char >  regionVec;
  std::vector< unsigned char >  typeVec;
  std::vector< unsigned char >  regionPairVec;
  std::vector< unsigned char >  typePairVec;
  std::vector< REGIONTYPEPAIR > regionTypePairVec;
  std::vector< REGIONTYPEPAIR > regionTypePairVec2;

  /*
RightLung	UndefinedType
LeftLung	UndefinedType
UndefinedRegion	Airway : 512
RightLung	Airway : 514
LeftLung	Airway : 515 
RightLung	Vessel : 770
LeftLung	Vessel : 771
   */

  //include only 512 and 770. 512 will stay and 770 will dissapear, keep a size 2
  REGIONTYPEPAIR regionTypePairTemp;
  regionTypePairTemp.region = 0;
  regionTypePairTemp.type   = 2;
  regionTypePairVec.push_back( regionTypePairTemp );
  regionTypePairTemp.region = 2;
  regionTypePairTemp.type   = 3;
  regionTypePairVec.push_back( regionTypePairTemp );

  
  //make the expected output

 
  cip::LabelMapType::RegionType region;
  cip::LabelMapType::IndexType start;
  start[0] = 0;
  start[1] = 0;
 
  cip::LabelMapType::SizeType size = inputLabelMap->GetLargestPossibleRegion().GetSize();
 
  region.SetSize(size);
  region.SetIndex(start);
  expectedOutputLabelMap->SetRegions(region);
  expectedOutputLabelMap->Allocate();
  
  
  LabelMapIteratorType it_expected( expectedOutputLabelMap,expectedOutputLabelMap->GetBufferedRegion() );
  LabelMapIteratorType it_input( inputLabelMap,inputLabelMap->GetBufferedRegion() );

  it_input.GoToBegin();
  it_expected.GoToBegin();
  while ( !it_input.IsAtEnd() )
    {
      if ( it_input.Get() == 770 )
	{
	  it_expected.Set( 0 ); //expected
	}
      else
	{
	  it_expected.Set( it_input.Get()  );
	}
      ++it_expected;
      ++it_input;
    }



  std::string eval = "vol";
  outputLabelMap = FilterConnectedComponents(inputLabelMap, 2, regionVec, typeVec, regionTypePairVec,  eval, true, false);
 

  LabelMapIteratorType it_output( outputLabelMap,outputLabelMap->GetBufferedRegion() );
  it_output.GoToBegin();
  it_expected.GoToBegin();
  while ( !it_output.IsAtEnd() )
    {
      if ( it_output.Get() != it_expected.Get() )
	{
	  std::cout << "FAILED on first test" << std::endl;
	  return 1;
	}
      ++it_output;
      ++it_expected;
    }


  /** test #2, same region and type inclusion, just on a slice by slice. Both should disseappear
  ***/
  std::string eval2 = "axial";
  
  std::cout<<eval2<<std::endl;
  //make the expected output. 512 and 770 will dissapear, keep a size 2
  it_input.GoToBegin();
  it_expected.GoToBegin();
  while ( !it_input.IsAtEnd() )
    {
      if (( it_input.Get() == 770 ) || ( it_input.Get() == 512))
	{
	  it_expected.Set( 0 ); //expected
	}
      else
	{
	  it_expected.Set( it_input.Get()  );
	}
      ++it_expected;
      ++it_input;
    }



  outputLabelMap = FilterConnectedComponents(inputLabelMap, 2, regionVec, typeVec, regionTypePairVec,  eval2, true, false);
 

  it_output.GoToBegin();
  it_expected.GoToBegin();
  while ( !it_output.IsAtEnd() )
    {
      if ( it_output.Get() != it_expected.Get() )
	{
	  std::cout << "FAILED on second test" << std::endl;
	  return 1;
	}
      ++it_output;
      ++it_expected;
    }


  /***
      last test, no inclusions or exclusions sagittal, again 512 should not dissapear. but everything else should. 
   ***/
  std::string eval3 = "sagittal";
  
  std::cout<<eval3<<std::endl;
  it_input.GoToBegin();
  it_expected.GoToBegin();
  while ( !it_input.IsAtEnd() )
    {
      if ( it_input.Get() != 512 )
	{
	  it_expected.Set( 0 ); //expected
	}
      else
	{
	  it_expected.Set( it_input.Get()  );
	}
      ++it_expected;
      ++it_input;
    }



  outputLabelMap = FilterConnectedComponents(inputLabelMap, 2, regionVec, typeVec, regionTypePairVec2,  eval3, false, false);
 

  it_output.GoToBegin();
  it_expected.GoToBegin();
  while ( !it_output.IsAtEnd() )
    {
      if ( it_output.Get() != it_expected.Get() )
	{
	  std::cout << "FAILED on third test" << std::endl;
	  return 1;
	}
      ++it_output;
      ++it_expected;
    }

  
  std::cout << "PASSED" << std::endl;
  return 0;
}

//ctest -V -R FilterConnect in CIP-build

