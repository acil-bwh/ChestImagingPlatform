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

  /*
RightLung	UndefinedType
LeftLung	UndefinedType
UndefinedRegion	Airway : 512
RightLung	Airway : 514
LeftLung	Airway : 515 
RightLung	Vessel : 770
LeftLung	Vessel : 771
   */





  REGIONTYPEPAIR regionTypePairTemp;
  DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage(inputLabelMap);
  duplicator->Update();
  expectedOutputLabelMap = duplicator->GetOutput();

  DuplicatorType::Pointer duplicator2 = DuplicatorType::New();
  duplicator2->SetInputImage(inputLabelMap);
  duplicator2->Update();
  outputLabelMap = duplicator2->GetOutput();
  

  /************
   Test # 1
   ***********/
  //include only 512 and 770. region and pair. volumetric. 512 will stay and 770 will dissapear, keep a size 2
  
  std::vector< unsigned char >  regionPairVec1;
  std::vector< unsigned char >  typePairVec1;
  std::vector< REGIONTYPEPAIR > regionTypePairVec1;
  std::vector< unsigned char >  regionVec1;
  std::vector< unsigned char >  typeVec1;

  regionTypePairTemp.region = 0;
  regionTypePairTemp.type   = 2;
  regionTypePairVec1.push_back( regionTypePairTemp );
  regionTypePairTemp.region = 2;
  regionTypePairTemp.type   = 3;
  regionTypePairVec1.push_back( regionTypePairTemp );
  ///*
  
  LabelMapIteratorType it_output( outputLabelMap,outputLabelMap->GetBufferedRegion() );
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
  FilterConnectedComponents(inputLabelMap, outputLabelMap, 2, regionVec1, typeVec1, regionTypePairVec1,  eval, true, false);
  

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
  

   std::cout << "PASSED first test" << std::endl;
  
  
  /************
   Test # 2
   region and type inclusion and a region inclusion.  on a slice by slice basis. 
   the types should dissapear but the region should be maintained.

  //~/ChestImagingPlatformPrivate-build/CIP-build/bin/FilterConnectedComponents -i ~/ChestImagingPlatformPrivate/Testing/Data/Input/simple_lm.nrrd -o ~/Documents/Data/tempdata/testoutconnected.nrrd --regionPairVecInclude 0,2 --typePairVecInclude 2,3 --regionVecInclude 2 -s 2 --ax
   ***********/

  std::vector< unsigned char >  regionPairVec2;
  std::vector< unsigned char >  typePairVec2;
  std::vector< REGIONTYPEPAIR > regionTypePairVec2;
  std::vector< unsigned char >  regionVec2;
  std::vector< unsigned char >  typeVec2;

  regionTypePairTemp.region = 0;
  regionTypePairTemp.type   = 2;
  regionTypePairVec2.push_back( regionTypePairTemp );
  regionTypePairTemp.region = 2;
  regionTypePairTemp.type   = 3;
  regionTypePairVec2.push_back( regionTypePairTemp );

  regionVec2.push_back(2);
  std::string eval2 = "axial";
  
  std::cout<<eval2<<std::endl;

  FilterConnectedComponents(inputLabelMap, outputLabelMap,  2, regionVec2, typeVec2, regionTypePairVec2,  eval2, true, false);
 
  LabelMapIteratorType it_output2( outputLabelMap,outputLabelMap->GetBufferedRegion() );
  LabelMapIteratorType it_expected2( expectedOutputLabelMap,expectedOutputLabelMap->GetBufferedRegion() );
  LabelMapIteratorType it_input2( inputLabelMap,inputLabelMap->GetBufferedRegion() );
  
  //make the expected output. 512 and 770 will dissapear but will be replaced with the value 2 (lung, undefined type), keep a size 2

  it_input2.GoToBegin();
  it_expected2.GoToBegin();
  while ( !it_input2.IsAtEnd() )
    {
      if ( it_input2.Get() == 770 )  
	{
	  it_expected2.Set( 2 ); //expected
	}
      if( it_input2.Get() == 512)
	{
	  it_expected2.Set( 0 ); //there was no lung in region

	}
      if(( it_input2.Get() != 512) && ( it_input2.Get() != 770 ) ) 
	{
	  it_expected2.Set( it_input2.Get()  );
	}
      ++it_expected2;
      ++it_input2;
    }

  it_output2.GoToBegin();
  it_expected2.GoToBegin();
  while ( !it_output2.IsAtEnd() )
    {
      if ( it_output2.Get() != it_expected2.Get() )
	{
	  std::cout << "FAILED on second test" << std::endl;
	  return 1;
	}
      ++it_output2;
      ++it_expected2;
    }

  std::cout << "PASSED second test" << std::endl;
  
  /***
      3rd test, no inclusions or exclusions sagittal,  512 should  dissapear. but everything else should stay the same. 
   ***/

  std::vector< unsigned char >  regionPairVec3;
  std::vector< unsigned char >  typePairVec3;
  std::vector< REGIONTYPEPAIR > regionTypePairVec3;
  std::vector< unsigned char >  regionVec3;
  std::vector< unsigned char >  typeVec3;

  std::string eval3 = "sagittal";
  std::cout<<eval3<<std::endl;
  FilterConnectedComponents(inputLabelMap, outputLabelMap, 3, regionVec3, typeVec3, regionTypePairVec3,  eval3, false, false);

  LabelMapIteratorType it_output3( outputLabelMap,outputLabelMap->GetBufferedRegion() );
  LabelMapIteratorType it_expected3( expectedOutputLabelMap,expectedOutputLabelMap->GetBufferedRegion() );
  LabelMapIteratorType it_input3( inputLabelMap,inputLabelMap->GetBufferedRegion() );
  
  it_input3.GoToBegin();
  it_expected3.GoToBegin();
  while ( !it_input3.IsAtEnd() )
    {
      if ( it_input3.Get() == 512 )
	{
	  it_expected3.Set( 0 ); //expected
	}
      else
	{
	  it_expected3.Set( it_input3.Get()  );
	}
      ++it_expected3;
      ++it_input3;
    }

  it_output3.GoToBegin();
  it_expected3.GoToBegin();
  while ( !it_output3.IsAtEnd() )
    {
      if ( it_output3.Get() != it_expected3.Get() )
	{
	  std::cout << "FAILED on third test" <<  it_output3.Get()<<" "<<it_expected3.Get()<<std::endl;
	  return 1;
	}
      ++it_output3;
      ++it_expected3;
    }
  
  /***
      4th test, exclusions coronal,  everything should dissapear except for label 2, 512 and 770. 514 should become 2.
   ***/
  std::vector< unsigned char >  regionPairVec4;
  std::vector< unsigned char >  typePairVec4;
  std::vector< REGIONTYPEPAIR > regionTypePairVec4;
  std::vector< unsigned char >  regionVec4;
  std::vector< unsigned char >  typeVec4;


  regionTypePairTemp.region = 0;
  regionTypePairTemp.type   = 2;
  regionTypePairVec4.push_back( regionTypePairTemp );
  regionTypePairTemp.region = 2;
  regionTypePairTemp.type   = 3;
  regionTypePairVec4.push_back( regionTypePairTemp );

  regionVec4.push_back(2);

  std::string eval4 = "sagittal";
  std::cout<<eval4<<std::endl;
  FilterConnectedComponents(inputLabelMap, outputLabelMap, 6, regionVec4, typeVec4, regionTypePairVec4,  eval4, false, true);

  LabelMapIteratorType it_output4( outputLabelMap,outputLabelMap->GetBufferedRegion() );
  LabelMapIteratorType it_expected4( expectedOutputLabelMap,expectedOutputLabelMap->GetBufferedRegion() );
  LabelMapIteratorType it_input4( inputLabelMap,inputLabelMap->GetBufferedRegion() );
  
  it_input4.GoToBegin();
  it_expected4.GoToBegin();
  while ( !it_input4.IsAtEnd() )
    {
      it_expected4.Set(0 ); 

      if (( it_input4.Get() == 512 )|| ( it_input4.Get() == 770 )|| ( it_input4.Get() == 2 ))
	{
	  it_expected4.Set( it_input4.Get() ); //expected
	}
      if (( it_input4.Get() == 514 ))
	{
	  it_expected4.Set(2); //expected
	}


      ++it_expected4;
      ++it_input4;
    }

  it_output4.GoToBegin();
  it_expected4.GoToBegin();
  while ( !it_output4.IsAtEnd() )
    {
      if ( it_output4.Get() != it_expected4.Get() )
	{
	  std::cout << "FAILED on 4th test" <<  it_output3.Get()<<" "<<it_expected3.Get()<<std::endl;
	  return 1;
	}
      ++it_output4;
      ++it_expected4;
    }

  std::cout << "PASSED" << std::endl;
  return 0;
}

//ctest -V -R FilterConnect in CIP-build

