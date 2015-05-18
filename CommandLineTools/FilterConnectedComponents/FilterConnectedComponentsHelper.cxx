#include "FilterConnectedComponentsHelper.h"
#include <itkImageDuplicator.h>


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

bool GetSliceHasForeground(cip::LabelMapType::Pointer labelMap, unsigned int whichSlice, std::string slicePlane )
{
  cip::LabelMapType::SizeType size = labelMap->GetBufferedRegion().GetSize();

  cip::LabelMapType::RegionType region;
  cip::LabelMapType::IndexType  start;
  cip::LabelMapType::SizeType   regionSize;


  if (slicePlane.compare("axial") == 0)
    {
      start[0] = 0;
      start[1] = 0;
      start[2] = whichSlice;
      
      regionSize[0] = size[0];
      regionSize[1] = size[1];
      regionSize[2] = 1;
    }
  else if (slicePlane.compare("coronal") == 0)
    {      
      start[0] = 0;
      start[1] = whichSlice;
      start[2] = 0;
      
      regionSize[0] = size[0];
      regionSize[1] = 1;
      regionSize[2] = size[2];
    }
  else if (slicePlane.compare("sagittal") == 0)
    {      
      start[0] = whichSlice;
      start[1] = 0;
      start[2] = 0;
      
      regionSize[0] = 1;
      regionSize[1] = size[1];
      regionSize[2] = size[2];
    }

  region.SetSize( regionSize );
  region.SetIndex( start );

  LabelMapIteratorType it( labelMap, region );

  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
      if ( it.Get() > 0 )
	{
	  return true;
	}

      ++it;
    }

  return false;
}

/**
 * Extract a slice from the input label map image, depending on desired direction
 */
void ExtractLabelMapSlice( cip::LabelMapType::Pointer image, LabelMapSliceType::Pointer sliceImage, int whichSlice, std::string slicePlane )
{
  cip::LabelMapType::SizeType size = image->GetBufferedRegion().GetSize();

  LabelMapSliceType::SizeType sliceSize;
  
  cip::LabelMapType::IndexType sliceStartIndex;

  cip::LabelMapType::SizeType sliceExtractorSize;
 

  if (slicePlane.compare("axial") == 0)
    {
      sliceStartIndex[0] = 0;
      sliceStartIndex[1] = 0;
      sliceStartIndex[2] = whichSlice;
 
      sliceExtractorSize[0] = size[0];
      sliceExtractorSize[1] = size[1];
      sliceExtractorSize[2] = 0;
     
      sliceSize[0] = size[0];
      sliceSize[1] = size[1];

    }
  else if (slicePlane.compare("coronal") == 0)
    {      
      sliceStartIndex[0] = 0;
      sliceStartIndex[1] = whichSlice;
      sliceStartIndex[2] = 0;

      sliceExtractorSize[0] = size[0];
      sliceExtractorSize[1] = 0;
      sliceExtractorSize[2] = size[2];
      
      sliceSize[0] = size[0];
      sliceSize[1] =  size[2];
    }
  else if (slicePlane.compare("sagittal") == 0)
    {      
      sliceStartIndex[0] = whichSlice;
      sliceStartIndex[1] = 0;
      sliceStartIndex[2] = 0;
      
      sliceExtractorSize[0] = 0;
      sliceExtractorSize[1] = size[1];
      sliceExtractorSize[2] = size[2];

      sliceSize[0] = size[1];
      sliceSize[1] = size[2];
    }



  sliceImage->SetRegions( sliceSize );
  sliceImage->Allocate();

  
  cip::LabelMapType::RegionType sliceExtractorRegion;
  sliceExtractorRegion.SetSize( sliceExtractorSize );
  sliceExtractorRegion.SetIndex( sliceStartIndex );
  
  LabelMapSliceExtractorType::Pointer sliceExtractor = LabelMapSliceExtractorType::New();
  sliceExtractor->SetInput( image );
  sliceExtractor->SetDirectionCollapseToIdentity();
  sliceExtractor->SetExtractionRegion( sliceExtractorRegion );
  try
    {   
      sliceExtractor->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
      std::cerr << "Exception caught extracting slice:";
      std::cerr << excp << std::endl;
    }   


  LabelMapSliceIteratorType eIt( sliceExtractor->GetOutput(), sliceExtractor->GetOutput()->GetBufferedRegion() );
  LabelMapSliceIteratorType sIt( sliceImage, sliceImage->GetBufferedRegion() );

  sIt.GoToBegin();
  eIt.GoToBegin();
  while ( !sIt.IsAtEnd() )
    {
      sIt.Set( eIt.Get() );

      ++sIt;
      ++eIt;
    }
}



//Performs the whole filtering
void FilterConnectedComponents(cip::LabelMapType::Pointer inputLabelMap, cip::LabelMapType::Pointer outputLabelMap, int sizeThreshold, std::vector< unsigned char> regionVec, std::vector< unsigned char> typeVec, std::vector<REGIONTYPEPAIR> regionTypePairVec, std::string  evalMethod, bool isInclude, bool isExclude)
{
  cip::ChestConventions conventions;
  if( (isInclude == true) && (isExclude == true))
    {
      std::cerr <<"Cannot specify inclusion and exclusion criteria"<< std::endl;
      return;
    }
    
  cip::LabelMapType::IndexType input_start;

  input_start[0] = 0;
  input_start[1] = 0;
 
  cip::LabelMapType::SizeType input_size = inputLabelMap->GetLargestPossibleRegion().GetSize();
  cip::LabelMapType::SpacingType inputSpacing = inputLabelMap->GetSpacing();
  cip::LabelMapType::RegionType inputRegion;
  inputRegion.SetSize(input_size);
  inputRegion.SetIndex(input_start);

  /* set output to input by default
   */
    LabelMapIteratorType it_output0(outputLabelMap,outputLabelMap->GetBufferedRegion());
  LabelMapIteratorType it_input0(inputLabelMap,inputLabelMap->GetBufferedRegion());
  it_output0.GoToBegin();
  it_input0.GoToBegin();

  while ( !it_output0.IsAtEnd() )
    {
      it_output0.Set(it_input0.Get());
      
      ++it_output0;
      ++it_input0;
    }
  

 cip::LabelMapType::IndexType tempIndex;
 cip::LabelMapType::RegionType boundingBox;
 boundingBox = cip::GetLabelMapChestRegionChestTypeBoundingBoxRegion(inputLabelMap);


  if(isInclude == true) 
    {
      //here we start off with the output being identical to the input.  so nothing
      //needs to be changed to the output labelmap as far as initialization

      for ( unsigned int i=0; i<typeVec.size(); i++ )
	{
	  // perform connected components on the region of interest only, depending on
	  // whether we want volumetric, axial ...
	  // return a labelmap with the small components removed
	  LabelMapChestExtractorType::Pointer extractor = LabelMapChestExtractorType::New();
	  extractor->SetInput( inputLabelMap );
	  extractor->SetChestType(typeVec[i]);
	  extractor->Update();

	  //DuplicatorType::Pointer duplicator2 = DuplicatorType::New();
	  //duplicator2->SetInputImage(inputLabelMap);
	  //duplicator2->Update();
	  cip::LabelMapType::Pointer connectedLabelMap = cip::LabelMapType::New();//duplicator2->GetOutput();
	  //If the region / type does not exist in the image, then the connected labelmap should be empty


	  performConnectedComponents(extractor->GetOutput(), connectedLabelMap, sizeThreshold, evalMethod);
	    
	  // Here we expect to get a labelmap identical to extractor->GetOutput(), except with 
	  // the small components washed away. Now, given the list of inclusion priorities
	  // remove type values from label if it been washed away

	  LabelMapIteratorType it_components(connectedLabelMap,connectedLabelMap->GetBufferedRegion());
	  LabelMapIteratorType it_inclusion(extractor->GetOutput(),extractor->GetOutput()->GetBufferedRegion());
	  LabelMapIteratorType it_output(outputLabelMap,outputLabelMap->GetBufferedRegion());
	  it_components.GoToBegin();
	  it_inclusion.GoToBegin();
	  it_output.GoToBegin();

	  while ( !it_output.IsAtEnd() )
	    {
	      if(( it_components.Get() == 0) && (it_inclusion.Get()>0))
		{
		  //get the present voxel value and translate to region and type. Then remove the region value
		  unsigned char chest_region = conventions.GetChestRegionFromValue( it_output.Get() );
		  unsigned char chest_type = conventions.GetChestTypeFromValue( it_output.Get() );
		  unsigned short final_value = conventions.GetValueFromChestRegionAndType(  (unsigned char)( cip::UNDEFINEDREGION ), chest_type ); 
		      
		   it_output.Set(final_value);
		}
	      ++it_components;
	      ++it_inclusion;
	      ++it_output;
	    }
	}

      for ( unsigned int i=0; i<regionVec.size(); i++ )
	{
	  // perform connected components on the region of interest only, depending on
	  // whether we want volumetric, axial ...
	  // return a labelmap with the small components removed
	  LabelMapChestExtractorType::Pointer extractor = LabelMapChestExtractorType::New();
	  extractor->SetInput( inputLabelMap );
	  extractor->SetChestRegion(regionVec[i]);
	  extractor->Update();

	  DuplicatorType::Pointer duplicator2 = DuplicatorType::New();
	  duplicator2->SetInputImage(inputLabelMap);
	  duplicator2->Update();
	  cip::LabelMapType::Pointer connectedLabelMap = duplicator2->GetOutput();	    

	  performConnectedComponents(extractor->GetOutput(), connectedLabelMap, sizeThreshold, evalMethod);
	  // Here we expect to get a labelmap identical to extractor->GetOutput(), except with 
	  // the small components washed away. Now, given the list of inclusion priorities
	  // remove region values from label if it been washed away

	  LabelMapIteratorType it_components(connectedLabelMap,connectedLabelMap->GetBufferedRegion());
	  LabelMapIteratorType it_inclusion(extractor->GetOutput(),extractor->GetOutput()->GetBufferedRegion());
	  LabelMapIteratorType it_output(outputLabelMap,outputLabelMap->GetBufferedRegion());
	  it_components.GoToBegin();
	  it_inclusion.GoToBegin();
	  it_output.GoToBegin();

	  while ( !it_output.IsAtEnd() )
	    {
	      if(( it_components.Get() == 0) && (it_inclusion.Get()>0))
		{
		  //get the present voxel value and translate to region and type. Then remove the region value
		  unsigned char chest_region = conventions.GetChestRegionFromValue( it_output.Get() );
		  unsigned char chest_type = conventions.GetChestTypeFromValue( it_output.Get() );
		  unsigned short final_value = conventions.GetValueFromChestRegionAndType( chest_region , (unsigned char)( cip::UNDEFINEDTYPE ) ) ;
		      
		  it_output.Set(final_value);
		}
	      ++it_components;
	      ++it_inclusion;
	      ++it_output;
	    }
	}

      for ( unsigned int i=0; i<regionTypePairVec.size(); i++ )
	{
	  // perform connected components on the region of interest only, depending on
	  // whether we want volumetric, axial ...
	  // return a labelmap with the small components removed


	  DuplicatorType::Pointer duplicator2 = DuplicatorType::New();
	  duplicator2->SetInputImage(inputLabelMap);
	  duplicator2->Update();
	  cip::LabelMapType::Pointer connectedLabelMap = duplicator2->GetOutput(); //= cip::LabelMapType::New();//

	  LabelMapChestExtractorType::Pointer extractor = LabelMapChestExtractorType::New();
	  extractor->SetInput( inputLabelMap );
	  extractor->SetRegionAndType( regionTypePairVec[i].region,regionTypePairVec[i].type );
	  extractor->Update();

	   boundingBox = cip::GetLabelMapChestRegionChestTypeBoundingBoxRegion(extractor->GetOutput());
  	    
	  performConnectedComponents(extractor->GetOutput(), connectedLabelMap,  sizeThreshold, evalMethod);
	   
	  // Here we expect to get a labelmap identical to extractor->GetOutput(), except with 
	  // the small components washed away. Now, if the voxel's region has not been set in the 
	  // region list remove it from voxel. otherwise keep.
	  // do the same for the type. I think the final removal value should be the same for all 
	  // voxels in this region/type, because all voxels should have the same value... 
	  LabelMapIteratorType it_inclusion(extractor->GetOutput(),extractor->GetOutput()->GetBufferedRegion());
	  LabelMapIteratorType it_components(connectedLabelMap,connectedLabelMap->GetBufferedRegion());
	  LabelMapIteratorType it_output(outputLabelMap,outputLabelMap->GetBufferedRegion());
	  LabelMapIteratorType it_input(inputLabelMap,inputLabelMap->GetBufferedRegion());
	  it_components.GoToBegin();
	  it_inclusion.GoToBegin();
	  it_output.GoToBegin();
	  it_input.GoToBegin();
	  while ( !it_input.IsAtEnd() )
	    {
	      if(( it_components.Get() == 0) && (it_inclusion.Get()>0))
		{
		  //We first need to see if it was in the region list or the type list
		  // if so, do not touch.
		  unsigned char chest_region = conventions.GetChestRegionFromValue( it_output.Get() );
		  unsigned char chest_type = conventions.GetChestTypeFromValue( it_output.Get() );
		  unsigned char final_region;
		  unsigned char final_type;
		  if(std::find(regionVec.begin(), regionVec.end(), chest_region)!=regionVec.end())
		    {
		      final_region = chest_region;
		    }
		  else
		    final_region = (unsigned char)( cip::UNDEFINEDREGION );
		      
		  if(std::find(typeVec.begin(), typeVec.end(), chest_type)!=typeVec.end())
		    final_type = chest_type;
		      
		  else
		    final_type = (unsigned char)( cip::UNDEFINEDTYPE );


		  unsigned short final_value = conventions.GetValueFromChestRegionAndType( final_region ,final_type ); 
		  it_output.Set(final_value); //this is messing up bounding box

		}

	      ++it_components;
	      ++it_inclusion;
	      ++it_output;
	      ++it_input;
	    }
	}

    }

  else
    {
      std::cout<<"performing connected components on all thresholded labels"<<std::endl;
      DuplicatorType::Pointer duplicator2 = DuplicatorType::New();
      duplicator2->SetInputImage(inputLabelMap);
      duplicator2->Update();
      cip::LabelMapType::Pointer connectedLabelMap = duplicator2->GetOutput();

      performConnectedComponents(inputLabelMap, connectedLabelMap, sizeThreshold, evalMethod);

      // here the output label map has a copy of the input label map.
      // First label output based on all labels. Then add back exclusions.
      // set all voxels deleted by connected components to 0, otherwise keep as is
      LabelMapIteratorType it_inclusion(inputLabelMap,inputLabelMap->GetBufferedRegion());
      LabelMapIteratorType it_components(connectedLabelMap,connectedLabelMap->GetBufferedRegion());
      LabelMapIteratorType it_output(outputLabelMap,outputLabelMap->GetBufferedRegion());
      it_components.GoToBegin();
      it_inclusion.GoToBegin();
      it_output.GoToBegin();
      while ( !it_output.IsAtEnd() )
	{		  
	  if(( it_components.Get() == 0) && (it_inclusion.Get()>0))
	    {
	       it_output.Set(0);
	    }
	  else
	    {
	        it_output.Set(it_inclusion.Get());
	    }
	  ++it_components;
	  ++it_inclusion;
	  ++it_output;
	}
      
      if(isExclude == true) 
	{

	  //read through the list of exclusion type. If connected = 0 and extractor >0
	  // add only type value back to the label (has been removed by the -all option). 
	  for ( unsigned int i=0; i<typeVec.size(); i++ )
	    {
	      // extract type labels
	      LabelMapChestExtractorType::Pointer extractor = LabelMapChestExtractorType::New();
	      extractor->SetInput( inputLabelMap );
	      extractor->SetChestType(typeVec[i]);
	      extractor->Update();
	      
	      LabelMapIteratorType it_exclusion(extractor->GetOutput(),extractor->GetOutput()->GetBufferedRegion());
	      LabelMapIteratorType it_input(inputLabelMap,inputLabelMap->GetBufferedRegion());
	      LabelMapIteratorType it_components2(connectedLabelMap,connectedLabelMap->GetBufferedRegion());
	      LabelMapIteratorType it_output2(outputLabelMap,outputLabelMap->GetBufferedRegion());

	      it_components2.GoToBegin();
	      it_exclusion.GoToBegin();
	      it_output2.GoToBegin();
	      it_input.GoToBegin();

	      while ( !it_output2.IsAtEnd() )
		{
		  if(( it_components2.Get() == 0) && (it_exclusion.Get()>0))
		    {
		      //get the present old label value and translate to region and type. Then add the exclusion type value to the label
		      unsigned char chest_region = conventions.GetChestRegionFromValue( it_input.Get() );
		      unsigned char chest_type = conventions.GetChestTypeFromValue( it_input.Get() );

		      unsigned short final_value = conventions.GetValueFromChestRegionAndType(  (unsigned char)( cip::UNDEFINEDREGION ), chest_type ); 
		       it_output2.Set(final_value);
		    }
		  ++it_components2;
		  ++it_exclusion;
		  ++it_output2;
		  ++it_input;
		}
	    }
	  //read through the list of exclusion region. If connected = 0 and extractor >0 add only region value to the label
	  for ( unsigned int i=0; i<regionVec.size(); i++ )
	    {
	      // extract region labels

	      LabelMapChestExtractorType::Pointer extractor = LabelMapChestExtractorType::New();
	      extractor->SetInput( inputLabelMap );
	      extractor->SetChestRegion(regionVec[i]);
	      extractor->Update();
	      
	      LabelMapIteratorType it_exclusion(extractor->GetOutput(),extractor->GetOutput()->GetBufferedRegion());
	      LabelMapIteratorType it_input(inputLabelMap,inputLabelMap->GetBufferedRegion());
	      LabelMapIteratorType it_components2(connectedLabelMap,connectedLabelMap->GetBufferedRegion());
	      LabelMapIteratorType it_output2(outputLabelMap,outputLabelMap->GetBufferedRegion());
	      it_components2.GoToBegin();
	      it_exclusion.GoToBegin();
	      it_output2.GoToBegin();
	      it_input.GoToBegin();

	      while ( !it_output2.IsAtEnd() )
		{
		  if(( it_components2.Get() == 0) && (it_exclusion.Get()>0))
		    {
		      //get the present old label value and translate to region and type. Then add the exclusion type value to the label
		      unsigned char chest_region = conventions.GetChestRegionFromValue( it_input.Get() );
		      unsigned char chest_type = conventions.GetChestTypeFromValue( it_output2.Get() ); //the type that has already been placed

		      unsigned short final_value = conventions.GetValueFromChestRegionAndType( chest_region ,chest_type); 

		       it_output2.Set(final_value);
		    }
		  ++it_components2;
		  ++it_exclusion;
		  ++it_output2;
		  ++it_input;
		}
	    }

	  //read through the list of exclusion region and type. If connected = 0 and extractor >0
	  // if not in region list add region value
	  // if not in type list add type value
	  for ( unsigned int i=0; i<regionTypePairVec.size(); i++ )
	    {
		// extract region labels
		LabelMapChestExtractorType::Pointer extractor = LabelMapChestExtractorType::New();
		extractor->SetInput( inputLabelMap );
		extractor->SetRegionAndType(regionTypePairVec[i].region, regionTypePairVec[i].type);
		extractor->Update();

		LabelMapIteratorType it_exclusion(extractor->GetOutput(),extractor->GetOutput()->GetBufferedRegion());
		LabelMapIteratorType it_input(inputLabelMap,inputLabelMap->GetBufferedRegion());
		LabelMapIteratorType it_components2(connectedLabelMap,connectedLabelMap->GetBufferedRegion());
		LabelMapIteratorType it_output2(outputLabelMap,outputLabelMap->GetBufferedRegion());
		it_components2.GoToBegin();
		it_exclusion.GoToBegin();
		it_output2.GoToBegin();
		it_input.GoToBegin();

		while ( !it_output2.IsAtEnd() )
		  {
		    if(( it_components2.Get() == 0) && (it_exclusion.Get()>0))
		      {
			//get the present old label value and translate to region and type. Then add the exclusion region & type value to the label
			// Can add both back blindly since if it was before then was added, if it was not, then should be added.
			it_output2.Set(it_input.Get());
		      }
		    ++it_components2;
		    ++it_exclusion;
		    ++it_output2;
		    ++it_input;
		  }
	    } 
	  
	}     
      
    }
  
}




  // Performs connected components given an extracted region of interest and a specified slice direction / volumetric
  void performConnectedComponents(cip::LabelMapType::Pointer unconnectedLabelMap,cip::LabelMapType::Pointer connectedLabelMap, int sizeThreshold, std::string  evalMethod)
    {
       
       //Set output to input by default in case the labelmap is empty
       LabelMapIteratorType it_connected( connectedLabelMap,connectedLabelMap->GetBufferedRegion() ); 
       LabelMapIteratorType it_unconnected( unconnectedLabelMap,unconnectedLabelMap->GetBufferedRegion() ); 
       it_connected.GoToBegin();
       it_unconnected.GoToBegin();
       while ( !it_connected.IsAtEnd() )
	 {	  
	   it_connected.Set(it_unconnected.Get());
	   
	   ++it_connected;
	   ++it_unconnected;
	 }
       

     //perform connected components on the labels from the original  set
      if (evalMethod.compare("vol") == 0)
	{  	  
	  ConnectedComponent3DType::Pointer connected =  ConnectedComponent3DType::New ();
	  connected->SetInput(unconnectedLabelMap);
	  connected->Update();
	  
	  std::cout << "Volumetric, Number of objects: " << connected->GetObjectCount() << std::endl;
	  //remove labels with small size (or create a volume with the value 1 if we want to delete the labels)
	  RelabelFilter3DType::Pointer relabel =  RelabelFilter3DType::New();
	  
	  relabel->SetInput(connected->GetOutput());
	  relabel->SetMinimumObjectSize(sizeThreshold);
	  relabel->Update();
	  
	  //delete the labels from the input labelmap that have been removed post connected components
	  LabelMapIteratorType it_components( relabel->GetOutput(),relabel->GetOutput()->GetBufferedRegion() );
	  LabelMapIteratorType it_original( unconnectedLabelMap,unconnectedLabelMap->GetBufferedRegion() ); 	  
	  LabelMapIteratorType it_connected( connectedLabelMap,connectedLabelMap->GetBufferedRegion() ); 	  

	  //store the output in lieu of the input labelmap
	  it_components.GoToBegin();
	  it_original.GoToBegin();
	  it_connected.GoToBegin();
	  
	  while ( !it_components.IsAtEnd() )
	    {
	      if(( it_components.Get() == 0) &&(it_original.Get()>0))
		{
		  it_connected.Set(0 );
		}
	      ++it_components;
	      ++it_original;
	      ++it_connected;
	    }
	}
      else
	{ //not vol
	  cip::LabelMapType::IndexType tempIndex;
	  cip::LabelMapType::RegionType boundingBox;
	  boundingBox = cip::GetLabelMapChestRegionChestTypeBoundingBoxRegion(unconnectedLabelMap);
	  cip::LabelMapType::SizeType unconnected_size = unconnectedLabelMap->GetBufferedRegion().GetSize();
	  cip::LabelMapType::SpacingType unconnected_spacing = unconnectedLabelMap->GetSpacing();
      
	  unsigned int sliceMin;
	  unsigned int sliceMax;
      
	  if (evalMethod.compare("axial") == 0)
	    {  
	      sliceMin = boundingBox.GetIndex()[2];
	      sliceMax = boundingBox.GetIndex()[2] + boundingBox.GetSize()[2] - 1;
	     }
	   else if (evalMethod.compare("coronal") == 0)
	     {      	  
	       sliceMin = boundingBox.GetIndex()[1];
	       sliceMax = boundingBox.GetIndex()[1] + boundingBox.GetSize()[1] - 1;
	     }
	   else if (evalMethod.compare("sagittal") == 0)
	    {      	  
	      sliceMin = boundingBox.GetIndex()[0];
	      sliceMax = boundingBox.GetIndex()[0] + boundingBox.GetSize()[0] - 1;	  
	    }

	  int numImages = sliceMax - sliceMin + 1;
	  //cip::LabelMapType::IndexType slice_start = inputRegion.GetIndex();
	  
	  //if there are no labels in the volume, then  boundingBox.GetSize() =0. do no perform in that case
	  if((boundingBox.GetSize()[0]==0) || (boundingBox.GetSize()[1]==0) ||(boundingBox.GetSize()[2]==0))
	    {
	      std::cout<<" no labels in labelmap "<<std::endl;
	      sliceMin = 0;
	      sliceMax = 0;
	    }

	  //perform connected components on each slice separately, 
	  for ( unsigned int n=sliceMin; n<=sliceMax; n++ )
	    {
	      if(GetSliceHasForeground(unconnectedLabelMap, n, evalMethod))
		{  
		  //Create a 2D image by extracting the appropriate slice in the right direction
		  LabelMapSliceType::Pointer unconnectedLabelMapSlice = LabelMapSliceType::New();
	      
		  ExtractLabelMapSlice(unconnectedLabelMap, unconnectedLabelMapSlice, n , evalMethod);

		  typedef itk::ImageDuplicator< LabelMapSliceType > DuplicatorType;
		  DuplicatorType::Pointer duplicatorSlice = DuplicatorType::New();
		  duplicatorSlice->SetInputImage(unconnectedLabelMapSlice);
		  duplicatorSlice->Update();
		  LabelMapSliceType::Pointer connectedLabelMapSlice = duplicatorSlice->GetOutput();

		  ConnectedComponent2DType::Pointer connected = ConnectedComponent2DType::New();    
		  connected->SetInput(unconnectedLabelMapSlice); 
		  connected->Update();

		  //RelabelComponentImageFilter
		  //remove labels with small size (or create a volume with the value 1 if we want to delete the labels)
		  RelabelFilter2DType::Pointer relabel = RelabelFilter2DType::New();
		  RelabelFilter2DType::ObjectSizeType minSize;
		  minSize = sizeThreshold;
		      
		  relabel->SetInput(connected->GetOutput());
		  relabel->SetMinimumObjectSize(sizeThreshold);
		  relabel->Update();  
		  std::cout << "slice "<<n<<" eval " <<evalMethod <<" Number of objects: " << connected->GetObjectCount() << std::endl;
	      
		  //delete the labels from the input labelmap that have been removed post connected components
		  LabelMapIterator2DType it_components2d( relabel->GetOutput(),relabel->GetOutput()->GetBufferedRegion() );
		  LabelMapIterator2DType it_original2d( unconnectedLabelMapSlice,unconnectedLabelMapSlice->GetBufferedRegion() ); 	  
		  LabelMapIteratorType it_connected( connectedLabelMap,connectedLabelMap->GetBufferedRegion() ); 	// the volume to be output  

		  // the tricky part here is to get the 2 iterators right, depending on slice direction and location
		  //store the output in lieu of the input labelmap
		  it_components2d.GoToBegin();
		  it_original2d.GoToBegin();
		  it_connected.GoToBegin();
	      
		  while ( !it_components2d.IsAtEnd() )
		    {
		      cip::LabelMapType::IndexType tempIndex;
		      if(evalMethod.compare("axial") == 0)
			{
			  tempIndex[0] = it_components2d.GetIndex()[0];
			  tempIndex[1] = it_components2d.GetIndex()[1];
			  tempIndex[2] = n;
			}
		      if(evalMethod.compare("coronal") == 0)
			{
			  tempIndex[0] = it_components2d.GetIndex()[0];
			  tempIndex[1] = n; 
			  tempIndex[2] = it_components2d.GetIndex()[1];
			}
		      if(evalMethod.compare("sagittal") == 0)
			{
			  tempIndex[0] = n; 
			  tempIndex[1] = it_components2d.GetIndex()[0];			 
			  tempIndex[2] = it_components2d.GetIndex()[1];
			}


		      if(( it_components2d.Get() == 0) &&(it_original2d.Get()>0))
			{
			  connectedLabelMap->SetPixel( tempIndex, 0 );

			}
		      ++it_components2d;
		      ++it_original2d;
		      ++it_connected;
		    }	      
		}
	    }
	}
      return;
    }
 
