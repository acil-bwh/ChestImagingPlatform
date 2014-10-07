#include "FilterConnectedComponentsHelper.h"
#include <itkImageDuplicator.h>


//bool GetSliceHasForeground(cip::LabelMapType::Pointer, unsigned int, std::string );

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
      std::cout<<"in axial"<<std::endl;
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
  sliceExtractor->SetExtractionRegion( sliceExtractorRegion );
  sliceExtractor->Update();

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



//Performs filtering
cip::LabelMapType::Pointer FilterConnectedComponents(cip::LabelMapType::Pointer inputLabelMap, int sizeThreshold, std::vector< unsigned char> regionVec, std::vector< unsigned char> typeVec, std::vector<REGIONTYPEPAIR> regionTypePairVec, std::string  evalMethod, bool isInclude, bool isExclude)
{
  
  //cip::LabelMapType::Pointer inclusionLabelMap = cip::LabelMapType::New();
  cip::LabelMapType::IndexType input_start;

  input_start[0] = 0;
  input_start[1] = 0;
 
  cip::LabelMapType::SizeType input_size = inputLabelMap->GetLargestPossibleRegion().GetSize();
  cip::LabelMapType::SpacingType inputSpacing = inputLabelMap->GetSpacing();
  cip::LabelMapType::RegionType inputRegion;
  inputRegion.SetSize(input_size);
  inputRegion.SetIndex(input_start);

  typedef itk::ImageDuplicator< cip::LabelMapType > DuplicatorType;
  DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage(inputLabelMap);
  duplicator->Update();
  cip::LabelMapType::Pointer inclusionLabelMap = duplicator->GetModifiableOutput();



  if( (isInclude == true) && (isExclude == true))
    {
      std::cerr <<"Cannot specify inclusion and exclusion criteria"<< std::endl;
      return inclusionLabelMap;
    }
    

  //extract the appropriate region and type labels to include and create a new volume that only has the labels 
  // for which the connected components will be computed
  LabelMapChestExtractorType::Pointer extractor = LabelMapChestExtractorType::New();
  extractor->SetInput( inputLabelMap );

  if(isInclude == true)
    {
      for ( unsigned int i=0; i<regionVec.size(); i++ )
	{
	  extractor->SetChestRegion(regionVec[i]);
	}
      if ( typeVec.size() > 0 )
	{
	  for ( unsigned int i=0; i<typeVec.size(); i++ )
	    {
	      extractor->SetChestType( typeVec[i] );
	    }
	}
      if ( regionTypePairVec.size()>0 )
	{
	  for ( unsigned int i=0; i<regionTypePairVec.size(); i++ )
	    {
	      extractor->SetRegionAndType( regionTypePairVec[i].region,regionTypePairVec[i].type );
	    }
	}
	
      extractor->Update();

      // Set all inclusion values to the appropriate in the volume for which we are going to compute connected components
	
      LabelMapIteratorType itinclusion( inclusionLabelMap,inclusionLabelMap->GetBufferedRegion() );
      LabelMapIteratorType it( extractor->GetOutput(),extractor->GetOutput()->GetBufferedRegion() );
      it.GoToBegin();
      itinclusion.GoToBegin();
      while ( !it.IsAtEnd() )
	{
	  if ( it.Get() != 0 )
	    {
	      itinclusion.Set( it.Get() );
	    }
	  else
	    itinclusion.Set( 0 );
	  ++it;
	  ++itinclusion;
	}

    }

  if( isExclude == true)
    {
      //extract the appropriate region and type labels to exclude

      for ( unsigned int i=0; i<regionVec.size(); i++ )
	{
	  extractor->SetChestRegion(regionVec[i]);
	}
      if ( typeVec.size() > 0 )
	{
	  for ( unsigned int i=0; i<typeVec.size(); i++ )
	    {
	      extractor->SetChestType( typeVec[i] );
	    }
	}
      if ( regionTypePairVec.size()>0 )
	{
	  for ( unsigned int i=0; i<regionTypePairVec.size(); i++ )
	    {
	      extractor->SetRegionAndType( regionTypePairVec[i].region,regionTypePairVec[i].type );
	    }
	}
	
      extractor->Update();

      // Set all exclusion values to 0 in the volume for which we are going to compute connected components
      LabelMapIteratorType it( extractor->GetOutput(),extractor->GetOutput()->GetBufferedRegion() );
      LabelMapIteratorType itexclusion( inclusionLabelMap,inclusionLabelMap->GetBufferedRegion() );
	
      it.GoToBegin();
      itexclusion.GoToBegin();
      while ( !it.IsAtEnd() )
	{
	  if ( it.Get() != 0 )
	    {
	      itexclusion.Set(0 );
	    }
	  else
	    itexclusion.Set( it.Get() );
	  ++it;
	  ++itexclusion;
	}
    }

  if( (isInclude == false) && (isExclude == false))
    {
      //then include all labelmaps by default.
      std::cout<<"performing connected components on all labels"<<std::endl;

      LabelMapIteratorType itinclusiondefault( inclusionLabelMap,inclusionLabelMap->GetBufferedRegion() );
      LabelMapIteratorType itdefault( inputLabelMap,inputLabelMap->GetBufferedRegion() );
      itdefault.GoToBegin();
      itinclusiondefault.GoToBegin();
      while ( !itdefault.IsAtEnd() )
	{
	  itinclusiondefault.Set(itdefault.Get() );
	  ++itdefault;
	  ++itinclusiondefault;
	}
    }
    

  //get a list of unique labels in order to compute connected components for each separately
  std::list< unsigned short > labelsList;

  LabelMapIteratorType it_forlabels(inclusionLabelMap,inclusionLabelMap->GetBufferedRegion());

  it_forlabels.GoToBegin();
  while ( !it_forlabels.IsAtEnd() )
    {
      if ( it_forlabels.Get() != 0 )
	{
	  labelsList.push_back( it_forlabels.Get() );
	}

      ++it_forlabels;
    }

  labelsList.unique();
  labelsList.sort();
  labelsList.unique();

  //perform connected components on the labels from the original  set
  if (evalMethod.compare("vol") == 0)
    {  
      //for each label in label list, perform connected components separately
      for (std::list<unsigned short>::iterator it_labels = labelsList.begin(); it_labels != labelsList.end(); it_labels++)
	{
	  typedef itk::ImageDuplicator< cip::LabelMapType > DuplicatorType;
	  DuplicatorType::Pointer duplicator2 = DuplicatorType::New();
	  duplicator2->SetInputImage(inclusionLabelMap);
	  duplicator2->Update();
	  cip::LabelMapType::Pointer forconnectedLabelMap = duplicator2->GetModifiableOutput();
	  
	  LabelMapIteratorType it_forconnected(forconnectedLabelMap,forconnectedLabelMap->GetBufferedRegion());
	  it_forconnected.GoToBegin();
	  while ( !it_forconnected.IsAtEnd() )
	    {
	      if ( it_forconnected.Get() != *it_labels )
		{
		  it_forconnected.Set(0);
		}
	      
	      ++it_forconnected;
	    }

	  ConnectedComponent3DType::Pointer connected =  ConnectedComponent3DType::New ();
	  connected->SetInput(forconnectedLabelMap);
	  connected->Update();
	  
	  std::cout << "Volumetric, Number of objects: " << connected->GetObjectCount() << std::endl;
	  
	  //remove labels with small size (or create a volume with the value 1 if we want to delete the labels)
	  RelabelFilter3DType::Pointer relabel =	RelabelFilter3DType::New();
	  
	  relabel->SetInput(connected->GetOutput());
	  relabel->SetMinimumObjectSize(sizeThreshold);
	  relabel->Update();
	  
	  //delete the labels from the input labelmap that have been removed post connected components
	  LabelMapIteratorType it_components( relabel->GetOutput(),relabel->GetOutput()->GetBufferedRegion() );
	  LabelMapIteratorType it_original( inputLabelMap,inputLabelMap->GetBufferedRegion() ); 
	  LabelMapIteratorType it_inclusion( forconnectedLabelMap,forconnectedLabelMap->GetBufferedRegion() ); 
	  
	  
	  //store the output in lieu of the input labelmap
	  it_components.GoToBegin();
	  it_original.GoToBegin();
	  it_inclusion.GoToBegin();
	  
	  
	  while ( !it_components.IsAtEnd() )
	    {
	      if(( it_components.Get() == 0) && (it_inclusion.Get()>0) &&(it_original.Get()>0))
		{
		  std::cout<<"removing label "<<it_original.Get()<<std::endl;		  
		  it_original.Set(0 );
		}
	      ++it_components;
	      ++it_original;
	      ++it_inclusion;
	    }
	}
    }
  cip::LabelMapType::RegionType boundingBox;
  boundingBox = cip::GetLabelMapChestRegionChestTypeBoundingBoxRegion(inclusionLabelMap);
  cip::LabelMapType::SizeType inclusion_size = inclusionLabelMap->GetBufferedRegion().GetSize();
  cip::LabelMapType::SpacingType inclusion_spacing = inclusionLabelMap->GetSpacing();

  //  cip::LabelMapType::SizeType sliceSize;
  //cip::LabelMapType::SpacingType sliceSpacing;

  unsigned int sliceMin;
  unsigned int sliceMax;

  if (evalMethod.compare("axial") == 0)
    {  
      std::cout<<"performing axial connected components"<<std::endl;

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

 

  if (evalMethod.compare("vol") != 0)
    {
      int numImages = sliceMax - sliceMin + 1;
      std::cout<<"max = "<<sliceMax<<" min = "<<sliceMin<<std::endl;
      cip::LabelMapType::IndexType slice_start = inputRegion.GetIndex();

      //perform connected components on each slice separately
      for ( unsigned int n=sliceMin; n<=sliceMax; n++ )
	{
	  std::cout<<" checking slice "<<n<<std::endl; 
	  if(GetSliceHasForeground(inclusionLabelMap, n, evalMethod))
	    {
	      std::cout<<"evaluating slice "<<n<<" using "<<evalMethod<<std::endl;
	      //Create a 2D image by extracting the appropriate slice in the right direction
	      LabelMapSliceType::Pointer inclusionSlice = LabelMapSliceType::New();
	      
	      ExtractLabelMapSlice(inclusionLabelMap, inclusionSlice, n , evalMethod);

	      for (std::list<unsigned short>::iterator it_labels = labelsList.begin(); it_labels != labelsList.end(); it_labels++)
		{
		  typedef itk::ImageDuplicator< LabelMapSliceType > DuplicatorType;
		  DuplicatorType::Pointer duplicator2 = DuplicatorType::New();
		  duplicator2->SetInputImage(inclusionSlice);
		  duplicator2->Update();
		  LabelMapSliceType::Pointer forconnectedLabelMapSlice = duplicator2->GetModifiableOutput();
		  
		  LabelMapIterator2DType it_forconnected(forconnectedLabelMapSlice,forconnectedLabelMapSlice->GetBufferedRegion());
		  it_forconnected.GoToBegin();
		  while ( !it_forconnected.IsAtEnd() )
		    {
		      if ( it_forconnected.Get() != *it_labels )
			{
			  it_forconnected.Set(0);
			}
		      
		      ++it_forconnected;
		    }

		  ConnectedComponent2DType::Pointer connected = ConnectedComponent2DType::New();    
		  connected->SetInput(forconnectedLabelMapSlice); 
		  connected->Update();
	      
		  //RelabelComponentImageFilter
		  //remove labels with small size (or create a volume with the value 1 if we want to delete the labels)
		  RelabelFilter2DType::Pointer relabel = RelabelFilter2DType::New();
		  RelabelFilter2DType::ObjectSizeType minSize = 10;
		  minSize = sizeThreshold;
	      
		  relabel->SetInput(connected->GetOutput());
		  relabel->SetMinimumObjectSize(sizeThreshold);
		  relabel->Update();
		  
		  //delete the labels from the input labelmap that have been removed post connected components.
		  // the tricky part here is to get the 2 iterators right, depending on slice direction and location
		  
		  //iterate over the 2D relabeler, and place at right location in 3D image
		  LabelMapIterator2DType it_components( relabel->GetOutput(),relabel->GetOutput()->GetBufferedRegion() ); //2D
		  LabelMapIteratorType it_original( inputLabelMap,inputLabelMap->GetBufferedRegion() ); //these 2 will vary
		  LabelMapIterator2DType it_inclusion( forconnectedLabelMapSlice,forconnectedLabelMapSlice->GetBufferedRegion() ); //2D now..
		  
		  it_components.GoToBegin();
		  it_inclusion.GoToBegin();
		  while ( !it_components.IsAtEnd() )
		    {
		      cip::LabelMapType::IndexType tempIndex;
		      if(evalMethod.compare("axial") == 0)
			{
			  tempIndex[0] = it_components.GetIndex()[0];
			  tempIndex[1] = it_components.GetIndex()[1];
			  tempIndex[2] = n;
			}
		      if(evalMethod.compare("coronal") == 0)
			{
			  tempIndex[0] = it_components.GetIndex()[0];
			  tempIndex[1] = n; 
			  tempIndex[2] = it_components.GetIndex()[1];
			}
		      if(evalMethod.compare("sagittal") == 0)
			{
			  tempIndex[0] = n; 
			  tempIndex[1] = it_components.GetIndex()[1];			 
			  tempIndex[2] = it_components.GetIndex()[2];
			}
		      if(( it_components.Get() == 0) && (it_inclusion.Get()>0) &&(inputLabelMap->GetPixel(tempIndex)>0))
			{
			  std::cout<<"removing label "<<inputLabelMap->GetPixel(tempIndex)<<std::endl;
			  inputLabelMap->SetPixel( tempIndex, 0 );
			}
		      ++it_components;
		      ++it_inclusion;
		    }
		}
	    }
	}
    }
      // return output label map
      return  inputLabelMap; 
    }
