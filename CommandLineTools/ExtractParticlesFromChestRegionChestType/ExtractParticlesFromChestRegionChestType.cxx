#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include "ExtractParticlesFromChestRegionChestTypeCLP.h"

void GetOutputParticlesUsingLabelMap( std::string, std::vector< std::string >, vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData > );
void GetOutputParticlesUsingChestRegionChestTypeArray( std::vector< std::string >, std::vector< std::string >,
							vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData > );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  vtkSmartPointer< vtkPolyData > outParticles = vtkSmartPointer< vtkPolyData >::New();

  std::cout << "Reading polydata..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > particlesReader = vtkSmartPointer< vtkPolyDataReader >::New();
    particlesReader->SetFileName( inParticlesFileName.c_str() );
    particlesReader->Update();

  // First make sure that the particles have the 'ChestRegionChestType' array
  cip::AssertChestRegionChestTypeArrayExistence( particlesReader->GetOutput() );

  std::cout << "Extracting..." << std::endl;
  if ( labelMapFileName.compare( "NA" ) != 0 )
    {
      GetOutputParticlesUsingLabelMap( labelMapFileName, cipRegions, particlesReader->GetOutput(), outParticles );
    }
  else
    {
      GetOutputParticlesUsingChestRegionChestTypeArray( cipRegions, cipTypes, particlesReader->GetOutput(), outParticles );
    }

  // Transfer the field array data 
  cip::TransferFieldData( particlesReader->GetOutput(), outParticles );

  std::cout << "Writing extracted particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > particlesWriter = vtkSmartPointer< vtkPolyDataWriter >::New();
    particlesWriter->SetFileName( outParticlesFileName.c_str() );
    particlesWriter->SetInputData( outParticles );
    particlesWriter->Write();

  std::cout << "DONE." << std::endl;

  return 0;
}

// A) If no region is specified, all particles falling within the foreground
// region will be retained. The particles' ChestRegion and ChestType array values
// will be preserved.
// B) If a region is specified, all particles falling inside the specified label
// map's region will be retained. The particles' chest type array is preserved, but
// the chest region is overwritten with the specified desired region
void GetOutputParticlesUsingLabelMap( std::string fileName, std::vector< std::string > cipRegions,
				      vtkSmartPointer< vtkPolyData > inParticles, vtkSmartPointer< vtkPolyData > outParticles )
{
  cip::ChestConventions conventions;

  std::cout << "Reading label map..." << std::endl;
  cip::LabelMapReaderType::Pointer labelMapReader = cip::LabelMapReaderType::New();
    labelMapReader->SetFileName( fileName );
  try
    {
    labelMapReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading label map:";
    std::cerr << excp << std::endl;
    }

  unsigned int numberPointDataArrays = inParticles->GetPointData()->GetNumberOfArrays();
  unsigned int numberParticles       = inParticles->GetNumberOfPoints();

  vtkSmartPointer< vtkPoints > outputPoints  = vtkSmartPointer< vtkPoints >::New();

  std::vector< vtkSmartPointer< vtkFloatArray > > arrayVec;

  for ( unsigned int i=0; i<numberPointDataArrays; i++ )
    {
    vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
      array->SetNumberOfComponents( inParticles->GetPointData()->GetArray(i)->GetNumberOfComponents() );
      array->SetName( inParticles->GetPointData()->GetArray(i)->GetName() );

    arrayVec.push_back( array );
    }

  unsigned int inc = 0;
  cip::LabelMapType::PointType point;
  cip::LabelMapType::IndexType index;

  for ( unsigned int i=0; i<numberParticles; i++ )
    {
    point[0] = inParticles->GetPoint(i)[0];
    point[1] = inParticles->GetPoint(i)[1];
    point[2] = inParticles->GetPoint(i)[2];

    labelMapReader->GetOutput()->TransformPhysicalPointToIndex( point, index );

    unsigned short labelValue   = labelMapReader->GetOutput()->GetPixel( index );
    unsigned char  labelRegion  = conventions.GetChestRegionFromValue( labelValue );

    if ( labelValue > 0 )
      {
	for ( unsigned int k=0; k<cipRegions.size(); k++ )
	  {
	    unsigned char cipRegion = conventions.GetChestRegionValueFromName( cipRegions[k] );

	    // If the label map chest region is a subordinate of the requested
	    // chest region, then add this particle to the output
	    if ( cipRegion == (unsigned char)(cip::UNDEFINEDREGION ) ||
		 conventions.CheckSubordinateSuperiorChestRegionRelationship( labelRegion, cipRegion ) )
	      {
		outputPoints->InsertNextPoint( inParticles->GetPoint(i) );

		for ( unsigned int j=0; j<numberPointDataArrays; j++ )
		  {
		    std::string name( arrayVec[j]->GetName() );
		    if ( name.compare( "ChestRegionChestType" ) == 0 )
		      {
			unsigned short chestRegionChestTypeValue = 
			  (unsigned short)(inParticles->GetPointData()->GetArray(j)->GetTuple(i)[0]);
			unsigned char cipType = conventions.GetChestTypeFromValue( chestRegionChestTypeValue );
			float newChestRegionChestTypeValue = 
			  float(conventions.GetValueFromChestRegionAndType( cipRegion, cipType ));

			arrayVec[j]->InsertTuple( inc, &newChestRegionChestTypeValue );
		      }
		    else
		      {
			arrayVec[j]->InsertTuple( inc, inParticles->GetPointData()->GetArray(j)->GetTuple(i) );
		      }
		  }
		inc++;

		break;
	      }
	  }
      }
    }

  outParticles->SetPoints( outputPoints );
  for ( unsigned int j=0; j<numberPointDataArrays; j++ )
    {
    outParticles->GetPointData()->AddArray( arrayVec[j] );
    }
}

void GetOutputParticlesUsingChestRegionChestTypeArray( std::vector< std::string > cipRegions, std::vector< std::string > cipTypes,
							vtkSmartPointer< vtkPolyData > inParticles, vtkSmartPointer< vtkPolyData > outParticles )
{
  cip::ChestConventions conventions;

  unsigned int numberPointDataArrays = inParticles->GetPointData()->GetNumberOfArrays();
  unsigned int numberParticles       = inParticles->GetNumberOfPoints();

  vtkSmartPointer< vtkPoints > outputPoints  = vtkSmartPointer< vtkPoints >::New();
  std::vector< vtkSmartPointer< vtkFloatArray > > pointArrayVec;

  for ( unsigned int i=0; i<numberPointDataArrays; i++ )
    {
    vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
      array->SetNumberOfComponents( inParticles->GetPointData()->GetArray(i)->GetNumberOfComponents() );
      array->SetName( inParticles->GetPointData()->GetArray(i)->GetName() );

    pointArrayVec.push_back( array );
    }

  unsigned int inc = 0;
  for ( unsigned int i=0; i<numberParticles; i++ )
    {
      unsigned short chestRegionChestTypeValue = 
	(unsigned short)(inParticles->GetPointData()->GetArray( "ChestRegionChestType" )->GetTuple(i)[0]);
      unsigned char cipRegion = conventions.GetChestRegionFromValue( chestRegionChestTypeValue );
      unsigned char cipType = conventions.GetChestTypeFromValue( chestRegionChestTypeValue );

      bool add = false;
      for ( unsigned int j=0; j<cipTypes.size(); j++ )
	{
	  if ( cipType == conventions.GetChestTypeValueFromName( cipTypes[j] ) )
	    {
	      add = true;
	      break;
	    }
	}
      if ( !add )
	{
	  for ( unsigned int j=0; j<cipRegions.size(); j++ )
	    {
	      if ( cipRegion == conventions.GetChestRegionValueFromName( cipRegions[j] ) )
		{
		  add = true;
		  break;
		}
	    }
	}
      if ( add )
	{
	  outputPoints->InsertNextPoint( inParticles->GetPoint(i) );
	  for ( unsigned int j=0; j<numberPointDataArrays; j++ )
	    {
	      pointArrayVec[j]->InsertTuple( inc, inParticles->GetPointData()->GetArray(j)->GetTuple(i) );
	    }
	  inc++;
	}
    }

  outParticles->SetPoints( outputPoints );
  for ( unsigned int j=0; j<numberPointDataArrays; j++ )
    {
    outParticles->GetPointData()->AddArray( pointArrayVec[j] );
    }
}

#endif
