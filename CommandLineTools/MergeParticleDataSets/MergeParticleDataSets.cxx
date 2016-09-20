/** \file
 *  \ingroup commandLineTools
 *  \details This program accepts as input multiple particle
 *  datasets and merges them into one dataset for output. 
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "cipHelper.h"
#include "vtkSmartPointer.h"
#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"
#include "MergeParticleDataSetsCLP.h"

void MergeParticles( vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData > );
void CopyParticles( vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData > );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  vtkSmartPointer< vtkPolyData > mergedParticles = vtkSmartPointer< vtkPolyData >::New();

  for ( unsigned int i=0; i<inFileNamesVec.size(); i++ )
    {
      std::cout << "Reading particles..." << std::endl;
      vtkSmartPointer< vtkPolyDataReader > particlesReader = vtkSmartPointer< vtkPolyDataReader >::New();
        particlesReader->SetFileName( inFileNamesVec[i].c_str() );
	particlesReader->Update();
      
      // If the particles data sets have different field data, don't merge them. We will eventually
      // want a way to handle merging particles data sets with different field data, but
      // in such a case the field data of both data sets should be reflected in the merged
      // data set.
      if ( mergedParticles->GetFieldData()->GetNumberOfArrays() == 0 && i == 0 )
	{
	  cip::TransferFieldData( particlesReader->GetOutput(), mergedParticles );
	}
      else if ( !cip::HaveSameFieldData( mergedParticles, particlesReader->GetOutput() ) )
	{
	  std::cout << inFileNamesVec[i] << 
	    " has a different number of field data arrays than a previously loaded data set. Skipping this data set..." 
		    << std::endl;
	  continue;
	}

      if ( mergedParticles->GetNumberOfPoints() == 0 )
	{
	  std::cout << "Copying..." << std::endl;
	  CopyParticles( particlesReader->GetOutput(), mergedParticles );
	}
      else
	{
	  std::cout << "Merging..." << std::endl;
	  MergeParticles( particlesReader->GetOutput(), mergedParticles );
	}
    }
  
  std::cout << "Writing merged particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > particlesWriter = vtkSmartPointer< vtkPolyDataWriter >::New();
    particlesWriter->SetInputData( mergedParticles );
    particlesWriter->SetFileName( outFileName.c_str() );
    particlesWriter->Write();      

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

void MergeParticles( vtkSmartPointer< vtkPolyData > particles, vtkSmartPointer< vtkPolyData > mergedParticles )
{
  unsigned int numberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();;
  vtkSmartPointer< vtkPoints > points = vtkSmartPointer< vtkPoints >::New();
  std::vector< vtkSmartPointer< vtkFloatArray > > arrayVec;

  for ( unsigned int i=0; i<numberOfPointDataArrays; i++ )
    {
      vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
        array->SetNumberOfComponents( particles->GetPointData()->GetArray(i)->GetNumberOfComponents() );
	array->SetName( particles->GetPointData()->GetArray(i)->GetName() );
      
      arrayVec.push_back( array );
    }

  double point1[3];
  double point2[3];
  bool addPoint;
  unsigned int inc = 0;
  for ( int i=0; i<particles->GetNumberOfPoints(); i++ )
    {
      addPoint = true;
      
      point1[0] = particles->GetPoint( i )[0];
      point1[1] = particles->GetPoint( i )[1];
      point1[2] = particles->GetPoint( i )[2];
      
      for ( int j=0; j<mergedParticles->GetNumberOfPoints(); j++ )
	{
	  point2[0] = mergedParticles->GetPoint( j )[0];
	  point2[1] = mergedParticles->GetPoint( j )[1];
	  point2[2] = mergedParticles->GetPoint( j )[2];
	  
	  if ( point1[0] == point2[0] && point1[1] == point2[1] && point1[2] == point2[2] )
	    {
	      addPoint = false;
	      break;
	    }
	}
      if ( addPoint )
	{
	  for ( unsigned int k=0; k<numberOfPointDataArrays; k++ )
	    {
	      mergedParticles->GetPointData()->GetArray(k)->InsertNextTuple( particles->GetPointData()->GetArray(k)->GetTuple(i) );
	    }
	  mergedParticles->GetPoints()->InsertNextPoint( point1 );
	}
    }
}

void CopyParticles( vtkSmartPointer< vtkPolyData > particles, vtkSmartPointer< vtkPolyData > mergedParticles )
{
  unsigned int numberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();
  vtkSmartPointer< vtkPoints > points = vtkSmartPointer< vtkPoints >::New();
  std::vector< vtkSmartPointer< vtkFloatArray > > arrayVec;

  for ( unsigned int i=0; i<numberOfPointDataArrays; i++ )
    {
      vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
        array->SetNumberOfComponents( particles->GetPointData()->GetArray(i)->GetNumberOfComponents() );
	array->SetName( particles->GetPointData()->GetArray(i)->GetName() );

      arrayVec.push_back( array );
    }

  unsigned int inc = 0;
  for ( int i=0; i<particles->GetNumberOfPoints(); i++ )
    {
      for ( unsigned int k=0; k<numberOfPointDataArrays; k++ )
	{
	  arrayVec[k]->InsertTuple( inc, particles->GetPointData()->GetArray(k)->GetTuple(i) );
	}
      inc++;
      points->InsertNextPoint( particles->GetPoint(i) );
    }

  mergedParticles->SetPoints( points );
  for ( unsigned int j=0; j<numberOfPointDataArrays; j++ )
    {
      mergedParticles->GetPointData()->AddArray( arrayVec[j] );
    }
}

#endif
