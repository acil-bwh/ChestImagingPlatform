#include "cipParticleConnectedComponentFilter.h"
#include "vtkIdList.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"
#include "vtkSmartPointer.h"
#include <cfloat>
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "cipHelper.h"

cipParticleConnectedComponentFilter::cipParticleConnectedComponentFilter()
{
  this->OutputPolyData        = vtkPolyData::New();
  this->Locator               = vtkPointLocator::New();

  this->NumberInputParticles       = 0;
  this->NumberOutputParticles      = 0;
  this->ComponentSizeThreshold     = 10;
  this->SelectedComponent          = 0;
  this->MaximumComponentSize       = USHRT_MAX;
  this->ParticleDistanceThreshold  = LONG_MAX;
}


void cipParticleConnectedComponentFilter::SetParticleAngleThreshold( double threshold )
{
  this->ParticleAngleThreshold = threshold;
}


void cipParticleConnectedComponentFilter::SetMaximumComponentSize( unsigned int maxSize )
{
  this->MaximumComponentSize = maxSize;
}


unsigned int cipParticleConnectedComponentFilter::GetMaximumComponentSize()
{
  return this->MaximumComponentSize;
}


double cipParticleConnectedComponentFilter::GetParticleAngleThreshold()
{
  return this->ParticleAngleThreshold;
}


void cipParticleConnectedComponentFilter::SetParticleDistanceThreshold( double distance )
{
  this->ParticleDistanceThreshold = distance;
}


double cipParticleConnectedComponentFilter::GetParticleDistanceThreshold()
{
  return this->ParticleDistanceThreshold;
}


unsigned int cipParticleConnectedComponentFilter::GetComponentLabelFromParticleID( unsigned int id )
{
  return this->ParticleToComponentMap[id];
}


unsigned int cipParticleConnectedComponentFilter::GetComponentSizeFromParticleID( unsigned int id )
{  
  return this->ComponentSizeMap[this->ParticleToComponentMap[id]];
}


unsigned int cipParticleConnectedComponentFilter::GetNumberOfOutputParticles()
{
  return this->NumberOutputParticles;
}


void cipParticleConnectedComponentFilter::SetSelectedComponent( unsigned int selectedComponent )
{
  this->SelectedComponent = selectedComponent;
}


unsigned int cipParticleConnectedComponentFilter::GetSelectedComponent()
{
  return this->SelectedComponent;
}


void cipParticleConnectedComponentFilter::SetInput( vtkPolyData* polyData )
{
  this->InputPolyData           = polyData;
  this->NumberInputParticles    = this->InputPolyData->GetNumberOfPoints();
  this->NumberOfPointDataArrays = this->InputPolyData->GetPointData()->GetNumberOfArrays();

  this->Locator->SetDataSet( this->InputPolyData );
  this->Locator->BuildLocator();
  
  for ( unsigned int i=0; i<this->NumberInputParticles; i++ )
    {
    this->ParticleToComponentMap[i] = 0;
    }

  // Transfer field data from input to output
  cip::TransferFieldData( this->InputPolyData, this->OutputPolyData );
}


vtkPolyData* cipParticleConnectedComponentFilter::GetOutput()
{
  return this->OutputPolyData;
}


unsigned int cipParticleConnectedComponentFilter::GetComponentSizeThreshold()
{
  return this->ComponentSizeThreshold;
}


void cipParticleConnectedComponentFilter::SetComponentSizeThreshold( unsigned int componentSizeThreshold )
{
  this->ComponentSizeThreshold = componentSizeThreshold;
}


double cipParticleConnectedComponentFilter::GetVectorMagnitude( double vector[3] )
{
  double magnitude = std::sqrt( std::pow( vector[0], 2 ) + std::pow( vector[1], 2 ) + std::pow( vector[2], 2 ) );

  return magnitude;
}


double cipParticleConnectedComponentFilter::GetVectorMagnitude( double point1[3], double point2[3] )
{
  double vec[3];
    vec[0] = point1[0] - point2[0];
    vec[1] = point1[1] - point2[1];
    vec[2] = point1[2] - point2[2];

  double magnitude = this->GetVectorMagnitude( vec );

  return magnitude;
}


double cipParticleConnectedComponentFilter::GetAngleBetweenVectors( double vec1[3], double vec2[3], bool returnDegrees )
{
  double vec1Mag = this->GetVectorMagnitude( vec1 );
  double vec2Mag = this->GetVectorMagnitude( vec2 );

  double arg = (vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2])/(vec1Mag*vec2Mag);

  if ( std::abs( arg ) > 1.0 )
    {
    arg = 1.0;
    }

  double angle = std::acos( arg );

  if ( !returnDegrees )
    {
    return angle;
    }

  double angleInDegrees = (180.0/vnl_math::pi)*angle;

  if ( angleInDegrees > 90.0 )
    {
    angleInDegrees = 180.0 - angleInDegrees;
    }

  return angleInDegrees;
}


void cipParticleConnectedComponentFilter::ComputeComponentSizes()
{
  //
  // Start by setting the component map entries to zero.
  //
  for ( unsigned int comp=1; comp<=this->LargestComponentLabel; comp++ )
    {
    this->ComponentSizeMap[comp] = 0;
    }

  //
  // Now determine each component's size
  //
  for ( unsigned int i=0; i<this->NumberInputParticles; i++ )
    {
    this->ComponentSizeMap[this->ParticleToComponentMap[i]]++;
    }
}


unsigned int cipParticleConnectedComponentFilter::GetComponentSize( unsigned int comp )
{
  if ( comp > this->LargestComponentLabel )
    {
    return 0;
    }
  else
    {
    return this->ComponentSizeMap[comp];
    }
}


vtkPolyData* cipParticleConnectedComponentFilter::GetComponent( unsigned int comp )
{
  vtkPolyData* componentPolyData = vtkPolyData::New();
  vtkPoints*   outputPoints      = vtkPoints::New();
  
  std::vector< vtkFloatArray* > arrayVec;

  for ( unsigned int i=0; i<this->NumberOfPointDataArrays; i++ )
    {
    vtkFloatArray* array = vtkFloatArray::New();
      array->SetNumberOfComponents( this->InputPolyData->GetPointData()->GetArray(i)->GetNumberOfComponents() );
      array->SetName( this->InputPolyData->GetPointData()->GetArray(i)->GetName() );

    arrayVec.push_back( array );
    }
   
  unsigned int inc = 0;
  for ( unsigned int i=0; i<this->NumberInputParticles; i++ )
    {
    if ( this->ParticleToComponentMap[i] == comp )
      {
      outputPoints->InsertNextPoint( this->InputPolyData->GetPoint(i) );
      }

    for ( unsigned int j=0; j<this->NumberOfPointDataArrays; j++ )
      {
      arrayVec[j]->InsertTuple( inc, this->InputPolyData->GetPointData()->GetArray(j)->GetTuple(i) );
      }

    inc++;
    }

  componentPolyData->SetPoints( outputPoints );
  for ( unsigned int j=0; j<this->NumberOfPointDataArrays; j++ )
    {
    componentPolyData->GetPointData()->AddArray( arrayVec[j] );    
    }

  return componentPolyData;
}


void cipParticleConnectedComponentFilter::GetComponentParticleIndices( unsigned int comp, std::vector< unsigned int >* indicesVec )
{
  for ( unsigned int i=0; i<this->NumberInputParticles; i++ )
    {
    if ( this->ParticleToComponentMap[i] == comp )
      {
      indicesVec->push_back( i );
      }
    }  
}


void cipParticleConnectedComponentFilter::QueryNeighborhood( unsigned int particleID,
							     unsigned int componentLabel,
							     unsigned int* currentComponentSize )
{
  if ( this->ParticleToComponentMap[particleID] == 0 )
    {
      this->ParticleToComponentMap[particleID] = componentLabel;
      (*currentComponentSize)++;
      
      vtkIdList* IdList = vtkIdList::New();
      this->Locator->FindPointsWithinRadius( this->ParticleDistanceThreshold,
					     this->InputPolyData->GetPoint(particleID), IdList );
      
      for ( unsigned int i=0; i < IdList->GetNumberOfIds(); i++ )
	{
	  if ( this->ParticleToComponentMap[IdList->GetId(i)] == 0 )
	    {
	      bool connected = this->EvaluateParticleConnectedness( particleID, IdList->GetId(i) );
	      if ( connected && (*currentComponentSize < this->MaximumComponentSize) )
		{
		  this->QueryNeighborhood( IdList->GetId(i), componentLabel, currentComponentSize );
		}
	    }
	}
    } 
}


bool cipParticleConnectedComponentFilter::EvaluateParticleConnectedness( unsigned int tmp1, unsigned int tmp2 )
{
  return true;
}


void cipParticleConnectedComponentFilter::Update()
{
  unsigned int componentLabel = 1;
  for ( unsigned int i=0; i<this->NumberInputParticles; i++ )
    {
      unsigned int componentSize = 0;
      this->QueryNeighborhood( i, componentLabel, &componentSize );
      componentLabel++;
    }

  this->LargestComponentLabel = componentLabel-1;

  // Now update component sizes
  this->ComputeComponentSizes();

  // At this point, we have a set of connected components, and we are
  // ready to create the output particles data
  vtkPoints* outputPoints  = vtkPoints::New();  

  std::vector< vtkSmartPointer< vtkFloatArray > > pointDataArrayVec;
  for ( unsigned int i=0; i<this->NumberOfPointDataArrays; i++ )
    {
    vtkFloatArray* array = vtkFloatArray::New();
      array->SetNumberOfComponents( this->InputPolyData->GetPointData()->GetArray(i)->GetNumberOfComponents() );
      array->SetName( this->InputPolyData->GetPointData()->GetArray(i)->GetName() );

    pointDataArrayVec.push_back( array );
    }

  vtkFloatArray* unmergedComponentsArray = vtkFloatArray::New();
    unmergedComponentsArray->SetNumberOfComponents( 1 );
    unmergedComponentsArray->SetName( "unmergedComponents" );

  unsigned int inc = 0;
  for ( unsigned int i=0; i<this->NumberInputParticles; i++ )
    {
    componentLabel = this->ParticleToComponentMap[i];

    if ( (this->SelectedComponent != 0 && this->ParticleToComponentMap[i] == this->SelectedComponent) ||
         (this->SelectedComponent == 0 && this->ComponentSizeMap[componentLabel] >=this->ComponentSizeThreshold) )
      {
      outputPoints->InsertNextPoint( this->InputPolyData->GetPoint(i) );

      for ( unsigned int j=0; j<this->NumberOfPointDataArrays; j++ )
        {
        pointDataArrayVec[j]->InsertTuple( inc, this->InputPolyData->GetPointData()->GetArray(j)->GetTuple(i) );
        }

      float temp = static_cast< float >( componentLabel );
      unmergedComponentsArray->InsertTuple( inc, &temp );

      inc++;
      }
    }

  this->NumberOutputParticles = inc;

  this->OutputPolyData->SetPoints( outputPoints );
  for ( unsigned int j=0; j<this->NumberOfPointDataArrays; j++ )
    {
      this->OutputPolyData->GetPointData()->AddArray( pointDataArrayVec[j] );    
    }
  this->OutputPolyData->GetPointData()->AddArray( unmergedComponentsArray );
}
