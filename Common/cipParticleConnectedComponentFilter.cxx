#include "cipParticleConnectedComponentFilter.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"
#include "vtkSmartPointer.h"
#include <cfloat>
#include "itkImageFileWriter.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "cipHelper.h"

cipParticleConnectedComponentFilter::cipParticleConnectedComponentFilter()
{
  this->OutputPolyData        = vtkPolyData::New();
  this->InternalInputPolyData = vtkPolyData::New();
  this->DataStructureImage    = ImageType::New();

  this->NumberInputParticles       = 0;
  this->NumberOutputParticles      = 0;
  this->InterParticleSpacing       = 0.0;
  this->ComponentSizeThreshold     = 10;
  this->SelectedComponent          = 0;
  this->MaximumComponentSize       = USHRT_MAX;
  this->ParticleDistanceThreshold  = this->InterParticleSpacing;
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


void cipParticleConnectedComponentFilter::SetInterParticleSpacing( double spacing )
{
  this->InterParticleSpacing = spacing;

  if ( this->NumberInputParticles > 0 )
    {
    this->InitializeDataStructureImageAndInternalInputPolyData();
    }
}


double cipParticleConnectedComponentFilter::GetInterParticleSpacing()
{
  return this->InterParticleSpacing;
}


void cipParticleConnectedComponentFilter::SetInput( vtkPolyData* polyData )
{
  this->InputPolyData           = polyData;
  this->NumberInputParticles    = this->InputPolyData->GetNumberOfPoints();
  this->NumberOfPointDataArrays = this->InputPolyData->GetPointData()->GetNumberOfArrays();

  if ( this->InterParticleSpacing != 0 )
    {
    this->InitializeDataStructureImageAndInternalInputPolyData();
    }

  for ( unsigned int i=0; i<this->NumberInternalInputParticles; i++ )
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
  double magnitude = vcl_sqrt( std::pow( vector[0], 2 ) + std::pow( vector[1], 2 ) + std::pow( vector[2], 2 ) );

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

  if ( vcl_abs( arg ) > 1.0 )
    {
    arg = 1.0;
    }

  double angle = vcl_acos( arg );

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
  for ( unsigned int i=0; i<this->NumberInternalInputParticles; i++ )
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
      array->SetNumberOfComponents( this->InternalInputPolyData->GetPointData()->GetArray(i)->GetNumberOfComponents() );
      array->SetName( this->InternalInputPolyData->GetPointData()->GetArray(i)->GetName() );

    arrayVec.push_back( array );
    }
   
  unsigned int inc = 0;
  for ( unsigned int i=0; i<this->NumberInternalInputParticles; i++ )
    {
    if ( this->ParticleToComponentMap[i] == comp )
      {
      outputPoints->InsertNextPoint( this->InternalInputPolyData->GetPoint(i) );
      }

    for ( unsigned int j=0; j<this->NumberOfPointDataArrays; j++ )
      {
      arrayVec[j]->InsertTuple( inc, this->InternalInputPolyData->GetPointData()->GetArray(j)->GetTuple(i) );
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
  for ( unsigned int i=0; i<this->NumberInternalInputParticles; i++ )
    {
    if ( this->ParticleToComponentMap[i] == comp )
      {
      indicesVec->push_back( i );
      }
    }  
}


void cipParticleConnectedComponentFilter::InitializeDataStructureImageAndInternalInputPolyData()
{
  double xMin = DBL_MAX;
  double yMin = DBL_MAX;
  double zMin = DBL_MAX;

  double xMax = -DBL_MAX;
  double yMax = -DBL_MAX;
  double zMax = -DBL_MAX;

  for ( unsigned int i=0; i<this->NumberInputParticles; i++ )
    {
    if ( (this->InputPolyData->GetPoint(i))[0] > xMax )
      {
      xMax = (this->InputPolyData->GetPoint(i))[0];
      }
    if ( (this->InputPolyData->GetPoint(i))[1] > yMax )
      {
      yMax = (this->InputPolyData->GetPoint(i))[1];
      }
    if ( (this->InputPolyData->GetPoint(i))[2] > zMax )
      {
      zMax = (this->InputPolyData->GetPoint(i))[2];
      }

    if ( (this->InputPolyData->GetPoint(i))[0] < xMin )
      {
      xMin = (this->InputPolyData->GetPoint(i))[0];
      }
    if ( (this->InputPolyData->GetPoint(i))[1] < yMin )
      {
      yMin = (this->InputPolyData->GetPoint(i))[1];
      }
    if ( (this->InputPolyData->GetPoint(i))[2] < zMin )
      {
      zMin = (this->InputPolyData->GetPoint(i))[2];
      }
    }

  //
  // The spacing of the data structure image is set to 1/2 of
  // the inter-particle spacing. This is somewhat arbitrary, but is chosed to give
  // (approximately) one voxel to each particle. In some cases,
  // multiple particles will get assigned to the same voxel. In this
  // case, the new particle will simply overwrite the old particle.
  //
  ImageType::PointType origin;
    origin[0] = xMin;
    origin[1] = yMin;
    origin[2] = zMin;

  ImageType::SpacingType spacing;
    spacing[0] = this->InterParticleSpacing/2.0;
    spacing[1] = this->InterParticleSpacing/2.0;
    spacing[2] = this->InterParticleSpacing/2.0;

  ImageType::SizeType  size;
    size[0] = static_cast< unsigned int >( vcl_ceil( (xMax-xMin)/spacing[0] ) ) + 1;
    size[1] = static_cast< unsigned int >( vcl_ceil( (yMax-yMin)/spacing[1] ) ) + 1;
    size[2] = static_cast< unsigned int >( vcl_ceil( (zMax-zMin)/spacing[2] ) ) + 1;

  this->DataStructureImage->SetRegions( size );
  this->DataStructureImage->Allocate();
  this->DataStructureImage->FillBuffer( 0 );
  this->DataStructureImage->SetSpacing( spacing );
  this->DataStructureImage->SetOrigin( origin );

  ImageType::PointType point;
  ImageType::IndexType index;

  for ( unsigned int i=0; i<this->NumberInputParticles; i++ )
    {      
    point[0] = this->InputPolyData->GetPoint(i)[0];
    point[1] = this->InputPolyData->GetPoint(i)[1];
    point[2] = this->InputPolyData->GetPoint(i)[2];

    this->DataStructureImage->TransformPhysicalPointToIndex( point, index );
    this->DataStructureImage->SetPixel( index, static_cast< unsigned int >( i+1 ) );
    }

  //
  // Now that the data structure image has been created, we can fill
  // the internal input poly data to be used throughout the rest of
  // the filter. The need for doing this is that only a subset of the
  // input particles are actually registered in the data structure
  // image (given that some particles overwrite old particles as the
  // image is filled). So we need our InternalInputPolyData to
  // refer to those particles that remain.
  //
  vtkPoints* points  = vtkPoints::New();

  std::vector< vtkFloatArray* > pointDataArrayVec;
  for ( unsigned int i=0; i<this->NumberOfPointDataArrays; i++ )
    {
    vtkFloatArray* array = vtkFloatArray::New();
      array->SetNumberOfComponents( this->InputPolyData->GetPointData()->GetArray(i)->GetNumberOfComponents() );
      array->SetName( this->InputPolyData->GetPointData()->GetArray(i)->GetName() );

    pointDataArrayVec.push_back( array );
    }

  IteratorType it( this->DataStructureImage, this->DataStructureImage->GetBufferedRegion() );

  unsigned int inc = 0;
  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
    if ( it.Get() != 0 )
      {
      unsigned int i = it.Get()-1;

      points->InsertNextPoint( this->InputPolyData->GetPoint(i) );
 
      for ( unsigned int j=0; j<this->NumberOfPointDataArrays; j++ )
        {
        pointDataArrayVec[j]->InsertTuple( inc, this->InputPolyData->GetPointData()->GetArray(j)->GetTuple(i) );
        }
      inc++;    
      it.Set( inc ); // Ensures that image's voxel value points to new
                     // particle structure, not the old one.
      }

    ++it;
    }

  this->NumberInternalInputParticles = inc;

  this->InternalInputPolyData->SetPoints( points );
  for ( unsigned int j=0; j<this->NumberOfPointDataArrays; j++ )
    {
    this->InternalInputPolyData->GetPointData()->AddArray( pointDataArrayVec[j] ); 
    }
}


void cipParticleConnectedComponentFilter::QueryNeighborhood( ImageType::IndexType index, unsigned int componentLabel, unsigned int* currentComponentSize )
{
  int searchRadius = 3;

  unsigned int particleIndex = this->DataStructureImage->GetPixel( index ) - 1;

  this->ParticleToComponentMap[particleIndex] = componentLabel;

  (*currentComponentSize)++;

  //
  // The ParticleToComponentMap will eventually get overwritten when
  // the component merging stage takes place. We will use the
  // following to keep a record of the unmerged component
  // labeling. This will be useful when we write our final data, since
  // we will want to indicate both the original component and the
  // final component in the output
  //
  this->ParticleToComponentMap[particleIndex] = componentLabel;

  this->DataStructureImage->SetPixel( index, 0 );

  ImageType::IndexType neighborIndex;

  for ( int x=-searchRadius; x<=searchRadius; x++ )
    {
    neighborIndex[0] = index[0] + x;

    for ( int y=-searchRadius; y<=searchRadius; y++ )
      {
      neighborIndex[1] = index[1] + y;

      for ( int z=-searchRadius; z<=searchRadius; z++ )
        {
        neighborIndex[2] = index[2] + z;

        if ( this->DataStructureImage->GetBufferedRegion().IsInside( neighborIndex ) )
          {
          if ( this->DataStructureImage->GetPixel( neighborIndex ) != 0 )
            {
            unsigned int neighborParticleIndex = this->DataStructureImage->GetPixel( neighborIndex ) - 1;

            bool connected = this->EvaluateParticleConnectedness( particleIndex, neighborParticleIndex );

            if ( connected && (*currentComponentSize < this->MaximumComponentSize) )
              {
              this->QueryNeighborhood( neighborIndex, componentLabel, currentComponentSize );
              }
            }
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
  IteratorType it( this->DataStructureImage, this->DataStructureImage->GetBufferedRegion() );

  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
    if ( it.Get() != 0 )
      {
      unsigned int componentSize = 0;
      this->QueryNeighborhood( it.GetIndex(), componentLabel, &componentSize );
      componentLabel++;
      }
    
    ++it;
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
      array->SetNumberOfComponents( this->InternalInputPolyData->GetPointData()->GetArray(i)->GetNumberOfComponents() );
      array->SetName( this->InternalInputPolyData->GetPointData()->GetArray(i)->GetName() );

    pointDataArrayVec.push_back( array );
    }

  vtkFloatArray* unmergedComponentsArray = vtkFloatArray::New();
    unmergedComponentsArray->SetNumberOfComponents( 1 );
    unmergedComponentsArray->SetName( "unmergedComponents" );

  unsigned int inc = 0;
  for ( unsigned int i=0; i<this->NumberInternalInputParticles; i++ )
    {
    componentLabel = this->ParticleToComponentMap[i];

    if ( (this->SelectedComponent != 0 && this->ParticleToComponentMap[i] == this->SelectedComponent) ||
         (this->SelectedComponent == 0 && this->ComponentSizeMap[componentLabel] >=this->ComponentSizeThreshold) )
      {
      outputPoints->InsertNextPoint( this->InternalInputPolyData->GetPoint(i) );

      for ( unsigned int j=0; j<this->NumberOfPointDataArrays; j++ )
        {
        pointDataArrayVec[j]->InsertTuple( inc, this->InternalInputPolyData->GetPointData()->GetArray(j)->GetTuple(i) );
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
