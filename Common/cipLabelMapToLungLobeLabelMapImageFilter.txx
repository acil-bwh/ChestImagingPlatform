#ifndef _cipLabelMapToLungLobeLabelMapImageFilter_txx
#define _cipLabelMapToLungLobeLabelMapImageFilter_txx

// DEB
#include "itkImageFileWriter.h"

#include <limits>
#include "itkImageRegionIterator.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "cipLabelMapToLungLobeLabelMapImageFilter.h"
#include "cipMacro.h"
#include "cipChestConventions.h"

#include "vtkDelaunay3D.h"
#include "vtkDataSetSurfaceFilter.h"
#include "vtkAbstractArray.h"
#include "vtkIndent.h"
#include "vtkFieldData.h"

#include <math.h>

cipLabelMapToLungLobeLabelMapImageFilter
::cipLabelMapToLungLobeLabelMapImageFilter()
{
  this->m_ThinPlateSplineSurfaceFromPointsLambda = 0.1;

  this->LeftObliqueThinPlateSplineSurface     = new cipThinPlateSplineSurface;
  this->RightObliqueThinPlateSplineSurface    = new cipThinPlateSplineSurface;
  this->RightHorizontalThinPlateSplineSurface = new cipThinPlateSplineSurface;

  this->LeftObliqueThinPlateSplineSurfaceFromPoints     = new cipThinPlateSplineSurface;
  this->RightObliqueThinPlateSplineSurfaceFromPoints    = new cipThinPlateSplineSurface;
  this->RightHorizontalThinPlateSplineSurfaceFromPoints = new cipThinPlateSplineSurface;

  this->LeftObliqueBlendMap     = BlendMapType::New();
  this->RightObliqueBlendMap    = BlendMapType::New();
  this->RightHorizontalBlendMap = BlendMapType::New();

  this->BlendSlope     = 1.0/98.9;
  this->BlendIntercept = -1.0/49.0;

  this->LeftObliqueObbTree = vtkSmartPointer< vtkOBBTree >::New();
  this->RightHorizontalObbTree = vtkSmartPointer< vtkOBBTree >::New();
  this->RightObliqueObbTree = vtkSmartPointer< vtkOBBTree >::New();  
}


bool
cipLabelMapToLungLobeLabelMapImageFilter
::IsFissure(unsigned int i, unsigned int j, unsigned char chestRegion, unsigned char chestType)
{
  //cip::OBLIQUEFISSURE
  //cip::HORIZONTALFISSURE

  InputImageType::SpacingType spacing = this->GetInput()->GetSpacing();
  InputImageType::PointType   origin  = this->GetInput()->GetOrigin();
  InputImageType::SizeType    size    = this->GetInput()->GetBufferedRegion().GetSize();  

  double source[3];
    source[0] = double(i)*spacing[0] + origin[0];
    source[1] = double(j)*spacing[1] + origin[1];
    source[2] = origin[2];    
  
  double target[3];
    target[0] = double(i)*spacing[0] + origin[0];
    target[1] = double(j)*spacing[1] + origin[1];
    target[2] = double(size[2])*spacing[2] + origin[2];      

  vtkSmartPointer< vtkPoints > intersectPoints = vtkSmartPointer<vtkPoints>::New();

  bool intersects;
  if ( chestRegion == (unsigned char)(cip::LEFTLUNG) )
    {
      intersects = this->LeftObliqueObbTree->IntersectWithLine(source, target, intersectPoints, NULL);
    }
  else if ( chestRegion == (unsigned char)(cip::RIGHTLUNG) &&
    chestType == (unsigned char)(cip::OBLIQUEFISSURE) )
    {
      intersects = this->RightObliqueObbTree->IntersectWithLine(source, target, intersectPoints, NULL);
    }
  else
    {
      intersects = this->RightHorizontalObbTree->IntersectWithLine(source, target, intersectPoints, NULL);
    }    

  return intersects;
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetLeftObliqueFissureParticles( vtkPolyData* particles )
{
  double irad = 0;
  for ( unsigned int i=0; i<particles->GetFieldData()->GetNumberOfArrays(); i++ )
    {
      std::string name = particles->GetFieldData()->GetArray(i)->GetName();
      if ( name.compare( "irad" ) == 0 )
        {
	  irad = particles->GetFieldData()->GetArray(i)->GetTuple(0)[0];
        }
    }

  vtkSmartPointer< vtkDelaunay3D > delaunay3D = vtkSmartPointer< vtkDelaunay3D >::New();
    delaunay3D->SetInputData( particles );
    delaunay3D->SetAlpha( 2*irad );
    delaunay3D->Update();

  vtkSmartPointer< vtkDataSetSurfaceFilter > surfFilter = vtkSmartPointer< vtkDataSetSurfaceFilter >::New();
    surfFilter->SetInputConnection( delaunay3D->GetOutputPort() );
    surfFilter->Update();

  this->LeftObliqueObbTree->SetDataSet( surfFilter->GetOutput() );
  this->LeftObliqueObbTree->BuildLocator();
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetRightObliqueFissureParticles( vtkPolyData* particles )
{
  vtkSmartPointer< vtkDelaunay3D > delaunay3D = vtkSmartPointer< vtkDelaunay3D >::New();
    delaunay3D->SetInputData( particles );
    delaunay3D->SetAlpha( particles->GetFieldData()->GetArray("irad")->GetTuple(0)[0] );
    delaunay3D->Update();

  vtkSmartPointer< vtkDataSetSurfaceFilter > surfFilter = vtkSmartPointer< vtkDataSetSurfaceFilter >::New();
    surfFilter->SetInputConnection( delaunay3D->GetOutputPort() );
    surfFilter->Update();

  this->RightObliqueObbTree->SetDataSet( surfFilter->GetOutput() );
  this->RightObliqueObbTree->BuildLocator();
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetRightHorizontalFissureParticles( vtkPolyData* particles )
{
  vtkSmartPointer< vtkDelaunay3D > delaunay3D = vtkSmartPointer< vtkDelaunay3D >::New();
    delaunay3D->SetInputData( particles );
    delaunay3D->SetAlpha( particles->GetFieldData()->GetArray("irad")->GetTuple(0)[0] );
    delaunay3D->Update();

  vtkSmartPointer< vtkDataSetSurfaceFilter > surfFilter = vtkSmartPointer< vtkDataSetSurfaceFilter >::New();
    surfFilter->SetInputConnection( delaunay3D->GetOutputPort() );
    surfFilter->Update();

  this->RightHorizontalObbTree->SetDataSet( surfFilter->GetOutput() );
  this->RightHorizontalObbTree->BuildLocator();
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::GenerateData()
{
  bool segmentLeftLobes  = false;
  bool segmentRightLobes = false;
  if ( this->LeftObliqueThinPlateSplineSurface->GetNumberSurfacePoints() > 0 || 
       this->LeftObliqueThinPlateSplineSurfaceFromPoints->GetNumberSurfacePoints() > 0 )
    {
      segmentLeftLobes = true;
    }
  if ( (this->RightObliqueThinPlateSplineSurface->GetNumberSurfacePoints() > 0 || 
	this->RightObliqueThinPlateSplineSurfaceFromPoints->GetNumberSurfacePoints() > 0) &&
       (this->RightHorizontalThinPlateSplineSurface->GetNumberSurfacePoints() > 0 || 
	this->RightHorizontalThinPlateSplineSurfaceFromPoints->GetNumberSurfacePoints() > 0) )
    {
      segmentRightLobes = true;
    }

  // Allocate the output buffer
  this->GetOutput()->SetBufferedRegion( this->GetOutput()->GetRequestedRegion() );
  this->GetOutput()->Allocate();
  this->GetOutput()->FillBuffer( 0 );

  cip::ChestConventions conventions;

  InputImageType::SpacingType spacing = this->GetInput()->GetSpacing();
  InputImageType::PointType   origin  = this->GetInput()->GetOrigin();
  InputImageType::SizeType    size    = this->GetInput()->GetBufferedRegion().GetSize();  
  InputImageType::PointType   point;
  InputImageType::IndexType   index;

  // Allocate the blend maps and update them if necessary. 
  BlendMapType::SpacingType blendMapSpacing;
    blendMapSpacing[0] = spacing[0];
    blendMapSpacing[1] = spacing[1];

  BlendMapType::PointType blendMapOrigin;
    blendMapOrigin[0] = origin[0];
    blendMapOrigin[1] = origin[1];

  BlendMapType::SizeType blendMapSize;
    blendMapSize[0] = size[0];
    blendMapSize[1] = size[1];

  this->LeftObliqueBlendMap->SetRegions( blendMapSize );
  this->LeftObliqueBlendMap->Allocate();
  this->LeftObliqueBlendMap->FillBuffer( 0.0 );
  this->LeftObliqueBlendMap->SetSpacing( blendMapSpacing );
  this->LeftObliqueBlendMap->SetOrigin( blendMapOrigin );
  if ( this->LeftObliqueThinPlateSplineSurface->GetNumberSurfacePoints() > 0 &&
       this->LeftObliqueThinPlateSplineSurfaceFromPoints->GetNumberSurfacePoints() > 0 )
    {
      this->UpdateBlendMap( this->LeftObliqueThinPlateSplineSurfaceFromPoints, this->LeftObliqueBlendMap );
    }

  this->RightObliqueBlendMap->SetRegions( blendMapSize );
  this->RightObliqueBlendMap->Allocate();
  this->RightObliqueBlendMap->FillBuffer( 0.0 );
  this->RightObliqueBlendMap->SetSpacing( blendMapSpacing );
  this->RightObliqueBlendMap->SetOrigin( blendMapOrigin );
  if ( this->RightObliqueThinPlateSplineSurface->GetNumberSurfacePoints() > 0 &&
       this->RightObliqueThinPlateSplineSurfaceFromPoints->GetNumberSurfacePoints() > 0 )
    {
      this->UpdateBlendMap( this->RightObliqueThinPlateSplineSurfaceFromPoints, this->RightObliqueBlendMap );
    }

  this->RightHorizontalBlendMap->SetRegions( blendMapSize );
  this->RightHorizontalBlendMap->Allocate();
  this->RightHorizontalBlendMap->FillBuffer( 0.0 );
  this->RightHorizontalBlendMap->SetSpacing( blendMapSpacing );
  this->RightHorizontalBlendMap->SetOrigin( blendMapOrigin );
  if ( this->RightHorizontalThinPlateSplineSurface->GetNumberSurfacePoints() > 0 &&
       this->RightHorizontalThinPlateSplineSurfaceFromPoints->GetNumberSurfacePoints() > 0 )
    {
      this->UpdateBlendMap( this->RightHorizontalThinPlateSplineSurfaceFromPoints, this->RightHorizontalBlendMap );
    }

  int loZ, roZ, rhZ;  // The z index values for each of the fissures
  unsigned short newValue;
  unsigned char cipRegion, cipType;

  for ( int i=0; i < int( size[0] ); i++ )
    {
      for ( int j=0; j < int( size[1] ); j++ )
	{
	  if ( segmentLeftLobes )
	    {
	      loZ = this->GetBoundaryHeightIndex( this->LeftObliqueThinPlateSplineSurface,
						  this->LeftObliqueThinPlateSplineSurfaceFromPoints,
						  this->LeftObliqueBlendMap, i, j );
	    }

	  if ( segmentRightLobes )
	    {
	      roZ = this->GetBoundaryHeightIndex( this->RightObliqueThinPlateSplineSurface,
						  this->RightObliqueThinPlateSplineSurfaceFromPoints,
						  this->RightObliqueBlendMap, i, j );

	      rhZ = this->GetBoundaryHeightIndex( this->RightHorizontalThinPlateSplineSurface,
						  this->RightHorizontalThinPlateSplineSurfaceFromPoints,
						  this->RightHorizontalBlendMap, i, j );
	    }

	  for ( int z=0; z < int( size[2] ); z++ )
	    {
	      index[0] = i;
	      index[1] = j;	     
	      index[2] = z;
	      
	      this->GetOutput()->SetPixel( index, this->GetInput()->GetPixel( index ) );
	      
	      if ( this->GetInput()->GetPixel( index ) != 0 )
		{
		  cipRegion = conventions.GetChestRegionFromValue( this->GetInput()->GetPixel( index ) );
		  cipType   = conventions.GetChestTypeFromValue( this->GetInput()->GetPixel( index ) );
		  
		  if ( segmentLeftLobes && 
		       conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, (unsigned char)( cip::LEFTLUNG ) ) )
		    {
		      if ( z == loZ )
		        {
			  if ( this->IsFissure( i, j, (unsigned char)(cip::LEFTLUNG), (unsigned char)(cip::OBLIQUEFISSURE) ) )
			    {
			      cipType = (unsigned char)( cip::OBLIQUEFISSURE );
			    }
			}
			
		      if ( z < loZ )
			{
			  cipRegion = (unsigned char)( cip::LEFTINFERIORLOBE );
			}
		      else
			{
			  cipRegion = (unsigned char)( cip::LEFTSUPERIORLOBE );
			}
		      
		      newValue = conventions.GetValueFromChestRegionAndType( cipRegion, cipType );
		      
		      this->GetOutput()->SetPixel( index, newValue );
		    }
		  else if ( segmentRightLobes && 
			    conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, (unsigned char)( cip::RIGHTLUNG ) ) )
		    {
		      if ( z == rhZ && z > roZ )
		        {
			  if ( this->IsFissure( i, j, (unsigned char)(cip::RIGHTLUNG), (unsigned char)(cip::HORIZONTALFISSURE) ) )
			    {
                              cipType = (unsigned char)( cip::HORIZONTALFISSURE );
			    }
			}
		      if ( z == roZ )
		        {
			  if ( this->IsFissure( i, j, (unsigned char)(cip::RIGHTLUNG), (unsigned char)(cip::OBLIQUEFISSURE) ) )
			    {
                              cipType = (unsigned char)( cip::OBLIQUEFISSURE );
			    }
			}                                           

		      if ( z <= roZ )
			{
			  cipRegion = (unsigned char)( cip::RIGHTINFERIORLOBE );
			}
		      else if ( z > roZ && z <= rhZ )
			{
			  cipRegion = (unsigned char)( cip::RIGHTMIDDLELOBE );
			}
		      else
			{
			  cipRegion = (unsigned char)( cip::RIGHTSUPERIORLOBE );
			}
		      
		      newValue = conventions.GetValueFromChestRegionAndType( cipRegion, cipType );		      
		      this->GetOutput()->SetPixel( index, newValue );
		    }
		}
	    }
	}
    }
}

void 
cipLabelMapToLungLobeLabelMapImageFilter
::UpdateBlendMap( cipThinPlateSplineSurface* tps, BlendMapType::Pointer blendMap )
{
  typedef itk::Image< float, 2 >                                                     DistanceImageType;
  typedef itk::SignedMaurerDistanceMapImageFilter< BlendMapType, DistanceImageType > DistanceMapType;
  typedef itk::ImageRegionIterator< BlendMapType >                                   BlendMapIteratorType;
  typedef itk::ImageRegionIterator< DistanceImageType >                              DistanceMapIteratorType;

  // First create foreground points on the blend map
  bool isInside;
  for ( unsigned int i=0; i<tps->GetSurfacePoints().size(); i++ )
    {
      BlendMapType::PointType point;
      BlendMapType::IndexType index;

      point[0] = tps->GetSurfacePoints()[i][0];
      point[1] = tps->GetSurfacePoints()[i][1];

      isInside = blendMap->TransformPhysicalPointToIndex( point, index );
      cipAssert( isInside );
      blendMap->SetPixel( index, 1 );
    }

  DistanceMapType::Pointer distanceMap = DistanceMapType::New();
    distanceMap->SetInput( blendMap );
    distanceMap->SetSquaredDistance( false );
    distanceMap->SetUseImageSpacing( true );
    distanceMap->SetInsideIsPositive( true );
  try
    {
    distanceMap->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught generating distance map:";
    std::cerr << excp << std::endl;
    }

  DistanceMapIteratorType dIt( distanceMap->GetOutput(), distanceMap->GetOutput()->GetBufferedRegion() );
  BlendMapIteratorType bIt( blendMap, blendMap->GetBufferedRegion() );

  dIt.GoToBegin();
  bIt.GoToBegin();
  while ( !bIt.IsAtEnd() )
    {
      cipAssert( -dIt.Get() >= 0.0 );

      bIt.Set( -dIt.Get()  );

      ++dIt;
      ++bIt;
    }

  /* // DEB */
  /* typedef itk::ImageFileWriter< DistanceImageType > WriterType; */
  /* std::cout << "---Writing blend map..." << std::endl; */
  /* WriterType::Pointer writer = WriterType::New(); */
  /* writer->SetFileName( "/Users/jross/tmp/fooBlend.nhdr" ); */
  /* writer->SetInput( blendMap ); */
  /* writer->Update(); */
}

void
cipLabelMapToLungLobeLabelMapImageFilter
::SetLeftObliqueThinPlateSplineSurface( cipThinPlateSplineSurface* tps )
{
  this->LeftObliqueThinPlateSplineSurface = tps;
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetRightObliqueThinPlateSplineSurface( cipThinPlateSplineSurface* tps )
{
  this->RightObliqueThinPlateSplineSurface = tps;
}


int
cipLabelMapToLungLobeLabelMapImageFilter
::GetBoundaryHeightIndex( cipThinPlateSplineSurface* tps, cipThinPlateSplineSurface* tpsFromPoints, 
		       BlendMapType::Pointer blendMap, unsigned int i, unsigned int j )
{
  InputImageType::SpacingType spacing = this->GetInput()->GetSpacing();
  InputImageType::PointType   origin  = this->GetInput()->GetOrigin();

  double x = double(i)*spacing[0] + origin[0];
  double y = double(j)*spacing[1] + origin[1];
  double z;

  if ( tps->GetNumberSurfacePoints() > 0 &&
       tpsFromPoints->GetNumberSurfacePoints() == 0 )
    {
      z = tps->GetSurfaceHeight( x, y );
    }
  else if ( tps->GetNumberSurfacePoints() == 0 &&
	    tpsFromPoints->GetNumberSurfacePoints() > 0 )
    {
      z = tpsFromPoints->GetSurfaceHeight( x, y );
    }
  else
    {
      BlendMapType::IndexType index;
        index[0] = i;
	index[1] = j;

      double blendVal = this->BlendSlope*blendMap->GetPixel( index ) + this->BlendIntercept;
      if ( blendVal <= 0.0 )
	{
	  z = tpsFromPoints->GetSurfaceHeight( x, y );
	}
      else if ( blendVal >= 1.0 )
	{
	  z = tps->GetSurfaceHeight( x, y );
	}
      else
	{
	  z = blendVal*tps->GetSurfaceHeight( x, y ) + (1.0 - blendVal)*tpsFromPoints->GetSurfaceHeight( x, y );
	}
    }

  return int( (z - origin[2])/spacing[2] );
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetRightHorizontalThinPlateSplineSurface( cipThinPlateSplineSurface* tps )
{
  this->RightHorizontalThinPlateSplineSurface = tps;
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetLeftObliqueFissureIndices( std::vector< InputImageType::IndexType > indices )
{
  Superclass::InputImagePointer inputPtr = const_cast< InputImageType* >( this->GetInput() );

  for ( unsigned int i=0; i<indices.size(); i++ )
    {
    this->LeftObliqueFissureIndices.push_back( indices[i] );
    }
  if ( inputPtr.IsNotNull() )
    {
    InputImageType::PointType physicalPoint;

    for ( unsigned int i=0; i<indices.size(); i++ )
      {
      this->GetInput()->TransformIndexToPhysicalPoint( indices[i], physicalPoint );
    
      cip::PointType point(3);
        point[0] = physicalPoint[0];
        point[1] = physicalPoint[1];
        point[2] = physicalPoint[2];

      this->LeftObliqueFissurePoints.push_back( point );
      }  

    this->LeftObliqueThinPlateSplineSurfaceFromPoints->SetSurfacePoints( this->LeftObliqueFissurePoints );
    }
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetLeftObliqueFissurePoints( const std::vector< cip::PointType >& points )
{
  for ( unsigned int i=0; i<points.size(); i++ )
    {
    this->LeftObliqueFissurePoints.push_back( points[i] );
    }
  this->LeftObliqueThinPlateSplineSurfaceFromPoints->SetSurfacePoints( this->LeftObliqueFissurePoints );
  this->LeftObliqueThinPlateSplineSurfaceFromPoints->SetLambda( this->m_ThinPlateSplineSurfaceFromPointsLambda );
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetRightObliqueFissureIndices( std::vector< InputImageType::IndexType > indices )
{
  Superclass::InputImagePointer inputPtr = const_cast< InputImageType* >( this->GetInput() );

  for ( unsigned int i=0; i<indices.size(); i++ )
    {
    this->RightObliqueFissureIndices.push_back( indices[i] );
    }
  if ( inputPtr.IsNotNull() )
    {
    InputImageType::PointType physicalPoint;

    for ( unsigned int i=0; i<indices.size(); i++ )
      {
      this->GetInput()->TransformIndexToPhysicalPoint( indices[i], physicalPoint );
    
      cip::PointType point(3);
        point[0] = physicalPoint[0];
        point[1] = physicalPoint[1];
        point[2] = physicalPoint[2];

      this->RightObliqueFissurePoints.push_back( point );
      }  

    this->RightObliqueThinPlateSplineSurfaceFromPoints->SetSurfacePoints( this->RightObliqueFissurePoints );
    this->RightObliqueThinPlateSplineSurfaceFromPoints->SetLambda( this->m_ThinPlateSplineSurfaceFromPointsLambda );
    }
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetRightObliqueFissurePoints( const std::vector< cip::PointType >& points )
{
  for ( unsigned int i=0; i<points.size(); i++ )
    {
    this->RightObliqueFissurePoints.push_back( points[i] );
    }

  this->RightObliqueThinPlateSplineSurfaceFromPoints->SetSurfacePoints( this->RightObliqueFissurePoints );
  this->RightObliqueThinPlateSplineSurfaceFromPoints->SetLambda( this->m_ThinPlateSplineSurfaceFromPointsLambda );
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetRightHorizontalFissureIndices( std::vector< InputImageType::IndexType > indices )
{
  Superclass::InputImagePointer inputPtr = const_cast< InputImageType* >( this->GetInput() );

  for ( unsigned int i=0; i<indices.size(); i++ )
    {
    this->RightHorizontalFissureIndices.push_back( indices[i] );
    }
  if ( inputPtr.IsNotNull() )
    {
    InputImageType::PointType physicalPoint;

    for ( unsigned int i=0; i<indices.size(); i++ )
      {
      this->GetInput()->TransformIndexToPhysicalPoint( indices[i], physicalPoint );
    
      cip::PointType point(3);
        point[0] = physicalPoint[0];
        point[1] = physicalPoint[1];
        point[2] = physicalPoint[2];

      this->RightHorizontalFissurePoints.push_back( point );
      }  

    this->RightHorizontalThinPlateSplineSurfaceFromPoints->SetSurfacePoints( this->RightHorizontalFissurePoints );
    this->RightHorizontalThinPlateSplineSurfaceFromPoints->SetLambda( this->m_ThinPlateSplineSurfaceFromPointsLambda );
    }
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetRightHorizontalFissurePoints( const std::vector< cip::PointType >& points )
{
  for ( unsigned int i=0; i<points.size(); i++ )
    {
    this->RightHorizontalFissurePoints.push_back( points[i] );
    }

  this->RightHorizontalThinPlateSplineSurfaceFromPoints->SetSurfacePoints( this->RightHorizontalFissurePoints );
  this->RightHorizontalThinPlateSplineSurfaceFromPoints->SetLambda( this->m_ThinPlateSplineSurfaceFromPointsLambda );
}


/**
 * Standard "PrintSelf" method
 */
void
cipLabelMapToLungLobeLabelMapImageFilter
::PrintSelf(
  std::ostream& os, 
  itk::Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}

#endif
