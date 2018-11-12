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
}


double
cipLabelMapToLungLobeLabelMapImageFilter
::GetLeftObliqueFissureCompleteness()
{
  

  return 0;
}


double
cipLabelMapToLungLobeLabelMapImageFilter
::GetRightObliqueFissureCompleteness()
{
  return 0;
}


double
cipLabelMapToLungLobeLabelMapImageFilter
::GetRightHorizontalFissureCompleteness()
{
  return 0;
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
  double loAreaTmp, roAreaTmp, rhAreaTmp; // Temp TPS surface area values for the three fissures
  double loArea = 0.0;
  double roArea = 0.0;
  double lhArea = 0.0;  

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

              loAreaTmp = this->GetLocalSurfaceArea( this->LeftObliqueThinPlateSplineSurface,
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
			  loArea += loAreaTmp;
			  std::cout << loAreaTmp << std::endl;
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


double
cipLabelMapToLungLobeLabelMapImageFilter
::GetLocalSurfaceArea( cipThinPlateSplineSurface* tps, cipThinPlateSplineSurface* tpsFromPoints, 
			  BlendMapType::Pointer blendMap, unsigned int i, unsigned int j )
{
  InputImageType::SpacingType spacing = this->GetInput()->GetSpacing();
  InputImageType::PointType   origin  = this->GetInput()->GetOrigin();

  double zUL, zUR, zLL, zLR, zM;

  double xUL = double(i)*spacing[0] + origin[0];
  double yUL = double(j)*spacing[1] + origin[1];

  double xUR = double(i+1)*spacing[0] + origin[0];
  double yUR = double(j)*spacing[1] + origin[1];

  double xLL = double(i)*spacing[0] + origin[0];
  double yLL = double(j+1)*spacing[1] + origin[1];

  double xLR = double(i+1)*spacing[0] + origin[0];
  double yLR = double(j+1)*spacing[1] + origin[1];

  double xM = (xUL + xUR)/2.;
  double yM = (yUL + yLL)/2.;

  if ( tps->GetNumberSurfacePoints() > 0 &&
       tpsFromPoints->GetNumberSurfacePoints() == 0 )
    {
      zUL = tps->GetSurfaceHeight( xUL, yUL );
      zUR = tps->GetSurfaceHeight( xUR, yUR );
      zLL = tps->GetSurfaceHeight( xLL, yLL );
      zLR = tps->GetSurfaceHeight( xLR, yLR );
      zM  = tps->GetSurfaceHeight( xM, yM );      
    }
  else if ( tps->GetNumberSurfacePoints() == 0 &&
	    tpsFromPoints->GetNumberSurfacePoints() > 0 )
    {
      zUL = tpsFromPoints->GetSurfaceHeight( xUL, yUL );
      zUR = tpsFromPoints->GetSurfaceHeight( xUR, yUR );
      zLL = tpsFromPoints->GetSurfaceHeight( xLL, yLL );
      zLR = tpsFromPoints->GetSurfaceHeight( xLR, yLR );
      zM  = tpsFromPoints->GetSurfaceHeight( xM, yM );      
    }
  else
    {
      BlendMapType::IndexType index;
        index[0] = i;
	index[1] = j;

      double blendVal = this->BlendSlope*blendMap->GetPixel( index ) + this->BlendIntercept;
      if ( blendVal <= 0.0 )
	{
	  zUL = tpsFromPoints->GetSurfaceHeight( xUL, yUL );
	  zUR = tpsFromPoints->GetSurfaceHeight( xUR, yUR );
	  zLL = tpsFromPoints->GetSurfaceHeight( xLL, yLL );
	  zLR = tpsFromPoints->GetSurfaceHeight( xLR, yLR );
	  zM  = tpsFromPoints->GetSurfaceHeight( xM, yM );	  
	}
      else if ( blendVal >= 1.0 )
	{
	  zUL = tps->GetSurfaceHeight( xUL, yUL );
	  zUR = tps->GetSurfaceHeight( xUR, yUR );
	  zLL = tps->GetSurfaceHeight( xLL, yLL );
	  zLR = tps->GetSurfaceHeight( xLR, yLR );
	  zM  = tps->GetSurfaceHeight( xM, yM );	  
	}
      else
	{
	  zUL = blendVal*tps->GetSurfaceHeight( xUL, yUL ) + (1.0 - blendVal)*tpsFromPoints->GetSurfaceHeight( xUL, yUL );
	  zUR = blendVal*tps->GetSurfaceHeight( xUR, yUR ) + (1.0 - blendVal)*tpsFromPoints->GetSurfaceHeight( xUR, yUR );
	  zLL = blendVal*tps->GetSurfaceHeight( xLL, yLL ) + (1.0 - blendVal)*tpsFromPoints->GetSurfaceHeight( xLL, yLL );
	  zLR = blendVal*tps->GetSurfaceHeight( xLR, yLR ) + (1.0 - blendVal)*tpsFromPoints->GetSurfaceHeight( xLR, yLR );
	  zM = blendVal*tps->GetSurfaceHeight( xM, yM ) + (1.0 - blendVal)*tpsFromPoints->GetSurfaceHeight( xM, yM );	  
	}
    }

  // Now compute the area of each triangle. Use Heron's Formula:
  double sideA, sideB, sideC, S;
  double surfaceArea = 0.;
  
  // UL, LL, M
  sideA = sqrt(pow(xM - xLL, 2) + pow(yM - yLL, 2) + pow(zM - zLL, 2));
  sideB = sqrt(pow(xM - xUL, 2) + pow(yM - yUL, 2) + pow(zM - zUL, 2));
  sideC = sqrt(pow(xLL - xUL, 2) + pow(yLL - yUL, 2) + pow(zLL - zUL, 2));
  S = (sideA + sideB + sideC)/2.;
  surfaceArea += sqrt(S*(S - sideA)*(S - sideB)*(S - sideC));

  // UL, UR, M
  sideA = sqrt(pow(xM - xUR, 2) + pow(yM - yUR, 2) + pow(zM - zUR, 2));
  sideB = sqrt(pow(xM - xUL, 2) + pow(yM - yUL, 2) + pow(zM - zUL, 2));
  sideC = sqrt(pow(xUR - xUL, 2) + pow(yUR - yUL, 2) + pow(zUR - zUL, 2));
  S = (sideA + sideB + sideC)/2.;
  surfaceArea += sqrt(S*(S - sideA)*(S - sideB)*(S - sideC));

  // UR, LR, M
  sideA = sqrt(pow(xM - xUR, 2) + pow(yM - yUR, 2) + pow(zM - zUR, 2));
  sideB = sqrt(pow(xM - xLR, 2) + pow(yM - yLR, 2) + pow(zM - zLR, 2));
  sideC = sqrt(pow(xUR - xLR, 2) + pow(yUR - yLR, 2) + pow(zUR - zLR, 2));
  S = (sideA + sideB + sideC)/2.;
  surfaceArea += sqrt(S*(S - sideA)*(S - sideB)*(S - sideC));  

  // LL, LR, M
  sideA = sqrt(pow(xM - xLL, 2) + pow(yM - yLL, 2) + pow(zM - zLL, 2));
  sideB = sqrt(pow(xM - xLR, 2) + pow(yM - yLR, 2) + pow(zM - zLR, 2));
  sideC = sqrt(pow(xLL - xLR, 2) + pow(yLL - yLR, 2) + pow(zLL - zLR, 2));
  S = (sideA + sideB + sideC)/2.;
  surfaceArea += sqrt(S*(S - sideA)*(S - sideB)*(S - sideC));  


  return surfaceArea;
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
