#include "cipChestDataViewer.h"
#include "vtkTransformPolyDataFilter.h"
#include "vtkTransform.h"
#include "vtkDataSetMapper.h"
#include "vtkTriangleStrip.h"
#include "itkImageFileWriter.h"
#include "vtkDiscreteMarchingCubes.h"
#include "vtkWindowedSincPolyDataFilter.h"
#include "vtkImageImport.h"
#include "vtkImageData.h"
#include "vtkDecimatePro.h"
#include "vtkProperty.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "vtkPoints.h"
#include "vtkPointData.h"
#include "vtkGlyph3D.h"
#include "vtkCellArray.h"
#include "vtkCylinderSource.h"
#include "vtkSuperquadricTensorGlyphFilter.h"
#include "vtkFloatArray.h"
#include "vtkUnstructuredGrid.h"
#include "vtkPolyDataNormals.h"
#include "vtkDataSetReader.h"
#include "vtkDoubleArray.h"
#include "vtkSphereSource.h"
#include "vtkTensorGlyph.h"
#include "vtkVectorText.h"
#include "vtkFollower.h"
#include "vtkLinearExtrusionFilter.h"
#include "vtkGlyph3DWithScaling.h"

cipChestDataViewer::cipChestDataViewer()
{
  this->Conventions = new cip::ChestConventions();

  this->RenderWindow                 = vtkRenderWindow::New();
  this->Renderer                     = vtkRenderer::New();
  this->RenderWindowInteractor       = vtkRenderWindowInteractor::New();
  this->TrackballCameraStyle         = vtkInteractorStyleTrackballCamera::New();

  this->GrayscaleImage    = GrayscaleImageType::New();
  this->vtkGrayscaleImage = vtkImageData::New();

  this->PlaneWidgetX = vtkImagePlaneWidget::New();
  this->PlaneWidgetY = vtkImagePlaneWidget::New();
  this->PlaneWidgetZ = vtkImagePlaneWidget::New();

  this->Renderer->SetBackground( 1, 1, 1 );

  this->RenderWindowInteractor->SetRenderWindow( this->RenderWindow );
  this->RenderWindowInteractor->SetInteractorStyle( this->TrackballCameraStyle );

  this->RenderWindow->AddRenderer( this->Renderer );
  this->RenderWindow->SetSize( 1400, 1400 );

  this->ParticleGlyphThetaResolution = 12;
  this->ParticleGlyphPhiResolution   = 12;

  this->LeftObliqueThinPlateSplineSurface      = new cipThinPlateSplineSurface;
  this->RightObliqueThinPlateSplineSurface     = new cipThinPlateSplineSurface;
  this->RightHorizontalThinPlateSplineSurface  = new cipThinPlateSplineSurface;

  this->ViewerCallbackCommand = vtkCallbackCommand::New();
  this->ViewerCallbackCommand->SetCallback( ViewerKeyCallback );
  this->ViewerCallbackCommand->SetClientData( (void*)this );

  this->RenderWindowInteractor->AddObserver( vtkCommand::KeyPressEvent, this->ViewerCallbackCommand );

  this->DisplayActorNames = false;

  this->ActorsVisible = true;
}

void cipChestDataViewer::SetBackgroundColor( double r, double g, double b )
{
  this->Renderer->SetBackground( r, g, b );
}

void cipChestDataViewer::SetPlaneWidgetXShowing( bool showing )
{
  this->PlaneWidgetXShowing = showing;

  if ( showing )
    {
    this->PlaneWidgetX->On();
    }
  else
    {
    this->PlaneWidgetX->Off();
    }
}

void cipChestDataViewer::SetPlaneWidgetYShowing( bool showing )
{
  this->PlaneWidgetYShowing = showing;

  if ( showing )
    {
    this->PlaneWidgetY->On();
    }
  else
    {
    this->PlaneWidgetY->Off();
    }
}

void cipChestDataViewer::SetPlaneWidgetZShowing( bool showing )
{
  this->PlaneWidgetZShowing = showing;

  if ( showing )
    {
    this->PlaneWidgetZ->On();
    }
  else
    {
    this->PlaneWidgetZ->Off();
    }
}

void cipChestDataViewer::SetLabelMapImage( LabelMapImageType::Pointer labelMapImage )
{
  this->LabelMapImage = LabelMapImageType::New();
  this->LabelMapImage = labelMapImage;
}

void cipChestDataViewer::SetGrayscaleImage( GrayscaleImageType::Pointer grayscaleImage )
{
  this->GrayscaleImage = grayscaleImage;

  //
  // Get the size of the image so that we can set initial slice values
  // for the plane widgets
  //
  GrayscaleImageType::SizeType size = this->GrayscaleImage->GetBufferedRegion().GetSize();

  unsigned int xSlice = size[0]/2;
  unsigned int ySlice = size[1]/2;
  unsigned int zSlice = size[2]/2;

  //
  // These are reasonable window / level settings for viewing lung CT
  // images. Used by the plane widgets
  //
  short window =  500;
  short level  = -800;

  //
  // Export the grayscale image to VTK land so that we can set up the
  // plane widgets
  //
  vtkImageImport* refImporter = vtkImageImport::New();

  ExportType::Pointer refExporter = ExportType::New();
    refExporter->SetInput( this->GrayscaleImage );

  this->ConnectPipelines( refExporter, refImporter );
  refImporter->Update();

  this->vtkGrayscaleImage->DeepCopy( refImporter->GetOutput() );

  //
  // Set up the X plane widget
  //
  this->PlaneWidgetX->SetInteractor( this->RenderWindowInteractor );
  this->PlaneWidgetX->RestrictPlaneToVolumeOn();
  this->PlaneWidgetX->DisplayTextOff();
  this->PlaneWidgetX->SetInputData( this->vtkGrayscaleImage );
  this->PlaneWidgetX->SetWindowLevel( window, level );
  this->PlaneWidgetX->SetPlaneOrientationToXAxes();
  this->PlaneWidgetX->SetSliceIndex( xSlice );
  this->PlaneWidgetX->SetKeyPressActivationValue( 'x' );
  this->PlaneWidgetX->On();
  this->PlaneWidgetX->InteractionOn();

  this->PlaneWidgetXShowing = false;

  //
  // Set up the Y plane widget
  //
  this->PlaneWidgetY->SetInteractor( this->RenderWindowInteractor );
  this->PlaneWidgetY->RestrictPlaneToVolumeOn();
  this->PlaneWidgetY->DisplayTextOff();
  this->PlaneWidgetY->SetInputData( this->vtkGrayscaleImage );
  this->PlaneWidgetY->SetWindowLevel( window, level );
  this->PlaneWidgetY->SetPlaneOrientationToYAxes();
  this->PlaneWidgetY->SetSliceIndex( ySlice );
  this->PlaneWidgetY->SetKeyPressActivationValue( 'y' );
  this->PlaneWidgetY->On();
  this->PlaneWidgetY->InteractionOn();

  this->PlaneWidgetYShowing = false;

  //
  // Set up the Z plane widget
  //
  this->PlaneWidgetZ->SetInteractor( this->RenderWindowInteractor );
  this->PlaneWidgetZ->RestrictPlaneToVolumeOn();
  this->PlaneWidgetZ->DisplayTextOff();
  this->PlaneWidgetZ->SetInputData( this->vtkGrayscaleImage );
  this->PlaneWidgetZ->SetWindowLevel( window, level );
  this->PlaneWidgetZ->SetPlaneOrientationToZAxes();
  this->PlaneWidgetZ->SetSliceIndex( zSlice );
  this->PlaneWidgetZ->SetKeyPressActivationValue( 'z' );
  this->PlaneWidgetZ->On();
  this->PlaneWidgetZ->InteractionOn();

  this->PlaneWidgetZShowing = true;
}

vtkSmartPointer< vtkActor > cipChestDataViewer::SetPolyData( vtkPolyData* polyData, std::string name )
{
  vtkPolyDataMapper* mapper = vtkPolyDataMapper::New();
    mapper->SetInputData( polyData );

  vtkSmartPointer< vtkActor > actor = vtkSmartPointer< vtkActor >::New();
    actor->SetMapper( mapper );
    actor->GetProperty()->SetColor( 0, 0, 0 );
    actor->GetProperty()->SetOpacity( 1.0 );

  this->ActorMap[name] = actor;
  this->Renderer->AddActor( this->ActorMap[name] );

  mapper->Delete();

  return actor;
}

void cipChestDataViewer::SetActorColor( std::string name, double r, double g, double b )
{
  this->ActorMap[name]->GetProperty()->SetColor( r, g, b );
}

void cipChestDataViewer::GetActorColor( std::string name, double color[3] )
{
  this->ActorMap[name]->GetProperty()->GetColor( color );
}

void cipChestDataViewer::SetActorOpacity( std::string name, double opacity )
{
  this->ActorMap[name]->GetProperty()->SetOpacity( opacity );
}

void cipChestDataViewer::ToggleActorVisibility()
{
  std::map< std::string, vtkActor* >::iterator it = this->ActorMap.begin();

  while ( it != this->ActorMap.end() )
    {
    if ( this->ActorsVisible )
      {
      this->Renderer->RemoveActor( (*it).second );
      }
    else
      {
      this->Renderer->AddActor( (*it).second );
      }

    ++it;
    }

  this->ActorsVisible = !this->ActorsVisible;

  this->RenderWindow->Render();
}

void cipChestDataViewer::SetLeftObliqueThinPlateSplineSurface( cipThinPlateSplineSurface* tps, std::string name )
{
  this->LeftObliqueThinPlateSplineSurface = tps;

  this->GenerateFissureActor( this->LeftObliqueThinPlateSplineSurface, (unsigned char)( cip::OBLIQUEFISSURE ),
                              (unsigned char)( cip::LEFTLUNG ), name );
}

void cipChestDataViewer::SetRightObliqueThinPlateSplineSurface( cipThinPlateSplineSurface* tps, std::string name )
{
  this->RightObliqueThinPlateSplineSurface = tps;

  this->GenerateFissureActor( this->RightObliqueThinPlateSplineSurface, (unsigned char)( cip::OBLIQUEFISSURE ),
                              (unsigned char)( cip::RIGHTLUNG ), name );
}

void cipChestDataViewer::SetRightHorizontalThinPlateSplineSurface( cipThinPlateSplineSurface* tps, std::string name )
{
  this->RightHorizontalThinPlateSplineSurface = tps;

  this->GenerateFissureActor( this->RightHorizontalThinPlateSplineSurface, (unsigned char)( cip::HORIZONTALFISSURE ),
                              (unsigned char)( cip::RIGHTLUNG ), name );
}

void cipChestDataViewer::SetLeftObliqueFissurePoints( const std::vector< cip::PointType >& pointsVec, std::string name )
{
  for ( unsigned int i=0; i<pointsVec.size(); i++ )
    {
    this->LeftObliqueFissurePoints.push_back( pointsVec[i] );
    }

  this->LeftObliqueThinPlateSplineSurface->SetSurfacePoints( pointsVec );
  this->GenerateFissureActor( this->LeftObliqueThinPlateSplineSurface, (unsigned char)( cip::OBLIQUEFISSURE ),
                              (unsigned char)( cip::LEFTLUNG ), name );
}

void cipChestDataViewer::SetRightObliqueFissurePoints( const std::vector< cip::PointType >& pointsVec, std::string name )
{
  for ( unsigned int i=0; i<pointsVec.size(); i++ )
    {
    this->RightObliqueFissurePoints.push_back( pointsVec[i] );
    }

  this->RightObliqueThinPlateSplineSurface->SetSurfacePoints( pointsVec );
  this->GenerateFissureActor( this->RightObliqueThinPlateSplineSurface, (unsigned char)( cip::OBLIQUEFISSURE ),
                              (unsigned char)( cip::RIGHTLUNG ), name );
}

void cipChestDataViewer::SetRightHorizontalFissurePoints( const std::vector< cip::PointType >& pointsVec, std::string name )
{
  for ( unsigned int i=0; i<pointsVec.size(); i++ )
    {
    this->RightHorizontalFissurePoints.push_back( pointsVec[i] );
    }

  this->RightHorizontalThinPlateSplineSurface->SetSurfacePoints( pointsVec );
  this->GenerateFissureActor( this->RightHorizontalThinPlateSplineSurface, (unsigned char)( cip::HORIZONTALFISSURE ),
                              (unsigned char)( cip::RIGHTLUNG ), name );
}

void cipChestDataViewer::Render()
{
  this->RenderWindow->Render();
  this->RenderWindowInteractor->Initialize();
  this->RenderWindowInteractor->Start();
}

bool cipChestDataViewer::Exists( std::string name )
{
  std::map< std::string, vtkActor* >::iterator it;
  for ( it = this->ActorMap.begin(); it != this->ActorMap.end(); it++ )
    {
    if (  name.compare( (*it).first ) == 0 )
      {
      return true;
      }
    }

  return false;
}

void cipChestDataViewer::SetAirwayParticles( vtkPolyData* polyData, double scaleFactor, std::string actorName )
{
  this->SetParticles( polyData, scaleFactor, actorName, false );
}

void cipChestDataViewer::SetVesselParticles( vtkPolyData* polyData, double scaleFactor, std::string actorName )
{
  this->SetParticles( polyData, scaleFactor, actorName, false );
}

void cipChestDataViewer::SetFissureParticles( vtkPolyData* polyData, double scaleFactor, std::string actorName )
{
  this->SetParticles( polyData, scaleFactor, actorName, true );
}

void cipChestDataViewer::SetPointsAsSpheres( vtkPolyData* polyData, double radius, std::string actorName )
{
  vtkSphereSource* sphereSource = vtkSphereSource::New();
    sphereSource->SetRadius( radius );
    sphereSource->SetCenter( 0, 0, 0 );

  vtkGlyph3D* glyph = vtkGlyph3D::New();
    glyph->SetInputData( polyData );
    glyph->SetSourceData( sphereSource->GetOutput() );
//     glyph->SetScaleModeToScaleByScalar();
//     glyph->SetScaleFactor( scaleFactor );
    glyph->Update();

  vtkPolyDataMapper* mapper = vtkPolyDataMapper::New();
    mapper->SetInputData( glyph->GetOutput() );

  vtkActor* actor = vtkActor::New();
    actor->SetMapper( mapper );

  this->ActorMap[actorName] = actor;
  this->Renderer->AddActor( this->ActorMap[actorName] );
}

vtkSmartPointer< vtkActor > cipChestDataViewer::SetAirwayParticlesAsCylinders( vtkPolyData* polyData, double scaleFactor, std::string actorName )
{
  return this->SetParticlesAsCylinders( polyData, scaleFactor, actorName, (unsigned char)( cip::AIRWAY ), false );
}

vtkSmartPointer< vtkActor > cipChestDataViewer::SetAirwayParticlesAsDiscs( vtkPolyData* polyData, double scaleFactor, std::string actorName )
{
  return this->SetParticlesAsDiscs( polyData, scaleFactor, actorName, (unsigned char)( cip::AIRWAY ), true );
}

vtkSmartPointer< vtkActor > cipChestDataViewer::SetVesselParticlesAsDiscs( vtkPolyData* polyData, double scaleFactor, std::string actorName )
{
  return this->SetParticlesAsDiscs( polyData, scaleFactor, actorName, (unsigned char)( cip::VESSEL ), true );
}

vtkSmartPointer< vtkActor > cipChestDataViewer::SetFissureParticlesAsDiscs( vtkPolyData* polyData, double scaleFactor, std::string actorName )
{
  return this->SetParticlesAsDiscs( polyData, scaleFactor, actorName, (unsigned char)( cip::FISSURE ), true );
}

vtkSmartPointer< vtkActor > cipChestDataViewer::SetParticlesAsDiscs( vtkPolyData* polyData, double scaleFactor, std::string actorName,
								     unsigned char particlesType, bool scaleGlyphsByParticlesScale )
{
  polyData->GetPointData()->SetScalars( polyData->GetPointData()->GetArray( "scale" ) );

  if ( particlesType == (unsigned char)( cip::AIRWAY ) )
    {
    polyData->GetPointData()->SetNormals( polyData->GetPointData()->GetArray( "hevec2" ) );
    }
  if ( particlesType == (unsigned char)( cip::VESSEL ) )
    {
    polyData->GetPointData()->SetNormals( polyData->GetPointData()->GetArray( "hevec0" ) );
    }
  if ( particlesType == (unsigned char)( cip::FISSURE ) )
    {
    polyData->GetPointData()->SetNormals( polyData->GetPointData()->GetArray( "hevec2" ) );
    }

  vtkCylinderSource* cylinderSource = vtkCylinderSource::New();
    cylinderSource->SetHeight( 0.4 ); //25
    cylinderSource->SetRadius( 1.0 );
    cylinderSource->SetCenter( 0, 0, 0 );
    cylinderSource->SetResolution( 20 );
    cylinderSource->CappingOn();

  vtkTransform* cylinderRotator = vtkTransform::New();
    cylinderRotator->RotateZ( 90 );

  vtkTransformPolyDataFilter* polyFilter = vtkTransformPolyDataFilter::New();
    polyFilter->SetInputConnection( cylinderSource->GetOutputPort() );
    polyFilter->SetTransform( cylinderRotator );
    polyFilter->Update();

  vtkGlyph3DWithScaling* glyph = vtkGlyph3DWithScaling::New();
    glyph->SetInputData( polyData );
    glyph->SetSourceData( polyFilter->GetOutput() );
    glyph->SetVectorModeToUseNormal();
    glyph->SetScaleModeToScaleByScalar();
    glyph->ScalingXOff();
    glyph->ScalingYOn();
    glyph->ScalingZOn();
    // #glyph SetScaleModeToDataScalingOff
    glyph->SetScaleFactor( scaleFactor );
    glyph->Update();

  vtkPolyDataMapper* mapper = vtkPolyDataMapper::New();
    mapper->SetInputConnection( glyph->GetOutputPort() );
    mapper->ScalarVisibilityOff();

  vtkActor* actor = vtkActor::New();
    actor->SetMapper( mapper );

  if ( particlesType == (unsigned char)( cip::AIRWAY ) )
    {
    this->AirwayParticlesActorMap[actorName] = actor;
    }
  else if ( particlesType == (unsigned char)( cip::VESSEL ) )
    {
    this->VesselParticlesActorMap[actorName] = actor;
    }
  else if ( particlesType == (unsigned char)( cip::FISSURE ) )
    {
    this->FissureParticlesActorMap[actorName] = actor;
    }
  this->ActorMap[actorName] = actor;
  this->Renderer->AddActor( this->ActorMap[actorName] );

  return actor;
}

void cipChestDataViewer::SetVesselParticlesAsCylinders( vtkPolyData* polyData, double scaleFactor, std::string actorName )
{
  this->SetParticlesAsCylinders( polyData, scaleFactor, actorName, (unsigned char)( cip::VESSEL ), false );
}

vtkActor* cipChestDataViewer::SetParticlesAsCylinders( vtkPolyData* polyData, double scaleFactor, std::string actorName,
                                                       unsigned char particlesType, bool scaleGlyphsByParticlesScale )
{
  if ( scaleGlyphsByParticlesScale )
    {
    polyData->GetPointData()->SetScalars( polyData->GetPointData()->GetArray( "scale" ) );
    }

  if ( particlesType == (unsigned char)( cip::AIRWAY ) )
    {
    polyData->GetPointData()->SetNormals( polyData->GetPointData()->GetArray( "hevec2" ) );
    }
  else if ( particlesType == (unsigned char)( cip::VESSEL ) )
    {
    polyData->GetPointData()->SetNormals( polyData->GetPointData()->GetArray( "hevec0" ) );
    }

  vtkCylinderSource* cylinderSource = vtkCylinderSource::New();
    cylinderSource->SetHeight( 20 );
    cylinderSource->SetRadius( 2 );
    cylinderSource->SetCenter( 0, 0, 0 );
    cylinderSource->SetResolution( 10 );
    cylinderSource->CappingOn();

  vtkTransform* cylinderRotator = vtkTransform::New();
    cylinderRotator->RotateZ( 90 );

  vtkTransformPolyDataFilter* polyFilter = vtkTransformPolyDataFilter::New();
    polyFilter->SetInputConnection( cylinderSource->GetOutputPort() );
    polyFilter->SetTransform( cylinderRotator );
    polyFilter->Update();

  vtkGlyph3D* glyph = vtkGlyph3D::New();
    glyph->SetInputData( polyData );
    glyph->SetSourceData( polyFilter->GetOutput() );
    glyph->SetVectorModeToUseNormal();
  if ( scaleGlyphsByParticlesScale )
    {
    glyph->SetScaleModeToScaleByScalar();
    }
    glyph->SetScaleFactor( scaleFactor );
    glyph->Update();
    glyph->GetOutput()->GetPointData()->SetScalars( NULL );

  vtkPolyDataMapper* mapper = vtkPolyDataMapper::New();
    mapper->SetInputConnection( glyph->GetOutputPort() );

  vtkActor* actor = vtkActor::New();
    actor->SetMapper( mapper );

  if ( particlesType == (unsigned char)( cip::AIRWAY ) )
    {
    this->AirwayParticlesActorMap[actorName] = actor;
    }
  else if ( particlesType == (unsigned char)( cip::VESSEL ) )
    {
    this->VesselParticlesActorMap[actorName] = actor;
    }
  this->ActorMap[actorName] = actor;
  this->Renderer->AddActor( this->ActorMap[actorName] );

  return actor;
}

void cipChestDataViewer::SetParticles( vtkPolyData* polyData, double scaleFactor, std::string actorName, bool fissureParticles )
{
  unsigned int numberOfParticles = polyData->GetNumberOfPoints();

  vtkFloatArray* vecs0 = vtkFloatArray::SafeDownCast( polyData->GetPointData()->GetArray( "hevec0" ) );
  vtkFloatArray* vecs1 = vtkFloatArray::SafeDownCast( polyData->GetPointData()->GetArray( "hevec1" ) );
  vtkFloatArray* vecs2 = vtkFloatArray::SafeDownCast( polyData->GetPointData()->GetArray( "hevec2" ) );

  vtkFloatArray* vals0 = vtkFloatArray::SafeDownCast( polyData->GetPointData()->GetArray( "h0" ) );
  vtkFloatArray* vals1 = vtkFloatArray::SafeDownCast( polyData->GetPointData()->GetArray( "h1" ) );
  vtkFloatArray* vals2 = vtkFloatArray::SafeDownCast( polyData->GetPointData()->GetArray( "h2" ) );

  vtkFloatArray* scaleArray = vtkFloatArray::SafeDownCast( polyData->GetPointData()->GetArray( "scale" ) );

  vtkDoubleArray *dbar = vtkDoubleArray::New();
    dbar->SetNumberOfTuples( numberOfParticles );
    dbar->SetNumberOfComponents( 9 );

  double centerOfMass[3];
    centerOfMass[0] = 0;
    centerOfMass[1] = 0;
    centerOfMass[2] = 0;


  for ( unsigned int i=0; i<numberOfParticles; i++ )
    {
    centerOfMass[0] += polyData->GetPoint(i)[0];
    centerOfMass[1] += polyData->GetPoint(i)[1];
    centerOfMass[2] += polyData->GetPoint(i)[2];

    double val0[1];
    double val1[1];
    double val2[1];
    vals0->GetTuple( i, val0 );
    vals1->GetTuple( i, val1 );
    vals2->GetTuple( i, val2 );

    float mag = std::sqrt( std::pow( *val0, 2 ) + std::pow( *val1, 2 ) + std::pow( *val2, 2 ) );

    double vec0[3];
    double vec1[3];
    double vec2[3];
    vecs0->GetTuple( i, vec0 );
    vecs1->GetTuple( i, vec1 );
    vecs2->GetTuple( i, vec2 );

    vnl_matrix_fixed< double, 3, 3 > M1;
    M1(0,0) = vec0[0]*vec0[0];   M1(0,1) = vec0[0]*vec0[1];   M1(0,2) = vec0[0]*vec0[2];
    M1(1,0) = vec0[1]*vec0[0];   M1(1,1) = vec0[1]*vec0[1];   M1(1,2) = vec0[1]*vec0[2];
    M1(2,0) = vec0[2]*vec0[0];   M1(2,1) = vec0[2]*vec0[1];   M1(2,2) = vec0[2]*vec0[2];

    vnl_matrix_fixed< double, 3, 3 > M2;
    M2(0,0) = vec1[0]*vec1[0];   M2(0,1) = vec1[0]*vec1[1];   M2(0,2) = vec1[0]*vec1[2];
    M2(1,0) = vec1[1]*vec1[0];   M2(1,1) = vec1[1]*vec1[1];   M2(1,2) = vec1[1]*vec1[2];
    M2(2,0) = vec1[2]*vec1[0];   M2(2,1) = vec1[2]*vec1[1];   M2(2,2) = vec1[2]*vec1[2];

    vnl_matrix_fixed< double, 3, 3 > M3;
    M3(0,0) = vec2[0]*vec2[0];   M3(0,1) = vec2[0]*vec2[1];   M3(0,2) = vec2[0]*vec2[2];
    M3(1,0) = vec2[1]*vec2[0];   M3(1,1) = vec2[1]*vec2[1];   M3(1,2) = vec2[1]*vec2[2];
    M3(2,0) = vec2[2]*vec2[0];   M3(2,1) = vec2[2]*vec2[1];   M3(2,2) = vec2[2]*vec2[2];

    double scale[1];
    if ( fissureParticles )
      {
      *scale = 1.0;
      M1 *= (1 - std::abs( *val0/mag ));
      M2 *= (1 - std::abs( *val1/mag ));
      M3 *= (1 - std::abs( *val2/mag ));
      }
    else
      {
      scaleArray->GetTuple( i, scale );
      M1 *= std::abs( *val0/mag );
      M2 *= std::abs( *val1/mag );
      M3 *= std::abs( *val2/mag );
      }

    vnl_matrix_fixed< double, 3, 3 > tensorMat;
    tensorMat = M1 + M2 + M3;
    tensorMat *= *scale;

    dbar->InsertTuple9( i, tensorMat(0,0), tensorMat(0,1), tensorMat(0,2),
                           tensorMat(1,0), tensorMat(1,1), tensorMat(1,2),
                           tensorMat(2,0), tensorMat(2,1), tensorMat(2,2) );
    }

  vtkPolyData* sculptedTensorsPolyData = vtkPolyData::New();
    sculptedTensorsPolyData->SetPoints( polyData->GetPoints() );
    sculptedTensorsPolyData->GetPointData()->SetTensors( dbar );

  vtkSuperquadricTensorGlyphFilter* epp = vtkSuperquadricTensorGlyphFilter::New();
    epp->SetInputData( sculptedTensorsPolyData );
    epp->SetThetaRoundness( 0.95 ); //0 is rect, 1 is sphere
    epp->SetPhiRoundness( 0.95 ); //0 is rect, 1 is sphere
    epp->SetThetaResolution( this->ParticleGlyphThetaResolution );
    epp->SetPhiResolution( this->ParticleGlyphPhiResolution );
    epp->SetScaleFactor( scaleFactor );
    epp->SetExtractEigenvalues( true );

  vtkPolyDataNormals* norm = vtkPolyDataNormals::New();
    norm->SetInputConnection( epp->GetOutputPort() );

  vtkPolyDataMapper *mapper = vtkPolyDataMapper::New();
    mapper->SetInputConnection( norm->GetOutputPort() );

  vtkActor* actor = vtkActor::New();
    actor->SetMapper( mapper );

  this->ActorMap[actorName] = actor;
  this->Renderer->AddActor( this->ActorMap[actorName] );

  if ( this->DisplayActorNames )
    {
    centerOfMass[0] /= static_cast< double >( numberOfParticles );
    centerOfMass[1] /= static_cast< double >( numberOfParticles );
    centerOfMass[2] /= static_cast< double >( numberOfParticles );

    vtkVectorText* vecText = vtkVectorText::New();
      vecText->SetText( actorName.c_str() );

    vtkPolyDataMapper* textMapper = vtkPolyDataMapper::New();
      textMapper->SetInputConnection( vecText->GetOutputPort());

    vtkFollower* textActor = vtkFollower::New();
      textActor->SetMapper( textMapper );
      textActor->SetScale( 1, 1, 1 );
      textActor->AddPosition( centerOfMass[0]+1, centerOfMass[1]+1, centerOfMass[2]+1 );

    this->Renderer->AddActor( textActor );
    }

  mapper->Delete();
  norm->Delete();
  epp->Delete();
  sculptedTensorsPolyData->Delete();
  dbar->Delete();
}

void cipChestDataViewer::GenerateFissureActor( cipThinPlateSplineSurface* tpsSurface, unsigned char whichFissure,
                                           unsigned char whichLung, std::string name )
{
 //  if ( whichFissure == HORIZONTALFISSURE && tpsSurface->GetNumberSurfacePoints() <= 0 )
//     {
//     return;
//     }
  if ( this->LabelMapImage.IsNotNull() )
    {
    LabelMapImageType::SpacingType spacing = this->LabelMapImage->GetSpacing();
    LabelMapImageType::PointType   origin  = this->LabelMapImage->GetOrigin();

    vtkPoints*    points          = vtkPoints::New();
    vtkCellArray* linesCellArray  = vtkCellArray::New();

    typedef itk::Image< unsigned int, 2 >                         DomainImageType;
    typedef itk::ImageRegionIteratorWithIndex< DomainImageType >  DomainIteratorType;

    cip::ChestConventions conventions;

    //
    // Create the domain image. This will be a 2D image that will be
    // responsible for keeping track on the point numberings
    //
    LabelMapImageType::SizeType size = this->LabelMapImage->GetBufferedRegion().GetSize();

    DomainImageType::SizeType domainSize;
      domainSize[0] = size[0];
      domainSize[1] = size[1];

    DomainImageType::Pointer domainImage = DomainImageType::New();
      domainImage->SetRegions( domainSize );
      domainImage->Allocate();
      domainImage->FillBuffer( 0 );

    DomainIteratorType dIt( domainImage, domainImage->GetBufferedRegion() );

    LabelMapImageType::IndexType  index;
    LabelMapImageType::PointType  physicalPoint;

    unsigned int inc = 1;

    dIt.GoToBegin();
    while ( !dIt.IsAtEnd() )
      {
      double x = dIt.GetIndex()[0]*spacing[0] + origin[0];
      double y = dIt.GetIndex()[1]*spacing[1] + origin[1];
      double z = tpsSurface->GetSurfaceHeight( x, y );

      index[0] = dIt.GetIndex()[0];
      index[1] = dIt.GetIndex()[1];
      index[2] = static_cast< unsigned int >( (z-origin[2])/spacing[2] );

      if ( this->LabelMapImage->GetBufferedRegion().IsInside( index ) )
        {
        unsigned short labelValue = this->LabelMapImage->GetPixel( index );
        unsigned char  lungRegion = conventions.GetChestRegionFromValue( labelValue );

        if ( ((whichLung == cip::LEFTLUNG) &&
              (lungRegion == (unsigned char)( cip::LEFTLUNG ) ||
               lungRegion == (unsigned char)( cip::LEFTUPPERTHIRD ) ||
               lungRegion == (unsigned char)( cip::LEFTMIDDLETHIRD ) ||
               lungRegion == (unsigned char)( cip::LEFTLOWERTHIRD ) ||
               lungRegion == (unsigned char)( cip::LEFTSUPERIORLOBE ) ||
               lungRegion == (unsigned char)( cip::LEFTINFERIORLOBE ))) ||
             ((whichLung == cip::RIGHTLUNG) &&
              (lungRegion == (unsigned char)( cip::RIGHTLUNG ) ||
               lungRegion == (unsigned char)( cip::RIGHTUPPERTHIRD ) ||
               lungRegion == (unsigned char)( cip::RIGHTMIDDLETHIRD ) ||
               lungRegion == (unsigned char)( cip::RIGHTLOWERTHIRD ) ||
               lungRegion == (unsigned char)( cip::RIGHTSUPERIORLOBE ) ||
               lungRegion == (unsigned char)( cip::RIGHTINFERIORLOBE ))) )
          {
          bool addPoint = true;

//           if ( whichFissure == HORIZONTALFISSURE )
//             {
//             double zRO = RightObliqueThinPlateSplineSurface->GetSurfaceHeight( x, y );

//             if ( zRO > z )
//               {
//               addPoint = false;
//               }
//             }
          if ( addPoint )
            {
            this->LabelMapImage->TransformIndexToPhysicalPoint( index, physicalPoint );

            points->InsertNextPoint( physicalPoint[0], physicalPoint[1], physicalPoint[2] );

            dIt.Set( inc );
            inc++;
            }
          }
        }

      ++dIt;
      }

    //
    // Create the poly data that will hold our triangle strips
    //
    vtkPolyData* triangleStripPolyData = vtkPolyData::New();
      triangleStripPolyData->Allocate();
      triangleStripPolyData->SetPoints( points );

    DomainImageType::IndexType domainIndex;

    for ( unsigned int y=0; y<domainSize[1]; y++ )
      {
      //
      // Initialize our three points and the triangle counter to keep
      // track of the triangle construction
      //
      unsigned int triangleCounter = 0;
      unsigned int firstPoint      = 0;
      unsigned int secondPoint     = 0;
      unsigned int thirdPoint      = 0;

      for ( unsigned int x=0; x<domainSize[0]; x++ )
        {
        domainIndex[0] = x;

        //
        // Create a new triangle strip for this row
        //
        vtkTriangleStrip* triangleStrip = vtkTriangleStrip::New();

        for ( unsigned int yInc=y; yInc<=y+1; yInc++ )
          {
          domainIndex[1] = yInc;

          if ( domainImage->GetBufferedRegion().IsInside( domainIndex ) )
            {
            if ( domainImage->GetPixel( domainIndex ) )
              {
              if ( triangleCounter == 0 )
                {
                firstPoint = domainImage->GetPixel( domainIndex ) - 1;
                triangleCounter++;
                }
              else if ( triangleCounter == 1 )
                {
                secondPoint = domainImage->GetPixel( domainIndex ) - 1;
                triangleCounter++;
                }
              else if ( triangleCounter == 2 )
                {
                thirdPoint = domainImage->GetPixel( domainIndex ) - 1;
                triangleCounter++;
                }

              //
              // If 'triangleCounter' is up to 3, that means we have
              // found three consecutive points that are non-zero in the
              // domain image. These three should form the next triangle
              // in our triangle strip
              //
              if ( triangleCounter == 3 )
                {
                //
                // Add these points to the triangle strip
                //
                triangleStrip->GetPointIds()->InsertNextId( firstPoint );
                triangleStrip->GetPointIds()->InsertNextId( secondPoint );
                triangleStrip->GetPointIds()->InsertNextId( thirdPoint );

                //
                // Now reset the triangle counter to two, indicating
                // that we only need one more non-zero domain location
                // point in order to add the next triangle. Also, the
                // 'thirdPoint' becomes the 'secondPoint', and the
                // 'secondPoint' becomes the 'firstPoint'.
                //
                triangleCounter = 2;

                firstPoint = secondPoint;  secondPoint = thirdPoint;
                }
              }
            else
              {
              triangleCounter = 0;

              break;
              }
            }
          }

        //
        // At this point, we have created one triangle stip. Add this
        // strip to our unstructured grid
        //
        triangleStripPolyData->InsertNextCell( triangleStrip->GetCellType(), triangleStrip->GetPointIds() );
        }
      }

    vtkDataSetMapper* triangleStripMapper = vtkDataSetMapper::New();
      triangleStripMapper->SetInputData( triangleStripPolyData );

    vtkActor* triangleStripActor = vtkActor::New();
      triangleStripActor->SetMapper( triangleStripMapper );

    this->ActorMap[name] = triangleStripActor;
    this->Renderer->AddActor( this->ActorMap[name] );
    }
}

void cipChestDataViewer::SetLeftObliqueFissurePCAModeAndVariance( std::vector< double > vec, double variance, unsigned int whichMode )
{
  if ( whichMode >= this->LeftObliqueFissurePCAVariances.size() )
    {
    while ( whichMode >= this->LeftObliqueFissurePCAVariances.size() )
      {
      this->LeftObliqueFissurePCAVariances.push_back( 0.0 );

      std::vector< double > tempModeVec;
      this->LeftObliqueFissurePCAModes.push_back( tempModeVec );
      }
    }

  this->LeftObliqueFissurePCAVariances[whichMode] = variance;

  for ( unsigned int i=0; i<vec.size(); i++ )
    {
    this->LeftObliqueFissurePCAModes[whichMode].push_back( vec[i] );
    }
}

void cipChestDataViewer::SetRightObliqueFissurePCAModeAndVariance( std::vector< double > vec, double variance, unsigned int whichMode )
{
  if ( whichMode >= this->RightObliqueFissurePCAVariances.size() )
    {
    while ( whichMode >= this->RightObliqueFissurePCAVariances.size() )
      {
      this->RightObliqueFissurePCAVariances.push_back( 0.0 );

      std::vector< double > tempModeVec;
      this->RightObliqueFissurePCAModes.push_back( tempModeVec );
      }
    }

  this->RightObliqueFissurePCAVariances[whichMode] = variance;

  for ( unsigned int i=0; i<vec.size(); i++ )
    {
    this->RightObliqueFissurePCAModes[whichMode].push_back( vec[i] );
    }
}

void cipChestDataViewer::SetRightHorizontalFissurePCAModeAndVariance( std::vector< double > vec, double variance, unsigned int whichMode )
{
  if ( whichMode >= this->RightHorizontalFissurePCAVariances.size() )
    {
    while ( whichMode >= this->RightHorizontalFissurePCAVariances.size() )
      {
      this->RightHorizontalFissurePCAVariances.push_back( 0.0 );

      std::vector< double > tempModeVec;
      this->RightHorizontalFissurePCAModes.push_back( tempModeVec );
      }
    }

  this->RightHorizontalFissurePCAVariances[whichMode] = variance;

  for ( unsigned int i=0; i<vec.size(); i++ )
    {
    this->RightHorizontalFissurePCAModes[whichMode].push_back( vec[i] );
    }
}

void cipChestDataViewer::ModifyLeftObliqueFissureByPCAMode( unsigned int whichMode, double stdMultiplier )
{
//   for ( unsigned int i=0; i<(this->LeftObliqueFissurePCAModes[whichMode]).size(); i++ )
//     {
//     (this->LeftObliqueFissureIndices[i])[2] += static_cast< unsigned int >( stdMultiplier*std::sqrt( this->LeftObliqueFissurePCAVariances[whichMode] )*
//                                                                               (this->LeftObliqueFissurePCAModes[whichMode])[i] );
//     }
//   this->Renderer->RemoveActor( this->ActorMap["LEFTLUNGOBLIQUEFISSURE"] );
//   this->ActorMap["LEFTLUNGOBLIQUEFISSURE"]->Delete();
//   this->GenerateFissureActor( (unsigned char)( OBLIQUEFISSURE ), (unsigned char)( LEFTLUNG ) );
}

void cipChestDataViewer::ModifyRightObliqueFissureByPCAMode( unsigned int whichMode, double stdMultiplier )
{
//   for ( unsigned int i=0; i<(this->RightObliqueFissurePCAModes[whichMode]).size(); i++ )
//     {
//     (this->RightObliqueFissureIndices[i])[2] += static_cast< unsigned int >( stdMultiplier*std::sqrt( this->RightObliqueFissurePCAVariances[whichMode] )*
//                                                                               (this->RightObliqueFissurePCAModes[whichMode])[i] );
//     }
//   this->Renderer->RemoveActor( this->ActorMap["RIGHTLUNGOBLIQUEFISSURE"] );
//   this->ActorMap["RIGHTLUNGOBLIQUEFISSURE"]->Delete();
//   this->GenerateFissureActor( (unsigned char)( OBLIQUEFISSURE ), (unsigned char)( RIGHTLUNG ) );
}

void cipChestDataViewer::ModifyRightHorizontalFissureByPCAMode( unsigned int whichMode, double stdMultiplier )
{
//   for ( unsigned int i=0; i<(this->RightHorizontalFissurePCAModes[whichMode]).size(); i++ )
//     {
//     (this->RightHorizontalFissureIndices[i])[2] += static_cast< unsigned int >( stdMultiplier*std::sqrt( this->RightHorizontalFissurePCAVariances[whichMode] )*
//                                                                                   (this->RightHorizontalFissurePCAModes[whichMode])[i] );
//     }
//   this->Renderer->RemoveActor( this->ActorMap["RIGHTLUNGHORIZONTALFISSURE"] );
//   this->ActorMap["RIGHTLUNGHORIZONTALFISSURE"]->Delete();
//   this->GenerateFissureActor( (unsigned char)( HORIZONTALFISSURE ), (unsigned char)( RIGHTLUNG ) );
}

// void cipChestDataViewer::ExtractAndViewLungRegionModel( unsigned char lungRegion, std::string name )
// {
//   if ( this->LabelMapImage.IsNotNull() )
//     {
//     typedef itk::ExtractLungLabelMapImageFilter LabelMapExtractorType;

//     LungConventions conventions;

//     unsigned short foregroundLabel = conventions.GetValueFromLungRegionAndType( lungRegion, (unsigned char)( UNDEFINEDTYPE) );

//     LabelMapExtractorType::Pointer extractor = LabelMapExtractorType::New();
//       extractor->SetInput( this->LabelMapImage );
//       extractor->SetLungRegion( lungRegion );
//       extractor->Update();

//     vtkPolyData* lungRegionPolyData = this->GetModelFromLabelMap( extractor->GetOutput(), foregroundLabel );

//     this->SetPolyData( lungRegionPolyData, name );
//     }
// }

// void cipChestDataViewer::ExtractAndViewLungTypeModel( unsigned char lungType, std::string name )
// {
//   if ( this->LabelMapImage.IsNotNull() )
//     {
//     typedef itk::ExtractLungLabelMapImageFilter LabelMapExtractorType;

//     LungConventions conventions;

//     unsigned short foregroundLabel = conventions.GetValueFromLungRegionAndType( (unsigned char)( UNDEFINEDREGION ), lungType );

//     LabelMapExtractorType::Pointer extractor = LabelMapExtractorType::New();
//       extractor->SetInput( this->LabelMapImage );
//       extractor->SetLungType( lungType );
//       extractor->Update();

//     vtkPolyData* lungTypePolyData = this->GetModelFromLabelMap( extractor->GetOutput(), foregroundLabel );

//     this->SetPolyData( lungTypePolyData, name );
//     }
// }

vtkPolyData* cipChestDataViewer::GetModelFromLabelMap( LabelMapImageType::Pointer labelMap, unsigned short foregroundLabel )
{
    LabelMapExportType::Pointer refExporter = LabelMapExportType::New();
      refExporter->SetInput( labelMap );

    vtkImageImport* refImporter = vtkImageImport::New();

    this->ConnectLabelMapPipelines( refExporter, refImporter );

    //
    // Perform marching cubes on the reference binary image and then
    // decimate
    //
    vtkDiscreteMarchingCubes* cubes = vtkDiscreteMarchingCubes::New();
      cubes->SetInputConnection( refImporter->GetOutputPort() );
      cubes->SetValue( 0, foregroundLabel );
      cubes->ComputeNormalsOff();
      cubes->ComputeScalarsOff();
      cubes->ComputeGradientsOff();
      cubes->Update();

    vtkWindowedSincPolyDataFilter* smoother = vtkWindowedSincPolyDataFilter::New();
      smoother->SetInputConnection( cubes->GetOutputPort() );
      smoother->SetNumberOfIterations( 2 );
      smoother->BoundarySmoothingOff();
      smoother->FeatureEdgeSmoothingOff();
      smoother->SetPassBand( 0.001 );
      smoother->NonManifoldSmoothingOn();
      smoother->NormalizeCoordinatesOn();
      smoother->Update();

    vtkDecimatePro* decimator = vtkDecimatePro::New();
      decimator->SetInputConnection( smoother->GetOutputPort() );
      decimator->SetTargetReduction( 0.9 );
      decimator->PreserveTopologyOn();
      decimator->BoundaryVertexDeletionOff();
      decimator->Update();

    vtkPolyDataNormals* normals = vtkPolyDataNormals::New();
      normals->SetInputConnection( decimator->GetOutputPort() );
      normals->SetFeatureAngle( 90 );
      normals->Update();

    normals->Delete();
//    decimator->Delete();
    smoother->Delete();
    cubes->Delete();
    refImporter->Delete();

    return decimator->GetOutput();
}

void cipChestDataViewer::ConnectPipelines( ExportType::Pointer exporter, vtkImageImport* importer )
{
  importer->SetUpdateInformationCallback(exporter->GetUpdateInformationCallback());
  importer->SetPipelineModifiedCallback(exporter->GetPipelineModifiedCallback());
  importer->SetWholeExtentCallback(exporter->GetWholeExtentCallback());
  importer->SetSpacingCallback(exporter->GetSpacingCallback());
  importer->SetOriginCallback(exporter->GetOriginCallback());
  importer->SetScalarTypeCallback(exporter->GetScalarTypeCallback());
  importer->SetNumberOfComponentsCallback(exporter->GetNumberOfComponentsCallback());
  importer->SetPropagateUpdateExtentCallback(exporter->GetPropagateUpdateExtentCallback());
  importer->SetUpdateDataCallback(exporter->GetUpdateDataCallback());
  importer->SetDataExtentCallback(exporter->GetDataExtentCallback());
  importer->SetBufferPointerCallback(exporter->GetBufferPointerCallback());
  importer->SetCallbackUserData(exporter->GetCallbackUserData());
}

void cipChestDataViewer::ConnectLabelMapPipelines( LabelMapExportType::Pointer exporter, vtkImageImport* importer )
{
  importer->SetUpdateInformationCallback(exporter->GetUpdateInformationCallback());
  importer->SetPipelineModifiedCallback(exporter->GetPipelineModifiedCallback());
  importer->SetWholeExtentCallback(exporter->GetWholeExtentCallback());
  importer->SetSpacingCallback(exporter->GetSpacingCallback());
  importer->SetOriginCallback(exporter->GetOriginCallback());
  importer->SetScalarTypeCallback(exporter->GetScalarTypeCallback());
  importer->SetNumberOfComponentsCallback(exporter->GetNumberOfComponentsCallback());
  importer->SetPropagateUpdateExtentCallback(exporter->GetPropagateUpdateExtentCallback());
  importer->SetUpdateDataCallback(exporter->GetUpdateDataCallback());
  importer->SetDataExtentCallback(exporter->GetDataExtentCallback());
  importer->SetBufferPointerCallback(exporter->GetBufferPointerCallback());
  importer->SetCallbackUserData(exporter->GetCallbackUserData());
}

void ViewerKeyCallback( vtkObject* obj, unsigned long b, void* clientData, void* d )
{
  cipChestDataViewer* dataViewer = reinterpret_cast< cipChestDataViewer* >( clientData );

  char pressedKey = dataViewer->GetRenderWindowInteractor()->GetKeyCode();

  if ( pressedKey == 'x' )
    {
    if ( dataViewer->GetPlaneWidgetXShowing() )
      {
      dataViewer->SetPlaneWidgetXShowing( false );
      }
    else
      {
      dataViewer->SetPlaneWidgetXShowing( true );
      }
    }

  if ( pressedKey == 'y' )
    {
    if ( dataViewer->GetPlaneWidgetYShowing() )
      {
      dataViewer->SetPlaneWidgetYShowing( false );
      }
    else
      {
      dataViewer->SetPlaneWidgetYShowing( true );
      }
    }

  if ( pressedKey == 'z' )
    {
    if ( dataViewer->GetPlaneWidgetZShowing() )
      {
      dataViewer->SetPlaneWidgetZShowing( false );
      }
    else
      {
      dataViewer->SetPlaneWidgetZShowing( true );
      }
    }
}
