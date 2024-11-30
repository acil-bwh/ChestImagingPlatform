/**
 *  \class cipChestDataViewer
 *  \ingroup common
 *  \brief  This class serves as a basic viewer for the various forms
 *  of lung data that we are interested in. Models of structures (read
 *  or generated internally from an input label map) can be rendered
 *  with arbitrary color and opacity. Particle data can also be
 *  rendered. 
 *
 *  $Date$
 *  $Revision: 188 $
 *  $Author: jross $
 *
 *  TODO:
 *
 */

#ifndef __cipChestDataViewer_h
#define __cipChestDataViewer_h

#include "vtkPolyData.h"
#include "vtkActorCollection.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkImageImport.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "itkImage.h"
#include "itkVTKImageExport.h"
#include "itkContinuousIndex.h"
#include "cipThinPlateSplineSurface.h"
#include "vtkImagePlaneWidget.h"
#include "itkVTKImageExport.h"
#include "vtkImageImport.h"
#include "vtkImageData.h"
#include "vtkCallbackCommand.h"
#include "cipChestConventions.h"
#include "vtkSmartPointer.h"

void ViewerKeyCallback( vtkObject*, unsigned long, void*, void* );


class cipChestDataViewer
{
public:
  ~cipChestDataViewer(){};
  cipChestDataViewer();

  typedef itk::Image< unsigned short, 3 >    LabelMapImageType;
  typedef itk::ContinuousIndex< double, 3 >  ContinuousIndexType;
  typedef itk::Image< short, 3 >             GrayscaleImageType;

  void SetGrayscaleImage( GrayscaleImageType::Pointer );
  void SetLabelMapImage( LabelMapImageType::Pointer );

  void ToggleActorVisibility();

  void SetBackgroundColor( double, double, double );

  void SetLeftObliqueFissurePoints( const std::vector< cip::PointType >&, std::string );
  void SetLeftObliqueFissurePCAModeAndVariance( std::vector< double >, double, unsigned int );
  void ModifyLeftObliqueFissureByPCAMode( unsigned int, double );

  void SetRightObliqueFissurePoints( const std::vector< cip::PointType >&, std::string );
  void SetRightObliqueFissurePCAModeAndVariance( std::vector< double >, double, unsigned int );
  void ModifyRightObliqueFissureByPCAMode( unsigned int, double );

  void SetRightHorizontalFissurePoints( const std::vector< cip::PointType >&, std::string );
  void SetRightHorizontalFissurePCAModeAndVariance( std::vector< double >, double, unsigned int );
  void ModifyRightHorizontalFissureByPCAMode( unsigned int, double );

  void SetActorColor( std::string, double, double, double );
  void GetActorColor( std::string, double[3] );

  void SetActorOpacity( std::string, double );

  void SetAirwayParticles( vtkPolyData*, double, std::string );
  void SetVesselParticles( vtkPolyData*, double, std::string );
  void SetFissureParticles( vtkPolyData*, double, std::string );

  vtkSmartPointer< vtkActor > SetAirwayParticlesAsCylinders( vtkPolyData*, double, std::string );
  void SetVesselParticlesAsCylinders( vtkPolyData*, double, std::string );
  void SetPointsAsSpheres( vtkPolyData*, double, std::string );

  vtkSmartPointer< vtkActor > SetAirwayParticlesAsDiscs( vtkPolyData*, double, std::string );
  vtkSmartPointer< vtkActor > SetVesselParticlesAsDiscs( vtkPolyData*, double, std::string );
  vtkSmartPointer< vtkActor > SetFissureParticlesAsDiscs( vtkPolyData*, double, std::string );
  vtkSmartPointer< vtkActor > SetParticlesAsDiscs( vtkPolyData*, double, std::string, unsigned char, bool );

  vtkSmartPointer< vtkActor > SetPolyData( vtkPolyData*, std::string );

  void SetLeftObliqueThinPlateSplineSurface( cipThinPlateSplineSurface*, std::string );
  void SetRightObliqueThinPlateSplineSurface( cipThinPlateSplineSurface*, std::string );
  void SetRightHorizontalThinPlateSplineSurface( cipThinPlateSplineSurface*, std::string );

  void SetPlaneWidgetXShowing( bool );
  bool GetPlaneWidgetXShowing()
    {
      return PlaneWidgetXShowing;
    };

  void SetPlaneWidgetYShowing( bool );
  bool GetPlaneWidgetYShowing()
    {
      return PlaneWidgetYShowing;
    };

  void SetPlaneWidgetZShowing( bool );
  bool GetPlaneWidgetZShowing()
    {
      return PlaneWidgetZShowing;
    };

//  void ExtractAndViewLungRegionModel( unsigned char, std::string );

//  void ExtractAndViewLungTypeModel( unsigned char, std::string );

  void SetParticleGlyphThetaResolution( double resolution )
    {
      ParticleGlyphThetaResolution = resolution;
    }

  void SetParticlePhiThetaResolution( double resolution )
    {
      ParticleGlyphPhiResolution = resolution;
    }

  void Render();

  //
  // Returns true is the passed actor name is among the actors being
  // viewed
  //
  bool Exists( std::string );

  vtkRenderWindowInteractor* GetRenderWindowInteractor()
    {
      return RenderWindowInteractor;
    };

private:
  typedef itk::VTKImageExport< LabelMapImageType >   LabelMapExportType;

  void SetParticles( vtkPolyData*, double, std::string, bool );
  void GenerateFissureActor( cipThinPlateSplineSurface*, unsigned char, unsigned char, std::string );
  void ConnectLabelMapPipelines( LabelMapExportType::Pointer, vtkImageImport* );
  vtkPolyData* GetModelFromLabelMap( LabelMapImageType::Pointer, unsigned short );

  vtkInteractorStyleTrackballCamera*  TrackballCameraStyle;

  LabelMapImageType::Pointer          LabelMapImage;

  std::vector< cip::PointType >  LeftObliqueFissurePoints;
  std::vector< cip::PointType >  RightObliqueFissurePoints;
  std::vector< cip::PointType >  RightHorizontalFissurePoints;

  std::vector< double >                  LeftObliqueFissurePCAVariances;
  std::vector< std::vector< double > >   LeftObliqueFissurePCAModes;

  std::vector< double >                  RightObliqueFissurePCAVariances;
  std::vector< std::vector< double > >   RightObliqueFissurePCAModes;

  std::vector< double >                  RightHorizontalFissurePCAVariances;
  std::vector< std::vector< double > >   RightHorizontalFissurePCAModes;

protected:
  typedef itk::VTKImageExport< GrayscaleImageType >  ExportType;

  vtkActor* SetParticlesAsCylinders( vtkPolyData*, double, std::string, unsigned char, bool );

  void ConnectPipelines( ExportType::Pointer, vtkImageImport* );

  GrayscaleImageType::Pointer GrayscaleImage;
  vtkImageData* vtkGrayscaleImage;

  vtkCallbackCommand*     ViewerCallbackCommand;

  vtkImagePlaneWidget*    PlaneWidgetX;
  vtkImagePlaneWidget*    PlaneWidgetY;
  vtkImagePlaneWidget*    PlaneWidgetZ;

  bool PlaneWidgetXShowing;
  bool PlaneWidgetYShowing;
  bool PlaneWidgetZShowing;

  cip::ChestConventions* Conventions;

  vtkRenderWindowInteractor*           RenderWindowInteractor;
  vtkRenderer*                         Renderer;
  vtkRenderWindow*                     RenderWindow;
  std::map< std::string, vtkActor* >   ActorMap;
  std::map< std::string, vtkActor* >   AirwayParticlesActorMap;
  std::map< std::string, vtkActor* >   VesselParticlesActorMap;
  std::map< std::string, vtkActor* >   FissureParticlesActorMap;
  double                               ParticleGlyphThetaResolution;
  double                               ParticleGlyphPhiResolution;
  cipThinPlateSplineSurface*           LeftObliqueThinPlateSplineSurface;
  cipThinPlateSplineSurface*           RightObliqueThinPlateSplineSurface;
  cipThinPlateSplineSurface*           RightHorizontalThinPlateSplineSurface;
  bool                                 DisplayActorNames;
  bool                                 ActorsVisible;
};

#endif
