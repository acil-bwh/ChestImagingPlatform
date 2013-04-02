#ifndef ACILAssistantBase_h
#define ACILAssistantBase_h

#include "itkImage.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkVTKImageExport.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryDilateImageFilter.h"

#include "vtkImageImport.h"


class ACILAssistantBase
{
public:
  ACILAssistantBase();
  ~ACILAssistantBase();

  typedef itk::Image< short, 3 >           GrayscaleImageType;
  typedef itk::Image< unsigned short, 3 >  LabelMapType;

  void InitializeLabelMapImage( LabelMapType::SizeType, LabelMapType::SpacingType, LabelMapType::PointType );

  void SetLabelMapImage( LabelMapType::Pointer );

  LabelMapType::Pointer GetLabelMapImage()
    {
      return LabelMap;
    };

  void SetGrayscaleImage( GrayscaleImageType::Pointer );

  GrayscaleImageType::Pointer GetGrayscaleImage()
    {
      return GrayscaleImage;
    };

  void PaintLabelMapSlice( LabelMapType::IndexType, unsigned char, unsigned char, unsigned int, short, short, unsigned int );

  void EraseLabelMapSlice( LabelMapType::IndexType, unsigned char, unsigned char, unsigned int, short, short, bool, unsigned int );

  void WritePaintedRegionTypePoints( std::string );

  short GetGrayscaleImageIntensity( GrayscaleImageType::IndexType );

  double GetAirwayness( GrayscaleImageType::IndexType, double );

  void Clear();

  std::vector< LabelMapType::IndexType >* GetPaintedIndices()
    {
      return &PaintedIndices;
    }

  bool SegmentLungLobes();

  bool LabelLeftLungRightLung();

  bool LabelLungThirds();

  bool SegmentAirwaysViaMinCostPaths( double, double, double );

  bool CloseLeftLungRightLung();

  /** Indicate that the scan is head first */
  void SetScanIsHeadFirst();

  /** Indicate that the scan is feet first */
  void SetScanIsFeetFirst();

  /** Indicate that the scan is a supine scan */
  void SetScanIsSupine();

  /** Indicate that the scan is a prone scan */
  void SetScanIsProne();

private:
  typedef itk::ImageRegionIteratorWithIndex< GrayscaleImageType >                  GrayscaleIteratorType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapType >                        LabelMapIteratorType;
  typedef itk::VTKImageExport< GrayscaleImageType >                                ExportType;
  typedef itk::ConnectedComponentImageFilter< LabelMapType, LabelMapType >         ConnectedComponentType;
  typedef itk::RelabelComponentImageFilter< LabelMapType, LabelMapType >           RelabelComponentType;
  typedef itk::BinaryBallStructuringElement< unsigned short, 3 >                   ElementType;
  typedef itk::BinaryDilateImageFilter< LabelMapType, LabelMapType, ElementType >  DilateType;
  typedef itk::BinaryErodeImageFilter< LabelMapType, LabelMapType, ElementType >   ErodeType;

  LabelMapType::Pointer        LabelMap;
  GrayscaleImageType::Pointer  GrayscaleImage;

  bool HeadFirst;
  bool Supine;
  bool FeetFirst;
  bool Prone;

  void CloseLabelMap( LabelMapType::Pointer, unsigned short );

  std::vector< LabelMapType::IndexType > PaintedIndices;

  void ConnectPipelines( ExportType::Pointer, vtkImageImport* );
};

#endif
