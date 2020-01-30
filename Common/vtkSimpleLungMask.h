/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkSimpleLungMask.h,v $

=========================================================================*/
// .NAME vtkSimpleLungMask - Simple lung mask extraction method
// .SECTION Description
// vtkSimpleLungMask performs simple lung mask segmentation based on thresholding
// and morphological operations.

#ifndef __vtkSimpleLungMask_h
#define __vtkSimpleLungMask_h

#include "vtkCIPCommonConfigure.h"

#include <vtkImageAlgorithm.h>
#include <vtkVersion.h>
#include "vtkMatrix4x4.h"
#include "vtkIntArray.h"
#include "vtkShortArray.h"

class VTK_CIP_COMMON_EXPORT vtkSimpleLungMask : public vtkImageAlgorithm
{
public:
  static vtkSimpleLungMask *New();
  vtkTypeMacro(vtkSimpleLungMask,vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  // Description:
  // Get left Centroid of the lung
  //vtkGet3Macro(LCentroid, int);
  // Get right Centroid of the lung
  //vtkGet3Macro(RCentroid,int);

  vtkGetMacro(LungThreshold,int);

  vtkSetMacro(WholeLungLabel,int);
  vtkGetMacro(WholeLungLabel,int);

  vtkSetMacro(LeftLungLabel,int);
  vtkGetMacro(LeftLungLabel,int);

  vtkSetMacro(RightLungLabel,int);
  vtkGetMacro(RightLungLabel,int);

  vtkSetMacro(TracheaLabel,int);
  vtkGetMacro(TracheaLabel,int);

  vtkSetMacro(VesselsLabel,int);
  vtkGetMacro(VesselsLabel,int);

  vtkSetMacro(NumberOfErosions,int);
  vtkGetMacro(NumberOfErosions,int);

  vtkSetMacro(NumberOfDilatations,int);
  vtkGetMacro(NumberOfDilatations,int);

  vtkSetMacro(ExtractVessels,int);
  vtkGetMacro(ExtractVessels,int);
  vtkBooleanMacro(ExtractVessels,int);

  vtkSetMacro(VesselsThreshold,int);
  vtkGetMacro(VesselsThreshold,int);

  //Density Mask Get methods
  vtkGetMacro(NumVoxelWholeLung,int)
  vtkGetMacro(NumVoxelLeftLung,int);
  vtkGetMacro(NumVoxelRightLung,int);
  vtkGetMacro(NumVoxelTrachea,int);
  vtkGetObjectMacro(LeftDMTable,vtkIntArray);
  vtkGetObjectMacro(RightDMTable,vtkIntArray);

  vtkGetObjectMacro(ThresholdTable,vtkShortArray);

  vtkSetObjectMacro(RasToVtk,vtkMatrix4x4);
  vtkGetObjectMacro(RasToVtk,vtkMatrix4x4);

protected:
  vtkSimpleLungMask();
  ~vtkSimpleLungMask();

  void ExecuteDataWithInformation(vtkDataObject *, vtkInformation *) override;

  void ComputeCentroids(vtkImageData *in, int LC[3], int RC[3]);
  void ComputeCentroid(vtkImageData *in, int ext[6], int C[3]);
  vtkImageData *PreVolumeProcessing(vtkImageData *in, int &ZCentroid);
  int SliceProcessing(vtkImageData *in,vtkImageData *out, int z);
  void PostVolumeProcessing(vtkImageData *in, vtkImageData *out);
  void AppendOutput(vtkImageData *slice,int z);
  void SplitLung(vtkImageData *out);
  void ExtractTrachea(vtkImageData *in);
  void DensityMaskAnalysis();
  void ExtractUpperTrachea(vtkImageData *outData);
  void FindTracheaTopCoordinates(vtkImageData *in,int initZ, int endZ,int sign, int C[3]);
  int CountPixels(vtkImageData *in,short cc);
  void Histogram(vtkImageData *in, int *hist, int minbin, int maxbin);
  void CopyToBuffer(vtkImageData *in, vtkImageData *out, int copyext[6]);
  void ExtractTracheaOLD(vtkImageData *in);

  int LungThreshold;
  int LCentroid[3];
  int RCentroid[3];

  int NumberOfDilatations;
  int NumberOfErosions;

  short WholeLungLabel; //greater than 1
  short LeftLungLabel; //greater than 1
  short RightLungLabel; //greater than 1
  unsigned char UcharTracheaLabel;
  short TracheaLabel;
  short VesselsLabel;
  short BodyLabel;
  short AirLabel;
  short UpperTracheaLabel;

  int BaseLabelWholeLung;
  int BaseLabelLeftLung;
  int BaseLabelRightLung;

  int NumVoxelWholeLung;
  int NumVoxelLeftLung;
  int NumVoxelRightLung;
  int NumVoxelTrachea;

  int TopLungZ;
  int BottomLungZ;

  int TracheaInitZ;
  int TracheaEndZ;

  int TracheaAreaTh;

  int ExtractVessels;
  int VesselsThreshold;

  vtkShortArray *ThresholdTable;
  vtkIntArray *LeftDMTable;
  vtkIntArray *RightDMTable;

  vtkMatrix4x4 *RasToVtk;

  int AirIntensityBaseline;

private:
  vtkSimpleLungMask(const vtkSimpleLungMask&);  // Not implemented.
  void operator=(const vtkSimpleLungMask&);  // Not implemented.
};

#endif
