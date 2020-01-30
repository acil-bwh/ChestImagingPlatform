/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkTubularScaleSelection.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkTubularScaleSelection - Computes a confidence map for tubular objects
// in an image (either valley lines like airways or ridge lines like vessels).
// .SECTION Description
// vtkTubularScaleSelection calculates a confidence map bounded between 0 and 1 for
// the chances of a given voxel in the image grid belongs to a tubular object (either valley
// line or a ridge line).

#ifndef __vtkTubularScaleSelection_h
#define __vtkTubularScaleSelection_h

#include "vtkCIPCommonConfigure.h"

#include "vtkThreadedImageAlgorithm.h"
#include "vtkImageData.h"
#include "teem/gage.h"

#define VTK_VALLEY 1
#define VTK_RIDGE 2

// VTK6 migration note:
// Replaced suplerclass vtkImageToImageFilter with vtkThreadedImageAlgorithm.

class VTK_CIP_COMMON_EXPORT vtkTubularScaleSelection : public vtkThreadedImageAlgorithm
{
public:
  static vtkTubularScaleSelection *New();
  vtkTypeMacro(vtkTubularScaleSelection, vtkThreadedImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;
  // Description: Mask (short image) where the tubular confidence map will be computed.
  vtkSetObjectMacro(Mask,vtkImageData);
  vtkGetObjectMacro(Mask,vtkImageData);

 // Description: Type of tubular structure that we want to capture. Valleys are dark tubes (i.e. airways). Ridges are bright tubes (i.e. vessels)
  vtkGetMacro(TubularType,int);
  vtkSetMacro(TubularType,int);
  void SetTubularTypeToValley() {
    this->SetTubularType(VTK_VALLEY);};
  void SetTubularTypeToRidge() {
    this->SetTubularType(VTK_RIDGE);};

  // Description: Scale at which the confidence measurement will be computed. This value is used if OptimizeScale is off.
   vtkGetMacro(StepScale,double);
   vtkSetMacro(StepScale,double);

  // Description: Scale at which the confidence measurement will be computed. This value is used if OptimizeScale is off.
   vtkGetMacro(InitialScale,double);
   vtkSetMacro(InitialScale,double);

  // Description: Scale at which the confidence measurement will be computed. This value is used if OptimizeScale is off.
   vtkGetMacro(FinalScale,double);
   vtkSetMacro(FinalScale,double);

  double ScaleSelection(gageContext *gtx, gagePerVolume *pvl,
                        double Seed[3], double initS, double maxS, double deltaS);
  double ScaleSelection(gageContext **gtx, gagePerVolume **pvl,
                        double Seed[3], double initS, double maxS, double deltaS);
  int SettingContext(gageContext *gtx,gagePerVolume *pvl,double scale);
  double Mode(const double *w);
  double Strength(const double *w);

protected:

  vtkTubularScaleSelection();
  ~vtkTubularScaleSelection();

  virtual int RequestInformation (vtkInformation *, vtkInformationVector**,
                                  vtkInformationVector *) override;

  void ThreadedExecute (vtkImageData *inData, vtkImageData *outData,
                        int outExt[6], int id) override;

  int TubularType;
  vtkImageData *Mask;
  double InitialScale;
  double FinalScale;
  double StepScale;

private:
  vtkTubularScaleSelection(const vtkTubularScaleSelection&);  // Not implemented.
  void operator=(const vtkTubularScaleSelection&);  // Not implemented.
};

#endif

