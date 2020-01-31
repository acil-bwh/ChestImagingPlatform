/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageTubularConfidence.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkImageTubularConfidence - Computes a confidence map for tubular objects
// in an image (either valley lines like airways or ridge lines like vessels).
// .SECTION Description
// vtkImageTubularConfidence calculates a confidence map bounded between 0 and 1 for
// the chances of a given voxel in the image grid belongs to a tubular object (either valley
// line or a ridge line).

#ifndef __vtkImageTubularConfidence_h
#define __vtkImageTubularConfidence_h

#include "vtkCIPCommonConfigure.h"

#include "vtkThreadedImageAlgorithm.h"
#include "vtkImageData.h"
#include "teem/gage.h"

#define VTK_VALLEY 1
#define VTK_RIDGE 2

// VTK6 migration note:
// Replaced suplerclass vtkImageToImageFilter with vtkThreadedImageAlgorithm.

class VTK_CIP_COMMON_EXPORT vtkImageTubularConfidence : public vtkThreadedImageAlgorithm
{
public:
  static vtkImageTubularConfidence *New();
  vtkTypeMacro(vtkImageTubularConfidence, vtkThreadedImageAlgorithm);
  // Description: Mask (short image) where the tubular confidence map will be computed.
  vtkSetObjectMacro(Mask,vtkImageData);
  vtkGetObjectMacro(Mask,vtkImageData);

  // Description: Mask (short image) where the tubular confidence map will be computed.
  vtkSetObjectMacro(ScaleImage,vtkImageData);
  vtkGetObjectMacro(ScaleImage,vtkImageData);

 // Description: Type of tubular structure that we want to capture. Valleys are dark tubes (i.e. airways). Ridges are bright tubes (i.e. vessels)
  vtkGetMacro(TubularType,int);
  vtkSetMacro(TubularType,int);
  void SetTubularTypeToValley() {
    this->SetTubularType(VTK_VALLEY);};
  void SetTubularTypeToRidge() {
    this->SetTubularType(VTK_RIDGE);};

  // Description: Use scale optimization. If OptimizeScale is on, we will search for the optimal scale before computing the confidence measurement per voxel.
   vtkGetMacro(OptimizeScale,int);
   vtkSetMacro(OptimizeScale,int);
   vtkBooleanMacro(OptimizeScale,int);

  // Description: Scale at which the confidence measurement will be computed. This value is used if OptimizeScale is off.
   vtkGetMacro(Scale,double);
   vtkSetMacro(Scale,double);

  // Description: Number of steps that to take along the airway path to define the angular measurement.
  vtkGetMacro(NumberOfSteps,int);
  vtkSetMacro(NumberOfSteps,int);

  // Description: Size of each step (in pixel units).
  vtkGetMacro(StepSize, double);
  vtkSetMacro(StepSize, double);

  vtkGetMacro(ModeThreshold, double);
  vtkSetMacro(ModeThreshold, double);

  double ValleyConfidenceMeasurement(const double *heval);
  double RidgeConfidenceMeasurement(const double *heval);

  double DirectionalConfidence(gageContext *gtx, gagePerVolume *pvl, double xyz[3]);

  int SettingContext(gageContext *gtx,gagePerVolume *pvl,double scale);

protected:

  vtkImageTubularConfidence();
  ~vtkImageTubularConfidence();

  virtual int RequestInformation(vtkInformation *, vtkInformationVector**,
                                 vtkInformationVector *) override;
  void ThreadedExecute(vtkImageData *inData, vtkImageData *outData,
                       int extent[6], int id) override;
                       
  int TubularType;
  vtkImageData *Mask;
  vtkImageData *ScaleImage;
  int OptimizeScale;
  double Scale;
  int NumberOfSteps;
  double StepSize;
  double ModeThreshold;

private:
  vtkImageTubularConfidence(const vtkImageTubularConfidence&);  // Not implemented.
  void operator=(const vtkImageTubularConfidence&);  // Not implemented.
};

#endif

