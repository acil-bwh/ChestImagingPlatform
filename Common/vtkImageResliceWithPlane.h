/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageResliceWithPlane.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkImageResliceWithPlane - combine images via a cookie-cutter operation
// .SECTION Description
// vtkImageResliceWithPlane performs a intensity correciton of Lung CT scans. The
// intensity correction is based on a least squares fitting of a 1st order polynomial to
// the mean intensity along the y direction. The rationale is based on the fact that
// graviational issues makes the condense water in the lungs to accumulates on certain areas
// (for example, posterior is the scan is supine) yielding a increase of expected intensity in
// those areas. This class remove the trend in the intensity profile. The DC value of the correction
// can be set to the low , high or mean intensity of the profile.

#ifndef __vtkImageResliceWithPlane_h
#define __vtkImageResliceWithPlane_h

#include "vtkImageAlgorithm.h"
#include "vtkCIPCommonConfigure.h"

#include "vtkImageReslice.h"

// VTK6 migration note:
// Replaced suplerclass vtkImageToImageFilter with vtkImageAlgorithm
// instead of vtkThreadedImageAlgorithm since this class did not provide
// ThreadedExecute() method and overrided ExecuteData() originally.

class VTK_CIP_COMMON_EXPORT vtkImageResliceWithPlane : public vtkImageAlgorithm
{
public:
  static vtkImageResliceWithPlane *New();
  vtkTypeMacro(vtkImageResliceWithPlane, vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  // Get right Centroid of the lung
  vtkSetVector3Macro(Dimensions,int);
  vtkGetVector3Macro(Dimensions,int);
  vtkGetVector3Macro(Spacing,double);
  vtkSetVector3Macro(Spacing,double);
  // Set plane center in ijk coordinate systems
  vtkSetVector3Macro(Center,double);
  vtkGetVector3Macro(Center,double);

  vtkGetVector3Macro(XAxis,double);
  vtkSetVector3Macro(XAxis,double);
  vtkGetVector3Macro(YAxis,double);
  vtkSetVector3Macro(YAxis,double);
  vtkGetVector3Macro(ZAxis,double);
  vtkSetVector3Macro(ZAxis,double);

  // Description
  // Automatic computation of reslice axes and reslice center
  // based on Tube model using Hessian information.
  vtkGetMacro(ComputeAxes,int);
  vtkSetMacro(ComputeAxes,int);
  vtkBooleanMacro(ComputeAxes,int);
  vtkGetMacro(ComputeCenter,int);
  vtkSetMacro(ComputeCenter,int);
  vtkBooleanMacro(ComputeCenter,int);

  // Description:
  // Define reslice axes from Hessian (InPlaneOff) or in the
  // ij direction (InPlaneOn)
  vtkBooleanMacro(InPlane,int);
  vtkGetMacro(InPlane,int);
  vtkSetMacro(InPlane,int);

  void SetInterpolationModeToNearestNeighbor() {
    this->SetInterpolationMode(VTK_RESLICE_NEAREST);
  }

  void SetInterpolationModeToLinear() {
    this->SetInterpolationMode(VTK_RESLICE_LINEAR);
  }

  void SetInterpolationModeToCubic() {
    this->SetInterpolationMode(VTK_RESLICE_CUBIC);
  }

  vtkSetMacro(InterpolationMode,int);
  vtkGetMacro(InterpolationMode,int);

  void ComputeAxesAndCenterUsingTubeModel();

protected:
  vtkImageResliceWithPlane();
  ~vtkImageResliceWithPlane();

  // Description:
  // This is a convenience method that is implemented in many subclasses
  // instead of RequestData.  It is called by RequestData.
  virtual void ExecuteDataWithInformation(vtkDataObject *output,
                                          vtkInformation* outInfo) override;

  virtual int RequestInformation(vtkInformation *, vtkInformationVector**,
                                 vtkInformationVector *) override;

  int ComputeAxes;
  int ComputeCenter;
  int InPlane;

  double Center[3];
  double XAxis[3];
  double YAxis[3];
  double ZAxis[3];
  int Dimensions[3];
  double Spacing[3];

  int InterpolationMode;

  vtkImageReslice *Reslice;

private:
  vtkImageResliceWithPlane(const vtkImageResliceWithPlane&);  // Not implemented.
  void operator=(const vtkImageResliceWithPlane&);  // Not implemented.
};

#endif
