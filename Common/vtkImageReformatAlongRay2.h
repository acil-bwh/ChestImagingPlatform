/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageReformatAlongRay2.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkImageReformatAlongRay2 - compute the Monogenic signal
// .SECTION Description
// vtkImageReformatAlongRay2 computes the Monogenic signal of an image.
// The output of the filter is the monogenic signal in the spatial domain.
// The computation is performed in the Fourier domain.

#ifndef __vtkImageReformatAlongRay2_h
#define __vtkImageReformatAlongRay2_h

#include "vtkImageAlgorithm.h"
#include "vtkCIPCommonConfigure.h"

#include "vtkImageKernelSource.h"
#include "teem/nrrd.h"
#include "teem/gage.h"

#define NORMAL 1
#define TANGENT 2

// VTK6 migration note:
// Replaced suplerclass vtkImageToImageFilter with vtkImageAlgorithm
// instead of vtkThreadedImageAlgorithm since this class did not provide
// ThreadedExecute() method and overrided ExecuteData() originally.

class VTK_CIP_COMMON_EXPORT vtkImageReformatAlongRay2 : public vtkImageAlgorithm
{
public:
  static vtkImageReformatAlongRay2 *New();
  vtkTypeMacro(vtkImageReformatAlongRay2, vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  //Description:
  //Tanget of the ray
  vtkSetVectorMacro(Tangent,double,3);
  vtkGetVectorMacro(Tangent,double,3);

  //Description:
  //Normal of the ray
  vtkSetVectorMacro(Normal,double,3);
  vtkGetVectorMacro(Normal,double,3);

  //Description:
  // Angle of the ray in the normal plane
  vtkSetMacro(Theta,double);
  vtkGetMacro(Theta,double);

  vtkSetMacro(Mode,int);
  vtkGetMacro(Mode,int);

  void SetModeToNormal() {this->Mode=NORMAL;};
  void SetModeToTangent() {this->Mode=TANGENT;};

  // Description:
  // Minimum radius in mm
  vtkSetMacro(RMin,double);
  vtkGetMacro(RMin,double);

  // Description:
  // Maximum radius in mm
  vtkSetMacro(RMax,double);
  vtkGetMacro(RMax,double);

  // Description:
  // Center for the ray tracing (in ijk space)
  vtkSetVectorMacro(Center,double,3);
  vtkGetVectorMacro(Center,double,3);

  // Description:
  // Spacing along ray in mm
  vtkSetMacro(Spacing,double);
  vtkGetMacro(Spacing,double);

  // Description:
  // Scale for derivative computations
  vtkSetMacro(Scale,double);
  vtkGetMacro(Scale,double);

protected:
  vtkImageReformatAlongRay2();
  ~vtkImageReformatAlongRay2();

  // Description:
  // This is a convenience method that is implemented in many subclasses
  // instead of RequestData.  It is called by RequestData.
  virtual void ExecuteDataWithInformation(vtkDataObject *output,
                                          vtkInformation* outInfo) override;
                                          
  virtual int RequestInformation(vtkInformation *, vtkInformationVector**,
                                 vtkInformationVector *) override;
  void ComputeTangentFromNormal();

  double Theta;
  double RMin;
  double RMax;
  int VTKToNrrdPixelType( const int vtkPixelType );

  double Center[3];

  double Tangent[3];
  double Normal[3];

  double Spacing;

  double Scale;

  int Mode;

private:
  vtkImageReformatAlongRay2(const vtkImageReformatAlongRay2&);  // Not implemented.
  void operator=(const vtkImageReformatAlongRay2&);  // Not implemented.
  gageContext *gtx;
  Nrrd *nin;
};

#endif

