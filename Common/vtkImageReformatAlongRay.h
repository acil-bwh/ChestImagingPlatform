/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageReformatAlongRay.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkImageReformatAlongRay - compute the Monogenic signal
// .SECTION Description
// vtkImageReformatAlongRay computes the Monogenic signal of an image.
// The output of the filter is the monogenic signal in the spatial domain.
// The computation is performed in the Fourier domain.

#ifndef __vtkImageReformatAlongRay_h
#define __vtkImageReformatAlongRay_h

#include "vtkImageAlgorithm.h"
#include "vtkCIPCommonConfigure.h"

#include "vtkImageKernelSource.h"
#include "teem/nrrd.h"
#include "teem/gage.h"

// VTK6 migration note:
// Replaced suplerclass vtkImageToImageFilter with vtkImageAlgorithm
// instead of vtkThreadedImageAlgorithm since this class did not provide
// ThreadedExecute() method and overrided ExecuteData() originally.

class VTK_CIP_COMMON_EXPORT vtkImageReformatAlongRay : public vtkImageAlgorithm
{
public:
  static vtkImageReformatAlongRay *New();
  vtkTypeMacro(vtkImageReformatAlongRay, vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  vtkSetMacro(Theta,double);
  vtkGetMacro(Theta,double);

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
  // Spacing along ray in pixel units
  vtkSetMacro(Delta,double);
  vtkGetMacro(Delta,double);

  // Description:
  // Scale for derivative computations
  vtkSetMacro(Scale,double);
  vtkGetMacro(Scale,double);

protected:
  vtkImageReformatAlongRay();
  ~vtkImageReformatAlongRay();

  // Description:
  // This is a convenience method that is implemented in many subclasses
  // instead of RequestData.  It is called by RequestData.
  virtual void ExecuteDataWithInformation(vtkDataObject *output,
                                          vtkInformation* outInfo) override;

  virtual int RequestInformation(vtkInformation *, vtkInformationVector**,
                                 vtkInformationVector *) override;

  virtual int RequestUpdateExtent(vtkInformation *, vtkInformationVector **,
                                  vtkInformationVector *) override;

  double Theta;
  double RMin;
  double RMax;
  int VTKToNrrdPixelType( const int vtkPixelType );

  double Center[3];

  double Delta;

  double Scale;

private:
  vtkImageReformatAlongRay(const vtkImageReformatAlongRay&);  // Not implemented.
  void operator=(const vtkImageReformatAlongRay&);  // Not implemented.
  gageContext *gtx;
  Nrrd *nin;
};

#endif

