/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkComputeMonogenicSignal.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkComputeMonogenicSignal - compute the Monogenic signal
// .SECTION Description
// vtkComputeMonogenicSignal computes the Monogenic signal of an image.
// The output of the filter is the monogenic signal in the spatial domain.
// The computation is performed in the Fourier domain.

#ifndef __vtkComputeMonogenicSignal_h
#define __vtkComputeMonogenicSignal_h

#include "vtkImageAlgorithm.h"
#include "vtkCIPCommonConfigure.h"

#include "vtkGeneralizedQuadratureKernelSource.h"

// VTK6 migration note:
// Replaced suplerclass vtkImageToImageFilter with vtkImageAlgorithm
// instead of vtkThreadedImageAlgorithm since this class did not provide
// ThreadedExecute() method and overrided ExecuteData() originally.

class VTK_CIP_COMMON_EXPORT vtkComputeMonogenicSignal : public vtkImageAlgorithm
{
public:
  static vtkComputeMonogenicSignal *New();
  vtkTypeMacro(vtkComputeMonogenicSignal, vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  vtkSetObjectMacro(QuadratureFilter,vtkGeneralizedQuadratureKernelSource);
  vtkGetObjectMacro(QuadratureFilter,vtkGeneralizedQuadratureKernelSource);

  vtkSetMacro(InputInFourierDomain,int);
  vtkGetMacro(InputInFourierDomain,int);
  vtkBooleanMacro(InputInFourierDomain,int);

  // Description:
  // Get the dimensionality of the input signal. This is a variable set at
  // running time. The monogenic signal will be a vector field (as a multiple
  // components scalar vtkImageData) whose dimension is equal to
  //  Dimensionality+1.
  vtkGetMacro(Dimensionality,int);

protected:
  vtkComputeMonogenicSignal();
  ~vtkComputeMonogenicSignal();

  // VTK6 migration note:
  // Use RequestData instead of ExecuteDataWithInformation when input data is needed
  virtual int RequestData(vtkInformation* request, 
                          vtkInformationVector** inputVector, 
                          vtkInformationVector* outputVector) override;
                                          
  virtual int RequestInformation(vtkInformation *, vtkInformationVector**,
                                 vtkInformationVector *) override;

  void FillOutput(vtkImageData *out, vtkImageData *in, int comp);

  int InputInFourierDomain;
  vtkGeneralizedQuadratureKernelSource* QuadratureFilter;
  int Dimensionality;

private:
  vtkComputeMonogenicSignal(const vtkComputeMonogenicSignal&);  // Not implemented.
  void operator=(const vtkComputeMonogenicSignal&);  // Not implemented.
};

#endif

