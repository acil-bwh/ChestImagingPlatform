/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMultipleReconstructionKernelsPhaseCongruency.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkMultipleReconstructionKernelsPhaseCongruency - Phase congruency based on multiple reconstruction kernels
// .SECTION Description
// vtkMultipleReconstructionKernelsPhaseCongruency multiple CT reconstruction kernels and computes a measurement of
// the signal congruency.
//
//

#ifndef __vtkMultipleReconstructionKernelsPhaseCongruency_h
#define __vtkMultipleReconstructionKernelsPhaseCongruency_h

#include "vtkThreadedImageAlgorithm.h"
#include "vtkCIPCommonConfigure.h"

class vtkImageStencilData;

// VTK6 migration note:
// - replaced super class vtkImageMultipleInputFilter with vtkThreadedImageAlgorithm

class VTK_CIP_COMMON_EXPORT vtkMultipleReconstructionKernelsPhaseCongruency : public vtkThreadedImageAlgorithm
{
public:
  static vtkMultipleReconstructionKernelsPhaseCongruency *New();
  vtkTypeMacro(vtkMultipleReconstructionKernelsPhaseCongruency, vtkThreadedImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  // Description:
  // Set a stencil to apply when blending the data.
  virtual void SetStencil(vtkImageStencilData*);
  vtkGetObjectMacro(Stencil, vtkImageStencilData);

  // Description:
  // Specify a threshold in compound mode. Pixels with opacity*alpha less
  // or equal the threshold are ignored.
  vtkSetMacro(NoiseLevel,double);
  vtkGetMacro(NoiseLevel,double);
  
  // This is called by RequestData of the superclass.
  virtual void ThreadedRequestData(vtkInformation *request,
                                   vtkInformationVector **inputVector,
                                   vtkInformationVector *outputVector,
                                   vtkImageData ***inData,
                                   vtkImageData **outData,
                                   int extent[6], int threadId) override;

protected:
  vtkMultipleReconstructionKernelsPhaseCongruency();
  ~vtkMultipleReconstructionKernelsPhaseCongruency();

  virtual int FillInputPortInformation(int, vtkInformation*) override;

  virtual int RequestInformation(vtkInformation *, vtkInformationVector**,
                                 vtkInformationVector *) override;

  // This is called by the superclass.
  virtual int RequestData(vtkInformation* request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector) override;

  // Description:                                                         
  // Keep the old signature for the sake of simplicity.
  // This is called by ThreadedRequestData of superclass.
  virtual void ThreadedExecute(vtkImageData **inDatas,
                               vtkImageData *outData,
                               int extent[6], int threadId);

  vtkImageStencilData *Stencil;
  double NoiseLevel;
private:
  vtkMultipleReconstructionKernelsPhaseCongruency(const vtkMultipleReconstructionKernelsPhaseCongruency&);  // Not implemented.
  void operator=(const vtkMultipleReconstructionKernelsPhaseCongruency&);  // Not implemented.
};

#endif

