/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkGeneralizedPhaseCongruency.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkGeneralizedPhaseCongruency - compute the Monogenic signal
// .SECTION Description
// vtkGeneralizedPhaseCongruency computes the Monogenic signal of an image.
// The output of the filter is the monogenic signal in the spatial domain.
// The computation is performed in the Fourier domain.

#ifndef __vtkGeneralizedPhaseCongruency_h
#define __vtkGeneralizedPhaseCongruency_h

#include "vtkImageAlgorithm.h"
#include "vtkCIPCommonConfigure.h"

#include "vtkImageKernelSource.h"
#include "vtkDoubleArray.h"

// VTK6 migration note:
// Replaced suplerclass vtkImageToImageFilter with vtkImageAlgorithm
// instead of vtkThreadedImageAlgorithm since this class did not provide
// ThreadedExecute() method and overrided ExecuteData() originally.

class VTK_CIP_COMMON_EXPORT vtkGeneralizedPhaseCongruency : public vtkImageAlgorithm
{
public:
  static vtkGeneralizedPhaseCongruency *New();
  vtkTypeMacro(vtkGeneralizedPhaseCongruency, vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  vtkSetMacro(NumberOfScales,int);
  vtkGetMacro(NumberOfScales,int);

  vtkSetMacro(MinimumWavelength,double);
  vtkGetMacro(MinimumWavelength,double);

  vtkSetMacro(MultiplicativeFactor,double);
  vtkGetMacro(MultiplicativeFactor,double);

  // Set/Get relative bandwidht
  vtkSetMacro(RelativeBandwidth,double);
  vtkGetMacro(RelativeBandwidth,double);

  // Set/Get quadrature filter bandwidth (FWHM) in octave
  void SetBandwidth (double b);
  double GetBandwidth () {
     double rb = this->GetRelativeBandwidth();
     return -2.0*sqrt(2.0)/sqrt(log(2.0)) * log(rb);
  };

  vtkSetMacro(UseWeights,int);
  vtkGetMacro(UseWeights,int);
  vtkBooleanMacro(UseWeights,int);

  vtkSetObjectMacro(Weights,vtkDoubleArray);
  vtkGetObjectMacro(Weights,vtkDoubleArray);

  vtkSetMacro(UsePhysicalUnits,int);
  vtkGetMacro(UsePhysicalUnits,int);
  vtkBooleanMacro(UsePhysicalUnits,int);

protected:
  vtkGeneralizedPhaseCongruency();
  ~vtkGeneralizedPhaseCongruency();

  // Description:
  // This is a convenience method that is implemented in many subclasses
  // instead of RequestData.  It is called by RequestData.
  virtual void ExecuteDataWithInformation(vtkDataObject *output,
                                          vtkInformation* outInfo) override;
                                          
  virtual int RequestInformation(vtkInformation *, vtkInformationVector**,
                                 vtkInformationVector *) override;

  void FillOutput(vtkImageData *out, vtkImageData *in, int comp);

  int NumberOfScales;
  double MinimumWavelength;
  double MultiplicativeFactor;
  double RelativeBandwidth;
  int UsePhysicalUnits;

  int UseWeights;
  vtkDoubleArray *Weights;

private:
  vtkGeneralizedPhaseCongruency(const vtkGeneralizedPhaseCongruency&);  // Not implemented.
  void operator=(const vtkGeneralizedPhaseCongruency&);  // Not implemented.
};

#endif

