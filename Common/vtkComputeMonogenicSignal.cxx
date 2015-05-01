/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkComputeMonogenicSignal.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkComputeMonogenicSignal.h"

#include "vtkImageData.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"

#include "vtkImageFFT.h"
#include "vtkImageRFFT.h"
#include "vtkImageExtractComponents.h"
#include "vtkImageMathematics.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include <math.h>

vtkStandardNewMacro(vtkComputeMonogenicSignal);

//----------------------------------------------------------------------------
vtkComputeMonogenicSignal::vtkComputeMonogenicSignal()
{

  this->InputInFourierDomain = 1;
  this->QuadratureFilter = NULL;
  this->Dimensionality = 3;
}

//----------------------------------------------------------------------------
vtkComputeMonogenicSignal::~vtkComputeMonogenicSignal()
{
if (this->QuadratureFilter != NULL)
    this->QuadratureFilter->Delete();
}

// ---------------------------------------------------------------------------
// VTK6 migration note:
// - introduced this method combining ExecuteInformation() and
//   ExecuteInformation(vtkImageData*, vtkImageData*) 
// - before migration ExecuteInformation() called
//   vtkImageToImageFilter::ExecuteInformation() where it called the latter
//   (overrided) version

int vtkComputeMonogenicSignal::RequestInformation (
  vtkInformation       *  request,
  vtkInformationVector ** inputVector,
  vtkInformationVector *  outputVector)
{
  this->Superclass::RequestInformation(request, inputVector, outputVector);

  vtkInformation* inInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation* outInfo = outputVector->GetInformationObject(0);

  // Compute Dimensionality
  int ext[6];
  inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), ext);
  
  if (ext[5] == ext[4]) {
    if (ext[3] == ext[2]) {
      this->Dimensionality = 1;
    } else {
      this->Dimensionality = 2;
    }    
  } else {
     this->Dimensionality = 3;
  }    

  // Num of output components = signal dimensionality dimensions + 1
  vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_DOUBLE, this->Dimensionality+1);
  
  return 1;
}  
  
//----------------------------------------------------------------------------
// VTK6 migration note:
// - introduced to replace ExecuteData()
// - changed this->GetInput() to vtkImageData::GetData(inInfoVec[0])
// - changed this->GetOutput() to vtkPolyData::GetData(outInfoVec)
int vtkComputeMonogenicSignal::RequestData(vtkInformation *request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  vtkImageData* inData = vtkImageData::GetData(inputVector[0]);
  vtkImageData* outData = vtkImageData::GetData(outputVector);
  vtkInformation* inInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation* outInfo = outputVector->GetInformationObject(0);

  // Make sure the Input has been set.
  if (inData == NULL)
    {
    vtkErrorMacro(<< "RequestData: input is not set.");
    return 0;
    }
  
  // Too many filters have floating point exceptions to execute
  // with empty input/ no request.
  if (this->UpdateExtentIsEmpty(outInfo, outData))
    {
    vtkErrorMacro(<< "RequestData: update extent is empty");
    return 0;
    }

  if (this->QuadratureFilter == NULL)
    {
    vtkErrorMacro(<< "RequestData: a quadrature kernel has to be set");
    return 0;
    }

  int updateExtent[6];
  outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),
               updateExtent);
  this->AllocateOutputData(outData, outInfo, updateExtent);
  
  vtkImageData *fftInput;
  vtkImageFFT *fft = vtkImageFFT::New();
  fft->SetDimensionality(this->Dimensionality);

  if (this->InputInFourierDomain == 0) {
    vtkImageFFT *fft = vtkImageFFT::New();
    fft->SetInputData(inData);
    fft->Update();
    fftInput = fft->GetOutput();
  } else {
    fftInput = inData;
  }
  
  int fftWholeExt[6];
  inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), fftWholeExt);

  //Set up quadrature filter dimensions
  this->QuadratureFilter->SetWholeExtent(fftWholeExt);
  this->QuadratureFilter->SetVoxelSpacing(fftInput->GetSpacing());
  this->QuadratureFilter->SetZeroFrequencyLocationToOrigin();
  this->QuadratureFilter->Update();

  //Compute each monogenic signal component
  vtkDebugMacro("Compute Monogenic Signal");
  for (int k=0; k < (this->Dimensionality+1); k++) {
    vtkImageRFFT *rfft = vtkImageRFFT::New();
    rfft->SetDimensionality(this->Dimensionality);
    vtkImageMathematics *mult = vtkImageMathematics::New();
    vtkImageExtractComponents *quadComp = vtkImageExtractComponents::New();

    // SetUp Filter
    quadComp->SetInputConnection(this->QuadratureFilter->GetOutputPort());
    quadComp->SetComponents(2*k,2*k+1);
    quadComp->Update();

    mult->SetOperationToComplexMultiply();
    mult->SetInput1Data(fftInput);
    mult->SetInput2Data(quadComp->GetOutput());
    mult->Update();
    
    rfft->SetInputConnection(mult->GetOutputPort());
    rfft->Update();
    
    this->FillOutput(this->GetOutput(),rfft->GetOutput(),k);
    mult->Delete();
    quadComp->Delete();
    rfft->Delete();
  }

  //Delete Objects
  fft->Delete();

  return 1;
}


void vtkComputeMonogenicSignal::FillOutput(vtkImageData *out, vtkImageData *in, int comp) {

  double *outPtr = (double *) out->GetScalarPointer();
  double *inPtr = (double *) in->GetScalarPointer();

  outPtr = outPtr+comp;
  for (int k=0; k<out->GetNumberOfPoints(); k++) {
    //For a generalized quadrature filter response, either the real part or the
    // imaginary part is zero, so we sum both.
    *outPtr= *inPtr + *(inPtr+1);
    outPtr = outPtr + (this->Dimensionality+1);
    inPtr = inPtr + 2;
  }
}

void vtkComputeMonogenicSignal::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Input in Fourer Domain" << this->InputInFourierDomain << "\n";
}

