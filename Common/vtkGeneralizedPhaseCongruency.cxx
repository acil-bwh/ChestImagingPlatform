/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkGeneralizedPhaseCongruency.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkGeneralizedPhaseCongruency.h"

#include "vtkImageData.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"

#include "vtkImageFFT.h"
#include "vtkGeneralizedQuadratureKernelSource.h"
#include "vtkImageExtractComponents.h"
#include "vtkComputeMonogenicSignal.h"
#include "vtkDoubleArray.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include <math.h>

#define VTK_EPS 1e-15


vtkStandardNewMacro(vtkGeneralizedPhaseCongruency);

//----------------------------------------------------------------------------
vtkGeneralizedPhaseCongruency::vtkGeneralizedPhaseCongruency()
{

  this->NumberOfScales = 4;
  this->MinimumWavelength = 4;
  this->MultiplicativeFactor = 1.3;
  this->RelativeBandwidth = 0.65;
  this->UsePhysicalUnits = 0;
  this->UseWeights = 0;
  this->Weights = vtkDoubleArray::New();

}

//----------------------------------------------------------------------------
vtkGeneralizedPhaseCongruency::~vtkGeneralizedPhaseCongruency()
{

  this->Weights->Delete();

}

//----------------------------------------------------------------------------
void vtkGeneralizedPhaseCongruency::SetBandwidth (double b){
     this->SetRelativeBandwidth(exp(-0.25 * sqrt(2.0*log(2.0)) * b));
}

// ---------------------------------------------------------------------------
// VTK6 migration note:
// - introduced this method combining ExecuteInformation() and
//   ExecuteInformation(vtkImageData*, vtkImageData*) 
// - before migration ExecuteInformation() called
//   vtkImageToImageFilter::ExecuteInformation() where it called the latter
//   (overrided) version

int vtkGeneralizedPhaseCongruency::RequestInformation (
  vtkInformation       *  request,
  vtkInformationVector ** inputVector,
  vtkInformationVector *  outputVector)
{
  vtkInformation* outInfo = outputVector->GetInformationObject(0);
  
  // Num of output components
  // 0: Phase congruency
  // 1: Phase congruency for edge up
  // 2: Phase congruency for edge down
  // Not implemented yet
  // 3: Phase congruency for peak points
  // 4: Phase congruency for valley points
  
  vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_DOUBLE, 3);
  
  return 1;
}

//----------------------------------------------------------------------------
// VTK6 migration note:
// - introduced to replace ExecuteData()
void vtkGeneralizedPhaseCongruency::ExecuteDataWithInformation(vtkDataObject* out,
  vtkInformation* outInfo)
{
  vtkImageData* input = vtkImageData::SafeDownCast(this->GetInput());

  // Make sure the Input has been set.
  if ( input == NULL )
    {
    vtkErrorMacro(<< "ExecuteData: Input is not set.");
    return;
    }
    
  // Too many filters have floating point exceptions to execute
  // with empty input/ no request.
  if (this->UpdateExtentIsEmpty(outInfo, out))
    {
    return;
    }


  if (this->UseWeights) {
    if (this->Weights->GetNumberOfTuples() != this->NumberOfScales) {
      vtkErrorMacro(<<"Weights array number of tuples should be the same as number of scales");
      return;
    }
  }

  vtkImageData *outData = this->AllocateOutputData(out, outInfo);

  vtkImageFFT *fft = vtkImageFFT::New();
 
  fft->SetInputData(input);
  fft->Update();

  if (!this->GetUsePhysicalUnits()) {
    fft->GetOutput()->SetSpacing(1,1,1);
  }

  //Set monogenic signal to get dimensionality of the output
  vtkComputeMonogenicSignal *mono = vtkComputeMonogenicSignal::New();
  mono->SetInputData(fft->GetOutput());
  mono->InputInFourierDomainOn();
  mono->UpdateInformation();
  int dimensionality;
  dimensionality = mono->GetDimensionality();
  mono->Delete();
  
  //cout<<"Number of input points"<<this->GetInput()->GetNumberOfPoints()<<endl;
  //Allocate working DataArrays
  // F: Mean monogenic signal across scales
  vtkDoubleArray *F = vtkDoubleArray::New();
  F->SetNumberOfComponents(dimensionality+1);
  F->SetNumberOfTuples(input->GetNumberOfPoints());
  double *fPtr = (double *) F->GetVoidPointer(0);
  // A: Mean norm of the monogenic signal across scales
  vtkDoubleArray *A = vtkDoubleArray::New();
  A->SetNumberOfComponents(1);
  A->SetNumberOfTuples(input->GetNumberOfPoints());
  double *aPtr = (double *) A->GetVoidPointer(0);

  // Init arrays
  for (int ii = 0 ; ii < F->GetNumberOfTuples() ; ii++) {
    for (int k=0; k<dimensionality+1;k++) {
      *fPtr = 0.0; fPtr++;
    }
    *aPtr = 0.0;
    aPtr++;
  }

  // Loop through scale space and fill F and A arrays
  double freq,tmp;
  double *monoPtr;
  double wavelength = this->MinimumWavelength;
  double weight = 1;
  for (int k = 0 ; k < this->NumberOfScales; k++) {
    freq = 2*vtkMath::Pi()/wavelength;
    vtkDebugMacro("Scale: "<<k<<" Frequency: "<<freq);

    if (this->UseWeights) {
      weight = this->Weights->GetValue(k);
      if (weight == 0) {
        vtkDebugMacro("Skipping scale. Weight is zero");
        continue;
      }
    } else {
      weight = 1;
    }

    vtkGeneralizedQuadratureKernelSource *quad = vtkGeneralizedQuadratureKernelSource::New();
    vtkComputeMonogenicSignal *mono = vtkComputeMonogenicSignal::New();
 
    // Set params of Quadrature kernel that we know
    quad->SetRelativeBandwidth(this->RelativeBandwidth);
    quad->SetZeroFrequencyLocationToOrigin();
    quad->SetCenterFrequency(freq);

    mono->SetInputData(fft->GetOutput());
    mono->InputInFourierDomainOn();
    mono->SetQuadratureFilter(quad);
    mono->Update();
    //dimensionality = mono->GetDimensionality();
    wavelength *= this->MultiplicativeFactor;
    //if (k == 0) 
    //  this->FillOutput(this->GetOutput(),mono->GetOutput(),1);
    fPtr = (double *) F->GetVoidPointer(0);
    aPtr = (double *) A->GetVoidPointer(0);

    //monoPtr = (double *) mono->GetOutput()->GetPointData()->GetScalars()->GetVoidPointer(0);
    monoPtr = (double *) mono->GetOutput()->GetScalarPointer();

    for (int ii = 0 ; ii < F->GetNumberOfTuples() ; ii++) {
         tmp = 0.0;
         for (int comp = 0 ; comp < dimensionality+1 ; comp++) {
            *fPtr = *fPtr +  (*monoPtr) * weight;
            tmp = tmp + *(monoPtr) * *(monoPtr) * weight * weight;
            fPtr++;
            monoPtr++;
        }
        *aPtr += sqrt(tmp);
        aPtr++;
    }
   mono->Delete();
   quad->Delete();
  }

  // Go into the real computation of phase congruency
  vtkDebugMacro("Computation of phase congruency");
  fPtr = (double *) F->GetVoidPointer(0);
  aPtr = (double *) A->GetVoidPointer(0);

  double *outPtr = (double *) outData->GetScalarPointer();
  double PC, PC1, PC2, PC3, PC4, phase;
  double modfR,modF, f, mask,norm;
  int signFx;
  int ijk[3];
  double v[3];
  int dims[3];
  input->GetDimensions(dims);
  double cx = dims[0]/2;
  double cy = dims[1]/2;
  double cz = dims[2]/2; 

  for (int ii = 0; ii < F->GetNumberOfTuples() ; ii++) {
    modfR = modF= 0;
    tmp = 0;
    mask = 0;
  // compute outwards flow to do phase unwrapping
  //vtkImageData *in =this->GetInput();
  //in->ComputeStructuredCoordinates(in->FindPoint(ii),ijk,pcoord);
  //this->GetPositionFromIndex(ii,&x,&y,&z);
  ijk[0] = ii % dims[0];
  ijk[1] = (ii / dims[0]) % dims[1];
  ijk[2] = ii / (dims[0]*dims[1]);
  v[0] = (ijk[0]-cx);
  v[1] = (ijk[1]-cy);
  v[2] = (ijk[2]-cz);
  norm = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
  if (norm > VTK_EPS) {
    v[0] = v[0]/norm;
    v[1] = v[1]/norm;
    v[2] = v[2]/norm;
  } else {
    v[0] = 0;
    v[1] = 0;
    v[2] = 0;
  }
  
  //Grab sign of Fx (first component monogenic signal)
  //to fix phase wrapping in edge structures.
  //This will make Generalized PC works fine with 1D signals.
  if (*fPtr>=0)
     signFx = 1;
  else
     signFx = -1; 

   for (int comp = 0; comp < (dimensionality+1) ;  comp++) {
       tmp = *fPtr * *fPtr; 
       modF += tmp;
       if (comp < (dimensionality)) {
         modfR += tmp;
        mask += v[comp] * *fPtr; 
       } else {
         // Last component is the bandpass version of the input signal.
         f = *fPtr;
       }
       fPtr++;
    }
   modF = sqrt(modF);
   modfR = sqrt(modfR);

   if (mask >= 0)
    mask = 1;
   else
    mask = -1;

   // Compute phase
   
   /**
   if (f > VTK_EPS) {
        //phase = atan(modfR/f)*mask;
        phase = atan2(-signFx*modfR,f);
   } else {
        if (modfR > 0)
          {
            //phase = vtkMath::Pi() / 2 * mask;
            phase = vtkMath::Pi() /2;
          } 
       else
          {
            //phase = -vtkMath::Pi() / 2 * mask;
            phase = -vtkMath::Pi() / 2;
          }
   }
   **/
   
   phase = atan2(-signFx*modfR,f);
   
   //PC
   if ((*aPtr) > VTK_EPS)
        PC = modF / (*aPtr);
   else
        PC = 0;
        
   double sinph = sin(phase);
   if (sinph > 0 ) {
    PC3 = PC * sinph;
   } else {
    PC3 = 0; 
   }
   //PC3 = (-1* PC * sin(phase) + 1) /2;
   if (-1.0*sinph > 0 ) {
    PC4 = PC * (-1.0*sinph);
   } else {
    PC4 = 0;
   }
   //PC4 = (PC * sin(phase) + 1)/2; 
   *outPtr = PC; outPtr++;
   //*outPtr = sin(phase); outPtr++;
   *outPtr = PC3; outPtr++;
   *outPtr = PC4; outPtr++;

   aPtr++;

}


  //Delete Objects
  fft->Delete();
  F->Delete();
  A->Delete();
  //mono->Delete();
  //quad->Delete();

}


void vtkGeneralizedPhaseCongruency::FillOutput(vtkImageData *out, vtkImageData *in, int comp) {

  double *outPtr = (double *) out->GetScalarPointer();
  double *inPtr = (double *) in->GetScalarPointer();

  outPtr = outPtr+comp;
  inPtr = inPtr+comp;
  for (int k=0; k<out->GetNumberOfPoints(); k++) {
    *outPtr= *inPtr;
    outPtr = outPtr + out->GetNumberOfScalarComponents();
    inPtr = inPtr + in->GetNumberOfScalarComponents();
  }
}

void vtkGeneralizedPhaseCongruency::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Number of Scales: " << this->NumberOfScales << "\n";
  os << indent << "Minimum Wavelength: " << this->MinimumWavelength << "\n";
  os << indent << "Multiplicative factor between scales: " << this->MultiplicativeFactor << "\n";
  os << indent << "Relative bandwidth: "<< this->RelativeBandwidth << "\n";
}

