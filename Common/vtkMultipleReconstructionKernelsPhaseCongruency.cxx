/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMultipleReconstructionKernelsPhaseCongruency.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkMultipleReconstructionKernelsPhaseCongruency.h"

#include "vtkImageData.h"
#include "vtkImageStencilData.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkExecutive.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include <cmath>

#define VTK_EPS 1e-15

vtkStandardNewMacro(vtkMultipleReconstructionKernelsPhaseCongruency);
vtkCxxSetObjectMacro(vtkMultipleReconstructionKernelsPhaseCongruency, Stencil, vtkImageStencilData);

//----------------------------------------------------------------------------
vtkMultipleReconstructionKernelsPhaseCongruency::vtkMultipleReconstructionKernelsPhaseCongruency()
{
  this->Stencil = 0;
  this->NoiseLevel = 0;
}

//----------------------------------------------------------------------------
vtkMultipleReconstructionKernelsPhaseCongruency::~vtkMultipleReconstructionKernelsPhaseCongruency()
{
  this->SetStencil(0);
}

//----------------------------------------------------------------------------
// VTK6 migration note:
// Introduced to multiple input connection for port 0
int vtkMultipleReconstructionKernelsPhaseCongruency::FillInputPortInformation(int port, vtkInformation* info)
{
  if (port == 0)
    {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
    info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
    }
  else
    {
    vtkErrorMacro("Invalid input port is given in vtkMultipleReconstructionKernelsPhaseCongruency::FillInputPortInformation");
    return 0;
    }
  return 1;
}


//----------------------------------------------------------------------------
// VTK6 migration note:
// - introduced to replace ExecuteInformation()
int vtkMultipleReconstructionKernelsPhaseCongruency::RequestInformation (
  vtkInformation       *  vtkNotUsed(request),
  vtkInformationVector ** inputVector,
  vtkInformationVector *  outputVector)
{
  vtkImageData* input = vtkImageData::GetData(inputVector[0]);
  vtkImageStencilData *stencil = this->GetStencil();
  if (stencil)
    {
    stencil->SetSpacing(input->GetSpacing());
    stencil->SetOrigin(input->GetOrigin());
    }
  // Set output
  vtkInformation* outInfo = outputVector->GetInformationObject(0);
  // Num of output components
  // 0: Phase congruency
  // 1: Phase congruency for edge up
  // 2: Phase congruency for edge down
  vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_DOUBLE, 3);
  
  return 1;
}

//----------------------------------------------------------------------------
// This method checks to see if we can simply reference the input data
//
// VTK6 migration note:
// - introduced to replace ExecuteData()
int vtkMultipleReconstructionKernelsPhaseCongruency::RequestData(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  int result = 0;
  // check to see if we have more than one input
  int singleInput = 1;
  for (int idx = 1; idx < this->GetNumberOfInputConnections(0); idx++)
    {
    if (this->GetExecutive()->GetInputData(0, idx) != NULL)
      {
      singleInput = 0;
      }
    }
  if (singleInput)
    {
    vtkErrorMacro("ExecuteData: single input, cannot compute phase congruency from a single kernel");
    return 0;
    }
  else // multiple inputs
    {
    vtkImageStencilData *stencil = this->GetStencil();
    if (stencil)
      {
      vtkInformation* outInfo = outputVector->GetInformationObject(0);
      vtkInformation* stencilInfo = stencil->GetInformation();
      
      // FIX_ME_VTK6
      int extent[6];
      outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), extent);
      stencilInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), extent, 6);
      
      //stencil->SetUpdateExtent(((vtkImageData *)output)->GetUpdateExtent());
      //stencil->Update();
      }
    // this will call ThreadedRequestData
    result = this->Superclass::RequestData(request, inputVector, outputVector);
    }
  
  return result;
}

void vtkMultipleReconstructionKernelsPhaseCongruency::ThreadedRequestData(
  vtkInformation *request,
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector,
  vtkImageData ***inData,
  vtkImageData **outData,
  int extent[6], int threadId)
{
  this->ThreadedExecute(inData[0], outData[0], extent, threadId);
}
                                 
//vtkImageStencilData*&, int&, int&, int&, int&, int&, int&, double*&, vtkMultipleReconstructionKernelsPhaseCongruency::ThreadedExecute::VTK_TT**&, int&, int&, int&, int&
//----------------------------------------------------------------------------
// helper function for the stencil
template <class T>
inline int vtkMultipleReconstructionKernelsPhaseCongruencyGetNextExtent(vtkImageStencilData *stencil,
                                 int &r1, int &r2, int rmin, int rmax,
                                 int yIdx, int zIdx, 
                                 double *&outPtr, T **&inPtrs, int numKernels,
                                 int outScalars, int inScalars,
                                 int &iter)
{
  // trivial case if stencil is not set
  if (!stencil)
    {
    if (iter++ == 0)
      {
      r1 = rmin;
      r2 = rmax;
      return 1;
      }
    return 0;
    }

  // save r2
  int oldr2 = r2;
  if (iter == 0)
    { // if no 'last time', start just before rmin
    oldr2 = rmin - 1;
    }

  int rval = stencil->GetNextExtent(r1, r2, rmin, rmax, yIdx, zIdx, iter);
  int incr = r1 - oldr2 - 1;
  if (rval == 0)
    {
    incr = rmax - oldr2;
    }

  outPtr += incr*outScalars;
  inPtrs[0] += incr*inScalars;
  for(int k=1;k<numKernels;k++)
    inPtrs[k] +=incr*1;

  return rval;
}

//----------------------------------------------------------------------------
// This templated function executes the filter for any type of data.
template <class T>
void vtkMultipleReconstructionKernelsPhaseCongruencyExecute(vtkMultipleReconstructionKernelsPhaseCongruency *self, 
                          vtkImageData **inDatas, T **inPtrs,
                          vtkImageData *outData, double *outPtr,
                          int extent[6], int id)
{
  int idxX, idxY, idxZ, idxK;
  int minX = 0;
  int maxX = 0;
  int iter;
  vtkIdType inIncX, inIncY, inIncZ;
  vtkIdType inIncX0, inIncY0, inIncZ0;
  vtkIdType outIncX, outIncY, outIncZ;
  int in0C, outC;
  unsigned long count = 0;
  unsigned long target;

  double meanf, F, Fe,Fo,A,gradient,PC,PC3,PC4;
  double sign;
  int numKernels;

  vtkImageStencilData *stencil = self->GetStencil();

  in0C = inDatas[0]->GetNumberOfScalarComponents();
  outC = outData->GetNumberOfScalarComponents();

  target = (unsigned long)((extent[3] - extent[2] + 1)*
                           (extent[5] - extent[4] + 1)/50.0);
  target++;

   // Get increments to march through image data 
  inDatas[0]->GetContinuousIncrements(extent, inIncX, inIncY, inIncZ);
  outData->GetContinuousIncrements(extent, outIncX, outIncY, outIncZ);

  numKernels = self->GetNumberOfInputConnections(0);
  // Get increments to march through data 
  inDatas[0]->GetContinuousIncrements(extent, inIncX0, inIncY0, inIncZ0);
  inDatas[1]->GetContinuousIncrements(extent, inIncX, inIncY, inIncZ);
  outData->GetContinuousIncrements(extent, outIncX, outIncY, outIncZ);

  // Loop through output pixels
  for (idxZ = extent[4]; idxZ <= extent[5]; idxZ++)
    {
    for (idxY = extent[2]; !self->AbortExecute && idxY <= extent[3]; idxY++)
      {
      if (!id) 
        {
        if (!(count%target))
          {
          self->UpdateProgress(count/(50.0*target));
          }
        count++;
        }

      iter = 0;

      
      while (vtkMultipleReconstructionKernelsPhaseCongruencyGetNextExtent(stencil, minX, maxX, extent[0], extent[1],
                                     idxY, idxZ,
                                     outPtr, inPtrs, numKernels, outC, in0C, iter))
        {
        for (idxX = minX; idxX <= maxX; idxX++)
          {
            meanf = 0.0;
            for (idxK = 0; idxK < numKernels; idxK++)
              {
              //Compute mean signal value
              meanf += (double) *inPtrs[idxK];
              }
            meanf /= numKernels;
            //Compute phase congruency based on multiple kernels
            //A: mean of the vector norms
            //F: Energy (norm) of the sum vector across kernels
            A=0;Fe=0;Fo=0;
            for (idxK = 0; idxK< numKernels; idxK++)
              {
              A += std::sqrt(meanf*meanf+((double) *inPtrs[idxK]) * ((double) *inPtrs[idxK]) );
              if (*inPtrs[idxK]<meanf)
                sign = -1;
              else
                sign = 1;
              Fe += sign*meanf;
              Fo += (double) *inPtrs[idxK];
              }
            F = std::sqrt(Fe*Fe + Fo*Fo);

            if (A > VTK_EPS)
              PC = (F - self->GetNoiseLevel())/(A);
            else
              PC = 0;

            if (PC<0)
              PC=0;

            // Fill PC3 and PC4 based on phase given by the gradient
            PC3 = 0.0;
            PC4 = 0.0;
            if (in0C == 2)
              {
              gradient = *(inPtrs[0]+1);
              if (gradient > 0)
                PC3 = PC;
              else
                PC4 = PC;
              }

            outPtr[0] = PC;
            outPtr[1] = PC3;
            outPtr[2] = PC4;
            outPtr += outC; 
            inPtrs[0] += in0C;
            for (int k = 1; k<numKernels;k++)
              {
              inPtrs[k]++;
              }
            }//X loop
          }//While loop for stencil
       outPtr += outIncY;
       inPtrs[0]+= inIncY0;
       for (int k = 1; k < numKernels; k++)
         {
         inPtrs[k] += inIncY;
         }
      }
      outPtr += outIncZ;
      inPtrs[0]+= inIncZ0;
      for (int k = 1; k < numKernels; k++)
        {
        inPtrs[k] += inIncZ;
        }
    }
}


//----------------------------------------------------------------------------
// This method is passed a input and output regions, and executes the filter
// algorithm to fill the output from the inputs.
// It just executes a switch statement to call the correct function for
// the regions data types.
void vtkMultipleReconstructionKernelsPhaseCongruency::ThreadedExecute(
  vtkImageData **inDatas, 
  vtkImageData *outData,
  int outExt[6], int id)
{
  void **inPtrs;
  void *outPtr;

  // check

  if (inDatas[0]->GetNumberOfScalarComponents() > 2)
    {
    vtkErrorMacro("The first input can have a maximum of two components");
    return;
    }
  // this filter expects that output is float
  if (outData->GetScalarType() != VTK_DOUBLE)
    {
     vtkErrorMacro(<< "Execute: output ScalarType (" << 
        outData->GetScalarType() << 
        "), must be float");
     return;
    }

 // Loop through checking all inputs 
  for (int idx = 0; idx < this->GetNumberOfInputConnections(0); ++idx)
    {
      if (inDatas[idx] == NULL)
        {
        vtkErrorMacro(<< "Execute: input" << idx << " is NULL");
        return;
        }
     }

  inPtrs = new void*[this->GetNumberOfInputConnections(0)];

  // Loop through to fill input pointer array
  for (int idx = 0; idx < this->GetNumberOfInputConnections(0); ++idx)
    {
      // Lauren should we use out ext here?
      inPtrs[idx] = inDatas[idx]->GetScalarPointerForExtent(outExt);
    }

  outPtr = outData->GetScalarPointerForExtent(outExt);
  // call Execute method to handle all data at the same time
  switch (inDatas[0]->GetScalarType())
    {
      vtkTemplateMacro(
        vtkMultipleReconstructionKernelsPhaseCongruencyExecute(
          this, inDatas, (VTK_TT **)(inPtrs), outData, (double *)(outPtr), outExt, id
        )
      );
    default:
      vtkErrorMacro(<< "Execute: Unknown ScalarType");
      return;
    }
}


//----------------------------------------------------------------------------
void vtkMultipleReconstructionKernelsPhaseCongruency::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Stencil: " << this->GetStencil() << endl;
  os << indent << "Noise Level: " << this->GetNoiseLevel() << endl;
}

