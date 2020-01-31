/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkTubularScaleSelection.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifdef _WIN32
// to pick up M_SQRT2 and other nice things...
#define _USE_MATH_DEFINES
#endif

// But, if you are on VS6.0 you don't get the define...
#ifndef M_SQRT2
#define M_SQRT2    1.41421356237309504880168872421      /* sqrt(2) */
#endif

#include "vtkTubularScaleSelection.h"

#include "vtkImageData.h"
#include "vtkImageProgressIterator.h"
#include "vtkObjectFactory.h"
#include "vtkInformationVector.h"

#include "vnl/vnl_math.h"
#include <math.h>
#include "vtkNRRDExport.h"
#define VTK_EPS 1e-12

vtkStandardNewMacro(vtkTubularScaleSelection);

vtkTubularScaleSelection::vtkTubularScaleSelection()
{
  this->Mask = NULL;
  this->TubularType = VTK_VALLEY;
  this->InitialScale = 1.0;
  this->FinalScale = 6.0;
  this->StepScale = 0.1;
}

vtkTubularScaleSelection::~vtkTubularScaleSelection()
{
  if (this->Mask)
    {
    this->Mask->Delete();
    }
}

int vtkTubularScaleSelection::RequestInformation (
  vtkInformation       * vtkNotUsed( request ),
  vtkInformationVector ** vtkNotUsed( inputVector ),
  vtkInformationVector * outputVector)
{
  vtkDataObject::SetPointDataActiveScalarInfo(
    outputVector->GetInformationObject(0), VTK_FLOAT, 1);
  return 1;
}

//----------------------------------------------------------------------------
// This execute method handles boundaries.
// it handles boundaries. Pixels are just replicated to get values
// out of extent.
template <class T>
void vtkTubularScaleSelectionExecute(vtkTubularScaleSelection *self, vtkImageData *inData,
                              vtkImageData *outData,
                              int outExt[6], int id, T *)
{
  // double delta = 4.0;

  //vtkImageIterator<T> inIt(inData, outExt);
  //vtkImageProgressIterator<T> outIt(outData, outExt, self, id);

  //Loop variables
  int idxR, idxY, idxZ;
  int maxY, maxZ;
  vtkIdType outIncX, outIncY, outIncZ;
  vtkIdType inIncX, inIncY, inIncZ;
  int rowLength;
  // progress
  unsigned long count = 0;
  unsigned long target;
  // Coordinates to probe with gage
  double x,y,z;
  double xyz[3];
  // find the output region to loop over
  rowLength = (outExt[1] - outExt[0]+1);
  maxY = outExt[3] - outExt[2];
  maxZ = outExt[5] - outExt[4];
  target = (unsigned long)((maxZ+1)*(maxY+1)/50.0);
  target++;

  // Mask variables
  vtkImageData *inMask = NULL;
  short * inMaskPtr = NULL;
  int doMasking = 0;
  if (self->GetMask() != NULL)
    {
    inMask=self->GetMask();
    doMasking = 1;
    }

  // Get increments to march through output data
  outData->GetContinuousIncrements(outExt, outIncX, outIncY, outIncZ);
  inData->GetContinuousIncrements(outExt, inIncX, inIncY, inIncZ);
  float *outPtr = (float *) outData->GetScalarPointerForExtent(outExt);
  T *inPtr = (T *)inData->GetScalarPointerForExtent(outExt);

  if (inMask)
    {
    //inMaskPtr = (short *) inMask->GetScalarPointerForExtent(outExt);
    inMaskPtr = (short *) inMask->GetScalarPointer();
    }
  int inFullExt[6];
  vtkIdType inInc[3];
  inData->GetExtent(inFullExt);
  inData->GetIncrements(inInc);
  int inPtId = ((outExt[0] - inFullExt[0]) * inInc[0]
          + (outExt[2] - inFullExt[2]) * inInc[1]
          + (outExt[4] - inFullExt[4]) * inInc[2]);

  // Set up gage business
  //Export vtkImageData to Nrrd
  vtkNRRDExport * nrrdexport= vtkNRRDExport::New();
  nrrdexport->SetInputData(inData);
  Nrrd *nin = nrrdexport->GetNRRDPointer();

  //Create array of context for each scale
  double finalScale = self->GetFinalScale();
  double initScale = self->GetInitialScale();
  double scaleStep = self->GetStepScale();
  int numScales = (int) ceil((finalScale-initScale)/scaleStep)+1;

  gageContext **gtx = new gageContext *[numScales];
  for (int i=0;i<numScales;i++) {
    gtx[i]=NULL;
  }
  gagePerVolume **pvl = new gagePerVolume *[numScales];
  double scale = initScale;
  int E = 0;
  for (int i=0; i<numScales; i++) {
    //cout<<"Creating gtx for scale: "<<i<<"thread "<<id<<endl;
    //fflush(stdout);
    gtx[i] = gageContextNew();
    if (gtx[i]==NULL) {
      cout<<"We have a problem creating Context"<<endl;
    }
    gageParmSet(gtx[i], gageParmRenormalize, AIR_TRUE); // slows things down if true
    if (!E) E |= !(pvl[i] = gagePerVolumeNew(gtx[i], nin, gageKindScl));
    if (!E) E |= gagePerVolumeAttach(gtx[i], pvl[i]);
    if (!E) E |= self->SettingContext(gtx[i],pvl[i],scale);
    scale = scale+scaleStep;
    if (E) {
     cout<<"Error Setting Context for scale "<<i<<endl;
     break;
    }
  }

  //int testId = (int) vnl_math_rnd(double(self->GetNumberOfThreads())/double(2));

  double val;
  // Loop through
  for (idxZ = 0; idxZ <= maxZ; idxZ++)
    {
    z = idxZ + outExt[4] - inFullExt[4];
    for (idxY = 0; idxY <= maxY; idxY++)
      {
      y = idxY + outExt[2] - inFullExt[2];
      if (id==0)
        {
        if (!(count%target))
          {
          self->UpdateProgress(count/(50.0*target));
          cout<<"Progress Update: "<<count/(50.0*target)<<endl;
          }
        count++;
        }
      for (idxR = 0; idxR < rowLength; idxR++)
        {
        // Find x
        x = idxR + outExt[0] - inFullExt[0];

        if (doMasking )
          {
          if (*(inMaskPtr + inPtId) == 0)
            {
            *outPtr = (float) 0.0;
            outPtr++;
            inPtId++;
            continue;
            }
          }
        xyz[0] = x;
        xyz[1] = y;
        xyz[2] = z;

        scale = self->ScaleSelection(gtx,pvl,xyz,initScale,finalScale,scaleStep);
        *outPtr = scale;
        if (scale<1 && scale>=0)
          cout<<"Something is weird: "<<scale<<endl;
        outPtr++;
        inPtId++;
       }
       outPtr += outIncY;
       inPtId += inIncY;
     }
     outPtr += outIncZ;
     inPtId += inIncZ;
   }

  // Clean objects
  for (int i=0; i<numScales; i++) {
   gageContextNix(gtx[i]);
  }
  nrrdexport->Delete();
  delete [] gtx;
  delete [] pvl;
}

//----------------------------------------------------------------------------
// This method contains a switch statement that calls the correct
// templated function for the input data type.  The output data
// must match input type.  This method does handle boundary conditions.
void vtkTubularScaleSelection::ThreadedExecute(vtkImageData *inData,
                                        vtkImageData *outData,
                                        int outExt[6], int id)
{
  vtkDebugMacro(<< "Execute: inData = " << inData
  << ", outData = " << outData);

  // this filter expects that input is the same type as output.
  if (inData->GetNumberOfScalarComponents() != 1)
    {
    vtkErrorMacro(<< "Execute: input must have 1 components, but has " << inData->GetNumberOfScalarComponents());
    return;
    }
  if (this->GetMask()!= NULL)
   {
   if (this->GetMask()->GetScalarType() != VTK_SHORT)
      {
      vtkErrorMacro(<< "Execute: mask must be short ");
      return;
      }
    }

  switch (inData->GetScalarType())
    {
    vtkTemplateMacro(
      vtkTubularScaleSelectionExecute(
        this, inData, outData, outExt, id, static_cast<VTK_TT *>(0)
      )
    );
    default:
      vtkErrorMacro(<< "Execute: Unknown ScalarType");
      return;
    }
}

int vtkTubularScaleSelection::SettingContext(gageContext *gtx,gagePerVolume *pvl,double scale)
{
  double kparm[3] = {3.0, 1.0, 0.0};
  //Round scacle
  kparm[0] = scale;
  int E=0;
  // { scale, B, C}; (B,C)=(1,0): uniform cubic B-spline
  if (!E) E |= gageKernelSet(gtx, gageKernel00, nrrdKernelBCCubic, kparm);
  if (!E) E |= gageKernelSet(gtx, gageKernel11, nrrdKernelBCCubicD, kparm);
  if (!E) E |= gageKernelSet(gtx, gageKernel22, nrrdKernelBCCubicDD, kparm);
  //if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclValue);
  //if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclGradVec);
  if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclHessian);
  //if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclHessEvec);
  if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclHessEval);
  if (!E) E |= gageUpdate(gtx);
  //const double *valu = gageAnswerPointer(gtx, pvl, gageSclValue);
  //const double *grad = gageAnswerPointer(gtx, pvl, gageSclGradVec);
  //const double *hess = gageAnswerPointer(gtx, pvl, gageSclHessian);
  //const double *hevec = gageAnswerPointer(gtx, pvl, gageSclHessEvec);
  const double *heval = gageAnswerPointer(gtx, pvl, gageSclHessEval);

  return E;
}

double vtkTubularScaleSelection::ScaleSelection(gageContext **gtx, gagePerVolume **pvl, double Seed[3], double initS,double maxS, double deltaS)
{
  double S = initS;
  double Sopt = -1;
  double Sopt2 = -1;
  int scaleIdx = 0;
  double prevV,nextV;
  double maxval = -10000000;
  double featureSign = 1;

 // If we are working with vessels:
  // Eigenvalues corresponding to airway cross section are positive
  // Mode is positive.
  if (this->GetTubularType()==VTK_VALLEY) {
    featureSign = 1;
  }
  // If we are working with vessels:
  // Eigenvalues corresponding to vessel cross section are negative
  // Mode is positive.
  if (this->GetTubularType()==VTK_RIDGE) {
    featureSign = -1;
  }

 // Allocate strength array
  int numElem = 0;
  for (double k = initS;k <=maxS;k=k+deltaS) {
    numElem++;
  }
  double *strength = new double[numElem];

  const double *heval;

  prevV = featureSign;
  scaleIdx = 0;
  do {
    gageProbe(gtx[scaleIdx],Seed[0],Seed[1],Seed[2]);
    heval=gageAnswerPointer(gtx[scaleIdx], pvl[scaleIdx], gageSclHessEval);

    nextV = this->Mode(heval);
   if (-1.0*featureSign*nextV > -1.0*featureSign*prevV && featureSign*heval[1]> 0 && featureSign*heval[1] > fabs(heval[2])) {
         prevV = nextV;
         Sopt = S;
     }

    // Compute strenght for locations that make sense
    if (featureSign*heval[1]> 0 && featureSign*heval[1] > fabs(heval[2])) {
        strength[scaleIdx]=S*S*(heval[0]+heval[1])/2;
    } else {
        strength[scaleIdx]=0;
    }

    S = S+deltaS;
    scaleIdx++;
  }while(S<=maxS);

  S = initS;
  scaleIdx = 0;
  prevV = 0;
  // Find scale with maximun strenght in the range Sinit to Sopt
  do {
    if (strength[scaleIdx] > prevV) {
         prevV = strength[scaleIdx];
         Sopt2 = S;
      }
      S = S + deltaS;
      scaleIdx++;
    } while (S<=Sopt);
  // If maximun strength was positive, use Sopt2.
  if (prevV > 0)
    Sopt = Sopt2;

  delete[] strength;

  return Sopt;
}

double vtkTubularScaleSelection::ScaleSelection(gageContext *gtx, gagePerVolume *pvl, double Seed[3], double initS,double maxS, double deltaS)
{
  double kparm[3] = {3.0, 1.0, 0.0};
  //double kparm[3] = {3.0, 0.5, 0.25};
  double S = initS;
  double Sopt = -1;
  double Sopt2 = -1;
  double prevV,nextV;
  double maxval = -10000000;
  double featureSign = 1;

 // If we are working with vessels:
  // Eigenvalues corresponding to airway cross section are positive
  // Mode is positive.
  if (this->GetTubularType()==VTK_VALLEY) {
    featureSign = 1;
  }
  // If we are working with vessels:
  // Eigenvalues corresponding to vessel cross section are negative
  // Mode is positive.
  if (this->GetTubularType()==VTK_RIDGE) {
    featureSign = -1;
  }

 // Allocate strength array
  int numElem = 0;
  for (double k = initS;k <=maxS;k=k+deltaS) {
    numElem++;
  }
  double *strength = new double[numElem];

  kparm[0] = (double) S;
  int E = 0;
  if (!E) E |= gageKernelSet(gtx, gageKernel00, nrrdKernelBCCubic, kparm);
  if (!E) E |= gageKernelSet(gtx, gageKernel11, nrrdKernelBCCubicD, kparm);
  if (!E) E |= gageKernelSet(gtx, gageKernel22, nrrdKernelBCCubicDD, kparm);
  if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclHessEval);
  if (!E) E |= gageUpdate(gtx);
  const double *heval = gageAnswerPointer(gtx, pvl, gageSclHessEval);
  gageProbe(gtx,Seed[0],Seed[1],Seed[2]);
  // For being a valley: l1>>0, l2>>0 and l3 ~ 0
  // Our metric for valleiness is: (l1 + l2)/2 - abs(l3)
  // We want to maximize this metric.
  // Or better instead, mode: We want to minimize this metric.

  //prevV = (heval[0] + heval[1])/2 - fabs(heval[2]);

   prevV = featureSign;
   int scaleIdx = 0;

   prevV = -100000;
  do {
    kparm[0] = S;
    if (!E) E |= gageKernelSet(gtx, gageKernel00, nrrdKernelBCCubic, kparm);
    if (!E) E |= gageKernelSet(gtx, gageKernel11, nrrdKernelBCCubicD, kparm);
    if (!E) E |= gageKernelSet(gtx, gageKernel22, nrrdKernelBCCubicDD, kparm);
    if (!E) E |= gageUpdate(gtx);
    //heval = gageAnswerPointer(gtx, pvl, gageSclHessEval);
    gageProbe(gtx,Seed[0],Seed[1],Seed[2]);
    //cout<<"Testing Scale: "<<S<<"  Eigenvalues: "<<heval[0]<<" "<<heval[1]<<" "<<heval[2]<<"  Mode: "<<this->Mode(heval)<<"  Disparity measure: "<< (heval[0] + heval[1])/2 - fabs(heval[2])<<endl;

    nextV = S*S*featureSign*this->Strength(heval);
    if (nextV > prevV && featureSign*heval[1]> 0 && featureSign*heval[1] > fabs(heval[2])) {
         prevV = nextV;
         Sopt = S;
     }

    // Compute strenght for locations that make sense
    if (featureSign*heval[1]> 0 && featureSign*heval[1] > fabs(heval[2])) {
        strength[scaleIdx]=S*S*(heval[0]+heval[1])/2;
    } else {
        strength[scaleIdx]=0;
    }

    S = S+deltaS;
    scaleIdx++;
  } while( S <= maxS);
  S = initS;
  scaleIdx = 0;
  prevV = 0;
  // Find scale with maximun strenght in the range Sinit to Sopt
  do {
    if (strength[scaleIdx] > prevV) {
         prevV = strength[scaleIdx];
         Sopt2 = S;
      }
      S = S + deltaS;
      scaleIdx++;
    } while (S<=Sopt);
  // If maximun strength was positive, use Sopt2.
  if (prevV > 0)
    Sopt = Sopt2;

  delete[] strength;

  return Sopt;
}

double vtkTubularScaleSelection::Mode(const double *w)
{
  // see PhD thesis, Gordon Kindlmann
  double mean = (w[0] + w[1] + w[2])/3;
  double norm = ((w[0] - mean)*(w[0] - mean) +
                  (w[1] - mean)*(w[1] - mean) +
                  (w[2] - mean)*(w[2] - mean))/3;
  norm = sqrt(norm);
  norm = norm*norm*norm;
  if (norm < VTK_EPS)
     norm += VTK_EPS;
  // multiply by sqrt 2: range from -1 to 1
  return  (M_SQRT2*((w[0] + w[1] - 2*w[2]) *
                         (2*w[0] - w[1] - w[2]) *
                         (w[0] - 2*w[1] + w[2]))/(27*norm));
}

double vtkTubularScaleSelection::Strength(const double *w)
{
  return ((w[0]+w[1])/2 - fabs(w[3]));
}

void vtkTubularScaleSelection::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

