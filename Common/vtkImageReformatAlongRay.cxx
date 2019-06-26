/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageReformatAlongRay.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkImageReformatAlongRay.h"

#include "vtkImageData.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include "teem/ell.h"

#include <math.h>

vtkStandardNewMacro(vtkImageReformatAlongRay);

//----------------------------------------------------------------------------
vtkImageReformatAlongRay::vtkImageReformatAlongRay()
{
 this->Theta = 0;
 this->RMin = 0;
 this->RMax = 12.7;
 this->Center[0] = 32;
 this->Center[1] = 32;
 this->Center[2] = 0;
 this->Delta = 1;
 this->Scale = 3;
 this->gtx = NULL;
 this->nin = nrrdNew();
}

//----------------------------------------------------------------------------
vtkImageReformatAlongRay::~vtkImageReformatAlongRay()
{
nrrdNix(this->nin);
if (this->gtx != NULL)
  gageContextNix(this->gtx);
}

//----------------------------------------------------------------------------
// VTK6 migration note:
// - introduced to replace ExecuteInformation()
int vtkImageReformatAlongRay::RequestInformation (
  vtkInformation       *  request,
  vtkInformationVector ** inputVector,
  vtkInformationVector *  outputVector)
{
  this->Superclass::RequestInformation( request, inputVector, outputVector );

  vtkInformation* outInfo = outputVector->GetInformationObject(0);

  // 3 components in the output:
  // Comp 0: interpolated ray
  // Comp 1: first order derivative
  // Comp 2: secondr order derivative
  vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_DOUBLE, 3);

  vtkImageData* input = vtkImageData::GetData(inputVector[0]);

  double *insp = input->GetSpacing();
  double sp = sqrt(insp[0]*insp[0] * cos(this->Theta) * cos(this->Theta) +
                   insp[1]*insp[1] * sin(this->Theta) * sin(this->Theta));

  outInfo->Set(vtkDataObject::SPACING(), this->Delta*sp, 1, 1);

  int nsamples = int ((this->RMax/sp - this->RMin/sp + 1)/this->Delta);

  outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), 0,nsamples-1,0,0,0,0);
  //outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), 0,0,0,0,0,0);

  return 1;
}

//----------------------------------------------------------------------------
int vtkImageReformatAlongRay::RequestUpdateExtent(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
  int inExt[6], outExt[6];
  vtkInformation *outInfo = outputVector->GetInformationObject(0);
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);

  //outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), outExt);
  //this->HitInputExtent = 1;

  int wholeExtent[6];
  inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), wholeExtent);
  inInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), wholeExtent, 6);

  vtkImageData* input = vtkImageData::GetData(inputVector[0]);
  double *insp = input->GetSpacing();
  double sp = sqrt(insp[0]*insp[0] * cos(this->Theta) * cos(this->Theta) +
                   insp[1]*insp[1] * sin(this->Theta) * sin(this->Theta));

  int nsamples = int ((this->RMax/sp - this->RMin/sp + 1)/this->Delta);

  outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), 0,nsamples-1,0,0,0,0);
  outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), 0,nsamples-1,0,0,0,0);

  return 1;
}

//----------------------------------------------------------------------------
// VTK6 migration note:
// - introduced to replace ExecuteData()
void vtkImageReformatAlongRay::ExecuteDataWithInformation(vtkDataObject *out,
  vtkInformation* outInfo)
{
  vtkImageData *input = vtkImageData::SafeDownCast(this->GetInput());

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

  // Allocate Output
  double *insp = input->GetSpacing();
  double sp = sqrt(insp[0]*insp[0] * cos(this->Theta) * cos(this->Theta) +
                   insp[1]*insp[1] * sin(this->Theta) * sin(this->Theta));
  int nsamples = int ((this->RMax/sp - this->RMin/sp + 1)/this->Delta);
  this->GetOutput()->SetExtent(0,nsamples-1,0,0,0,0);

  vtkImageData *output = this->GetOutput();
  output->AllocateScalars(outInfo);
  //vtkIndent ident;
  //output->PrintSelf(cout,ident);

  // Convert input to nrrd
  int dims[3];
  double Spacing[3];
  input->GetDimensions(dims);
  input->GetSpacing(Spacing);
  void *data =  (void *) input->GetScalarPointer();
  if (data == NULL) {
     vtkErrorMacro("Input does not have Scalars");
     return;
  }
  const int type = this->VTKToNrrdPixelType(input->GetScalarType());
  size_t size[3];
  size[0]=dims[0];
  size[1]=dims[1];
  //Trick: So gage thinks we always have a 3D volume even if this is 2D.
  size[2]=dims[2];
  int dimensionality=3;

  if (dims[2] == 1) {
    dimensionality = 2;
  } else {
    dimensionality = 3;
  }

  if(nrrdWrap_nva(this->nin,data,type,3,size)) {
	cout<<"Error with nrrdWrap"<<endl;
        //sprintf(err,"%s:",me);
	//biffAdd(NRRD, err); return;
  }
  nrrdAxisInfoSet_nva(this->nin, nrrdAxisInfoSpacing, Spacing);
  this->nin->axis[0].center = nrrdCenterCell;
  this->nin->axis[1].center = nrrdCenterCell;
  this->nin->axis[2].center = nrrdCenterCell;
  // Create gage context
  int E = 0;
  this->gtx = gageContextNew();
  gageParmSet(this->gtx, gageParmRenormalize, AIR_TRUE); // slows things down if true
  gagePerVolume *pvl;
  if (!E) E |= !(pvl = gagePerVolumeNew(this->gtx, this->nin, gageKindScl));
  if (!E) E |= gagePerVolumeAttach(this->gtx, pvl);
  if (E) {
    fprintf(stderr, "%s: trouble:\n%s\n",this->GetClassName(), biffGetDone(GAGE));
    return;
  }

  double kparm[3];
  kparm[0] = this->Scale;
  kparm[1] = 0.5;
  kparm[2] = 0.25;

  if (!E) E |= gageKernelSet(this->gtx, gageKernel00, nrrdKernelBCCubic, kparm);
  if (!E) E |= gageKernelSet(this->gtx, gageKernel11, nrrdKernelBCCubicD, kparm);
  if (!E) E |= gageKernelSet(this->gtx, gageKernel22, nrrdKernelBCCubicDD, kparm);
  if (!E) E |= gageQueryItemOn(this->gtx, pvl, gageSclValue);
  if (!E) E |= gageQueryItemOn(this->gtx, pvl, gageSclGradVec);
  if (!E) E |= gageQueryItemOn(this->gtx, pvl, gageSclHessian);
  if (!E) E |= gageUpdate(this->gtx);
  const double *valu = gageAnswerPointer(this->gtx, pvl, gageSclValue);
  const double *grad = gageAnswerPointer(this->gtx, pvl, gageSclGradVec);
  const double *hess = gageAnswerPointer(this->gtx, pvl, gageSclHessian);

  //Loop through ray points
  double dp[3],vp[3],xp[3];
  // Delta increment in the point
  //cout<<"Theta: "<<this->Theta;
  //cout<<"Center: "<<this->Center[0]<<" "<<this->Center[1]<<" "<<this->Center[2]<<endl;
  vp[0] =  cos(this->Theta);
  vp[1] =  sin(this->Theta);
  vp[2] = 0;

  // Initial point
  for (int i=0; i<3 ; i++) {
    dp[i] = this->Delta * vp[i];
    xp[i] = this->Center[i] + vp[i] * this->RMin;
  }

  double *outPtr = (double *) output->GetScalarPointer();
  double hessvp[3];
  for (int k = 0; k < nsamples ; k++ ) {
    gageProbe(this->gtx,xp[0],xp[1],xp[2]);
    *outPtr = (double) valu[0];
    outPtr++;
    *outPtr = (double) (vp[0]*grad[0] + vp[1]*grad[1] + vp[2]*grad[2]);
    outPtr++;
    ELL_3MV_MUL(hessvp,hess,vp);
    *outPtr =  (double) (hessvp[0]*vp[0]+hessvp[1]*vp[1]+hessvp[2]*vp[2]);
    outPtr++;
    for (int i=0; i<3; i++)
      xp[i]=xp[i] + dp[i];
  }

  gageContextNix(this->gtx);
}

int vtkImageReformatAlongRay::VTKToNrrdPixelType( const int vtkPixelType )
  {
  switch( vtkPixelType )
    {
    default:
    case VTK_VOID:
      return nrrdTypeDefault;
      break;
    case VTK_CHAR:
      return nrrdTypeChar;
      break;
    case VTK_UNSIGNED_CHAR:
      return nrrdTypeUChar;
      break;
    case VTK_SHORT:
      return nrrdTypeShort;
      break;
    case VTK_UNSIGNED_SHORT:
      return nrrdTypeUShort;
      break;
      //    case nrrdTypeLLong:
      //      return LONG ;
      //      break;
      //    case nrrdTypeULong:
      //      return ULONG;
      //      break;
    case VTK_INT:
      return nrrdTypeInt;
      break;
    case VTK_UNSIGNED_INT:
      return nrrdTypeUInt;
      break;
    case VTK_FLOAT:
      return nrrdTypeFloat;
      break;
    case VTK_DOUBLE:
      return nrrdTypeDouble;
      break;
    }
 }

void vtkImageReformatAlongRay::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Theta sampling: " << this->Theta << "\n";
  os << indent << "Delta: "<<this->Delta << "\n";
  os << indent << "Scale: "<<this->Scale << "\n";
}

