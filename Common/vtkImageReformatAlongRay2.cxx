/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageReformatAlongRay2.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkImageReformatAlongRay2.h"

#include "vtkImageData.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include "teem/ell.h"

#include <math.h>

vtkStandardNewMacro(vtkImageReformatAlongRay2);

//----------------------------------------------------------------------------
vtkImageReformatAlongRay2::vtkImageReformatAlongRay2()
{

 this->Theta = 0;
 this->RMin = 0;
 this->RMax = 63;
 this->Center[0] = 32;
 this->Center[1] = 32;
 this->Center[2] = 0;
 this->Spacing = 0.05;
 this->Tangent[0]=1;
 this->Tangent[1]=0;
 this->Tangent[2]=0;
 this->Normal[0]=0;
 this->Normal[1]=0;
 this->Normal[2]=1;
 this->Mode = 1;
 this->Scale = 3;
 this->gtx = NULL;
 this->nin = nrrdNew();
}

//----------------------------------------------------------------------------
vtkImageReformatAlongRay2::~vtkImageReformatAlongRay2()
{
nrrdNix(this->nin);
if (this->gtx != NULL) 
  gageContextNix(this->gtx);
}

//----------------------------------------------------------------------------
// VTK6 migration note:
// - introduced to replace ExecuteInformation()
int vtkImageReformatAlongRay2::RequestInformation (
  vtkInformation       *  request,
  vtkInformationVector ** inputVector,
  vtkInformationVector *  outputVector)
{
  this->Superclass::RequestInformation(request, inputVector, outputVector);

  vtkInformation* outInfo = outputVector->GetInformationObject(0);
  // 3 components in the output:
  // Comp 0: interpolated ray
  // Comp 1: first order derivative
  // Comp 2: secondr order derivative
  vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_DOUBLE, 3);

  outInfo->Set(vtkDataObject::SPACING(), this->Spacing,1,1);

  int nsamples = int ((this->RMax - this->RMin)/this->Spacing);
  
  outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), 0,nsamples-1,0,0,0,0);
  outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), 0,0,0,0,0,0);
  
  return 1;
}  
  
//----------------------------------------------------------------------------
// VTK6 migration note:
// - introduced to replace ExecuteData()
void vtkImageReformatAlongRay2::ExecuteDataWithInformation(vtkDataObject *out, 
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

  // Allocate Output
  int nsamples = int ((this->RMax - this->RMin)/this->Spacing);
  this->GetOutput()->SetExtent(0,nsamples-1,0,0,0,0);

  vtkImageData *output = this->GetOutput();
  output->AllocateScalars(outInfo);
  //vtkIndent ident;
  //output->PrintSelf(cout,ident);

  // Convert input to nrrd
  int dims[3];
  double insp[3];
  input->GetDimensions(dims);
  input->GetSpacing(insp);
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
  nrrdAxisInfoSet_nva(this->nin, nrrdAxisInfoSpacing, insp);  
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
  

  if (this->Mode == NORMAL) {
    this->ComputeTangentFromNormal();
  }

  //Loop through ray points
  double dp[3],xp[3];
  
  // Initial point
  for (int i=0; i<3 ; i++) {
    dp[i] = this->Spacing/insp[i] * this->Tangent[i];
    xp[i] = this->Center[i] + this->Spacing/insp[i] * this->RMin * this->Tangent[i];
  }
   
  double *outPtr = (double *) output->GetScalarPointer();
  double hessvp[3];
  for (int k = 0; k < nsamples; k++ ) {
    gageProbe(this->gtx,xp[0],xp[1],xp[2]);
    *outPtr = (double) valu[0];
    outPtr++;
    *outPtr = (double) (this->Tangent[0]*grad[0] + this->Tangent[1]*grad[1] + this->Tangent[2]*grad[2]);
    outPtr++;
    ELL_3MV_MUL(hessvp,hess,this->Tangent);
    *outPtr =  (double) (hessvp[0]*this->Tangent[0]+hessvp[1]*this->Tangent[1]+hessvp[2]*this->Tangent[2]);
    outPtr++;
    for (int i=0; i<3; i++)
      xp[i]=xp[i] + dp[i];
  }

  gageContextNix(this->gtx);

}


int vtkImageReformatAlongRay2::VTKToNrrdPixelType( const int vtkPixelType )
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

void vtkImageReformatAlongRay2::ComputeTangentFromNormal()
{
  
  //Local frame for the plane containig the line
  double zAxis[3],yAxis[3],xAxis[3];
  double iAxis[3], jAxis[3];
  double normx,normy;
  double m[9]; 

  zAxis[0]=this->Normal[0];
  zAxis[1]=this->Normal[1];
  zAxis[2]=this->Normal[2];
  
  //Chech sign of z axis
  if (zAxis[2]<0) {
    zAxis[0]=-zAxis[0];
    zAxis[1]=-zAxis[1];
    zAxis[2]=-zAxis[2];
  }

  ELL_3MV_OUTER(m,zAxis,zAxis);
  m[0] = 1 - m[0];
  m[4] = 1 - m[4];
  m[8] = 1 - m[8];

  iAxis[0]=1;
  iAxis[1]=0;
  iAxis[2]=0;
  jAxis[0]=0;
  jAxis[1]=1;
  jAxis[2]=0;

  ELL_3MV_MUL(xAxis,m,iAxis);
  ELL_3MV_MUL(yAxis,m,jAxis);
  // Renormalize vectors
  normx=vtkMath::Normalize(xAxis);
  normy=vtkMath::Normalize(yAxis);

  // Compute the axis with smallest norms as the cross product
  if (normx > normy) {
    vtkMath::Cross(zAxis,xAxis,yAxis);
  } else {
    vtkMath::Cross(yAxis,zAxis,xAxis);
  }
  
  for (int k=0;k<3;k++)
    this->Tangent[k]=cos(this->Theta)*xAxis[k]+sin(this->Theta)*yAxis[k];

}

void vtkImageReformatAlongRay2::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Theta: " << this->Theta << "\n";
  os << indent << "Spacing: "<<this->Spacing << "\n";
  os << indent << "Scale: "<<this->Scale << "\n";
}

