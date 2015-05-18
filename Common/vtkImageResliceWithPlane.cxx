/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageResliceWithPlane.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkImageResliceWithPlane.h"

#include "vtkImageData.h"
#include "vtkImageStencilData.h"
#include "vtkDoubleArray.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include "vtkImageCast.h"
#include "vtkImageReslice.h"
#include "vtkCell.h"
#include "vtkExtractAirwayTree.h"
#include "vtkPointData.h"
#include "vtkNrrdReader.h"
#include "teem/nrrd.h"
#include "teem/gage.h"


#include <math.h>

vtkStandardNewMacro(vtkImageResliceWithPlane);

//----------------------------------------------------------------------------
vtkImageResliceWithPlane::vtkImageResliceWithPlane()
{
  
  this->Reslice = NULL;
  this->InPlane = 1;  // Reslice InPlane (1) or using the Hessian (0);
  this->ComputeAxes = 0;
  this->ComputeCenter = 0;
  this->Center[0] = 0;
  this->Center[1] = 0;
  this->Center[2] = 0;
  this->Dimensions[0] = 32;
  this->Dimensions[1] = 32;
  this->Dimensions[2] = 32;
  this->Spacing[0] = 1;
  this->Spacing[1] = 1;
  this->Spacing[1] = 1;
  this->InterpolationMode = VTK_RESLICE_CUBIC;
  this->XAxis[0] = 1;
  this->XAxis[1] = 0;
  this->XAxis[2] = 0;
  this->YAxis[0] = 0;
  this->YAxis[1] = 1;
  this->YAxis[2] = 0;
  this->ZAxis[0] = 0;
  this->ZAxis[1] = 0;
  this->ZAxis[2] = 1; 
}
//----------------------------------------------------------------------------
vtkImageResliceWithPlane::~vtkImageResliceWithPlane()
{
}

//----------------------------------------------------------------------------
// VTK6 migration note:
// - introduced to replace ExecuteInformation()

int vtkImageResliceWithPlane::RequestInformation (
  vtkInformation       *  vtkNotUsed(request),
  vtkInformationVector ** vtkNotUsed(inputVector),
  vtkInformationVector *  vtkNotUsed(outputVector))
{
  //if (this->GetOutput() != this->Reslice->GetOutput()) {
  //  this->GetOutput()->Delete();
  //  this->SetOutput(this->Reslice->GetOutput());
  //}
  return 1;
}

//----------------------------------------------------------------------------
// VTK6 migration note:
// - introduced to replace ExecuteData()
void vtkImageResliceWithPlane::ExecuteDataWithInformation(vtkDataObject *out, 
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

  if (this->ComputeAxes || this->ComputeCenter) {
    this->ComputeAxesAndCenterUsingTubeModel();
  }

  this->Reslice = vtkImageReslice::New();
  this->Reslice->SetInputData(input);

  // Compute center in global coordinates
  double spacing[3];
  double origin[3];
  double x[3];
  input->GetSpacing(spacing);
  input->GetOrigin(origin); 
  for (int i=0; i<3; i++)
    x[i] = this->Center[i]*spacing[i] + origin[i];

  //Set Reslice Information
  //cout<<"Reslice coordinates info: "<<endl;
  //cout<<"X: "<<XAxis[0]<<" "<<XAxis[1]<<" "<<XAxis[2]<<endl;
  //cout<<"Y: "<<YAxis[0]<<" "<<YAxis[1]<<" "<<YAxis[2]<<endl;
  //cout<<"Z: "<<ZAxis[0]<<" "<<ZAxis[1]<<" "<<ZAxis[2]<<endl;

  //cout<<"Center: "<<Center[0]<<" "<<Center[1]<<" "<<Center[2]<<endl;

  this->Reslice->SetResliceAxesDirectionCosines(XAxis[0],XAxis[1],XAxis[2],
                YAxis[0],YAxis[1],YAxis[2],ZAxis[0],ZAxis[1],ZAxis[2]); 
  this->Reslice->SetResliceAxesOrigin(x);
  this->Reslice->SetOutputDimensionality(2);
  this->Reslice->SetInterpolationMode(this->InterpolationMode);
  this->Reslice->SetOutputSpacing(this->Spacing);
  this->Reslice->SetOutputExtent(0,this->Dimensions[0]-1,
            0,this->Dimensions[1]-1,0,this->Dimensions[2]-1);
  this->Reslice->SetOutputOrigin(- (this->Dimensions[0]*0.5-0.5) * this->Spacing[0],
            - (this->Dimensions[1]*0.5 - 0.5) * this->Spacing[1],
            - (this->Dimensions[2]*0.5 - 0.5) * this->Spacing[2]);
  //this->Reslice->SetOutputOrigin(- (this->Dimensions[0]*0.5) * this->Spacing[0],
  //          - (this->Dimensions[1]*0.5) * this->Spacing[1],
  //          - (this->Dimensions[2]*0.5) * this->Spacing[2]);
  
  vtkDebugMacro("Updating reslice");
  this->Reslice->Update();

  this->GetOutput()->DeepCopy(this->Reslice->GetOutput());

  this->Reslice->Delete();
}

void vtkImageResliceWithPlane::ComputeAxesAndCenterUsingTubeModel() {

  vtkImageData *input = vtkImageData::SafeDownCast(this->GetInput());
  int dims[3];
  double Spacing[3], origin[3];
  double seed[3], seedNew[3],seed2[3],seedR[3];

  if (input == NULL)
    {
    vtkErrorMacro(<<"Input is NULL");
    return;
    }
  input->GetDimensions(dims);
  input->GetOrigin(origin);
  input->GetSpacing(Spacing);

 // Gage jazz to compute hessian   
 void *data =  (void *) input->GetScalarPointer();
 if (data == NULL) {
    vtkErrorMacro(<<"Scalars must be assigned in input data");
    return;
 }
 Nrrd *nin = nrrdNew();
 vtkExtractAirwayTree *helper = vtkExtractAirwayTree::New();
 const int type = vtkExtractAirwayTree::VTKToNrrdPixelType(input->GetScalarType());
 size_t size[3];
 size[0]=dims[0];
 size[1]=dims[1];
 size[2]=dims[2];


 if(nrrdWrap_nva(nin,data,type,3,size)) {
	//sprintf(err,"%s:",me);
	//biffAdd(NRRD, err); return;
  }
  nrrdAxisInfoSet_nva(nin, nrrdAxisInfoSpacing, Spacing);
  
  int E = 0;
  gageContext *gtx = gageContextNew();
  gageParmSet(gtx, gageParmRenormalize, AIR_TRUE); // slows things down if true
  gagePerVolume *pvl; 
  if (!E) E |= !(pvl = gagePerVolumeNew(gtx, nin, gageKindScl));
  if (!E) E |= gagePerVolumeAttach(gtx, pvl);   


  // Compute initial seed from Center
  for (int i=0; i<3; i++) {
    seed[i] = this->Center[i];
  }
  double scale = helper->ScaleSelection(gtx,pvl,seed,1.0,10.0,0.1);
  vtkDebugMacro("Optimal scale: "<<scale);

  if (scale == -1) {
    // A optimal scale was not found
    // Let us choose a fix scale for now. We should bail somehow
    scale = 2;
  }

  //double kparm[3] = {3.0,0.5,0.25};
  // Derivatives should be done with the uniform cubic B-spline
  helper->SettingContext(gtx,pvl,scale);

  const double *valu = gageAnswerPointer(gtx, pvl, gageSclValue);
  const double *grad = gageAnswerPointer(gtx, pvl, gageSclGradVec);
  const double *hess = gageAnswerPointer(gtx, pvl, gageSclHessian);
  const double *hevec = gageAnswerPointer(gtx, pvl, gageSclHessEvec);
  const double *hevec3 = gageAnswerPointer(gtx, pvl,gageSclHessEvec2);
  const double *heval = gageAnswerPointer(gtx, pvl, gageSclHessEval);

  // Find optimal center
  if (this->ComputeCenter) {
      if (this->InPlane) {
        // We assume the scans are axial.
        //cout<<"Scale: "<<scale<<endl;
        helper->RelocateSeedInPlane(gtx,pvl,seed,seedNew,2);
      } else {
        helper->RelocateSeed(gtx,pvl,seed,seedNew);
      }

  }
  else {
    seedNew[0]=seed[0];
    seedNew[1]=seed[1];
    seedNew[2]=seed[2];
  }

  // Set new center
  for (int i=0; i<3; i++)
    this->Center[i] = seedNew[i];

  // Find optimal axes
  // Probe hessian in new center and find optimal axes along tube
  if (this->ComputeAxes) 
   {
    gageProbe(gtx,seedNew[0],seedNew[1],seedNew[2]);
    // Choose xAxis and yAxis such as their are the projection of i-axis and j-axis
    // in the plane given by the normal whose value is the third eigenvector of the
    // hessian.
    double m[9]; 
    double xAxis[3], yAxis[3], zAxis[3];
    double iAxis[3], jAxis[3];
    double normx,normy;
    if (this->InPlane) {
      zAxis[0]=0;
      zAxis[1]=0;
      zAxis[2] = 1;
    } 
    else {
      //Take an step along hevec3 and relocate seed
      /*
      for (int k=0;k<3;k++)
        seed2[k] = seedNew[k]+0.25*hevec3[k];
      helper->RelocateSeed(gtx,pvl,seed2,seedR);
      for (int k=0;k<3;k++)
        zAxis[k]=seedR[k]-seedNew[k];
      vtkMath::Normalize(zAxis);
      */
      // Or use hevec3 directly
      zAxis[0] =hevec3[0];
      zAxis[1] =hevec3[1];
      zAxis[2] =hevec3[2];
    }

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

    //Asigned axes
    for (int i=0; i<3; i++) {
      this->XAxis[i] = xAxis[i];
      this->YAxis[i] = yAxis[i];
      this->ZAxis[i] = zAxis[i];
    }

  }

  // Delete gage stuff     
  gageContextNix(gtx);
  helper->Delete();
  //Remove nrrd structure but don't touch nin->data
  nrrdNix(nin);
  vtkDebugMacro("Done ComputeTubeModel");
}

void vtkImageResliceWithPlane::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os <<"X Axis: "<<XAxis[0]<<" "<<XAxis[1]<<" "<<XAxis[2]<<endl;
  os <<"Y Axis: "<<YAxis[0]<<" "<<YAxis[1]<<" "<<YAxis[2]<<endl;
  os <<"Z Axis: "<<ZAxis[0]<<" "<<ZAxis[1]<<" "<<ZAxis[2]<<endl;

  os <<"Center: "<<Center[0]<<" "<<Center[1]<<" "<<Center[2]<<endl;

}

