/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkSimpleLungMask.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkSimpleLungMask.h"

#include "vtkImageData.h"
#include "vtkImageStencilData.h"
#include "vtkDoubleArray.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"

//#include "vtkITKOtsuThresholdImageFilter.h"
#include "vtkImageCast.h"
#include "vtkImageSeedConnectivity.h"
#include "vtkImageErode.h"
#include "vtkImageThreshold.h"
#include "vtkImageReslice.h"
#include "vtkExtractVOI.h"
#include "vtkImageMathematics.h"
#include "vtkImageChangeInformation.h"
#include "vtkStructuredPointsWriter.h"
#include "vtkUnsignedCharArray.h"
#include "vtkPointData.h"
#include "vtkImageConnectivity.h"
#include "vtkImageStatistics.h"
#include <vtkInformation.h>
#include <vtkStreamingDemandDrivenPipeline.h>
#include <math.h>

vtkStandardNewMacro(vtkSimpleLungMask);

//----------------------------------------------------------------------------
vtkSimpleLungMask::vtkSimpleLungMask()
{
  this->TopLungZ = 0;
  this->BottomLungZ = 0;

  this->NumberOfDilatations = 1;
  this->NumberOfErosions = 1;

  this->WholeLungLabel = 4; //This is a tmp label to assign
  this->LeftLungLabel = 3;
  this->RightLungLabel = 2;
  this->UcharTracheaLabel = 1;
  this->TracheaLabel = 512;
  this->VesselsLabel = 768;
  this->BodyLabel = 29;
  this->AirLabel = 30;
  this->UpperTracheaLabel = 512; //This value does not conform to combentions

  this->TracheaInitZ = 0;
  this->TracheaEndZ = 0;

  this->NumVoxelWholeLung =0;
  this->NumVoxelLeftLung = 0;
  this->NumVoxelRightLung = 0;
  this->NumVoxelTrachea = 0;

  this->BaseLabelLeftLung = 9;
  this->BaseLabelRightLung = 12;
  this->BaseLabelWholeLung = 20;

  this->RasToVtk = NULL;

  this->LungThreshold = 600;

  this->TracheaAreaTh = 2500; //Area in mm^2
  this->ExtractVessels = 1;
  this->VesselsThreshold = 750;

  this->ThresholdTable = vtkShortArray::New();
  this->ThresholdTable->InsertNextValue(-950+1024);
  this->ThresholdTable->InsertNextValue(-925+1024);
  this->ThresholdTable->InsertNextValue(-910+1024);
  this->ThresholdTable->InsertNextValue(-905+1024);
  this->ThresholdTable->InsertNextValue(-900+1024);
  this->ThresholdTable->InsertNextValue(-875+1024);
  this->ThresholdTable->InsertNextValue(-856+1024);

  this->LeftDMTable = vtkIntArray::New();
  this->RightDMTable = vtkIntArray::New();
  this->RasToVtk=NULL;
  this->AirIntensityBaseline = 0;

  for (int i=0; i<3;i++)
    {
    this->LCentroid[i]=0;
    this->RCentroid[i]=0;
    }
}

//----------------------------------------------------------------------------
vtkSimpleLungMask::~vtkSimpleLungMask()
{
if (this->RasToVtk) {
  this->RasToVtk->Delete();
}
this->ThresholdTable->Delete();
this->LeftDMTable->Delete();
this->RightDMTable->Delete();
}

void vtkSimpleLungMask::ComputeCentroids(vtkImageData *in, int LC[3], int RC[3])
{
    int ext[6];
    in->GetExtent(ext);

    //Image Extent for the left side
    int extL[6];
    //Image Extent for the right side
    int extR[6];

    for (int i = 0; i<6;i++) {
        extL[i] = ext[i];
        extR[i]= ext[i];
    }

    //Define left
    double laxis[4] = {-1,0,0,0};
    double lvtk[4];
    this->GetRasToVtk()->MultiplyPoint(laxis,lvtk);

    int axis = 0;
    double max_comp = fabs(lvtk[0]);
    for (int i =1; i< 3; i++) {
        if(max_comp <fabs(lvtk[i])) {
            axis = i;
            max_comp = lvtk[i];
         }
    }

    if (lvtk[axis] > 0){
        extL[axis*2] = (int) (ext[axis*2 + 1]/2 + 0.5);
        extR[axis*2+1] = extL[axis*2] -1;
    } else {
        extR[axis*2] = (int) (ext[axis*2 + 1]/2 + 0.5);
        extL[axis*2+1] = extR[axis*2]-1;
    }
    cout<<"Extent left: "<<extL[0]<<"-"<<extL[1]<<" "<<extL[2]<<"-"<<extL[3]<<" "<<extL[4]<<"-"<<extL[5]<<endl;
    cout<<"Extent right: "<<extR[0]<<"-"<<extR[1]<<" "<<extR[2]<<"-"<<extL[3]<<" "<<extR[4]<<"-"<<extR[5]<<endl;

    //Before computing centroids:
    this->ComputeCentroid(in,extL,LC);
    this->ComputeCentroid(in,extR,RC);
}

template <class T>
void vtkSimpleLungMaskComputeCentroid(vtkSimpleLungMask *self,vtkImageData *in, T *inPtr, int ext[6], int C[3])
{
    vtkIdType incX,incY,incZ;
    //Compute Left Centroid looping through data
    in->GetContinuousIncrements(ext,incX,incY,incZ);
    double Ctmp[3];
    Ctmp[0]=0;
    Ctmp[1]=0;
    Ctmp[2]=0;
    double numC=0;

    for (int idxZ = ext[4];  idxZ <= ext[5]; idxZ++ )
        {
        for(int idxY = ext[2]; idxY <= ext[3]; idxY++)
            {
            for(int idxX =ext[0]; idxX <=ext[1]; idxX++)
                {
                 if(*inPtr > 0) {
                    numC++;
                    Ctmp[0] = Ctmp[0]+(idxX-Ctmp[0])/numC;
                    Ctmp[1] = Ctmp[1]+(idxY-Ctmp[1])/numC;
                    Ctmp[2] = Ctmp[2]+(idxZ-Ctmp[2])/numC;
                  }
                  inPtr++;
                  }
            inPtr += incY;
           }
        inPtr += incZ;
       }

  C[0] = (int) Ctmp[0];
  C[1] = (int) Ctmp[1];
  C[2] = (int) Ctmp[2];
}

void vtkSimpleLungMask::ComputeCentroid(vtkImageData *in, int ext[6], int C[3])
{
  switch (in->GetScalarType())
    {
    vtkTemplateMacro5(vtkSimpleLungMaskComputeCentroid, this, in,
                       static_cast<VTK_TT*>(in->GetScalarPointerForExtent(ext)), ext,
                       C);
    default:
      vtkGenericWarningMacro("Execute: Unknown input ScalarType");
      return;
    }
}

//----------------------------------------------------------------------------
vtkImageData *vtkSimpleLungMask::PreVolumeProcessing(vtkImageData *in, int &ZCentroid)
{
    /*
    // Otsu calculator and threshold
    vtkITKOtsuThresholdImageFilter *th = vtkITKOtsuThresholdImageFilter::New();

    th->SetInput(in);
    th->SetInsideValue(1);
    th->SetOutsideValue(0);
    cout<<"Updating Otsu"<<endl;
    th->Update();
    this->LungThreshold = th->GetThreshold();
    */

    vtkImageThreshold *th = vtkImageThreshold::New();
    th->SetInputData(in);
    th->SetInValue(1);
    th->SetOutValue(0);
    th->ThresholdBetween(-100 + this->AirIntensityBaseline,this->LungThreshold + this->AirIntensityBaseline);
    th->ReplaceInOn();
    th->ReplaceOutOn();
    th->SetOutputScalarTypeToUnsignedChar();
    th->Update();

    // Remove background doing Connected components in volume corners
    int dims[3];
    in->GetDimensions(dims);
    vtkImageSeedConnectivity *bgrm = vtkImageSeedConnectivity::New();
    bgrm->SetInputConnection(th->GetOutputPort());
    bgrm->SetInputConnectValue(1);
    bgrm->SetOutputConnectedValue(1);
    bgrm->SetOutputUnconnectedValue(0);
    bgrm->AddSeed(0,0,0);
    bgrm->AddSeed(dims[0]-1,0,0);
    bgrm->AddSeed(0,dims[1]-1,0);
    bgrm->AddSeed(dims[0]-1,dims[1]-1,0);
    bgrm->AddSeed(0,0,dims[2]-1);
    bgrm->AddSeed(dims[0]-1,0,dims[2]-1);
    bgrm->AddSeed(0,dims[1]-1,dims[2]-1);
    bgrm->AddSeed(dims[0]-1,dims[1]-1,dims[2]-1);
    // Add two more seed in the posterior part to remove the gantry bed
    bgrm->AddSeed(int(dims[0]/2),0,0);
    bgrm->AddSeed(int(dims[0]/2),dims[1]-1,0);
    bgrm->Update();

    unsigned char *mask = (unsigned char *)th->GetOutput()->GetScalarPointer();
    unsigned char *bg = (unsigned char *) bgrm->GetOutput()->GetScalarPointer();

    for (int i=0; i< th->GetOutput()->GetNumberOfPoints(); i++) {
    	if (*mask == 1 && *bg == 0) {
		*mask = this->WholeLungLabel;
	} else {
		*mask = 0;
	}
	mask++;
	bg++;
    }

    bgrm->Delete();

    //Compute centroid
    cout<<"Compute centroids"<<endl;
    this->ComputeCentroids(th->GetOutput(), this->LCentroid, this->RCentroid);
    cout<<"CentroidL: "<<LCentroid[0]<<" "<<LCentroid[1]<<" "<<LCentroid[2]<<endl;
    cout<<"CentroidR: "<<RCentroid[0]<<" "<<RCentroid[1]<<" "<<RCentroid[2]<<endl;

    // Make sure we get only the components connected to the centroid
    // Here we should have a decent lung mask
    vtkImageSeedConnectivity *connect = vtkImageSeedConnectivity::New();
    connect->SetInputConnection(th->GetOutputPort());
    connect->SetInputConnectValue(this->WholeLungLabel);
    //connect->SetOutputConnectedValue(1);
    connect->SetOutputConnectedValue(this->WholeLungLabel);
    connect->SetOutputUnconnectedValue(0);
    connect->AddSeed(this->LCentroid[0],this->LCentroid[1],this->LCentroid[2]);
    connect->AddSeed(this->RCentroid[0],this->RCentroid[1],this->RCentroid[2]);
    cout<<"CC"<<endl;

    connect->Update();

  //Here we have the initial lung mask without holes and in unsigned char format for further processing
  // We reuse the cast output registering it's output
  vtkImageData *out;
  connect->GetOutput()->Register(this);
  out = connect->GetOutput();

  th->Delete();
  connect->Delete();

  ZCentroid = int( (LCentroid[2] + RCentroid[2])/2);
  return out;

  /***************************
  vtkImageData *tmp = vtkImageData::New();
  tmp->DeepCopy(cast->GetOutput());

  // Dilate and Erore
  vtkImageErode *di_er = vtkImageErode::New();

  //1. Dilate
  di_er->SetForeground(0);
  //di_er->SetBackground(this->WholeLungLabel);
  di_er->SetNeighborTo8();
  for (int i=0; i<this->GetNumberOfDilatations(); i++) {
    di_er->SetInput(tmp);
    di_er->Update();
    tmp->DeepCopy(di_er->GetOutput());
  }

  //2. Erode
   vtkImageErode *di_er = vtkImageErode::New();
  di_er->SetBackground(0);
  di_er->SetForeground(this->WholeLungLabel);
  di_er->SetNeighborTo4();
  for (int i=0; i<this->GetNumberOfErosions(); i++) {
    di_er->SetInput(tmp);
    di_er->Update();
    tmp->DeepCopy(di_er->GetOutput());
  }

  di_er->Delete();

    //WARNING: out is UnsignedChar type
    tmp->Register(this);
    ZCentroid = int( (LCentroid[2] + RCentroid[2])/2);
    return tmp;
  **/
}

//-----------------------------------------------------------------------------
// This funtion returns 1 if we should keep processing slices
//otherwise the answer is zero.
int vtkSimpleLungMask::SliceProcessing(vtkImageData *in,vtkImageData *out, int z)
{
  // Dilate - Erode: Some morphological operations to improve result
  vtkImageErode *er = vtkImageErode::New();
  vtkImageErode *di = vtkImageErode::New();

  di->SetForeground(0);
  di->SetBackground(1);
  di->SetNeighborTo4();
  er->SetBackground(0);
  er->SetForeground(1);
  er->SetNeighborTo4();

  vtkImageData *tmp = vtkImageData::New();
  tmp->DeepCopy(in);
  for (int i=0; i<this->GetNumberOfDilatations(); i++) {
    di->SetInputData(tmp);
    di->Update();
    tmp->DeepCopy(di->GetOutput());
  }

  for (int i=0; i<this->GetNumberOfErosions(); i++) {
    er->SetInputData(tmp);

    er->Update();
    tmp->DeepCopy(er->GetOutput());
  }

  // Some Level sets to refine the mask close to boundaries: Proximity equation
  // Only attachment force and curvature smoothing

  // Compute Centroids to apply to CC
  int LC[3];
  int RC[3];
  this->ComputeCentroids(in, LC, RC);
  //cout<<"Centroid Slice processing: "<<LC[0]<<" "<<LC[1]<<" "<<RC[0]<<" "<<RC[1]<<endl;
  if ((LC[0] ==0 && LC[1] ==0) &&
        (RC[0]==0 && RC[1]==0))
        return 0;

  //Add black mid line to separeta left and right from adjacent lungs
  if (LC[0] != 0 && RC[0] != 0) {
     int mid = (int) (LC[0]+RC[0])/2;
     vtkIdType incX,incY,incZ;
     int dims[3];
     tmp->GetDimensions(dims);
     tmp->GetIncrements(incX,incY,incZ);
     unsigned char *tmpPtr = (unsigned char *)tmp->GetScalarPointer()+mid;
     for(int j=0;j<dims[1];j++) {
       *tmpPtr = 0;
       tmpPtr=tmpPtr+incY;
     }
  }

  vtkImageData *mathIn1;
  vtkImageData *mathIn2;

  vtkImageSeedConnectivity *connect = vtkImageSeedConnectivity::New();
  if (LC[0] !=0 && LC[1] !=0) {
     connect->SetInputData(tmp);
     connect->AddSeed(LC[0],LC[1],0);
     connect->SetInputConnectValue(1);
     connect->SetOutputConnectedValue(this->LeftLungLabel);
     connect->SetOutputUnconnectedValue(0);
     connect->Update();
     mathIn1=connect->GetOutput();
  } else {
     mathIn1=NULL;
  }

  vtkImageSeedConnectivity *connect2 = vtkImageSeedConnectivity::New();
  if (RC[0] !=0 && RC[1] !=0) {
    connect2->SetInputData(tmp);
    connect2->AddSeed(RC[0],RC[1],0);
    connect2->SetInputConnectValue(1);
    connect2->SetOutputConnectedValue(this->RightLungLabel);
    connect2->SetOutputUnconnectedValue(0);
    connect2->Update();
    mathIn2 = connect2->GetOutput();
  } else {
    mathIn2 = NULL;
  }
  if (mathIn1 ==NULL) {
    out->DeepCopy(mathIn2);
  } else if (mathIn2 == NULL) {
    out->DeepCopy(mathIn1);
  } else {
    vtkImageMathematics *math = vtkImageMathematics::New();
    math->SetInput1Data(mathIn1);
    math->SetInput2Data(mathIn2);
    math->SetOperationToAdd();
    math->Update();
    //vtkImageData *out;
    out->DeepCopy(math->GetOutput());
    math->Delete();
  }
  //out->Register(this);
  //connect2->UnRegisterAllOutputs();

  //Clean Objects
  di->Delete();
  er->Delete();
  tmp->Delete();
  connect->Delete();
  connect2->Delete();
  //return out;
  return 1;
}

//----------------------------------------------------------------------------
// In and out should be preallocated image objects
void vtkSimpleLungMask::PostVolumeProcessing(vtkImageData *in, vtkImageData *out)
{
  int ext[6];
  out->GetExtent(ext);
  vtkIdType incX,incY,incZ;
  out->GetContinuousIncrements(ext,incX,incY,incZ);
  short *outPtr = (short *) out->GetScalarPointerForExtent(ext);
  unsigned char *inPtr = (unsigned char *) in->GetScalarPointerForExtent(ext);

  for (int idxZ = ext[4]; idxZ<=ext[5]; idxZ++) {
    for (int idxY = ext[2]; idxY<=ext[3]; idxY++) {
        for(int idxX = ext[0]; idxX<=ext[1]; idxX++) {
            if (*inPtr == 1 && *outPtr == 0)
               *outPtr = this->TracheaLabel;
            inPtr++;
            outPtr++;
         }
      }
   }
}

void vtkSimpleLungMask::AppendOutput(vtkImageData *slice,int z)
{
  int ext[6];
  this->GetOutput()->GetExtent(ext);

  vtkIdType incX, incY, incZ;

  this->GetOutput()->GetIncrements(incX,incY,incZ);

  short * outPtr = (short *) this->GetOutput()->GetScalarPointer(0,0,z);

  unsigned char* slicePtr = (unsigned char*) slice->GetScalarPointer();

  for (int i=0 ;i <incZ; i++)
    {
    *outPtr = (short) *slicePtr;
    outPtr++;
    slicePtr++;
    }
}

void vtkSimpleLungMask::ExecuteDataWithInformation(vtkDataObject *output, vtkInformation* outInfo)
{
  vtkImageData *inData = vtkImageData::SafeDownCast(this->GetInput());
  vtkImageData *outData = this->AllocateOutputData(output, outInfo);

  int outExt[6];
  outInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), outExt);

  // Make sure the Input has been set.
  if ( inData == NULL )
    {
    vtkErrorMacro(<< "ExecuteData: Input is not set.");
    return;
    }

  // Too many filters have floating point exceptions to execute
  // with empty input/ no request.
  if (this->UpdateExtentIsEmpty(outInfo, output))
    {
    return;
    }

  this->UpdateProgress(0.0);
   vtkImageStatistics *tmpFilter = vtkImageStatistics::New();
   int orgExt[6];
   inData->GetExtent(orgExt);
   int mid = int ((orgExt[5]+orgExt[4])/2.0);
   // Set input extend for on slice
   inData->SetExtent(orgExt[0],orgExt[1],orgExt[2],orgExt[3],mid,mid);
   tmpFilter->SetInputData(inData);
   tmpFilter->Update();
   if (tmpFilter->GetMedian()<0)
      this->AirIntensityBaseline = -1024;
   else
      this->AirIntensityBaseline = 0;
   // Reset extent to original one
   inData->SetExtent(orgExt);
   tmpFilter->Delete();

  short *inPtr;
  short *outPtr;
  //Init output buffer b/c we use this internally
  outPtr = (short *)outData->GetScalarPointer();
  for (int i=0; i<outData->GetNumberOfPoints();i++) {
      *outPtr = 0;
       outPtr++;
  }
  vtkImageData *preMask = NULL;  //Mask after the preprocessing
  vtkImageData *sliceMask = NULL; //Mask for slice processing 2D
  int ZCentroid;

  //cout<<"Before prevolume processing"<<endl;
  this->UpdateProgress(0.1);
  preMask = this->PreVolumeProcessing(inData,ZCentroid);
  this->UpdateProgress(0.4);
  //cout<<"Done prevolume"<<endl;

  /*******************************
  int ext[6];
  int dims[3];
  double or[3],sp[3];
  preMask->GetExtent(ext);
  preMask->GetOrigin(or);
  preMask->GetSpacing(sp);
  preMask->GetDimensions(dims);

  sliceMask = vtkImageData::New();
  vtkIndent indent;
  //Forward slice processing.
    for (int z=ZCentroid; z<=ext[5]; z++) {
	 vtkExtractVOI *voi = vtkExtractVOI::New();
     voi->SetInput(preMask);
	//voi->SetOutputOrigin(or[0]+dims[0]/2*sp[0],or[1]+dims[1]/2*sp[1],or[2]+z*sp[2]);
	voi->SetVOI(ext[0],ext[1],ext[2],ext[3],z,z);
	voi->Update();
	int result =this->SliceProcessing(voi->GetOutput(),sliceMask,z);
        voi->Delete();
	if(result != NULL)
           {
            this->AppendOutput(sliceMask,z);
            //this->AppendOutput(voi->GetOutput(),z);
	    //sliceMask->UnRegister(this);
            //sliceMask->Delete();
            }
        else
            {
            this->TopLungZ = z;
            break;
            }
    }

  //Backward slice processing.
    for (int z=ZCentroid-1; z>=ext[4]; z--) {
	//voi->SetOutputOrigin(or[0],or[1],or[2]+z*sp[2]);
	vtkExtractVOI *voi = vtkExtractVOI::New();
        voi->SetInput(preMask);
	voi->SetVOI(ext[0],ext[1],ext[2],ext[3],z,z);
	voi->Update();

	int result = this->SliceProcessing(voi->GetOutput(),sliceMask,z);
        voi->Delete();
	if(result != NULL)
            {
            this->AppendOutput(sliceMask,z);
	    //this->AppendOutput(voi->GetOutput(),z);
             //sliceMask->UnRegister(this);
             //sliceMask->Delete();
             }
        else
            {
            this->BottomLungZ = z;
            break;
            }
    }
  sliceMask->Delete();

  // Post volume processing
  //cout<<"Post volume processing ..."<<endl;
  //this->PostVolumeProcessing(preMask,outData);
  //cout<<"postvolume DONE"<<endl;
  //Here final result should be in outData;
  *************************************************/

  // Extract trachea. Based on preMask analysis, relabel output.
  //cout<<"Extracting Trachea"<<endl;
  cout<<"Extracting Trachea"<<endl;
  this->ExtractTrachea(preMask);
  this->UpdateProgress(0.5);

  // Copy almost definitive mask to output buffer
  outPtr = (short *)outData->GetScalarPointer();
  unsigned char* prePtr = (unsigned char*)preMask->GetScalarPointer();
  for (int i=0; i<outData->GetNumberOfPoints();i++) {
    if (*prePtr == this->UcharTracheaLabel) {
      *outPtr = (short) this->TracheaLabel;
    } else {
      *outPtr = (short) *prePtr;
    }
       outPtr++;
       prePtr++;
  }

  //Clean data
  //preMask->UnRegister(this);
  preMask->Delete();

// Remove holes
  vtkImageData *tmp = vtkImageData::New();
  tmp->ShallowCopy(outData);

  vtkImageConnectivity *con  = vtkImageConnectivity::New();
  con->SetInputData(tmp);
  con->SetBackground(this->WholeLungLabel);
  con->SetMinForeground(-256);
  con->SetMaxForeground(256);
  con->SetFunctionToRemoveIslands();
  con->SetMinSize(1000);
  con->SliceBySliceOn();
  cout<<"Removing Island"<<endl;
  con->Update();

  this->UpdateProgress(0.6);

  //Clean potential crap from Image Connectivity
  short *conPtr = (short *) con->GetOutput()->GetScalarPointer();
  cout<<"Cleaning Remove Islands"<<endl;
  outPtr = (short *) outData->GetScalarPointer();
  for (int i=0; i<outData->GetNumberOfPoints();i++) {
    if (*conPtr == this->WholeLungLabel || *conPtr == this->TracheaLabel)
      {
      *outPtr = *conPtr;
      }
    else
      {
      *outPtr =0;
      }
    outPtr++;
    conPtr++;
  }

  con->Delete();
  tmp->Delete();

  // Extract Upper trachea
  //this->ExtractUpperTrachea(outData);

  this->UpdateProgress(0.7);

  // Extract Vessels (Simple thresholding)
  if (this->ExtractVessels == 1) {
	inPtr = (short *) inData->GetScalarPointer();
	outPtr = (short *) outData->GetScalarPointer();
	for (int i=0; i<outData->GetNumberOfPoints();i++) {
          if (*outPtr > 0 && *inPtr > this->VesselsThreshold + this->AirIntensityBaseline) {
            *outPtr = this->VesselsLabel;
          }
       outPtr++;
       inPtr++;
  	}
  }

  this->UpdateProgress(0.8);
  // Extract Air outside the body
  int dims[3];
  inData->GetDimensions(dims);
  vtkImageThreshold *th = vtkImageThreshold::New();
  th->SetInputData(inData);
  th->SetInValue(1);
  th->SetOutValue(0);
  th->ThresholdBetween(this->AirIntensityBaseline-100,this->LungThreshold + this->AirIntensityBaseline-1);
  th->ReplaceInOn();
  th->ReplaceOutOn();
  th->SetOutputScalarTypeToUnsignedChar();
  th->Update();
  vtkImageSeedConnectivity *air = vtkImageSeedConnectivity::New();
  air->SetInputData(th->GetOutput());
  air->SetInputConnectValue(1);
  air->SetOutputConnectedValue(1);
  air->SetOutputUnconnectedValue(0);
  air->AddSeed(0,0,(int) (dims[2]/2));
  air->AddSeed(dims[0]-1,0,(int) (dims[2]/2));
  air->AddSeed(0,dims[1]-1,(int) (dims[2]/2));
  air->AddSeed(dims[0]-1,dims[1]-1,(int) (dims[2]/2));
  air->AddSeed(0,(int) (dims[1]/2), (int) (dims[2]/2));
  air->AddSeed(dims[0]-1,(int) (dims[1]/2), (int) (dims[2]/2));

  air->Update();

  unsigned char *airPtr = (unsigned char*) air->GetOutput()->GetScalarPointer();
  outPtr = (short *) outData->GetScalarPointer();
  for (int i=0; i<outData->GetNumberOfPoints();i++) {
    if (*airPtr > 0 && *outPtr == 0) {
      *outPtr = this->AirLabel;
    }
    outPtr++;
    airPtr++;
  }

  //Extract body other than the lungs
  th->ThresholdByUpper(this->LungThreshold + this->AirIntensityBaseline + 1);
  th->Update();
  int LC[3];
  int RC[3];
  int C[3];
  this->ComputeCentroids(th->GetOutput(), LC, RC);
  for (int k=0;k<3;k++)
    C[k]=(LC[k]+RC[k])/2;

  air->RemoveAllSeeds();
  air->AddSeed(C[0],C[1],C[2]);
  air->Update();

  airPtr = (unsigned char*) air->GetOutput()->GetScalarPointer();
  outPtr = (short *) outData->GetScalarPointer();
  for (int i=0; i<outData->GetNumberOfPoints();i++) {
    if (*airPtr > 0 && *outPtr == 0) {
      *outPtr = this->BodyLabel;
    }
    outPtr++;
    airPtr++;
  }

  th->Delete();
  air->Delete();

  // Slip the lung in three regions
  cout<<"Split Lung in regions"<<endl;
  this->SplitLung(outData);
  this->UpdateProgress(0.95);

  //Compute some statistics
  this->DensityMaskAnalysis();

  this->UpdateProgress(1);
}

void vtkSimpleLungMask::CopyToBuffer(vtkImageData *in, vtkImageData *out, int copyext[6]) {
  //out->SetWholeExtent(copyext);
  out->SetExtent(copyext);
  out->SetSpacing(in->GetSpacing());
  //out->SetDimensions(in->GetDimensions()[0],in->GetDimensions()[1],0);
  out->SetDimensions(copyext[1]-copyext[0] + 1,copyext[3] - copyext[2] + 1,copyext[5] - copyext[4] + 1);
  out->SetScalarType(4, this->GetInformation());
  out->AllocateScalars(this->GetInformation());
  int numPoints = (copyext[1]-copyext[0] + 1) * (copyext[3] - copyext[2] + 1) * (copyext[5] - copyext[4] + 1);

  unsigned char *inPtr = (unsigned char *) in->GetScalarPointerForExtent(copyext);
  short *slicePtr = (short *) out->GetScalarPointer();
  for (int i=0; i< numPoints; i++) {
        *slicePtr = *inPtr;
         inPtr++;
         slicePtr++;
      }
}

void vtkSimpleLungMask::FindTracheaTopCoordinates(vtkImageData *in,int initZ, int endZ,int sign, int C[3]) {
  int k = initZ;
  int ext[6];
  in->GetExtent(ext);
  double sp[3];
  in->GetSpacing(sp);
  int testext[6];
  testext[0] = ext[0];
  testext[1] = ext[1];
  testext[2] = ext[2];
  testext[3] = ext[3];
  testext[4] = 0;
  testext[5] = 0;

  //Init Output Centroid
  C[0]=0;
  C[1]=0;
  C[2]=0;

  //Set up filters that we will use
  vtkImageConnectivity *conk  = vtkImageConnectivity::New();
  conk->SetBackground(0);
  conk->SetMinForeground(0);
  conk->SetMaxForeground(10);
  conk->SetFunctionToIdentifyIslands();

  vtkImageConnectivity *conkp1  = vtkImageConnectivity::New();
  conkp1->SetBackground(0);
  conkp1->SetMinForeground(0);
  conkp1->SetMaxForeground(10);
  conkp1->SetFunctionToIdentifyIslands();

  vtkImageThreshold *th = vtkImageThreshold::New();
  th->SetInValue(1);
  th->SetOutValue(0);
  th->ReplaceInOn();
  th->ReplaceOutOn();
  th->SetOutputScalarTypeToUnsignedChar();

  vtkImageData *slice1 = vtkImageData::New();
  //slice1->SetWholeExtent(testext);
  slice1->SetExtent(testext);
  slice1->SetSpacing(in->GetSpacing());
  slice1->SetDimensions(in->GetDimensions()[0],in->GetDimensions()[1],1);
  //vtkImageConnected need short input
  slice1->SetScalarType(4, this->GetInformation());
  slice1->AllocateScalars(this->GetInformation());

  vtkImageData *slice2 = vtkImageData::New();
  //slice2->SetWholeExtent(testext);
  slice2->SetExtent(testext);
  slice2->SetSpacing(in->GetSpacing());
  slice2->SetDimensions(in->GetDimensions()[0],in->GetDimensions()[1],1);
  //vtkImageConnected need short input
  slice2->SetScalarType(4, this->GetInformation());
  slice2->AllocateScalars(this->GetInformation());

  int *hist = new int[500];
  do {
      testext[4]=k;
      testext[5]=k;
      //cout<<"Ready to copy buffer to slice "<<k<<endl;
      this->CopyToBuffer(in,slice1,testext);
      testext[4]=0;
      testext[5]=0;
      slice1->SetExtent(testext);
      conk->SetInputData(slice1);
      //cout<<"Updating Identify island"<<endl;
      conk->Update();
      testext[4]=k+sign;
      testext[5]=k+sign;
      //cout<<"Ready to copy buffer to slice "<<k+sign<<endl;
      this->CopyToBuffer(in,slice2,testext);
      testext[4]=0;
      testext[5]=0;
      slice2->SetExtent(testext);
      conkp1->SetInputData(slice2);
      //cout<<"Updating Identify island"<<endl;
      conkp1->Update();

      //Loop through each CC in slice k
      int cckp1=0;
      //Check only the first 500 components.
      /*
      if (k==0) {
	vtkStructuredPointsWriter *writer = vtkStructuredPointsWriter::New();
	writer->SetInput(conk->GetOutput());
	writer->SetFileName("aa.vtk");
	writer->Write();
      }
      */
      this->Histogram(conk->GetOutput(),hist,0,499);
      cout<<"Histogram= ";
      for (int i=0;i<20;i++) {
	cout<<hist[i]<<" ";
      }
      cout<<endl;

      for (short cc=1;cc<500;cc++) {
	int countk=hist[cc];
	if (countk == 0) {
	  continue;
	}
        if (countk*sp[0]*sp[1] > this->TracheaAreaTh) {
	  cout<<"cc ="<<cc<<" Too big to be the trachea countk "<<countk<<endl;
          continue;
	}

	if (countk < 10) {
	  cout <<"cc="<<cc<<" Too small "<<countk<<endl;
	}

        th->SetInputConnection(conk->GetOutputPort());
	th->ThresholdBetween(cc,cc);
	th->Update();
	testext[4]=0;
	testext[5]=0;
	this->ComputeCentroid(th->GetOutput(),testext,C);
	if (C[0] + C[1] + C[2] == 0) {
	  //Continue to next label
	  continue;
        }
        //Compute volume for that cc
	cout<<"Tentative Centroid: "<<C[0]<<" "<<C[1]<<" "<<C[2]<<endl;
	//Check cc at centroid location for slice k+sign
	cckp1 = conkp1->GetOutput()->GetScalarComponentAsFloat(C[0],C[1],C[2],0);
	if (cckp1 == 0) {
	  continue;
	}
        int countkp1=this->CountPixels(conkp1->GetOutput(),cckp1);
	cout<<"Count for that centroid: "<<countkp1<<endl;

	if ((fabs(double(countkp1-countk))/(0.5*(countkp1+countk)) < 0.2) & (countkp1 * sp[0]*sp[1] <= this->TracheaAreaTh) & (countkp1 > 10) ) {
	  cout<<"Trachea found at "<<k<<endl;
	  C[2]=k;
	  delete [] hist;
          slice1->Delete();
          slice2->Delete();
          th->Delete();
          conk->Delete();
          conkp1->Delete();
	  return;
	}
      }
  k = k + sign;
  } while(k!=(endZ-sign));

  delete [] hist;
  slice1->Delete();
  slice2->Delete();
  th->Delete();
  conk->Delete();
  conkp1->Delete();
}

int vtkSimpleLungMask::CountPixels(vtkImageData *in, short cc) {
 int count = 0;
 int numPoints = in->GetNumberOfPoints();
 short *inPtr = (short *)in->GetScalarPointer(0,0,0);
 for (int i = 0; i< numPoints; i++) {
   if ((*inPtr) == cc) {
     count++;
   }
   inPtr++;
 }

 return count;
}

void vtkSimpleLungMask::Histogram(vtkImageData *in, int *hist, int minbin, int maxbin) {
  int nbins = maxbin-minbin+1;
  int numPoints = in->GetNumberOfPoints();
  short *inPtr = (short *)in->GetScalarPointer(0,0,0);

  for (int i=0;i<nbins;i++) {
    hist[i]=0;
  }

 for (int i = 0; i< numPoints; i++) {
   if (((*inPtr)>=minbin) & ((*inPtr)<=maxbin)) {
     hist[(*inPtr)+minbin]++;
   }
   inPtr++;
 }
}

void vtkSimpleLungMask::ExtractTrachea(vtkImageData *in) {
    int ext[6];
    in->GetExtent(ext);

    //Define superior
    double saxis[4] = {0,0,1,0};
    double svtk[4];
    this->GetRasToVtk()->MultiplyPoint(saxis,svtk);

    int axis = 0;
    double max_comp = fabs(svtk[0]);
    for (int i =1; i< 3; i++) {
        if(max_comp <fabs(svtk[i])) {
            axis = i;
            max_comp = svtk[i];
         }
    }
    int sign,initZ,endZ;
    if (svtk[axis] > 0) {
        initZ = ext[axis*2+1];
        endZ = ext[axis*2];
        sign = -1;
    } else {
        initZ = ext[axis*2];
        endZ = ext[axis*2+1];
        sign = 1;
    }

    //cout<<"Direction: "<<sign<<endl;
    //cout<<"Init Z: "<<initZ<<" End Z:"<<endZ<<endl;

    // Extract slices from initZ: do slice by slice analysis.
    int testext[6];
    int C[3];
    testext[0] = ext[0];
    testext[1] = ext[1];
    testext[2] = ext[2];
    testext[3] = ext[3];
    testext[4] = 0;
    testext[5] = 0;

	int flag = 0;

    int numPoints = (ext[1]-ext[0] + 1) * (ext[3] - ext[2] + 1);
    unsigned char *outPtr;
    unsigned char *inPtr, *slicePtr;
    int k = initZ;
    double sp[3];
    double org[3];
    in->GetSpacing(sp);
    in->GetOrigin(org);

    vtkImageData *slice = vtkImageData::New();
    //slice->SetWholeExtent(testext);
    slice->SetExtent(testext);
    slice->SetSpacing(in->GetSpacing());
    slice->SetDimensions(in->GetDimensions()[0],in->GetDimensions()[1],1);
    vtkUnsignedCharArray *data = vtkUnsignedCharArray::New();
    data->SetNumberOfTuples(numPoints);
    slice->GetPointData()->SetScalars(data);
    slice->SetScalarType(3, this->GetInformation());
    data->Delete();

    this->FindTracheaTopCoordinates(in,initZ,endZ,sign,C);

    if (C[0] + C[1] + C[2] == 0) {
      cout<<"Trachea not found"<<endl;
      slice->Delete();
      return;
    }

    k = C[2];

    do {
        testext[4] = k;
        testext[5] = k;
            // Check pixval in seed to see if we have to stop
            // We should be always inside the trachea until we branch off.
            //cout<<"Check slice #: "<<k<<endl;
            C[2]=k;
            if (in->GetScalarComponentAsFloat(C[0],C[1],C[2],0) == 0 ) {
                cout<<"Out of trachea boundaries at "<<k<<endl;
                break;
            }

            inPtr = (unsigned char *) in->GetScalarPointerForExtent(testext);
            slicePtr = (unsigned char *) slice->GetScalarPointer();
            for (int i=0; i< numPoints; i++) {
                *slicePtr = *inPtr;
                inPtr++;
                slicePtr++;
            }
           // Do initial eroding to avoid connectivity between trachea and lung lobes
           // (good for low res scans). We could put a low res conditions in here to avoid
           // this step.
           vtkImageErode *di_er = vtkImageErode::New();
           di_er->SetInputData(slice);
           di_er->SetBackground(0);
           di_er->SetForeground(this->WholeLungLabel);
           di_er->SetNeighborTo4();
           di_er->Update();

            vtkImageSeedConnectivity *cc = vtkImageSeedConnectivity::New();
            cc->SetInputData(di_er->GetOutput());
            cc->AddSeed(C[0],C[1]);
            cc->SetInputConnectValue(this->WholeLungLabel);
            cc->SetOutputConnectedValue(this->WholeLungLabel);
            cc->SetOutputUnconnectedValue(0);
            //cout<<"Doing CC"<<endl;
            cc->Update();
	          di_er->Delete();
            //cout<<"CC done"<<endl;
            //cout<<"Getting inPtr"<<endl;
            inPtr = (unsigned char *)cc->GetOutput()->GetScalarPointer(0,0,0);

	    //Second stop condition: number of pixel selected as trachea lower than a value
	    int count = 0;
	    for (int i = 0; i< numPoints; i++) {
                if ((*inPtr)>0)
                    count++;
               inPtr++;
            }
	    //Make the count area:
	    count = (int) (count * sp[0]*sp[1]);

	    if (count > this->TracheaAreaTh) {
        cout<<"We walk into the lung at "<<k<<endl;
        cc->Delete();
        break;
	    }
              // Dilate to compensate for the erosion:
	    vtkImageErode *di_er2 = vtkImageErode::New();
      di_er2->SetForeground(0);
      di_er2->SetBackground(this->WholeLungLabel);
      di_er2->SetNeighborTo8();
      di_er2->SetInputConnection(cc->GetOutputPort());
      di_er2->Update();
	    cc->Delete();
      
	    inPtr = (unsigned char *)di_er2->GetOutput()->GetScalarPointer(0,0,0);
	    outPtr = (unsigned char *)in->GetScalarPointerForExtent(testext);
	    //cout<<"Copy process input in output"<<endl;
      for (int i = 0; i< numPoints; i++) {
          if ((short) (*inPtr) == this->WholeLungLabel)
              *outPtr = (unsigned char) (this->UcharTracheaLabel);
          inPtr++;
          outPtr++;
      }

            // Recompute centroid for seed point for next iteration
	    // Now extent is for a single slice.
	    testext[4]=0;
	    testext[5]=0;
      this->ComputeCentroid(di_er2->GetOutput(),testext,C);
      //cout<<"New seed: "<<C[0]<<" "<<C[1]<<" "<<C[2]<<endl;
      // Delete Objects
      di_er2->Delete();

    k = k + sign;
    } while(k != endZ);
  slice->Delete();

  this->TracheaInitZ = initZ;
  this->TracheaEndZ = k-sign;
}

void vtkSimpleLungMask::ExtractTracheaOLD(vtkImageData *in) {
    int ext[6];
    in->GetExtent(ext);

    //Define superior
    double saxis[4] = {0,0,1,0};
    double svtk[4];
    this->GetRasToVtk()->MultiplyPoint(saxis,svtk);

    int axis = 0;
    double max_comp = fabs(svtk[0]);
    for (int i =1; i< 3; i++) {
        if(max_comp <fabs(svtk[i])) {
            axis = i;
            max_comp = svtk[i];
         }
    }
    int sign,initZ,endZ;
    if (svtk[axis] > 0) {
        initZ = ext[axis*2+1];
        endZ = ext[axis*2];
        sign = -1;
    } else {
        initZ = ext[axis*2];
        endZ = ext[axis*2+1];
        sign = 1;
    }

    //cout<<"Direction: "<<sign<<endl;
    //cout<<"Init Z: "<<initZ<<" End Z:"<<endZ<<endl;

    // Extract slices from initZ: do slice by slice analysis.
    int testext[6];
    int C[3];
    testext[0] = ext[0];
    testext[1] = ext[1];
    testext[2] = ext[2];
    testext[3] = ext[3];
    testext[4] = 0;
    testext[5] = 0;

	int flag = 0;

    int numPoints = (ext[1]-ext[0] + 1) * (ext[3] - ext[2] + 1);
    unsigned char *outPtr;
    unsigned char *inPtr, *slicePtr;
    int k = initZ;
    double sp[3];
    double org[3];
    in->GetSpacing(sp);
    in->GetOrigin(org);

    vtkImageData *slice = vtkImageData::New();
    //slice->SetWholeExtent(testext);
    slice->SetExtent(testext);
    slice->SetSpacing(in->GetSpacing());
    slice->SetDimensions(in->GetDimensions()[0],in->GetDimensions()[1],1);
    vtkUnsignedCharArray *data = vtkUnsignedCharArray::New();
    data->SetNumberOfTuples(numPoints);
    slice->GetPointData()->SetScalars(data);
    slice->SetScalarType(3, this->GetInformation());

    do {
        testext[4] = k;
        testext[5] = k;
        if (flag == 0) {
            C[0] = 0;
            C[1] = 0;
            C[2] = 0;
            this->ComputeCentroid(in,testext,C);
            //cout<<"Testing for init trachea: "<<C[0]<<" "<<C[1]<<" "<<C[2]<<endl;
            if (C[0] + C[1] + C[2] == 0) {
                k = k + sign;
                continue;
            }
	    if (in->GetScalarComponentAsFloat(C[0],C[1],C[2],0) == 0 ) {
	      k = k + sign;
	      continue;
            }
            cout<<"Trachea found at "<<k<<endl;
            flag = 1;
            k = k-sign;
        } else {
            // Check pixval in seed to see if we have to stop
            // We should be always inside the trachea until we branch off.
            //cout<<"Check slice #: "<<k<<endl;
            C[2]=k;
            if (in->GetScalarComponentAsFloat(C[0],C[1],C[2],0) == 0 ) {
                cout<<"Out of trachea boundaries at "<<k<<endl;
                break;
            }

            inPtr = (unsigned char *) in->GetScalarPointerForExtent(testext);
            slicePtr = (unsigned char *) slice->GetScalarPointer();
            for (int i=0; i< numPoints; i++) {
                *slicePtr = *inPtr;
                inPtr++;
                slicePtr++;
            }
           // Do initial eroding to avoid connectivity between trachea and lung lobes
           // (good for low res scans). We could put a low res conditions in here to avoid
           // this step.
           vtkImageErode *di_er = vtkImageErode::New();
           di_er->SetInputData(slice);
           di_er->SetBackground(0);
           di_er->SetForeground(this->WholeLungLabel);
           di_er->SetNeighborTo4();
           di_er->Update();

            vtkImageSeedConnectivity *cc = vtkImageSeedConnectivity::New();
            cc->SetInputConnection(di_er->GetOutputPort());
            cc->AddSeed(C[0],C[1]);
            cc->SetInputConnectValue(this->WholeLungLabel);
            cc->SetOutputConnectedValue(this->WholeLungLabel);
            cc->SetOutputUnconnectedValue(0);
            //cout<<"Doing CC"<<endl;
            cc->Update();
            //cout<<"CC done"<<endl;
            outPtr = (unsigned char *)in->GetScalarPointerForExtent(testext);
            //cout<<"Getting inPtr"<<endl;
            inPtr = (unsigned char *)cc->GetOutput()->GetScalarPointer(0,0,0);

	    //Second stop condition: number of pixel selected as trachea lower than a value
	    int count = 0;
	    for (int i = 0; i< numPoints; i++) {
                if ((*inPtr)>0)
                    count++;
               inPtr++;
            }
	    //Make the count area:
	    count = (int) (count * sp[0]*sp[1]);

	    if (count > this->TracheaAreaTh) {
	    	cout<<"We walk into the lung at "<<k<<endl;
		break;
	    }
              // Dilate to compensate for the erosion:
            di_er->SetForeground(0);
            di_er->SetBackground(this->WholeLungLabel);
            di_er->SetNeighborTo8();
            di_er->SetInputConnection(cc->GetOutputPort());
            di_er->Update();
	    cc->Delete();
	    inPtr = (unsigned char *)di_er->GetOutput()->GetScalarPointer(0,0,0);
	    //cout<<"Copy process input in output"<<endl;
            for (int i = 0; i< numPoints; i++) {
                if ((short) (*inPtr) == this->WholeLungLabel)
                    *outPtr = (unsigned char) (this->UcharTracheaLabel);
               inPtr++;
                outPtr++;
            }

            // Recompute centroid for seed point for next iteration
	    // Now extent is for a single slice.
	    testext[4]=0;
	    testext[5]=0;
            this->ComputeCentroid(di_er->GetOutput(),testext,C);
            //cout<<"New seed: "<<C[0]<<" "<<C[1]<<" "<<C[2]<<endl;
            // Delete Objects
            di_er->Delete();
        }
    k = k + sign;
    } while(k != endZ);

  slice->Delete();

  this->TracheaInitZ = initZ;
  this->TracheaEndZ = k-sign;
}

void vtkSimpleLungMask::ExtractUpperTrachea(vtkImageData *outData) {
    int ext[6];
    double sp[3];
    outData->GetExtent(ext);
    outData->GetSpacing(sp);
    int testext[6];
    short *outPtr;

    int numPoints = (ext[1]-ext[0] + 1) * (ext[3] - ext[2] + 1);
    testext[0] = ext[0];
    testext[1] = ext[1];
    testext[2] = ext[2];
    testext[3] = ext[3];
    testext[4] = 0;
    testext[5] = 0;

    int sign;
    if (this->TracheaInitZ == this->TracheaEndZ)
      return;

    if (this->TracheaInitZ > this->TracheaEndZ) {
      sign = -1;
    } else {
      sign = 1;
    }

    int k =this->TracheaInitZ;
    int foundTopLung =0;
    cout<<"Extracing upper trachea"<<endl;
    cout<<"Init Z: "<<k<<" sign: "<<sign<<endl;
    do {
    testext[4]=k;
    testext[5]=k;
    outPtr = (short *) outData->GetScalarPointerForExtent(testext);

    int count = 0;
    for (int ii=0; ii<numPoints; ii++) {
      if (foundTopLung == 0)
        {
        if ( this->WholeLungLabel == (short) (*outPtr))
          {
          //found=1;
          count++;
          }
        if (this->TracheaLabel == (short) (*outPtr))
          {
          *outPtr = (this->UpperTracheaLabel);
           //cout<<"Replacing lung label"<<endl;
          }
        }
     outPtr++;
    }
    count = (int) (count*sp[0]*sp[1]);
    if (count > this->TracheaAreaTh/8.0)
      foundTopLung = 1;

    k = k+sign;
  } while(k != this->TracheaEndZ);
}

void vtkSimpleLungMask::SplitLung(vtkImageData *outData) {
 short *outPtr = (short *) outData->GetScalarPointer();

 this->NumVoxelWholeLung = 0;
 this->NumVoxelLeftLung = 0;
 this->NumVoxelRightLung = 0;
 this->NumVoxelTrachea = 0;
 for(int i=0; i<outData->GetNumberOfPoints();i++) {
   if (*outPtr == this->WholeLungLabel)
     this->NumVoxelWholeLung++;
   if (*outPtr == this->LeftLungLabel)
     this->NumVoxelLeftLung++;
   if (*outPtr == this->RightLungLabel)
     this->NumVoxelRightLung++;
   if (*outPtr == this->TracheaLabel)
     this->NumVoxelTrachea++;
   outPtr++;
 }

     //Define superior
    double saxis[4] = {0,0,1,0};
    double svtk[4];
    this->GetRasToVtk()->MultiplyPoint(saxis,svtk);
    int IS; //flag to now if the Volumes in memory is IS or SI
    //cout<<"S vtk axis: "<<svtk[0]<<" "<<svtk[1]<<" "<<svtk[2]<<endl;
    if (svtk[2]<0)
        IS =0;
     else
        IS =1;

 int thirdW = (int) this->NumVoxelWholeLung/3;
 int thirdL = (int) this->NumVoxelLeftLung/3;
 int thirdR = (int) this->NumVoxelRightLung/3;

 outPtr = (short *) outData->GetScalarPointer();
 int labelW;
 int labelL;
 int labelR;
 if (IS == 0) {
   labelL = this->BaseLabelLeftLung;
   labelR = this->BaseLabelRightLung;
   labelW = this->BaseLabelWholeLung;
 } else {
   labelL = this->BaseLabelLeftLung+2;
   labelR = this->BaseLabelRightLung+2;
   labelW = this->BaseLabelWholeLung+2;
 }
 int cW = 0;
 int cL = 0;
 int cR = 0;

 for(int i=0; i<outData->GetNumberOfPoints();i++) {
   
   if (*outPtr == this->WholeLungLabel) {
     if (cW < thirdW)
       *outPtr = labelW;
     else {
       if (IS ==0)
         labelW++;
       else
         labelW--;
       cW = 0;
       *outPtr = labelW;
     }
     cW++;
   }
   
   if (*outPtr == this->LeftLungLabel) {
     if (cL < thirdL)
       *outPtr = labelL;
     else {
       if (IS ==0)
         labelL++;
       else
         labelL--;
       cL = 0;
       *outPtr = labelL;
     }
     cL++;
   }
   if (*outPtr == this->RightLungLabel) {
     if (cR < thirdR)
       *outPtr = labelR;
     else {
       if (IS==0)
         labelR++;
       else
         labelR--;
       cR = 0;
       *outPtr = labelR;
     }
     cR++;
   }
   outPtr++;
 }
}

void vtkSimpleLungMask::DensityMaskAnalysis()
{
  vtkImageData *inData = vtkImageData::SafeDownCast(this->GetInput());
  vtkImageData *outData = this->GetOutput();
  short *inPtr;
  short *outPtr;
  vtkShortArray *ThTable = this->GetThresholdTable();
  vtkIntArray *Ltable = this->GetLeftDMTable();
  Ltable->Reset();
  Ltable->SetNumberOfComponents(3);
  Ltable->SetNumberOfTuples(ThTable->GetNumberOfTuples());

  vtkIntArray *Rtable = this->GetRightDMTable();
  Rtable->Reset();
  Rtable->SetNumberOfComponents(3);
  Rtable->SetNumberOfTuples(ThTable->GetNumberOfTuples());

  int bLL = this->BaseLabelLeftLung;
  int bLR = this->BaseLabelRightLung;

  // Analysis for regions

  //For each threhold
  for (int thIdx = 0;thIdx<ThTable->GetNumberOfTuples();thIdx++) {
    int th = (short) ThTable->GetComponent(thIdx,0) + this->AirIntensityBaseline;
    inPtr = (short *) inData->GetScalarPointer();
    outPtr = (short *) outData->GetScalarPointer();

    //Reset tables
    for(int i=0;i<3;i++) {
      Ltable->SetComponent(thIdx,i,0);
      Rtable->SetComponent(thIdx,i,0);
    }

    //Loop through image collecting pixels
    for(int i=0;i<inData->GetNumberOfPoints();i++) {
       if(*inPtr<th) {
	  if(bLL==*outPtr) {
	        Ltable->SetComponent(thIdx,0,Ltable->GetComponent(thIdx,0)+1);
	  } else if (bLL+1 == *outPtr) {
	        Ltable->SetComponent(thIdx,1,Ltable->GetComponent(thIdx,1)+1);
          } else if (bLL+2 == *outPtr) {
	        Ltable->SetComponent(thIdx,2,Ltable->GetComponent(thIdx,2)+1);
	  } else if (bLR == *outPtr) {
	        Rtable->SetComponent(thIdx,0,Rtable->GetComponent(thIdx,0)+1);
	  } else if (bLR+1 == *outPtr) {
	        Rtable->SetComponent(thIdx,1,Rtable->GetComponent(thIdx,1)+1);
          } else if (bLR+2 == *outPtr) {
	        Rtable->SetComponent(thIdx,2,Rtable->GetComponent(thIdx,2)+1);
	  }
       }
     inPtr++;
     outPtr++;
    }
  } //end loop th
}

void vtkSimpleLungMask::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Whole Lung Label:  " << this->GetWholeLungLabel() << "\n";
  os << indent << "Left Lung Label:  " << this->GetLeftLungLabel() << "\n";
  os << indent << "Right Lung Label: " << (this->GetRightLungLabel() );
}

