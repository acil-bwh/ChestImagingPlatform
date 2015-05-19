/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageTubularConfidence.cxx,v $

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
#define M_SQRT2    1.41421356237309504880168872421     /*sqrt(2)*/ 
#endif


#include "vtkImageTubularConfidence.h"

#include "vtkImageData.h"
#include "vtkImageProgressIterator.h"
#include "vtkObjectFactory.h"
#include "vtkInformationVector.h"

#include "vnl/vnl_math.h"
#include <math.h>
#include "vtkNRRDExport.h"
#define VTK_EPS 1e-12

#include "vtkTubularScaleSelection.h"
#include "vtkExtractAirwayTree.h"
#include "vtkMath.h"

vtkStandardNewMacro(vtkImageTubularConfidence);

vtkImageTubularConfidence::vtkImageTubularConfidence()
{
  this->Mask = NULL;
  this->ScaleImage = NULL;
  this->TubularType = VTK_VALLEY;
  this->Scale = 3.0;
  this->OptimizeScale = 0;
  this->NumberOfSteps =5;
  this->StepSize =0.25;
  this->ModeThreshold = 0.3;

}

vtkImageTubularConfidence::~vtkImageTubularConfidence()
{

  if (this->Mask)
    {
    this->Mask->Delete();
    }
}

//----------------------------------------------------------------------------
// This method overrides information set by parent's ExecuteInformation.
// 
// VTK6 migration note:
// Introduced this method to replace ExecuteInformation().
int vtkImageTubularConfidence::RequestInformation (
  vtkInformation       *  vtkNotUsed(request),
  vtkInformationVector ** vtkNotUsed(inputVector),
  vtkInformationVector *  outputVector)
{
  // We save six channels:
  // Channel 0 = mode
  // Channel 1 = heval0
  // Channel 2 = heval1
  // Channel 3 = tubular metric
  // Channel 4 = diretional metric
  // Channel 5 = distance when minimizing in tangent plane
  vtkDataObject::SetPointDataActiveScalarInfo(
    outputVector->GetInformationObject(0), VTK_FLOAT, 6);
  return 1;
}

//----------------------------------------------------------------------------
// This execute method handles boundaries.
// it handles boundaries. Pixels are just replicated to get values 
// out of extent.
template <class T>
void vtkImageTubularConfidenceExecute(vtkImageTubularConfidence *self, vtkImageData *inData,
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
  double xyz2[3];

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

  // Scale map image variables
  vtkImageData *inScale = NULL;
  float * inScalePtr = NULL;
  int useScaleImage =0;
  if (self->GetScaleImage() != NULL)
    {
    inScale = self->GetScaleImage();
    useScaleImage =1;
    }

  vtkTubularScaleSelection *helperScaleSelection = vtkTubularScaleSelection::New();
  helperScaleSelection->SetTubularType(self->GetTubularType());
  vtkExtractAirwayTree *helperExtractAirway = vtkExtractAirwayTree::New();
  helperExtractAirway->SetTubularType(self->GetTubularType());

  // Get increments to march through output data 
  outData->GetContinuousIncrements(outExt, outIncX, outIncY, outIncZ);
  inData->GetContinuousIncrements(outExt, inIncX, inIncY, inIncZ);
  float *outPtr = (float *) outData->GetScalarPointerForExtent(outExt);
  T *inPtr = (T *)inData->GetScalarPointerForExtent(outExt);
  int numComp = outData->GetNumberOfScalarComponents();

  if (inMask)
    {
    //inMaskPtr = (short *) inMask->GetScalarPointerForExtent(outExt);
    inMaskPtr = (short *) inMask->GetScalarPointer();
    }

  if (inScale)
    {
    inScalePtr = (float *) inScale->GetScalarPointer();
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
  int E = 0;
  gageContext *gtx = gageContextNew();
  gageParmSet(gtx, gageParmRenormalize, AIR_TRUE); // slows things down if true
  gagePerVolume *pvl; 
  if (!E) E |= !(pvl = gagePerVolumeNew(gtx, nin, gageKindScl));
  if (!E) E |= gagePerVolumeAttach(gtx, pvl);
  
  // Set some default parms for kernels
  // { scale, B, C}; (B,C)=(1,0): uniform cubic B-spline

  double scale=self->GetScale();
  self->SettingContext(gtx,pvl,scale);
  const double *valu = gageAnswerPointer(gtx, pvl, gageSclValue);
  const double *grad = gageAnswerPointer(gtx, pvl, gageSclGradVec);
  const double *hess = gageAnswerPointer(gtx, pvl, gageSclHessian);
  const double *hevec = gageAnswerPointer(gtx, pvl, gageSclHessEvec);
  const double *heval = gageAnswerPointer(gtx, pvl, gageSclHessEval);
  const double *hevec3;
  const double *hevect;
  if (self->GetTubularType() == VTK_VALLEY)
    {
    hevec3 = hevec+6;
    hevect = hevec+0;
    }
  if (self->GetTubularType() == VTK_RIDGE)
    {
    hevec3 = hevec+0;
    hevect = hevec+3;
    }
  double hevalxyz[3],hevec3xyz[3], hevalxyz2[3],hevec3xyz2[3];
  //int testId = (int) vnl_math_rnd(double(self->GetNumberOfThreads())/double(2));

  double val,val2,orth;
  double dist1, dist2;

  double maxrange = 5;  //Dynamic range of an airway

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
            for (int k =0; k<numComp;k++) {
              *outPtr = (float) 0.0;
              outPtr++;
            }
            inPtId++;
            continue;
            }
          }
        xyz[0] = x;
        xyz[1] = y;
        xyz[2] = z;

        if (self->GetOptimizeScale()) {
          if (useScaleImage)
            {
            scale = *(inScalePtr + inPtId);
            }
          else
            {
            scale = helperScaleSelection->ScaleSelection(gtx,pvl,xyz,1.0,6.0,0.25);
            }
          // Set scale
          if (scale>=1)
            self->SettingContext(gtx,pvl,scale);
            grad = gageAnswerPointer(gtx, pvl, gageSclGradVec);
            heval = gageAnswerPointer(gtx, pvl, gageSclHessEval);
            hevec = gageAnswerPointer(gtx, pvl, gageSclHessEvec);
          if (self->GetTubularType() == VTK_VALLEY)
            {
            hevec3 = hevec+6;
            hevect = hevec+0;
            }
          if (self->GetTubularType() == VTK_RIDGE)
            {
            hevec3 = hevec+0;
            hevect = hevec+3;
            }
         }


        if (scale == -1 || scale<1) {
           for (int k =0; k<numComp;k++) {
              *outPtr = (float) 0.0;
              outPtr++;
            }
            inPtId++;
            continue;
        } else {
          gageProbe(gtx, x,y,z);

          for( int i=0; i<3; i++ )
            {
            hevalxyz[i]  = heval[i];
            hevec3xyz[i] = hevec3[i];
            }
          //double mode = helperScaleSelection->Mode(heval);

          //-------
          // Compute tubularness measurment
          //
          if (self->GetTubularType() == VTK_VALLEY)
            {
            val = self->ValleyConfidenceMeasurement( hevalxyz );
            }
          if (self->GetTubularType() == VTK_RIDGE)
            {
            val = self->RidgeConfidenceMeasurement( hevalxyz );
            }

          double measureTotal = self->DirectionalConfidence(gtx,pvl,xyz);
         // Last Channel: distance when point is projected along the tangent plane normalized by scale
          if (0) {
            if (helperExtractAirway->RelocateSeedInPlane(gtx,pvl,xyz,xyz2,2) == 1) {
              dist1 = sqrt(vtkMath::Distance2BetweenPoints(xyz,xyz2));
            } else {
              dist1 = 100;
            }
            if (helperExtractAirway->RelocateSeedInPlane(gtx,pvl,xyz,xyz2,1) == 1) {
              dist2 = sqrt(vtkMath::Distance2BetweenPoints(xyz,xyz2));
            } else {
              dist2 = 100;
            }
          }

          //Relocate point in 3D and check orthogonality conditions
          helperExtractAirway->RelocateSeed(gtx,pvl,xyz,xyz2);
          double dist;
          dist = sqrt(vtkMath::Distance2BetweenPoints(xyz,xyz2));
          gageProbe(gtx, xyz2[0],xyz2[1],xyz2[2]);

          for( int i=0; i<3; i++ )
            {
            hevalxyz2[i]  = heval[i];
            hevec3xyz2[i] = hevec3[i];
            }
          if (self->GetTubularType() == VTK_VALLEY)
            {
            val2 = self->ValleyConfidenceMeasurement( hevalxyz2 );
            }
          if (self->GetTubularType() == VTK_RIDGE)
            {
            val2 = self->RidgeConfidenceMeasurement( hevalxyz2 );
            }
          //Orthogonality condition at probing location
          if (0) {
          double gradn =vtkMath::Norm(grad);
          if (gradn < maxrange)
            gradn = maxrange;
          orth = 2.0/(gradn*gradn) * (pow((grad[0]*hevect[0]+grad[1]*hevect[1]+grad[2]*hevect[2]),2) + 
                    pow((grad[0]*hevect[3]+grad[1]*hevect[4]+grad[2]*hevect[5]),2));
          //orth = 1/(gradn*gradn) * pow((grad[0]*hevec3[0]+grad[1]*hevec3[1]+grad[2]*hevec3[2]),2);
          }

         // Option 1: 1 channle
         if (0) {
          //-------
          // The final value is the tubularness measurement adjusted
          // by a measure that indicates the "straightness"  of the
          // segment centered at the current location
          //
          //*outPtr = (double)val*measureTotal/(2.0*double(numSteps));
          //*outPtr = (float)val*measureTotal/(2.0);
         }

         //Option 2" 6 channels using different measuremetns
         if (0){
          // Save multiple channels
          //*outPtr = mode;
          //outPtr++;
          *outPtr = hevalxyz[0];
          outPtr++;
          *outPtr = hevalxyz[1];
          outPtr++;
          *outPtr = val;
          outPtr++;
          *outPtr = orth;
          outPtr++;
          *outPtr = measureTotal;
          outPtr++;
          if (dist1<dist2) {
            *outPtr = dist1/scale;
          } else {
            *outPtr = dist2/scale;
          }
         }
 
         // Option 3
         *outPtr = hevalxyz[0];
         outPtr++;
         *outPtr = hevalxyz2[0];
         outPtr++;
         *outPtr = val;
         outPtr++;
         *outPtr = val2;
         outPtr++;
         *outPtr = measureTotal;
         outPtr++;
         *outPtr = dist/scale;
         outPtr++;
         inPtId++;
        }

       }
       outPtr += outIncY;
       inPtId += inIncY;
     }
     outPtr += outIncZ;
     inPtId += inIncZ;
   }

  // Clean objects
  nrrdexport->Delete();
  gageContextNix(gtx);
  helperScaleSelection->Delete();
  helperExtractAirway->Delete();
}

//----------------------------------------------------------------------------
// This method contains a switch statement that calls the correct
// templated function for the input data type.  The output data
// must match input type.  This method does handle boundary conditions.
void vtkImageTubularConfidence::ThreadedExecute(vtkImageData *inData, 
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

  int dim[3];
  inData->GetDimensions(dim);

  if (this->GetMask()!= NULL)
   { 
   if (this->GetMask()->GetScalarType() != VTK_SHORT)
      {
      vtkErrorMacro(<< "Execute: mask must be short ");
      return;
      }
   int dim2[3];
   this->GetMask()->GetDimensions(dim2);
   if (dim[0] != dim2[0] || dim[1] != dim2[1] || dim[2] != dim2[2]) 
     {
     vtkErrorMacro(<<"Execute: Mask dimensions does not match Input image dimensions");
     return;
     }
   }

  if (this->GetScaleImage()!= NULL)
   { 
   if (this->GetScaleImage()->GetScalarType() != VTK_FLOAT)
      {
      vtkErrorMacro(<< "Execute: scale image must be float ");
      return;
      }
   int dim2[3];
   this->GetScaleImage()->GetDimensions(dim2);
   if (dim[0] != dim2[0] || dim[1] != dim2[1] || dim[2] != dim2[2]) 
     {
     vtkErrorMacro(<<"Execute: Mask dimensions does not match Input image dimensions");
     return;
     }
   }

  switch (inData->GetScalarType())
    {
    vtkTemplateMacro(
      vtkImageTubularConfidenceExecute(
        this, inData, outData, outExt, id, static_cast<VTK_TT *>(0)
      )
    );
    default:
      vtkErrorMacro(<< "Execute: Unknown ScalarType");
      return;
    }
}

double vtkImageTubularConfidence::ValleyConfidenceMeasurement(const double *heval)
{
  double ans,A , B, S;
  double alpha = 0.5;
  double beta = 0.5;
  double gamma = 5;
  double cc = 1e-6;

    if (heval[0] <0 || heval[1]<0) {
      ans = 0;
    }
    else if (AIR_ABS(heval[0])<1e-10 || AIR_ABS(heval[1])<1e-10) {
      ans = 0;
    }
    else {
      A = AIR_ABS(heval[1])/AIR_ABS(heval[0]);
      B = AIR_ABS(heval[2])/sqrt(AIR_ABS(heval[1]*heval[0]));
      S = sqrt(heval[0]*heval[0] + heval[1]*heval[1] + heval[2]*heval[2]);
      ans= (1-exp(-A*A/(2*alpha*alpha))) *
        exp(-B*B/(2*beta*beta)) *
        (1-exp(-S*S/(2*gamma*gamma))) *
        exp(-2*cc*cc/(AIR_ABS(heval[1])*heval[0]*heval[0]));
    }
  return ans;
}

double vtkImageTubularConfidence::RidgeConfidenceMeasurement(const double *heval)
{
  double ans, A, B, S;
  double alpha = 0.5;
  double beta = 0.5;
  double gamma = 5;
  double cc = 1e-6;

   if (heval[1] >0 || heval[2]>0) {
      ans = 0;
    }
    else if (AIR_ABS(heval[1])<1e-10 || AIR_ABS(heval[2])<1e-10) {
      ans = 0;
    }
    else {
      A = AIR_ABS(heval[1])/AIR_ABS(heval[2]);
      B = AIR_ABS(heval[0])/sqrt(AIR_ABS(heval[1]*heval[2]));
      S = sqrt(heval[0]*heval[0] + heval[1]*heval[1] + heval[2]*heval[2]);
      ans = (1-exp(-A*A/(2*alpha*alpha))) *
        exp(-B*B/(2*beta*beta)) *
        (1-exp(-S*S/(2*gamma*gamma))) *
        exp(-2*cc*cc/(AIR_ABS(heval[1])*heval[2]*heval[2]));
    }
  return ans;
}

double vtkImageTubularConfidence::DirectionalConfidence(gageContext *gtx,gagePerVolume *pvl,
                        double xyz[3]) {

  int    numSteps = this->GetNumberOfSteps();
  double stepSize = this->GetStepSize();
  double measureTotal = 0.0;
  double nextPos[3];
  double nextPos2[3];

  double nextVec[3];
  double currentPos[3];
  double currentVec[3];
  double originalPos[3];
  double originalVec[3];
  double x,y,z;
  x=xyz[0];
  y=xyz[1];
  z=xyz[2];
  originalPos[0] = xyz[0];   originalPos[1] = xyz[1];   originalPos[2] = xyz[2];
  currentPos[0] = xyz[0];    currentPos[1] = xyz[1];    currentPos[2] = xyz[2];

  const double *hevec3;
  const double *hevec = gageAnswerPointer(gtx, pvl, gageSclHessEvec);
  if (this->GetTubularType() == VTK_VALLEY)
    hevec3 = hevec+6;
  if (this->GetTubularType() == VTK_RIDGE)
    hevec3 = hevec+0;

  //Relocate initial point before starting tracking. The idea is to place the seed at the intensity minima to start
  // the tracking process from there.
  vtkExtractAirwayTree *helperExtractAirway = vtkExtractAirwayTree::New();
  helperExtractAirway->SetTubularType(this->GetTubularType());

  if (helperExtractAirway->RelocateSeed(gtx,pvl,xyz,currentPos) == 0) {
    currentPos[0] = xyz[0];
    currentPos[1] = xyz[1];
    currentPos[2] = xyz[2];
  }

  gageProbe(gtx, currentPos[0],currentPos[1],currentPos[2]);

  for ( int i=0; i<3; i++ )
    {
      originalPos[i] = currentPos[i];
      originalVec[i] = hevec3[i];
      currentVec[i]  = hevec3[i]; 
      nextVec[i] = hevec3[i];
     }

  for ( int s=0; s<=1; s++ )
    {
      for ( int n=0; n<numSteps; n++ )
        {
          for ( int i=0; i<3; i++ )
            {
              nextPos[i] = currentPos[i] + stepSize*currentVec[i];
            }

          //-------
          // Each time 'gageProbe' is called, 'hevec3' is reset to
          // reflect the value at the position passed to
          // 'gageProbe' 
          //
          if (helperExtractAirway->RelocateSeed(gtx,pvl,nextPos,nextPos2) == 1) {
            nextPos[0] = nextPos2[0];
            nextPos[1] = nextPos2[1];
            nextPos[2] = nextPos2[2];
          } 
          gageProbe( gtx, nextPos[0], nextPos[1], nextPos[2] );
          for (int i=0; i<3;i++) {
            currentVec[i]=nextPos[i]-currentPos[i];
          }
          vtkMath::Normalize(currentVec);
          // Stop tracking if mode is too positive 
          // We use a similar stopping condition than vtkExtractAirwayTree.
          // if (self->Mode(heval) > self->GetModeThreshold())
          //  break;
          for ( int i=0; i<3; i++ )
            {
            nextVec[i] = hevec3[i];
            }

          //-------
          // Check the angle between the current and the previous
          // vectors.  If it is greater than 90 and less than 235,
          // flip it to prevent "reversal of direction"
          //
          double dotProduct = nextVec[0]*currentVec[0] + nextVec[1]*currentVec[1] + nextVec[2]*currentVec[2];

          if ( dotProduct < 0 )
            {
            for ( int i=0; i<3; i++ )
              {
              nextVec[i] = -nextVec[i];
              }
            }

          //-------
          // Now that we have the original vector and the next
          // vector, we compute a measure related to the angle
          // between them.
          //
          measureTotal +=
          pow(fabs(originalVec[0]*nextVec[0]+originalVec[1]*nextVec[1]+originalVec[2]*nextVec[2]),2);

          // Alternatively, we can wait until the end to just use the final vectors.
          //if ( n == (numSteps-1) )
          //  {
          //  measureTotal += pow(fabs(originalVec[0]*nextVec[0]+originalVec[1]*nextVec[1]+originalVec[2]*nextVec[2]),2);
          //  }

          //-------
          // Now set the current position to the next position, and
          // the current vector to the next vector
          //
          for ( int i=0; i<3; i++ )
            {
            currentPos[i] = nextPos[i];
            currentVec[i] = nextVec[i];
            }
          }


          //-------
          // Now that we've traversed in the "forward" direction,
          // reset to the original position and the *negative* of
          // the original vector to traverse in the "backwards"
          // direction
          //
          for ( int i=0; i<3; i++ )
            {
            currentPos[i] =  originalPos[i]; 
            currentVec[i] = -originalVec[i]; 
            }
         }

  helperExtractAirway->Delete();
  return measureTotal/(2*numSteps);

}

int vtkImageTubularConfidence::SettingContext(gageContext *gtx,gagePerVolume *pvl,double scale) 
{

  double kparm[3] = {3.0, 1.0, 0.0};
  kparm[0] = scale;
  int E=0;
  if (!E) E |= gageKernelSet(gtx, gageKernel00, nrrdKernelBCCubic, kparm);
  if (!E) E |= gageKernelSet(gtx, gageKernel11, nrrdKernelBCCubicD, kparm);
  if (!E) E |= gageKernelSet(gtx, gageKernel22, nrrdKernelBCCubicDD, kparm);
  // Define items to query
  if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclValue);
  if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclGradVec);
  if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclHessian);
  if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclHessEvec);
  if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclHessEval);
  if (!E) E |= gageUpdate(gtx);

  return E;

}








