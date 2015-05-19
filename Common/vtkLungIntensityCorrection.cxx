/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkLungIntensityCorrection.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkLungIntensityCorrection.h"

#include "vtkImageData.h"
#include "vtkImageStencilData.h"
#include "vtkDoubleArray.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkExecutive.h"

#include <math.h>

vtkStandardNewMacro(vtkLungIntensityCorrection);

//----------------------------------------------------------------------------
// VTK6 migration note (following pattern of Imaging/Core/vtkImageBlend):
// - added this->SetNumberOfInputPorts(2)
//----------------------------------------------------------------------------
vtkLungIntensityCorrection::vtkLungIntensityCorrection()
{
  this->ReverseStencil = 0;
  this->ClampNegativeValues = 1;
  this->Parameters = vtkDoubleArray::New();
  this->DCValue = DCMEAN;
  this->UseWeightedLS = 1;

  // we have the image inputs and the optional stencil input
  this->SetNumberOfInputPorts(2);
}

//----------------------------------------------------------------------------
// VTK6 migration note:
// deleted this->Parameters->Delete();
//----------------------------------------------------------------------------
vtkLungIntensityCorrection::~vtkLungIntensityCorrection()
{
}

//----------------------------------------------------------------------------
// VTK6 migration note (following pattern of Imaging/Core/vtkImageBlend):
// - changed name from SetStencil to SetStencilData
// - changed SetNthInput(1, stencil) to SetInputDataInternal(1, stencil)
//----------------------------------------------------------------------------
void vtkLungIntensityCorrection::SetStencilData(vtkImageStencilData *stencil)
{
  this->SetInputDataInternal(1, stencil); 
}

//----------------------------------------------------------------------------
// VTK6 migration note (following pattern of Imaging/Core/vtkImageBlend):
// - changed NumberOfInputs to GetNumberOfInputConnections(0)
// - changed Inputs[1] to GetExecutive()->GetInputData(0, 1)
vtkImageStencilData *vtkLungIntensityCorrection::GetStencil()
{
  if (this->GetNumberOfInputConnections(1) < 1) // because the port is optional
    {
    return NULL;
    }
  return vtkImageStencilData::SafeDownCast(
    this->GetExecutive()->GetInputData(1, 0));
}

// ---------------------------------------------------------------------------
// VTK6 migration note:
// - introduced this method combining ExecuteInformation() and
//   ExecuteInformation(vtkImageData*, vtkImageData*) 
// - before migration ExecuteInformation() called
//   vtkImageToImageFilter::ExecuteInformation() where it called the latter
//   (overrided) version
int vtkLungIntensityCorrection::RequestInformation (
  vtkInformation       *  request,
  vtkInformationVector ** inputVector,
  vtkInformationVector *  outputVector)
{
  this->Superclass::RequestInformation(request, inputVector, outputVector);
  
  vtkImageStencilData* stencil = this->GetStencil();
  vtkImageData* input = vtkImageData::GetData(inputVector[0]);
  
  // need to set the spacing and origin of the stencil to match the output
  if (stencil)
    {
    stencil->SetSpacing(input->GetSpacing());
    stencil->SetOrigin(input->GetOrigin());
    }
    
  //Allocate array of parameters
  int dim[3];
  input->GetDimensions(dim);

  if (dim[2] != this->Parameters->GetNumberOfTuples())
    {
    this->Parameters->SetNumberOfComponents(2);
    this->Parameters->SetNumberOfTuples(dim[2]);
    }
    
  return 1;
}

//----------------------------------------------------------------------------
/* original implementation before VTK6 migration
void vtkLungIntensityCorrection::ExecuteInformation(vtkImageData *input, 
                                         vtkImageData *vtkNotUsed(output))
{
  // need to set the spacing and origin of the stencil to match the output
  vtkImageStencilData *stencil = this->GetStencil();
  if (stencil)
    {
    stencil->SetSpacing(input->GetSpacing());
    stencil->SetOrigin(input->GetOrigin());
    }

  //Allocate array of parameters
  int dim[3];
  input->GetDimensions(dim);
  
  if (dim[2]!= this->Parameters->GetNumberOfTuples())
    {
    this->Parameters->SetNumberOfComponents(2);
    this->Parameters->SetNumberOfTuples(dim[2]);
    }
  //this->Parameters->Reset();

}
*/


//----------------------------------------------------------------------------
template <class T>
void vtkLungIntensityCorrectionExecute(vtkLungIntensityCorrection *self,
                            vtkImageData *inData, T *inPtr,
                            vtkImageData *outData, T *outPtr,
                            int outExt[6], int id)
{
  int numscalars;
  int idX, idY, idZ;
  int r1, r2, cr1, cr2, iter, rval;
  vtkIdType outIncX, outIncY, outIncZ;
  int inExt[6];
  vtkIdType inInc[3], outInc[3];
  unsigned long count = 0;
  int idx;
  unsigned long target;
  T *tempPtr, *outtempPtr;
  double value,tmpvalue;
  double nvalue;
  double dcvalue;
  int minY,maxY;
 
  // get the clipping extents
  vtkImageStencilData *stencil = self->GetStencil();

 // get parameters array
  vtkDoubleArray *parameters = self->GetParameters();

  // find maximum input range
  inData->GetExtent(inExt);
  inData->GetIncrements(inInc);

  // Get Increments to march through data 
  outData->GetContinuousIncrements(outExt, outIncX, outIncY, outIncZ);
  outData->GetIncrements(outInc);
  numscalars = inData->GetNumberOfScalarComponents();

  target = (unsigned long)
    ((outExt[5]-outExt[4]+1)*(outExt[3]-outExt[2]+1)/50.0);
  target++;  
  

  // Allocate matrices for LS problem
    double **xt;
    double **yt;
    double **Mt;
    double *w;
    int numdimY = inExt[3]-inExt[2]+1;
    xt = new double* [numdimY];
    yt = new double* [numdimY];
    for (int i=0;i<numdimY;i++)
       {
        xt[i] = new double [2];
        yt[i] = new double [1];
        }
    Mt = new double* [2];
    double m,n;
    Mt[0] = &m;
    Mt[1] = &n;
    
    if (self->GetUseWeightedLS()) 
      {
       w = new double [numdimY];
      }
      

  //Init minY maxY
  minY = outExt[3];
  maxY = outExt[2];

  // Loop through output pixels
  for (idZ = outExt[4]; idZ <= outExt[5]; idZ++)
    {
    idx = 0;
    cout<<"Slice "<<idZ<<endl;
    for (idY = outExt[2]; idY <= outExt[3]; idY++)
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
      if (self->GetReverseStencil())
        { // flag that we want the complementary extents
        iter = -1;
        }

      cr1 = outExt[0];
      value = 0.0;
      nvalue = 0;
      rval = 0;
      for (;;)
        {

        r1 = outExt[1] + 1;
        r2 = outExt[1];
        if (stencil)
          {
          rval = stencil->GetNextExtent(r1, r2, outExt[0], outExt[1],
                                        idY, idZ, iter);
          if (rval == 0)
            {
             break;
            }
          }
        else
          {
	      
          r1 = outExt[0];
          r2 = outExt[1];
          if (rval == 1)
            {
             break;
            }
          rval = 1;
          }

	if (idY < minY)
	  minY = idY;
	if (idY > maxY)
	  maxY = idY;
  

        cr1 = r1;
        cr2 = r2;
        tempPtr = inPtr + (inInc[2]*(idZ - inExt[4]) +
                           inInc[1]*(idY - inExt[2]) +
                           numscalars*(r1 - inExt[0]));

        //Loop through region gathering data
        tmpvalue =0;
        for (idX = r1; idX<= r2; idX++)
          {
          // gather mean pixel value
          tmpvalue += (double) *tempPtr;
          tempPtr++;  
          } 
        value += tmpvalue/(r2-r1+1);
        nvalue = nvalue+1;

        } //end of loop along x stencil

        if (nvalue > 0)
          {
           value = value/nvalue;
           //cout <<"Value: "<<value<<endl;

           //Fill matrices to solve LS problem
           xt[idx][0] = idY;
           xt[idx][1] = 1.0;
           yt[idx][0] = value;
	   
	   if (self->GetUseWeightedLS())
	     {
	     w[idx] = nvalue/(outExt[1]-outExt[0]+1);
	     }
           idx++;
           }
        }
      
   
    //Solve LS problem
    if (idx > 0) 
      {
      
      if (self->GetUseWeightedLS())
        {
	 vtkLungIntensityCorrection::SolveWeightedLeastSquares(idx,xt,2,yt,1,w,Mt);
	}
      else 
        {	 
         vtkMath::SolveLeastSquares(idx,xt,2,yt,1,Mt);
        }
      parameters->SetComponent(idZ,0,m);
      parameters->SetComponent(idZ,1,n);
      }
    else {
      //Nothing to do. Flat line
      m = 0;
      n = 0;
      parameters->SetComponent(idZ,0,m);
      parameters->SetComponent(idZ,1,n);
      }


    // Update Output

    //Compute dc value
    switch (self->GetDCValue())
      {
      case DCLOW: 
         if (m>0)
           dcvalue = (m*minY+n);
         else
           dcvalue = (m*maxY+n);
         break;
      case DCMEAN:
         dcvalue = (m*(maxY+minY) + 2*n)/2.0;
         break;
      case DCHIGH:
         if (m>0)
           dcvalue = (m*maxY+n);
         else
           dcvalue = (m*minY+n);
         break;
      }
    cout<<"parameters (m,n): "<<m<<" "<<n<<endl;  
    cout<<"dcvalue: "<<dcvalue<<" minY: "<<minY<<" maxY:"<<maxY<<endl;

    for (idY = outExt[2]; idY <= outExt[3]; idY++)
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
      if (self->GetReverseStencil())
        { // flag that we want the complementary extents
        iter = -1;
        }

      rval = 0;
      for (;;)
        {

        r1 = outExt[1] + 1;
        r2 = outExt[1];
        if (stencil)
          {
          rval = stencil->GetNextExtent(r1, r2, outExt[0], outExt[1],
                                        idY, idZ, iter);
          if (rval == 0)
            {
             break;
            }
          }
        else
          {
          r1 = outExt[0];
          r2 = outExt[1];
          if (rval == 1)
            {
             break;
            }
          rval = 1;
          }
        tempPtr = inPtr + (inInc[2]*(idZ - inExt[4]) +
                           inInc[1]*(idY - inExt[2]) +
                           numscalars*(r1 - inExt[0]));
        outtempPtr = outPtr + (outInc[2]*(idZ - outExt[4]) +
                           outInc[1]*(idY - outExt[2]) +
                           numscalars*(r1 - outExt[0]));

        for (idX = r1; idX<= r2; idX++)
          {
          //Correction
	  *outtempPtr = *tempPtr - (T)(idY*m + n - dcvalue);
	  if (self->GetClampNegativeValues()==1)
	    {
	     if (*outtempPtr<0)
	       *outtempPtr = 0;
	    }
           outtempPtr++;
           tempPtr++;
           }
        
         }
   
      // Copy input data into output in the areas outside the stencil
      iter = -1;
      if (self->GetReverseStencil())
        { // flag that we want the complementary extents
        iter = 0;
        }     
      rval = 0;
      for (;;)
        {

        r1 = outExt[1] + 1;
        r2 = outExt[1];
        if (stencil)
          {
          rval = stencil->GetNextExtent(r1, r2, outExt[0], outExt[1],
                                        idY, idZ, iter);
          if (rval == 0)
            {
             break;
            }
          }
        else
          {
            //Data has been already copied to the output in the previous pass.
	    break;
          }
        tempPtr = inPtr + (inInc[2]*(idZ - inExt[4]) +
                           inInc[1]*(idY - inExt[2]) +
                           numscalars*(r1 - inExt[0]));
        outtempPtr = outPtr + (outInc[2]*(idZ - outExt[4]) +
                           outInc[1]*(idY - outExt[2]) +
                           numscalars*(r1 - outExt[0]));

        for (idX = r1; idX<= r2; idX++)
          {
          //Correction
          *outtempPtr = *tempPtr;
           outtempPtr++;
           tempPtr++;
           }
        
         }      
   
    //outPtr += outIncY;
   }

    // Jump to next slices
    //outPtr += outIncZ;
   }

//free data
for (int i=0;i<numdimY;i++)
 {
  delete xt[i];
  delete yt[i];
 }
 delete xt;
 delete yt;
 delete Mt;
}


//----------------------------------------------------------------------------
// VTK6 migration note:
// - changed vtkTemplateMacro7 to vtkTemplateMacro
void vtkLungIntensityCorrection::ThreadedExecute(vtkImageData *inData, 
                                      vtkImageData *outData,
                                      int outExt[6], int id)
{
  void *inPtr;
  void *outPtr;

  vtkDebugMacro("Execute: inData = " << inData << ", outData = " << outData);
  
  inPtr = inData->GetScalarPointer();
  outPtr = outData->GetScalarPointerForExtent(outExt);

  switch (inData->GetScalarType())
    {
    vtkTemplateMacro(
      vtkLungIntensityCorrectionExecute(
        this, inData, (VTK_TT *)(inPtr), outData, (VTK_TT *)(outPtr), outExt, id
      )
    );
    default:
      vtkErrorMacro("Execute: Unknown ScalarType");
      return;
    }
}

// Solves for the weighted least squares best fit matrix for the equation X'M' =  Y' with weights W=diag(w).
// Uses pseudoinverse to get the ordinary least squares. 
// The inputs and output are transposed matrices.
//    Dimensions: X' is numberOfSamples by xOrder,
//                Y' is numberOfSamples by yOrder,
//                w is the weight vector and is  numberofSamples vector.
//                M' dimension is xOrder by yOrder.
// M' should be pre-allocated. All matrices are row major. The resultant
// matrix M' should be pre-multiplied to X' to get Y', or transposed and
// then post multiplied to X to get Y.
// The solution is: M = inv(X'W'WX) X'W'WY
int vtkLungIntensityCorrection::SolveWeightedLeastSquares(int numberOfSamples, double **xt, int xOrder,
                               double **yt, int yOrder, double *w, double **mt)
			       
{
  // check dimensional consistency
  if ((numberOfSamples < xOrder) || (numberOfSamples < yOrder))
    {
    vtkGenericWarningMacro("Insufficient number of samples. Underdetermined.");
    return 0;
    }

  int i, j, k;
  
  // set up intermediate variables
  double **XWWXt = new double *[xOrder];     // size x by x
  double **XWWXtI = new double *[xOrder];    // size x by x
  double **XWWYt = new double *[xOrder];     // size x by y
  for (i = 0; i < xOrder; i++)
    {
    XWWXt[i] = new double[xOrder];
    XWWXtI[i] = new double[xOrder];

    for (j = 0; j < xOrder; j++)
      {
      XWWXt[i][j] = 0.0;
      XWWXtI[i][j] = 0.0;
      }

    XWWYt[i] = new double[yOrder];
    for (j = 0; j < yOrder; j++)
      {
      XWWYt[i][j] = 0.0;
      }
    }
  
   // first find the pseudoinverse matrix
   for (k = 0; k < numberOfSamples; k++)
    {
    for (i = 0; i < xOrder; i++)
      {
      // first calculate the XWWXt matrix, only do the upper half (symmetrical)
      for (j = i; j < xOrder; j++)
        {
        XWWXt[i][j] += xt[k][i] *w[k]*w[k]* xt[k][j];
        }
      // now calculate the XYt matrix
      for (j = 0; j < yOrder; j++)
        {
        XWWYt[i][j] += xt[k][i] *w[k]*w[k]* yt[k][j];
        }
      }
    }
    
   // now fill in the lower half of the XWWXt matrix
  for (i = 0; i < xOrder; i++)
    {
    for (j = 0; j < i; j++)
      {
      XWWXt[i][j] = XWWXt[j][i];
      }
    }
  
  // next get the inverse of XXt
  if (!(vtkMath::InvertMatrix(XWWXt, XWWXtI, xOrder)))
    {
    return 0;
    }
  
  // next get m
  for (i = 0; i < xOrder; i++)
    {
    for (j = 0; j < yOrder; j++)
      {
      mt[i][j] = 0.0;
      for (k = 0; k < xOrder; k++)
        {
        mt[i][j] += XWWXtI[i][k] * XWWYt[k][j];
        }
      }
    }     	
  // clean up:
  // set up intermediate variables
  for (i = 0; i < xOrder; i++)
    {
    delete [] XWWXt[i];
    delete [] XWWXtI[i];

    delete [] XWWYt[i];
    }
  delete [] XWWXt;
  delete [] XWWXtI;
  delete [] XWWYt;
  
  return 1;
}

//----------------------------------------------------------------------------
// VTK6 migration note:
// Introduced to specify the stencil data is optional
int vtkLungIntensityCorrection::FillInputPortInformation(int port, vtkInformation* info)
{
  if (port == 0)
    {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
    info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
    }
  if (port == 1)
    {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageStencilData");
    // the stencil input is optional
    info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
    }
  else
    {
    vtkErrorMacro("Invalid input port is given in vtkLungIntensityCorrection::FillInputPortInformation");
    return 0;
    }
  return 1;
}

//----------------------------------------------------------------------------
void vtkLungIntensityCorrection::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Stencil: " << this->GetStencil() << "\n";
  os << indent << "ReverseStencil: " << (this->ReverseStencil ?
                                         "On\n" : "Off\n");

}

