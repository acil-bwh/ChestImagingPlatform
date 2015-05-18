/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkComputeCentroid.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkComputeCentroid.h"

#include "vtkImageData.h"
#include "vtkImageStencilData.h"
#include "vtkDoubleArray.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

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

#include <math.h>

vtkStandardNewMacro(vtkComputeCentroid);

//----------------------------------------------------------------------------
vtkComputeCentroid::vtkComputeCentroid()
{
  
this->Centroid[0] = 0;
this->Centroid[1] = 0;
this->Centroid[2] = 0;
}

//----------------------------------------------------------------------------
vtkComputeCentroid::~vtkComputeCentroid()
{
}


template <class T>
void vtkComputeCentroidComputeCentroid(vtkComputeCentroid *self,
  vtkImageData *in, T *inPtr, int ext[6], double C[3])          
{
    vtkIdType incX,incY,incZ;
    //Compute Left Centroid looping through data
    in->GetContinuousIncrements(ext,incX,incY,incZ);
    unsigned int Ctmp[3];
    Ctmp[0]=0;
    Ctmp[1]=0;
    Ctmp[2]=0;
    unsigned int numC=0;

    for (int idxZ = ext[4];  idxZ <= ext[5]; idxZ++ )
        {
        for(int idxY = ext[2]; idxY <= ext[3]; idxY++)
            {
            for(int idxX =ext[0]; idxX <=ext[1]; idxX++)
                {
                 if(*inPtr > 0) {
                    Ctmp[0] += idxX;
                    Ctmp[1] += idxY;
                    Ctmp[2] += idxZ;
                    numC++;
                  }
                  inPtr++;
                  }
            inPtr += incY;
           }
        inPtr += incZ;
       }
       
    if (numC>0) {   
       C[0] =  (double) Ctmp[0]/numC;
       C[1] =  (double) Ctmp[1]/numC;
       C[2] =  (double) Ctmp[2]/numC;      
    } else {
       C[0] = 0;
       C[1] = 0;
       C[2] = 0;
    }   
  
}   
 
//----------------------------------------------------------------------------
// VTK6 migration note:
// - replaced vtkTemplateMacro5 with vtkTemplateMacro
void vtkComputeCentroid::ComputeCentroid(vtkImageData *in, int ext[6], double C[3])
{    

  switch (in->GetScalarType())
    {
    vtkTemplateMacro(
      vtkComputeCentroidComputeCentroid(
        this, in, (VTK_TT *) in->GetScalarPointerForExtent(ext), ext, C
      )
    );
    default:
      vtkGenericWarningMacro("Execute: Unknown input ScalarType");
      return;
    }
}

void vtkComputeCentroid::ComputeCentroid() {
  this->Update();
}

//----------------------------------------------------------------------------
// VTK6 migration note:
// - introduced to replace ExecuteInformation()
int vtkComputeCentroid::RequestInformation (
  vtkInformation       *  vtkNotUsed(request),
  vtkInformationVector ** vtkNotUsed(inputVector),
  vtkInformationVector *  outputVector)
{
  vtkInformation* outInfo = outputVector->GetInformationObject(0);
  // numComponents == -1 => value will not be changed
  vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_SHORT, -1);
   
  return 1;
}  
  
//----------------------------------------------------------------------------
// VTK6 migration note:
// - introduced to replace ExecuteData()
void vtkComputeCentroid::ExecuteDataWithInformation(vtkDataObject *out, 
  vtkInformation* outInfo)
{
  vtkImageData* input = vtkImageData::SafeDownCast(this->GetInput());
  
  // Make sure the Input has been set.
  if ( input == NULL )
    {
    vtkErrorMacro(<< "ExecuteData: Input is not set.");
    return;
    }
    

  //vtkImageData *outData = this->AllocateOutputData(out);
  int ext[6];
  input->GetExtent(ext);
  this->ComputeCentroid(input, ext, this->Centroid);
}

       
void vtkComputeCentroid::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Centroid " << this->GetCentroid()[0]<<" "<<
  this->GetCentroid()[1]<<" "<<this->GetCentroid()[2]<<endl;
}

