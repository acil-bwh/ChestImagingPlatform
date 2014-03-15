/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkImageErode.cxx,v $
  Date:      $Date: 2010-02-20 15:39:37 -0500 (Sat, 20 Feb 2010) $
  Version:   $Revision: 12195 $

=========================================================================auto=*/
#include "vtkImageErode.h"
#include <time.h>
#include "vtkObjectFactory.h"

#include "vtkDataArray.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkStreamingDemandDrivenPipeline.h"


//------------------------------------------------------------------------------
vtkStandardNewMacro(vtkImageErode);

//----------------------------------------------------------------------------
// Description:
// Constructor sets default values
vtkImageErode::vtkImageErode()
{

    this->Background = 0;
    this->Foreground = 1;
    this->HandleBoundaries = 1;
    this->SetNeighborTo4();
}


//----------------------------------------------------------------------------
vtkImageErode::~vtkImageErode()
{

}

//----------------------------------------------------------------------------
// Description:
// This templated function executes the filter for any type of data.
// For every pixel in the foreground, if a neighbor is in the background,
// then the pixel becomes background.
template <class T>
void vtkImageErodeExecute(vtkImageErode *self,
                     vtkImageData *inData, T *inPtr,
                     vtkImageData *outData, T *outPtr,
                     int outExt[6], int id,
                     vtkDataArray *inArray)
{
  // For looping though output (and input) pixels.
  int outMin0, outMax0, outMin1, outMax1, outMin2, outMax2;
  int outIdx0, outIdx1, outIdx2;
  vtkIdType inInc0, inInc1, inInc2;
  vtkIdType outInc0, outInc1, outInc2;
  T *inPtr0, *inPtr1, *inPtr2;
  T *outPtr0, *outPtr1, *outPtr2;
  int numComps, outIdxC;
  // For looping through hood pixels
  int hoodMin0, hoodMax0, hoodMin1, hoodMax1, hoodMin2, hoodMax2;
  int hoodIdx0, hoodIdx1, hoodIdx2;
  T *hoodPtr0, *hoodPtr1, *hoodPtr2;
  // For looping through the mask.
  unsigned char *maskPtr, *maskPtr0, *maskPtr1, *maskPtr2;
  vtkIdType maskInc0, maskInc1, maskInc2;
  // The extent of the whole input image
  int inImageMin0, inImageMin1, inImageMin2;
  int inImageMax0, inImageMax1, inImageMax2;
  // Other
  T backgnd = (T)(self->GetBackground());
  T foregnd = (T)(self->GetForeground());
  T pix;
  unsigned long count = 0;
  unsigned long target;

  clock_t tStart, tEnd, tDiff;
  tStart = clock();

  // Get information to march through data
  inData->GetIncrements(inInc0, inInc1, inInc2); 
  self->GetInput()->GetWholeExtent(inImageMin0, inImageMax0, inImageMin1,
    inImageMax1, inImageMin2, inImageMax2);
  outData->GetIncrements(outInc0, outInc1, outInc2); 
  outMin0 = outExt[0];   outMax0 = outExt[1];
  outMin1 = outExt[2];   outMax1 = outExt[3];
  outMin2 = outExt[4];   outMax2 = outExt[5];
  numComps = outData->GetNumberOfScalarComponents();

  // Neighborhood around current voxel
  self->GetRelativeHoodExtent(hoodMin0, hoodMax0, hoodMin1, 
    hoodMax1, hoodMin2, hoodMax2);

  // Set up mask info
  maskPtr = (unsigned char *)(self->GetMaskPointer());
  self->GetMaskIncrements(maskInc0, maskInc1, maskInc2);

  // in and out should be marching through corresponding pixels.
  inPtr = static_cast<T *>(inData->GetScalarPointer(outMin0, outMin1, outMin2));

  target = (unsigned long)(numComps*(outMax2-outMin2+1)*
    (outMax1-outMin1+1)/50.0);
  target++;


  // loop through components
  for (outIdxC = 0; outIdxC < numComps; ++outIdxC)
    {
    // loop through pixels of output
    outPtr2 = outPtr;
    inPtr2 = inPtr;
    for (outIdx2 = outMin2; outIdx2 <= outMax2; outIdx2++)
      {
      outPtr1 = outPtr2;
      inPtr1 = inPtr2;
      for (outIdx1 = outMin1; 
        !self->AbortExecute && outIdx1 <= outMax1; outIdx1++)
        {
        if (!id) {
          if (!(count%target))
            self->UpdateProgress(count/(50.0*target));
          count++;
        }
        outPtr0 = outPtr1;
        inPtr0 = inPtr1;
        for (outIdx0 = outMin0; outIdx0 <= outMax0; outIdx0++)
          {
          pix = *inPtr0;
          // Default output equal to input
          *outPtr0 = pix;

          if (pix == foregnd)
            {
            // Loop through neighborhood pixels (kernel radius=1)
            // Note: input pointer marches out of bounds.
            hoodPtr2 = inPtr0 + inInc0*hoodMin0 + inInc1*hoodMin1 
              + inInc2*hoodMin2;
            maskPtr2 = maskPtr;
            for (hoodIdx2 = hoodMin2; hoodIdx2 <= hoodMax2; ++hoodIdx2)
              {
              hoodPtr1 = hoodPtr2;
              maskPtr1 = maskPtr2;
              for (hoodIdx1 = hoodMin1; hoodIdx1 <= hoodMax1;    ++hoodIdx1)
                {
                hoodPtr0 = hoodPtr1;
                maskPtr0 = maskPtr1;
                for (hoodIdx0 = hoodMin0; hoodIdx0 <= hoodMax0; ++hoodIdx0)
                  {
                  if (*maskPtr0)
                    {
                    // handle boundaries
                    if (outIdx0 + hoodIdx0 >= inImageMin0 &&
                      outIdx0 + hoodIdx0 <= inImageMax0 &&
                      outIdx1 + hoodIdx1 >= inImageMin1 &&
                      outIdx1 + hoodIdx1 <= inImageMax1 &&
                      outIdx2 + hoodIdx2 >= inImageMin2 &&
                      outIdx2 + hoodIdx2 <= inImageMax2)
                      {
                      // If the neighbor is backgnd, then
                      // set the output to backgnd
                      if (*hoodPtr0 == backgnd)
                        *outPtr0 = backgnd;
                      }
                    }
                  hoodPtr0 += inInc0;
                  maskPtr0 += maskInc0;
                  }//for0
                hoodPtr1 += inInc1;
                maskPtr1 += maskInc1;
                }//for1
              hoodPtr2 += inInc2;
              maskPtr2 += maskInc2;
              }//for2
            }//if
          inPtr0 += inInc0;
          outPtr0 += outInc0;
          }//for0
        inPtr1 += inInc1;
        outPtr1 += outInc1;
        }//for1
      inPtr2 += inInc2;
      outPtr2 += outInc2;
      }//for2
    inPtr++;
    outPtr++;
    }

  tEnd = clock();
  tDiff = tEnd - tStart;
}

//----------------------------------------------------------------------------
// Description:
// This method contains the first switch statement that calls the correct
// templated function for the input and output region types.
void vtkImageErode::ThreadedRequestData(
                          vtkInformation *vtkNotUsed(request),
                          vtkInformationVector **inputVector,
                          vtkInformationVector *vtkNotUsed(outputVector),
                          vtkImageData ***inData,
                          vtkImageData **outData,
                          int outExt[6], int id)
{
  void *inPtr;
  void *outPtr = outData[0]->GetScalarPointerForExtent(outExt);

  vtkDataArray *inArray = this->GetInputArrayToProcess(0,inputVector);
  if (id == 0)
  {
    outData[0]->GetPointData()->GetScalars()->SetName(inArray->GetName());
  }
  
  inPtr = inArray->GetVoidPointer(0);

  switch (inArray->GetDataType())
    {
    vtkTemplateMacro(
      vtkImageErodeExecute(this,inData[0][0],
              static_cast<VTK_TT *>(inPtr),
              outData[0], static_cast<VTK_TT *>(outPtr),
              outExt, id,inArray));
  default:
    vtkErrorMacro(<< "Execute: Unknown input ScalarType");
    return;
    }
}

//----------------------------------------------------------------------------
void vtkImageErode::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  
  os << indent << "Background: " << this->Background;
  os << "\n";
  
  os << indent << "Foreground: " << this->Foreground;
  os << "\n";
  
}
