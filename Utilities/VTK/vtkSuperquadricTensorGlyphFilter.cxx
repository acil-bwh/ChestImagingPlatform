/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkSuperquadricTensorGlyphFilter.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// This plugin has been developed and contributed by Sven Buijssen, TU
// Dortmund, Germany.
// Thanks to Bryn Lloyd (blloyd@vision.ee.ethz.ch) at ETH Zuerich for
// developing and sharing vtkTensorGlyphFilter, the ancestor of this
// filter. That filter's output (i.e. spheres) can be mimicked by setting both
// ThetaRoundness and PhiRoundness to 1.0.
// Thanks to Gordon Kindlmann for pointing out at VisSym04 that superquadrics
// can be put to a good use to visualise tensors.

#include "vtkSuperquadricTensorGlyphFilter.h"

#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include "vtkSuperquadricSource.h"
#include "vtkTensorGlyph.h"

vtkStandardNewMacro(vtkSuperquadricTensorGlyphFilter);

vtkSuperquadricTensorGlyphFilter::vtkSuperquadricTensorGlyphFilter()
{
  this->SetNumberOfInputPorts(1);
  this->ThetaResolution = 10;
  this->PhiResolution = 10;
  this->ThetaRoundness = 0.3;
  this->PhiRoundness = 0.3;
  this->ScaleFactor = 0.125;
  this->ExtractEigenvalues = 0;

  // by default, process active point tensors
  this->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,
                               vtkDataSetAttributes::TENSORS);
}

vtkSuperquadricTensorGlyphFilter::~vtkSuperquadricTensorGlyphFilter()
{
}

//----------------------------------------------------------------------------
int vtkSuperquadricTensorGlyphFilter::RequestInformation(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **vtkNotUsed(inputVector),
  vtkInformationVector *outputVector)
{
  // get the info object
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  outInfo->Set(CAN_HANDLE_PIECE_REQUEST(), 1);
  return 1;
}

int vtkSuperquadricTensorGlyphFilter::RequestData(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
  // get the info objects
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  // get the input and output
  vtkDataSet *input = vtkDataSet::SafeDownCast(
    inInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkPolyData *output = vtkPolyData::SafeDownCast(
    outInfo->Get(vtkDataObject::DATA_OBJECT()));

  vtkPointData *pd;
  vtkPointData* outputPD = output->GetPointData();
  vtkIdType numPts, numSourcePts, ptIncr, inPtId, i;
  vtkDataArray *inTensors = NULL;

  numPts = input->GetNumberOfPoints();
  if (numPts < 1)
    {
    vtkDebugMacro(<<"No points to glyph!");
    return 1;
    }

  vtkSmartPointer<vtkSuperquadricSource> superquadric = vtkSmartPointer<vtkSuperquadricSource>::New();
  //vtkNew(vtkSuperquadricSource, superquadric);
  superquadric->SetThetaResolution(this->ThetaResolution);
  superquadric->SetPhiResolution(this->PhiResolution);
  superquadric->SetThetaRoundness(this->ThetaRoundness);
  superquadric->SetPhiRoundness(this->PhiRoundness);
  superquadric->ToroidalOff();

  // For some reason, it is necessary to set the active tensor despite
  // vtkAlgorithm's SetInputArrayToProcess being invoked as soon as
  // another tensor is chosen from the GUI dropdown box.

  inTensors = this->GetInputArrayToProcess(0,inputVector);
//   if (inTensors)
//     input->GetPointData()->SetActiveTensors( inTensors->GetName() );

  //vtkNew(vtkTensorGlyph, tensors);
  vtkSmartPointer<vtkTensorGlyph> tensors = vtkSmartPointer<vtkTensorGlyph>::New();
  if (this->ExtractEigenvalues)
    tensors->ExtractEigenvaluesOn();
  else
    tensors->ExtractEigenvaluesOff();
  tensors->SetInputData(input);
  tensors->SetSourceData(superquadric->GetOutput());
  tensors->SetScaleFactor(this->ScaleFactor);
  tensors->Update();

  // Copy tensor glyph filter's output to output
  vtkPolyData *tensorPD = tensors->GetOutput();
  output->ShallowCopy(tensorPD);
  // But keep in mind that the data fields created by vtkTensorGlyph
  // are not named. ParaView would not color them. Compensate here to
  // add the auto-created normals array.
  if (tensorPD->GetPointData()->GetNormals())
    tensorPD->GetPointData()->GetNormals()->SetName("Normals");

  // Copy point data from source (if possible)
  pd = input->GetPointData();
  if ( pd )
    {
    ptIncr=0;
    numSourcePts = tensorPD->GetNumberOfPoints() / numPts;
    outputPD->CopyAllocate(pd,numPts*numSourcePts);
    for (inPtId=0; inPtId < numPts; inPtId++)
      {
      for (i=0; i < numSourcePts; i++)
        {
        outputPD->CopyData(pd,inPtId,ptIncr+i);
        }
      ptIncr += numSourcePts;
      }
    }

  return 1;
}

void vtkSuperquadricTensorGlyphFilter::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

