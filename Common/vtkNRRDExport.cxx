/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkNRRDExport.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkNRRDExport.h"

#include "vtkImageData.h"
#include "vtkObjectFactory.h"
#include "vtkExecutive.h"

#include <ctype.h>
#include <string.h>

vtkStandardNewMacro(vtkNRRDExport);

//----------------------------------------------------------------------------
vtkNRRDExport::vtkNRRDExport()
{
  this->NRRD = nrrdNew();
  this->SetNumberOfInputPorts(1);
}

//----------------------------------------------------------------------------
vtkNRRDExport::~vtkNRRDExport()
{
// Do nrrd Nix
nrrdNix(this->NRRD);
}

//----------------------------------------------------------------------------
void vtkNRRDExport::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

}

//----------------------------------------------------------------------------
// VTK6 migration note:
// - changed name from SetInput(...)
// - changed SetNthInput(0, input) to SetInputDataInternal(0, input)
void vtkNRRDExport::SetInputData(vtkImageData *input)
{
  this->SetInputDataInternal(0, input);
}

//----------------------------------------------------------------------------
vtkImageData *vtkNRRDExport::GetInput()
{
  if (this->GetNumberOfInputConnections(0) < 1)
    {
    return NULL;
    }
  
  return vtkImageData::SafeDownCast(this->GetExecutive()->GetInputData(0, 0));
}


//----------------------------------------------------------------------------
// Exports all the data from the input.
// Provides a valid pointer to the data (only valid until the next
// update, though)

Nrrd *vtkNRRDExport::GetNRRDPointer()
{
  // Error checking
  if ( this->GetInput() == NULL )
    {
    vtkErrorMacro(<<"Export: Please specify an input!");
    return 0;
    }

  vtkImageData *input = this->GetInput();
  //input->UpdateInformation();
  //input->SetUpdateExtent(input->GetWholeExtent());
  //input->ReleaseDataFlagOff();

  //input->Update(); FIX_ME_VTK6
  this->UpdateProgress(0.0);
  this->UpdateProgress(1.0);

  void *data =  (void *) input->GetScalarPointer();
  const int type = this->VTKToNrrdPixelType(input->GetScalarType());
  size_t size[3];
  int dims[3];
  input->GetDimensions(dims);
  size[0]=dims[0];
  size[1]=dims[1];
  size[2]=dims[2];
  double Spacing[3];
  input->GetSpacing(Spacing);


  if(nrrdWrap_nva(this->NRRD,data,type,3,size)) {
	//sprintf(err,"%s:",me);
	//biffAdd(NRRD, err); return;
  }
  nrrdAxisInfoSet_nva(this->NRRD, nrrdAxisInfoSpacing, Spacing);

  return this->NRRD;
}

int vtkNRRDExport::VTKToNrrdPixelType( const int vtkPixelType )
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

