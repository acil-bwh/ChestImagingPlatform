/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkNRRDExport.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkNRRDExport - Export VTK images to third-party systems.
// .SECTION Description
// vtkNRRDExport provides a way of exporting image data at the end
// of a pipeline to a third-party system or to a simple C array.
// Applications can use this to get direct access to the image data
// in memory.  A callback interface is provided to allow connection
// of the VTK pipeline to a third-party pipeline.  This interface
// conforms to the interface of vtkImageImport.
// In Python it is possible to use this class to write the image data
// into a python string that has been pre-allocated to be the correct
// size.
// .SECTION See Also
// vtkImageImport

#ifndef __vtkNRRDExport_h
#define __vtkNRRDExport_h

#include "vtkAlgorithm.h"
#include "vtkCIPCommonConfigure.h"

#include "teem/nrrd.h"

class vtkImageData;

// VTK6 migration note:
// Replaced superclass vtkProcessObject with vtkAlgorithm

class VTK_CIP_COMMON_EXPORT vtkNRRDExport : public vtkAlgorithm
{
public:
  static vtkNRRDExport *New();
  vtkTypeMacro(vtkNRRDExport, vtkAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  // Description:
  // Set/Get the input object from the image pipeline.
  void SetInputData(vtkImageData *input);
  vtkImageData *GetInput();

  // Description:
  // The main interface: (Use with caution).   Update the
  // pipeline and return a pointer to the image memory.  The
  // pointer is only valid until the next time that the pipeline
  // is updated.
  Nrrd *GetNRRDPointer();

  int VTKToNrrdPixelType( const int vtkPixelType );

protected:
  vtkNRRDExport();
  ~vtkNRRDExport();
  Nrrd *NRRD;

private:
  vtkNRRDExport(const vtkNRRDExport&);  // Not implemented.
  void operator=(const vtkNRRDExport&);  // Not implemented.
};

#endif

