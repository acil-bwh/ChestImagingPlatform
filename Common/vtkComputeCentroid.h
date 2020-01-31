/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkComputeCentroid.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkComputeCentroid - combine images via a cookie-cutter operation
// .SECTION Description
// vtkComputeCentroid performs a intensity correciton of Lung CT scans. The
// intensity correction is based on a least squares fitting of a 1st order polynomial to
// the mean intensity along the y direction. The rationale is based on the fact that
// graviational issues makes the condense water in the lungs to accumulates on certain areas
// (for example, posterior is the scan is supine) yielding a increase of expected intensity in
// those areas. This class remove the trend in the intensity profile. The DC value of the correction
// can be set to the low , high or mean intensity of the profile.

#ifndef __vtkComputeCentroid_h
#define __vtkComputeCentroid_h

#include "vtkImageAlgorithm.h"
#include "vtkCIPCommonConfigure.h"

// VTK6 migration note:
// Replaced suplerclass vtkImageToImageFilter with vtkImageAlgorithm
// instead of vtkThreadedImageAlgorithm since this class did not provide
// ThreadedExecute() method and overrided ExecuteData() originally.

class VTK_CIP_COMMON_EXPORT vtkComputeCentroid : public vtkImageAlgorithm
{
public:
  static vtkComputeCentroid *New();
  vtkTypeMacro(vtkComputeCentroid, vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  // Description:
  // Get Image Centroid
  vtkGetVectorMacro(Centroid, double,3);

protected:
  vtkComputeCentroid();
  ~vtkComputeCentroid();

  // Description:
  // This is a convenience method that is implemented in many subclasses
  // instead of RequestData.  It is called by RequestData.
  virtual void ExecuteDataWithInformation(vtkDataObject *output,
                                          vtkInformation* outInfo) override;

  virtual int RequestInformation(vtkInformation *, vtkInformationVector**,
                                 vtkInformationVector *) override;

  void ComputeCentroid(vtkImageData *in, int ext[6], double C[3]);
  void ComputeCentroid();
  double Centroid[3];

private:
  vtkComputeCentroid(const vtkComputeCentroid&);  // Not implemented.
  void operator=(const vtkComputeCentroid&);  // Not implemented.
};

#endif

