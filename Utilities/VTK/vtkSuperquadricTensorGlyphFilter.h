/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkSuperquadricTensorGlyphFilter.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// .NAME vtkSuperquadricTensorGlyphFilter - scale and orient superquadric glyph according to tensor eigenvalues and eigenvectors

// .SECTION Description
// vtkSuperquadricTensorGlyphFilter is a filter that generates a superquadric
// glyph at every point in the input data set. The glyphs are oriented and
// scaled according to eigenvalues and eigenvectors of "tensor" data of the
// input data set, interpreting the entries of the 3x3 matrix as principal axes
// of the superquadric and their norm as the length of these axes. Set both
// roundness values to 0.0 to get rectangular glyphs, set them to 1.0 to get
// ellipsoidal glyphs, set theta roundness to 1.0 and phi roundness to 0.0 to
// get cylindrical glyphs. Other values lead to superquadric glyphs which are
// in general favorable as they can be distinguished easily for all view
// angles. The Superquadric Tensor Glyph filter operates on any type of data
// set. Its output is polygonal.

// .SECTION Thanks
// This plugin has been developed and contributed by Sven Buijssen, TU
// Dortmund, Germany.
// Thanks to Bryn Lloyd (blloyd@vision.ee.ethz.ch) at ETH Zuerich for
// developing and sharing vtkTensorGlyphFilter, the ancestor of this
// filter. That filter's output (i.e. spheres) can be mimicked by setting both
// ThetaRoundness and PhiRoundness to 1.0.
// Thanks to Gordon Kindlmann for pointing out at VisSym04 that superquadrics
// can be put to a good use to visualise tensors.

// .SECTION See Also
// vtkTensorGlyph

#ifndef vtkSuperquadricTensorGlyphFilter_h
#define vtkSuperquadricTensorGlyphFilter_h

#include "vtkPolyDataAlgorithm.h"
#include "vtkCIPUtilitiesConfigure.h"

class VTK_CIP_UTILITIES_EXPORT vtkSuperquadricTensorGlyphFilter : public vtkPolyDataAlgorithm
{
public:
  vtkTypeMacro(vtkSuperquadricTensorGlyphFilter,vtkPolyDataAlgorithm);

  void PrintSelf(ostream& os, vtkIndent indent) override;

  static vtkSuperquadricTensorGlyphFilter *New();

  vtkSetMacro(ThetaResolution,int);
  vtkSetMacro(PhiResolution,int);
  vtkSetMacro(ThetaRoundness,double);
  vtkSetMacro(PhiRoundness,double);
  vtkSetMacro(ScaleFactor,double);

  // Description:
  // If true, then extract eigenvalues from tensor. False by default.
  vtkGetMacro(ExtractEigenvalues, int);
  vtkSetMacro(ExtractEigenvalues, int);
  vtkBooleanMacro(ExtractEigenvalues, int);

protected:
  vtkSuperquadricTensorGlyphFilter();
  ~vtkSuperquadricTensorGlyphFilter();

  /* implementation of algorithm */
  int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *) override;
  int RequestInformation(vtkInformation *, vtkInformationVector **, vtkInformationVector *) override;

  //virtual int FillInputPortInformation(int port, vtkInformation* info);

  void SetActiveTensors(int, int, int, int, const char *);

  int ThetaResolution;
  int PhiResolution;
  double ThetaRoundness;
  double PhiRoundness;
  double ScaleFactor;
  int ExtractEigenvalues;

private:
  vtkSuperquadricTensorGlyphFilter(const vtkSuperquadricTensorGlyphFilter&);  // Not implemented.
  void operator=(const vtkSuperquadricTensorGlyphFilter&);  // Not implemented.
};

#endif
