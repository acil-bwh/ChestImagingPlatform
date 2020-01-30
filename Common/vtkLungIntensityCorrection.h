/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkLungIntensityCorrection.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkLungIntensityCorrection - combine images via a cookie-cutter operation
// .SECTION Description
// vtkLungIntensityCorrection performs a intensity correciton of Lung CT scans. The
// intensity correction is based on a least squares fitting of a 1st order polynomial to
// the mean intensity along the y direction. The rationale is based on the fact that
// graviational issues makes the condense water in the lungs to accumulates on certain areas
// (for example, posterior is the scan is supine) yielding a increase of expected intensity in
// those areas. This class remove the trend in the intensity profile. The DC value of the correction
// can be set to the low , high or mean intensity of the profile.

#ifndef __vtkLungIntensityCorrection_h
#define __vtkLungIntensityCorrection_h

#include "vtkThreadedImageAlgorithm.h"
#include "vtkCIPCommonConfigure.h"

#define DCLOW 0
#define DCMEAN 1
#define DCHIGH 2

class vtkImageStencilData;
class vtkDoubleArray;

// VTK6 migration note:
// Replaced suplerclass vtkImageToImageFilter with vtkThreadedImageAlgorithm.

class VTK_CIP_COMMON_EXPORT vtkLungIntensityCorrection : public vtkThreadedImageAlgorithm
{
public:
  static vtkLungIntensityCorrection *New();
  vtkTypeMacro(vtkLungIntensityCorrection, vtkThreadedImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  // Description:
  // Specify the stencil to use.  The stencil can be created
  // from a vtkImplicitFunction or a vtkPolyData.
  virtual void SetStencilData(vtkImageStencilData *stencil);
  vtkImageStencilData *GetStencil();

  // Description:
  // Reverse the stencil.
  vtkSetMacro(ReverseStencil, int);
  vtkBooleanMacro(ReverseStencil, int);
  vtkGetMacro(ReverseStencil, int);

  // Description:
  // Clamp to zero negative values.
  vtkSetMacro(ClampNegativeValues, int);
  vtkBooleanMacro(ClampNegativeValues, int);
  vtkGetMacro(ClampNegativeValues, int);

  //Description:
  // Set the DC value for the intensity correction. There is three possibilities: DC to Low, DC to Mean
  // and DC to high. DC to Low means that the DC value would be adjusted such that it corresponds to the minimum
  // intesity value of the fitted line.
  // DC to High means that the DC value will correspond to the maximum intensity value of the fitted line. DC to mean
  // will be the mean value of the fitted line.
  vtkGetMacro(DCValue, int);
  vtkSetMacro(DCValue, int);

  void SetDCValueToLow(void) {
     this->SetDCValue(DCLOW);
   };
  void SetDCValueToMean(void) {
     this->SetDCValue(DCMEAN);
  };
  void SetDCValueToHigh(void) {
     this->SetDCValue(DCHIGH);
  };

  vtkGetObjectMacro(Parameters,vtkDoubleArray);

  //Descrition:
  // Applied a weighted least square method to solve the fitting problem.
  vtkSetMacro(UseWeightedLS, int);
  vtkBooleanMacro(UseWeightedLS, int);
  vtkGetMacro(UseWeightedLS, int);

  // Description:
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
  static int SolveWeightedLeastSquares(int numberOfSamples, double **xt, int xOrder,
                               double **yt, int yOrder,double *w, double **mt);

protected:
  vtkLungIntensityCorrection();
  ~vtkLungIntensityCorrection();
  
  virtual int FillInputPortInformation(int, vtkInformation*) override;

  virtual int RequestInformation(vtkInformation *, vtkInformationVector**,
                                 vtkInformationVector *) override;
  void ThreadedExecute(vtkImageData *inData, vtkImageData *outData,
                       int extent[6], int id) override;

  int ReverseStencil;
  int ClampNegativeValues;
  vtkDoubleArray *Parameters;
  int DCValue;
  int UseWeightedLS;
private:
  vtkLungIntensityCorrection(const vtkLungIntensityCorrection&);  // Not implemented.
  void operator=(const vtkLungIntensityCorrection&);  // Not implemented.
};

#endif

