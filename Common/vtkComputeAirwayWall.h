/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkComputeAirwayWall.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkComputeAirwayWall - compute the Monogenic signal
// .SECTION Description
// vtkComputeAirwayWall computes the Monogenic signal of an image.
// The output of the filter is the monogenic signal in the spatial domain.
// The computation is performed in the Fourier domain.

#ifndef __vtkComputeAirwayWall_h
#define __vtkComputeAirwayWall_h

#include "vtkImageAlgorithm.h"
#include "vtkCIPCommonConfigure.h"

#include "vtkDoubleArray.h"
#include "vtkPolyData.h"
#include "vtkDataArrayCollection.h"

// VTK6 migration note:
// Replaced suplerclass vtkImageToImageFilter with vtkImageAlgorithm
// instead of vtkThreadedImageAlgorithm since this class did not provide
// ThreadedExecute() method and overrided ExecuteData() originally.

class VTK_CIP_COMMON_EXPORT vtkComputeAirwayWall : public vtkImageAlgorithm
{
public:
  static vtkComputeAirwayWall *New();
  vtkTypeMacro(vtkComputeAirwayWall, vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  vtkSetMacro(Method,int);
  vtkGetMacro(Method,int);

  // Wall Threshol: minimum intensity value at
  // candidate edget point to qualify
  vtkSetMacro(WallThreshold,int);
  vtkGetMacro(WallThreshold,int);

  // Gradient Treshold: minimum gradient treshold at the
  // candidate edge point to qualifiy.
  vtkSetMacro(GradientThreshold,double);
  vtkGetMacro(GradientThreshold,double);

  // Phase Congruency Treshold: minimum PC value at the
  // candidate edge point to qualifiy.
  vtkSetMacro(PCThreshold,double);
  vtkGetMacro(PCThreshold,double);
  
  vtkSetMacro(NumberOfScales,int);
  vtkGetMacro(NumberOfScales,int);

  vtkSetMacro(Bandwidth,double);
  vtkGetMacro(Bandwidth,double);

  vtkSetMacro(MinimumWavelength,double);
  vtkGetMacro(MinimumWavelength,double);

  vtkSetMacro(MultiplicativeFactor,double);
  vtkGetMacro(MultiplicativeFactor,double);

  vtkSetMacro(UseWeights,int);
  vtkGetMacro(UseWeights,int);
  vtkBooleanMacro(UseWeights,int);

  vtkSetObjectMacro(Weights,vtkDoubleArray);
  vtkGetObjectMacro(Weights,vtkDoubleArray);

  // Description: Set/Get Max and Min angle for sector
  // computations
  vtkSetMacro(ThetaMax,double);
  vtkGetMacro(ThetaMax,double);
  vtkSetMacro(ThetaMin,double);
  vtkGetMacro(ThetaMin,double);

  vtkSetMacro(RMin,double);
  vtkGetMacro(RMin,double);
  vtkSetMacro(RMax,double);
  vtkGetMacro(RMax,double);
  vtkSetMacro(Delta,double);
  vtkGetMacro(Delta,double);
  vtkSetMacro(Scale,double);
  vtkGetMacro(Scale,double);
  vtkGetMacro(NumberOfThetaSamples,int);
  vtkSetMacro(NumberOfThetaSamples,int);
    
  // Description: Set/Get factor to remove outlier rays
    // the criteria is mean + StdFactor * std where mean and std are computed over all the airway rays
  vtkSetMacro(StdFactor,double);
  vtkGetMacro(StdFactor,double);

  // Description: Set/Get alpha value. Alpha is a factor
  // to define the parenchymal extent of an airway to be used
  // to compute the mean parenchymal attenuation.
  // The parenchymal attenuation would be computed in the region
  // belonging to the first zero of the gradient after the outer wall
  // and inner_radius + Alpha * wall_thickness.
  // The default value is Alpha = 3;
  vtkSetMacro(Alpha,double);
  vtkGetMacro(Alpha,double);

  // Description: Set/Get T value. T defines
  // the parenchymal extent of an airway that would be used in the power
  // calculation.
  // The default value is T = 15 mm.
  vtkSetMacro(T,double);
  vtkGetMacro(T,double);

  vtkGetMacro(ActivateSector,int);
  vtkSetMacro(ActivateSector,int);
  vtkBooleanMacro(ActivateSector,int);

  vtkGetMacro(NumberOfQuantities,int);

  vtkGetObjectMacro(StatsMean,vtkDoubleArray);
  vtkGetObjectMacro(StatsStd,vtkDoubleArray);
  vtkGetObjectMacro(StatsMin,vtkDoubleArray);
  vtkGetObjectMacro(StatsMax,vtkDoubleArray);

  vtkGetObjectMacro(InnerContour,vtkPolyData);
  vtkGetObjectMacro(OuterContour,vtkPolyData);

  void RemoveOutliers(vtkDoubleArray *r);

  void FWHM(vtkDoubleArray *ray,vtkDoubleArray *values);
  void FWHM(vtkDoubleArray *c, vtkDoubleArray *cp, vtkDoubleArray *cpp, double &rmin, double &rmax);
  void ZeroCrossing(vtkDoubleArray *ray,vtkDoubleArray *values);
  void ZeroCrossing(vtkDoubleArray *c,vtkDoubleArray *cp, vtkDoubleArray *cpp, double &rmin, double &rmax);
  void PhaseCongruency(vtkDoubleArray *ray,vtkDoubleArray *values);
  void PhaseCongruency(vtkDoubleArray *c, vtkDoubleArray *cp, vtkDoubleArray *pc1,vtkDoubleArray *pc2,vtkDoubleArray *values);
  void PhaseCongruency(vtkDoubleArray *c, vtkDoubleArray *cp, vtkDoubleArray *pcV,vtkDoubleArray *values);
  void PhaseCongruencyMultipleKernels(vtkDataArrayCollection *signalCollection, vtkDoubleArray *values,double sp);

protected:
  vtkComputeAirwayWall();
  ~vtkComputeAirwayWall();

  // Description:
  // This is a convenience method that is implemented in many subclasses
  // instead of RequestData.  It is called by RequestData.
  virtual void ExecuteDataWithInformation(vtkDataObject *output,
                                          vtkInformation* outInfo) override;

  virtual int RequestInformation(vtkInformation *, vtkInformationVector**,
                                 vtkInformationVector *) override;

  double FindValue(vtkDoubleArray *c, int loc, double target);
  void FindZeros(vtkDoubleArray *c, vtkDoubleArray *cp, vtkDoubleArray *cpp, vtkDoubleArray *zeros);
  int FindZeroLocation(double fm1, double fm1p, double fm1pp, double f1,
                        double f1p, double f1pp, double delta, double & zero);
  int FindZeroLocation(double fm1, double fm1p, double f1, double f1p,
                        double delta, double & zero);
  int FindZeroLocation(double fm1, double f1, double delta, double & zero);
  int Method;
  int WallThreshold;
  double GradientThreshold;
  vtkDoubleArray *StatsMean;
  vtkDoubleArray *StatsStd;
  vtkDoubleArray *StatsMin;
  vtkDoubleArray *StatsMax;

  vtkPolyData *InnerContour;
  vtkPolyData *OuterContour;

  double PCThreshold;
  int NumberOfScales;
  double Bandwidth;
  double MultiplicativeFactor;
  double MinimumWavelength;
  int UseWeights;
  vtkDoubleArray *Weights;

  double ThetaMax;
  double ThetaMin;

  double RMin;
  double RMax;
  double Delta;
  double Scale;

  double Alpha;
  double T;
  int NumberOfThetaSamples;

  int ActivateSector;

  int NumberOfQuantities;
    
  double StdFactor;
    
private:
  vtkComputeAirwayWall(const vtkComputeAirwayWall&);  // Not implemented.
  void operator=(const vtkComputeAirwayWall&);  // Not implemented.
};

#endif

