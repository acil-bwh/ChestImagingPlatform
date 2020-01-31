// .NAME vtkEllipseFitting - fits an ellipse in a Least Squares sense to a set of points
// .SECTION Description
// vtkEllipseFitting is a filter that takes create a set of points (in the polydata)
// and fits an ellipse. The algorithm is based on the work:
// R. Halif and J. Flusser, "Numerically stable direct least squares fitting of ellipses", Dept of Software Engineering,
// Charles University, Czech Republic, 2000.

#ifndef __vtkEllipseFitting_h
#define __vtkEllipseFitting_h

#include "vtkPolyDataAlgorithm.h"
#include "vtkCIPCommonConfigure.h"

// VTK6 migration note:
// - replaced super class vtkPolyDataToPolyDataFilter with vtkPolyDataAlgorithm

class VTK_CIP_COMMON_EXPORT vtkEllipseFitting : public vtkPolyDataAlgorithm
{
public:
  static vtkEllipseFitting *New();
  void PrintSelf(ostream& os, vtkIndent indent) override;
  vtkTypeMacro(vtkEllipseFitting, vtkPolyDataAlgorithm);

  vtkGetMacro(MajorAxisLength,double);
  vtkGetMacro(MinorAxisLength,double);
  vtkGetVector2Macro(Center,double);
  vtkGetMacro(Angle,double);
  vtkGetVector2Macro(MajorAxis,double);
  vtkGetVector2Macro(MinorAxis,double);

protected:
  vtkEllipseFitting();
 ~vtkEllipseFitting();

  double MajorAxisLength;
  double MinorAxisLength;
  double Center[2];
  double Angle;
  double MajorAxis[2];
  double MinorAxis[2];

  // ellipse parameters in conic representation:
  //p[0]*x^2 + p[1]*x*y + p[2]*y^2 + p[3]*x + p[4]*y + p[5] = 0
  double P[6];

  // Usual data generation method
  virtual int RequestData(vtkInformation *request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector) override;
  virtual int RequestInformation(vtkInformation *, vtkInformationVector**,
                                 vtkInformationVector *) override;
  void MultiplyMatrix(double A1[3][3], double A2[3][3], double R[3][3]);

private:
  vtkEllipseFitting(const vtkEllipseFitting&);  // Not implemented.
  void operator=(const vtkEllipseFitting&);  // Not implemented.
};

#endif

