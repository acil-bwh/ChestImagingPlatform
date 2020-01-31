// .NAME vtkSmoothLines - classify the points in polylines cells.
// .SECTION Description
// vtkSmoothLines is a filter that takes create a point data scalar field.
// Each point is assigned a value in the field depending on the point type. There are
// three possible point types: regular point (value = 1), end point (value = -1) and
// junction point (value > 1).

#ifndef __vtkSmoothLines_h
#define __vtkSmoothLines_h

#include "vtkPolyDataAlgorithm.h"
#include "vtkDoubleArray.h"
#include "vtkCIPCommonConfigure.h"

// VTK6 migration note:
// - replaced super class vtkPolyDataToPolyDataFilter with vtkPolyDataAlgorithm

class VTK_CIP_COMMON_EXPORT vtkSmoothLines : public vtkPolyDataAlgorithm
{
public:
  static vtkSmoothLines *New();
  void PrintSelf(ostream& os, vtkIndent indent) override;
  vtkTypeMacro(vtkSmoothLines,vtkPolyDataAlgorithm);

  vtkGetMacro(Beta,double);
  vtkSetMacro(Beta,double);

  vtkGetMacro(NumberOfIterations,int);
  vtkSetMacro(NumberOfIterations,int);

  vtkGetMacro(Delta, double);
  vtkSetMacro(Delta, double);

  void SolveHeatEquation(vtkDoubleArray *in, vtkDoubleArray *out);
protected:
  vtkSmoothLines();
 ~vtkSmoothLines();

  double Beta;
  int NumberOfIterations;
  double Delta;

  // Usual data generation method
  virtual int RequestData(vtkInformation *request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector) override;
private:
  vtkSmoothLines(const vtkSmoothLines&);  // Not implemented.
  void operator=(const vtkSmoothLines&);  // Not implemented.
};

#endif

