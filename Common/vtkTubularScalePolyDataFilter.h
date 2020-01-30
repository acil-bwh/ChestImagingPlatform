// .NAME vtkTubularScalePolyDataFilter - classify the points in polylines cells.
// .SECTION Description
// vtkTubularScalePolyDataFilter is a filter that takes create a point data scalar field.
// Each point is assigned a value in the field depending on the point type. There are
// three possible point types: regular point (value = 1), end point (value = -1) and
// junction point (value > 1).

#ifndef __vtkTubularScalePolyDataFilter_h
#define __vtkTubularScalePolyDataFilter_h

#include "vtkCIPCommonConfigure.h"
#include "vtkPolyDataAlgorithm.h"
#include "vtkDoubleArray.h"
#include "vtkTubularScaleSelection.h"

// VTK6 migration note:
// Replaced superclass vtkPolyDataToPolyDataFilter with vtkPolyDataAlgorithm.

class VTK_CIP_COMMON_EXPORT vtkTubularScalePolyDataFilter : public vtkPolyDataAlgorithm
{
public:
  static vtkTubularScalePolyDataFilter *New();
  void PrintSelf(ostream& os, vtkIndent indent) override;
  vtkTypeMacro(vtkTubularScalePolyDataFilter, vtkPolyDataAlgorithm);

   // Description: Type of tubular structure that we want to capture. Valleys are dark tubes (i.e. airways). Ridges are bright tubes (i.e. vessels)
  vtkGetMacro(TubularType,int);
  vtkSetMacro(TubularType,int);
  void SetTubularTypeToValley() {
    this->SetTubularType(VTK_VALLEY);};
  void SetTubularTypeToRidge() {
    this->SetTubularType(VTK_RIDGE);};
  // Description: Scale at which the confidence measurement will be computed. This value is used if OptimizeScale is off.
   vtkGetMacro(StepScale,double);
   vtkSetMacro(StepScale,double);

  // Description: Scale at which the confidence measurement will be computed. This value is used if OptimizeScale is off.
   vtkGetMacro(InitialScale,double);
   vtkSetMacro(InitialScale,double);

  // Description: Scale at which the confidence measurement will be computed. This value is used if OptimizeScale is off.
   vtkGetMacro(FinalScale,double);
   vtkSetMacro(FinalScale,double);

   vtkSetObjectMacro(ImageData,vtkImageData);
   vtkGetObjectMacro(ImageData,vtkImageData);

protected:
  vtkTubularScalePolyDataFilter();
 ~vtkTubularScalePolyDataFilter();

  // Usual data generation method
  virtual int RequestData(vtkInformation *request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector) override;
                          
  int TubularType;
  vtkImageData *ImageData;
  double InitialScale;
  double FinalScale;
  double StepScale;
private:
  vtkTubularScalePolyDataFilter(const vtkTubularScalePolyDataFilter&);  // Not implemented.
  void operator=(const vtkTubularScalePolyDataFilter&);  // Not implemented.
};

#endif

