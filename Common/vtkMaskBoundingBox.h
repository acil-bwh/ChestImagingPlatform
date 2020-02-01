// .NAME vtkMaskBoundingBox - Bounding box that enclosed a mask.
//
// .SECTION Description
// This class computeds the bounding box that enclosed a mask.
// The input is a label image. The bounding box that enclose a label is
// return when the filter runs.

#ifndef __vtkMaskBoundingBox_h
#define __vtkMaskBoundingBox_h

#include "vtkAlgorithm.h"
#include "vtkImageStencilData.h"
#include "vtkCIPCommonConfigure.h"

class vtkImageData;

class VTK_CIP_COMMON_EXPORT vtkMaskBoundingBox : public vtkAlgorithm 
{
  public:
  static vtkMaskBoundingBox *New();

  vtkTypeMacro(vtkMaskBoundingBox, vtkAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  // Description:
  // Set the Input of a filter.
  void SetInputData(vtkImageData *input);
  vtkImageData *GetInput();

  void Compute();

  //Description:
  //Set/Get the label whos bounding box we are going to compute.
  vtkSetMacro(Label,short);
  vtkGetMacro(Label,short);

  //Descrition:
  //Get the bounding box values: [xmin xmax ymin ymax zmin zmax]
  vtkSetVector6Macro( BoundingBox, int );
  vtkGetVector6Macro( BoundingBox, int );

  vtkGetObjectMacro(Stencil, vtkImageStencilData);

protected:
  vtkMaskBoundingBox();
  ~vtkMaskBoundingBox();

  short Label;
  int BoundingBox[6];
  vtkImageStencilData *Stencil;
private:
  vtkMaskBoundingBox(const vtkMaskBoundingBox&) {}; //Not implemented
  void operator=(const vtkMaskBoundingBox&) {}; //Not implemented
};

#endif
