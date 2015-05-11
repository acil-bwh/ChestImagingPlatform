/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkComputeAirwayWallPolyData.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkComputeAirwayWallPolyData - sample subset of input polygonal data
// .SECTION Descriptions
// vtkComputeAirwayWallPolyData is a filter that sub-samples input polygonal data. The user
// specifies every nth item, with an initial offset to begin sampling.

#ifndef __vtkComputeAirwayWallPolyData_h
#define __vtkComputeAirwayWallPolyData_h

#include "vtkPolyDataAlgorithm.h"
#include "vtkComputeAirwayWall.h"
#include "vtkImageData.h"
#include "vtkDoubleArray.h"
#include "vtkEllipseFitting.h"

#define VTK_HESSIAN 0
#define VTK_POLYDATA 1
#define VTK_VECTOR 2
#define VTK_SMOOTH 0
#define VTK_SHARP 1

// VTK6 migration note:
// Replaced superclass vtkPolyDataToPolyDataFilter with vtkPolyDataAlgorithm.

class VTK_CIP_COMMON_EXPORT vtkComputeAirwayWallPolyData : public vtkPolyDataAlgorithm
{
public:
  static vtkComputeAirwayWallPolyData *New();
  vtkTypeMacro(vtkComputeAirwayWallPolyData, vtkPolyDataAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);
  
  // Description:
  // Set/Get Image data
  vtkSetObjectMacro(Image,vtkImageData);
  vtkGetObjectMacro(Image,vtkImageData);

  // Description:
  // Airway wall computation engine
  vtkSetObjectMacro(WallSolver,vtkComputeAirwayWall);
  vtkGetObjectMacro(WallSolver,vtkComputeAirwayWall);

  // Description:
  // Reformat airway along airway long axis
  vtkBooleanMacro(Reformat,int);
  vtkSetMacro(Reformat,int);
  vtkGetMacro(Reformat,int);
  
  // Description:
  // Axis computation model: 
  // 0 = Hessian.
  // 1 = from vktPolyData line.
  // 2 = from Vector field in PolyData pointData.
  vtkSetMacro(AxisMode,int);
  vtkGetMacro(AxisMode,int);
  void SetAxisModeToHessian() {this->SetAxisMode(VTK_HESSIAN);};
  void SetAxisModeToPolyData() {this->SetAxisMode(VTK_POLYDATA);};
  void SetAxisModeToVector() {this->SetAxisMode(VTK_VECTOR);};
  
  // Description:
  // Reconstruction kernel from image
  // 0 = Smooth
  // 1 = Sharp
  vtkSetMacro(Reconstruction,int);
  vtkGetMacro(Reconstruction,int);
  void SetReconstructionToSmooth() {this->SetReconstruction(VTK_SMOOTH);};
  void SetReconstructionToSharp() {this->SetReconstruction(VTK_SHARP);};
  
  // Description:
  // Save a png image with the airway segmentation results for quality control
  vtkBooleanMacro(SaveAirwayImage,int);
  vtkSetMacro(SaveAirwayImage,int);
  vtkGetMacro(SaveAirwayImage,int);
  
  // Description:
  // File prefix for the airway image
  vtkSetStringMacro(AirwayImagePrefix);
  vtkGetStringMacro(AirwayImagePrefix);
  
protected:
  vtkComputeAirwayWallPolyData();
  ~vtkComputeAirwayWallPolyData();

  // Usual data generation method
  virtual int RequestData(vtkInformation *request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector);
  //void ExecuteInformation();
  void ComputeCellData();
  int AxisMode;
  int Reformat;
  vtkComputeAirwayWall *WallSolver;
  vtkDoubleArray *AxisArray;
  double SelfTuneModelSmooth[3];
  double SelfTuneModelSharp[3];
  int Reconstruction;
  vtkImageData *Image;
  double SegmentPercentage;
  int SaveAirwayImage;
  char *AirwayImagePrefix;
  void SetWallSolver(vtkComputeAirwayWall *ref, vtkComputeAirwayWall *out);
  void ComputeAirwayAxisFromLines();
  void CreateAirwayImage(vtkImageData *resliceCT,vtkEllipseFitting *eifit,vtkEllipseFitting *eofit,vtkImageData *airwayImage);
  
private:
  vtkComputeAirwayWallPolyData(const vtkComputeAirwayWallPolyData&);  // Not implemented.
  void operator=(const vtkComputeAirwayWallPolyData&);  // Not implemented.
};

#endif


