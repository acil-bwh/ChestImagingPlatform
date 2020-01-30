/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkExtractAirwayTree.h,v $

=========================================================================*/
// .NAME vtkExtractAirwayTree - extract airway tree
// .SECTION Description
// vtkExtractAirwayTree is a filter that takes as input a volume (e.g., 3D
// structured point set) and generates on output one or more lines.

#ifndef __vtkExtractAirwayTree_h
#define __vtkExtractAirwayTree_h

#include "vtkPolyDataAlgorithm.h"
#include "vtkMath.h"
#include "vtkCIPCommonConfigure.h"
#include "teem/gage.h"

#define VTK_VALLEY 1
#define VTK_RIDGE 2

// VTK6 migration note:
// - gage_t was replaced with double based on:
//   http://teem.sourceforge.net/gage/#gage_t

//BTX
class vtkTrackingState {
public:
    double Pgrad[3];
    double Pheval[3];
    double Phevec[9];
    gageContext *gtx;
    const double *grad;
    const double *heval;
    const double *hevec;
    double Seed[3];
    double PSeed[3];
    double PTangent[3];
    double Tangent[3];
    int direction;
    int TubularType;
    void Update(double newS[3]){
      //Copy previous step
      Pgrad[0]=grad[0]; Pgrad[1]=grad[1]; Pgrad[2]=grad[2];
      Pheval[0]=heval[0]; Pheval[1]=heval[1]; Pheval[2]=heval[2];
      for(int i=0;i<9;i++)
        Phevec[i]=hevec[i];
      PSeed[0]=this->Seed[0]; PSeed[1]=this->Seed[1]; PSeed[2]=this->Seed[2];

     // Probe new seed
     gageProbe(this->gtx,newS[0],newS[1],newS[2]);
     this->Seed[0]=newS[0]; this->Seed[1]=newS[1]; this->Seed[2]=newS[2];

     for (int i=0;i<3;i++)
        PTangent[i] = (Seed[i]-PSeed[i]);
     vtkMath::Normalize(PTangent);
     // Update tracking direction
     if (this->TubularType == VTK_VALLEY)
       this->UpdateDirectionValley();
     if (this->TubularType == VTK_RIDGE)
       this->UpdateDirectionRidge();
    };
    void UpdateDirectionValley(){
        Tangent[0] = hevec[6];
        Tangent[1] = hevec[7];
        Tangent[2] = hevec[8];
        vtkMath::Normalize(Tangent);
        double tmp = (PTangent[0])*Tangent[0] +
                     (PTangent[1])*Tangent[1] +
                     (PTangent[2])*Tangent[2];
        if (tmp>=0)
          this->direction = 1;
        else
          this->direction = -1;
    };
    void UpdateDirectionRidge(){
        Tangent[0] = hevec[0];
        Tangent[1] = hevec[1];
        Tangent[2] = hevec[2];
        vtkMath::Normalize(Tangent);
        double tmp = (this->PTangent[0])*Tangent[0] +
                     (this->PTangent[1])*Tangent[1] +
                     (this->PTangent[2])*Tangent[2];
        if (tmp>=0)
          this->direction = 1;
        else
          this->direction = -1;
    };
};

//ETX

// VTK6 migration note:
// - replaced super class vtkStructuredPointsToPolyDataFilter with vtkPolyDataAlgorithm
class VTK_CIP_COMMON_EXPORT vtkExtractAirwayTree : public vtkPolyDataAlgorithm
{
public:
  static vtkExtractAirwayTree *New();
  vtkTypeMacro(vtkExtractAirwayTree, vtkPolyDataAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  vtkSetVector3Macro(Seed,double);
  vtkGetVector3Macro(Seed,double);

  vtkSetMacro(Scale,double);
  vtkGetMacro(Scale,double);

  vtkGetMacro(TubularType,int);
  vtkSetMacro(TubularType,int);
  void SetTubularTypeToValley() {
    this->SetTubularType(VTK_VALLEY);};
  void SetTubularTypeToRidge() {
    this->SetTubularType(VTK_RIDGE);};

  vtkGetMacro(RescaleAtStoppingPoint,int);
  vtkSetMacro(RescaleAtStoppingPoint,int);
  vtkBooleanMacro(RescaleAtStoppingPoint,int);
  vtkGetMacro(ModeThreshold,double);
  vtkSetMacro(ModeThreshold,double);
  vtkGetMacro(Delta,double);
  vtkSetMacro(Delta,double);

  void PrintState(vtkTrackingState *state);
  int StoppingCondition(vtkTrackingState *state);
  void ApplyUpdate(vtkTrackingState *state,double *newS);

  double ScaleSelection(gageContext *gtx, gagePerVolume *pvl,
                        double Seed[3], double initS, double maxS, double deltaS);
  int RelocateSeed(gageContext *gtx, gagePerVolume *pvl, double Seed[3], double SeedNew[3]);
  int RelocateSeedInPlane(gageContext *gtx, gagePerVolume *pvl, double Seed[3], double SeedNew[3], int axis);
  int SettingContext(gageContext *gtx,gagePerVolume *pvl,double scale);
  double Mode(const double *w);
  double Strength(const double *w);

  static int VTKToNrrdPixelType( const int vtkPixelType )
  {
  switch( vtkPixelType )
    {
    default:
    case VTK_VOID:
      return nrrdTypeDefault;
      break;
    case VTK_CHAR:
      return nrrdTypeChar;
      break;
    case VTK_UNSIGNED_CHAR:
      return nrrdTypeUChar;
      break;
    case VTK_SHORT:
      return nrrdTypeShort;
      break;
    case VTK_UNSIGNED_SHORT:
      return nrrdTypeUShort;
      break;
      //    case nrrdTypeLLong:
      //      return LONG ;
      //      break;
      //    case nrrdTypeULong:
      //      return ULONG;
      //      break;
    case VTK_INT:
      return nrrdTypeInt;
      break;
    case VTK_UNSIGNED_INT:
      return nrrdTypeUInt;
      break;
    case VTK_FLOAT:
      return nrrdTypeFloat;
      break;
    case VTK_DOUBLE:
      return nrrdTypeDouble;
      break;
    }
  }

protected:
  vtkExtractAirwayTree();
  ~vtkExtractAirwayTree();

  virtual int FillInputPortInformation(int port, vtkInformation *info) override;
  virtual int RequestData(vtkInformation *request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector) override;
  int SettingContextAndState(gageContext *gtx,gagePerVolume *pvl,vtkTrackingState *state,double scale);

  double Seed[3];
  int TubularType;
  double Scale;
  double ModeThreshold;
  double Delta;
  int RescaleAtStoppingPoint;

private:
  vtkExtractAirwayTree(const vtkExtractAirwayTree&);  // Not implemented.
  void operator=(const vtkExtractAirwayTree&);  // Not implemented.
};

#endif

