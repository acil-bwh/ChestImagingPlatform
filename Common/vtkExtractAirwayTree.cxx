/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkExtractAirwayTree.cxx,v $

=========================================================================*/
#ifdef _WIN32
// to pick up M_SQRT2 and other nice things...
#define _USE_MATH_DEFINES
#endif

// But, if you are on VS6.0 you don't get the define...
#ifndef M_SQRT2
#define M_SQRT2    1.41421356237309504880168872421      /* sqrt(2) */
#endif

#include "vtkExtractAirwayTree.h"

#include "vtkCellArray.h"
#include "vtkCharArray.h"
#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"
#include "vtkIntArray.h"
#include "vtkLongArray.h"
#include "vtkMath.h"
#include "vtkMergePoints.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkShortArray.h"
#include "vtkStructuredPoints.h"
#include "vtkUnsignedCharArray.h"
#include "vtkUnsignedIntArray.h"
#include "vtkUnsignedLongArray.h"
#include "vtkUnsignedShortArray.h"
#include "vtkInformation.h"

#include "teem/nrrd.h"
#include "teem/gage.h"

#define VTK_EPS 1e-12

#define ITERMAX 500

#define STOP 1
#define CONTINUE 0

vtkStandardNewMacro(vtkExtractAirwayTree);

// Description:
// Construct object with initial range (0,1) and single contour value
// of 0.0. ComputeNormal is on, ComputeGradients is off and ComputeScalars is on.
vtkExtractAirwayTree::vtkExtractAirwayTree()
{

  Seed[0] = 0.0;
  Seed[1] = 0.0;
  Seed[2] = 0.0;
  TubularType = VTK_VALLEY;
  Scale = 3.0;
  ModeThreshold = 0.25;
  Delta = 0.1;
  RescaleAtStoppingPoint = 0;

}

vtkExtractAirwayTree::~vtkExtractAirwayTree()
{

}


//
// Contouring filter specialized for volumes and "short int" data values.  
//
template <class T>
void vtkExtractAirwayTreeComputeGradient(vtkExtractAirwayTree *self,T *scalars, int dims[3], 
                                     double origin[3], double Spacing[3],
                                     vtkPointLocator *locator, 
                                     vtkDataArray *newScalars, 
                                     vtkDataArray *newGradients, 
                                     vtkDataArray *newNormals, 
                                     vtkCellArray *newPolys, double *values, 
                                     int numValues)
{

}

//
// VTK6 migration note:
// - introduced to inform that the input type of this algorithm is vtkImageData
//
int vtkExtractAirwayTree::FillInputPortInformation(int, vtkInformation *info)
{
  info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
  return 1;
}

// ---------------------------------------------------------------------------
// Contouring filter specialized for volumes and "short int" data values.  
// VTK6 migration note:
// - replaced Execute()
// - changed this->GetInput() to vtkImageData::GetData(inInfoVec[0])
// - changed this->GetOutput() to vtkPolyData::GetData(outInfoVec)
int vtkExtractAirwayTree::RequestData(vtkInformation* vtkNotUsed(request), 
  vtkInformationVector** inInfoVec, 
  vtkInformationVector* outInfoVec)
{
  vtkPoints *newPts;
  vtkCellArray *newPolys;
  vtkImageData *input = vtkImageData::GetData(inInfoVec[0]);
  vtkPointData *pd;
  vtkDataArray *inScalars;
  int dims[3];
  double Spacing[3], origin[3];
  int numIter;
  vtkPolyData *output = vtkPolyData::GetData(outInfoVec);
 
  vtkDebugMacro(<< "Executing airway tree extraction");

//
// Initialize and check input
//
  if (input == NULL)
    {
    vtkErrorMacro(<<"Input is NULL");
    return 0;
    }
  pd=input->GetPointData();
  if (pd ==NULL)
    {
    vtkErrorMacro(<<"PointData is NULL");
    return 0;
    }
  inScalars=pd->GetScalars();
  if ( inScalars == NULL )
    {
    vtkErrorMacro(<<"Scalars must be defined for tracking");
    return 0;
    }

  if ( input->GetDataDimension() != 3 )
    {
    vtkErrorMacro(<<"Cannot tracked data of dimension != 3");
    return 0;
    }
  input->GetDimensions(dims);
  input->GetOrigin(origin);
  input->GetSpacing(Spacing);

  newPts = vtkPoints::New(); 
  newPolys = vtkCellArray::New();

  //Call teem
  void *data =  (void *) input->GetScalarPointer();
  Nrrd *nin = nrrdNew();
  const int type = this->VTKToNrrdPixelType(input->GetScalarType());
  size_t size[3];
  size[0]=dims[0];
  size[1]=dims[1];
  size[2]=dims[2];
  
  if(nrrdWrap_nva(nin,data,type,3,size)) {
	//sprintf(err,"%s:",me);
	//biffAdd(NRRD, err); return;
  }
  nrrdAxisInfoSet_nva(nin, nrrdAxisInfoSpacing, Spacing);
  
  // Creating context and assigning volume to context.
  int E = 0;
  gageContext *gtx = gageContextNew();
  gageParmSet(gtx, gageParmRenormalize, AIR_TRUE); // slows things down if true
  gagePerVolume *pvl; 
  if (!E) E |= !(pvl = gagePerVolumeNew(gtx, nin, gageKindScl));
  if (!E) E |= gagePerVolumeAttach(gtx, pvl);
  
  if (E) {
   vtkErrorMacro("Error Setting Gage Context... Leaving Execute");
   newPts->Delete();
   newPolys->Delete();
   return 0;
  }

  // Creating state for the tracking.
  vtkTrackingState *state = new vtkTrackingState;
  state->TubularType = this->GetTubularType();

  // Choose the right scale
  double scaleAtSeed = this->ScaleSelection(gtx,pvl,this->Seed,1.0,10.0,0.1);
  if (scaleAtSeed == -1) {
    //No good scale was found
    // Do something, for now we choose something
    scaleAtSeed = 2;
  }
  double scale = scaleAtSeed;
  cout<<"Tracking at Optimal scale: "<<scale<<endl;

  // Setting up the context and the state according to the estimated scaled.
  if (this->SettingContextAndState(gtx,pvl,state,scaleAtSeed)) {
    vtkErrorMacro("Error Setting Gage Context... Leaving Execute");
    delete state;
    newPts->Delete();
    newPolys->Delete();
    return 0;
  }

  // for example
  gageProbe(gtx, Seed[0], Seed[1], Seed[2]);

  double newS[3];
  // Do initial reallocation and readjust scale.
  /*
  newS[0]=this->Seed[0];
  newS[1]=this->Seed[1];
  newS[2]=this->Seed[2];
  this->RelocateSeed(gtx,pvl,newS,newS);
  scale = this->ScaleSelection(gtx,pvl,newS,2.0,10.0,0.1);
  cout<<"New scale: "<<scale<<endl;
  kparm[0] = floor(scale);
  if (!E) E |= gageKernelSet(gtx, gageKernel00, nrrdKernelBCCubic, kparm);
  if (!E) E |= gageKernelSet(gtx, gageKernel11, nrrdKernelBCCubicD, kparm);
  if (!E) E |= gageKernelSet(gtx, gageKernel22, nrrdKernelBCCubicDD, kparm);
  if (!E) E |= gageUpdate(gtx);
  */
  int direction[2];
  direction[0]=-1;
  direction[1]=+1;
  int npts[3];
  npts[0]=0;

  // Run tracking twice: forward and backwards
  for (int forward =0; forward<2;forward++) {
    newS[0]=this->Seed[0];
    newS[1]=this->Seed[1];
    newS[2]=this->Seed[2];
    //Move the seed to the minimum in the d-plane
    this->RelocateSeed(gtx,pvl,newS,newS);
    state->direction = direction[forward];
    for (int i=0; i<3;i++)
      state->Seed[i]=newS[i];

    // If scale has changed, update context
    if (scale != scaleAtSeed) {
      if (this->SettingContextAndState(state->gtx,pvl,state,scaleAtSeed)) {
         vtkErrorMacro("Error Setting Gage Context... Leaving Execute");
         delete state;
         newPts->Delete();
         newPolys->Delete();
         return 0;
        }
    }
    // Probe the state seed to make sure that gtx has the right pointers
    gageProbe(state->gtx,state->Seed[0],state->Seed[1],state->Seed[2]);
    //Save point
    newPts->InsertNextPoint(state->Seed[0],state->Seed[1],
                              state->Seed[2]);

    //Take initial step
    this->ApplyUpdate(state,newS);
    this->RelocateSeed(gtx,pvl,newS,newS);

    numIter = 0;
    do {
      //1. Update state: 
      //   1.1 save seed and previous gradient
      //   1.2 Probe at current location
      //   1.3 Update direction
      //2. Check stopping condition at current location
      //3. ApplyUpdate
      //4. Apply Constrain: seed should be in a minimum

      //cout<<"NEW STEP: Direction "<<direction[forward]<<"  ----"<<endl;
      // Probing should be always done like this:
      state->Update(newS); //Save state at k-1, probe at k and compute direction of evolution

      //Check stopping criteria at current location
      if (this->StoppingCondition(state) == STOP) 
        {
        // Check for new scale, in case the object scale has changed and this is the reason for stopping
        if (this->GetRescaleAtStoppingPoint()) 
          {
          scale = this->ScaleSelection(gtx,pvl,state->Seed,1.0,6.0,0.1);
          if (scale == -1) {
            // No good scale was found. Let us break here
            break;
          }
          cout<<"Stop Condition Optimal scale: "<<scale<<endl;
          if (this->SettingContextAndState(gtx,pvl,state,scale)) 
            {
            vtkErrorMacro("Error Setting Gage Context... Leaving Execute");
            delete state;
            newPts->Delete();
            newPolys->Delete();
            return 0;
            }

          //Probe at the state->Seed location without updating the whole state
          gageProbe(state->gtx,state->Seed[0],state->Seed[1],state->Seed[2]);
          // Let us not save state at k-1.
          // If condition is stop, break, if not, keep moving.
          if (this->StoppingCondition(state) == STOP)
            {
            //cout<<"Bailing out..."<<endl;
            break;
            }
          }
        else 
          {
          //cout<<"Bailing out..."<<endl;
          break;
          }
        }

      newPts->InsertNextPoint(state->Seed[0],state->Seed[1],
                              state->Seed[2]);

      // Apply new Update
      cout<<"Seed in state (before step): "<<state->Seed[0]<<" "<<state->Seed[1]<<" "<<state->Seed[2]<<endl;
      this->ApplyUpdate(state,newS);
      cout<<"Seed after taking step: "<<newS[0]<<" "<<newS[1]<<" "<<newS[2]<<endl;
      cout<<"State direction: "<<state->direction<<endl;
      //Move the seed to the minimum in the d Dim -plane
      this->RelocateSeed(gtx,pvl,newS,newS);
      cout<<"Seed after relocation: "<<newS[0]<<" "<<newS[1]<<" "<<newS[2]<<endl;

      numIter++;
    }while(numIter < 2*ITERMAX);

    // Fill CellArray
    npts[forward+1] = newPts->GetNumberOfPoints();
    newPolys->InsertNextCell(npts[forward+1]-npts[forward]);
    for (int k=npts[forward]; k<npts[forward+1]; k++) 
    {
      newPolys->InsertCellPoint(k);
    }

  }

  //NrrdIoState *nio = nrrdIoStateNew();
  //nrrdSave("test.nrrd",nin,nio); 
  //nrrdNix(nin);
  cout<<"Num points backwards: "<<npts[1]<<endl;
  cout<<"Num points forward: "<<npts[2]-npts[1]<<endl;
  vtkDebugMacro(<<"Created: " 
               << newPts->GetNumberOfPoints() << " points, " 
               << newPolys->GetNumberOfCells() << " lines");

  //
  // Update ourselves.  Because we don't know up front how many triangles
  // we've created, take care to reclaim memory. 
  //
  output->SetPoints(newPts);
  newPts->Delete();

  output->SetLines(newPolys);
  newPolys->Delete();

  // Free memory
  delete state;
  gageContextNix(gtx);
  nrrdNix(nin);
  
  return 1;
}

void vtkExtractAirwayTree::ApplyUpdate(vtkTrackingState *state, double * newS) {

  newS[0] = newS[0]+ state->direction*this->Delta*state->Tangent[0];
  newS[1] = newS[1]+ state->direction*this->Delta*state->Tangent[1];
  newS[2] = newS[2]+ state->direction*this->Delta*state->Tangent[2];

}

int vtkExtractAirwayTree::SettingContextAndState(gageContext *gtx,gagePerVolume *pvl, vtkTrackingState *state,double scale) 
{
  int E=0;
  E = this->SettingContext(gtx,pvl,scale);
  const double *valu = gageAnswerPointer(gtx, pvl, gageSclValue);
  const double *grad = gageAnswerPointer(gtx, pvl, gageSclGradVec);
  const double *hess = gageAnswerPointer(gtx, pvl, gageSclHessian);
  const double *hevec = gageAnswerPointer(gtx, pvl, gageSclHessEvec);
  const double *heval = gageAnswerPointer(gtx, pvl, gageSclHessEval);
  state->gtx = gtx;
  state->grad= grad;
  state->hevec=hevec;
  state->heval=heval;
  return E;

}

int vtkExtractAirwayTree::SettingContext(gageContext *gtx,gagePerVolume *pvl,double scale)
{

  double kparm[3] = {3.0, 1.0, 0.0};
  //Round scacle 
  kparm[0] = scale;
  int E=0;
  // { scale, B, C}; (B,C)=(1,0): uniform cubic B-spline
  if (!E) E |= gageKernelSet(gtx, gageKernel00, nrrdKernelBCCubic, kparm);
  if (!E) E |= gageKernelSet(gtx, gageKernel11, nrrdKernelBCCubicD, kparm);
  if (!E) E |= gageKernelSet(gtx, gageKernel22, nrrdKernelBCCubicDD, kparm);
  if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclValue);
  if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclGradVec);
  if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclHessian);
  if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclHessEvec);
  if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclHessEval);
  if (!E) E |= gageUpdate(gtx);

  return E;
}

int vtkExtractAirwayTree::StoppingCondition(vtkTrackingState *state)
{
  const double *hevec;
  const double *heval;
  const double *grad;
  hevec=state->hevec;
  heval=state->heval;
  grad=state->grad;

  double orth;
  double mode;
  double maxrange = 1000;  //Dynamic range of an airway

  double featureSign=1;

  // If we are working with vessels:
  // Eigenvalues corresponding to airway cross section are positive
  // Mode is positive.
  if (this->GetTubularType()==VTK_VALLEY) {
    featureSign = 1;
  }
  // If we are working with vessels:
  // Eigenvalues corresponding to vessel cross section are negative
  // Mode is positive.
  if (this->GetTubularType()==VTK_RIDGE) {
    featureSign = -1;
  }


  // First check Seed Point is not out of bounds
  int dims[3];
  vtkImageData* input = vtkImageData::SafeDownCast(this->GetInput());
  input->GetDimensions(dims);
  for (int k=0; k<3; k++) {
    if (state->Seed[k] >=dims[k] || state->Seed[k]<0) {
      cout<<"Point is a out-of-bounds"<<endl;
      return STOP;
    }
  } 

  mode = this->Mode(heval);

  //Check orthogonality condition
  if (this->GetTubularType()==VTK_VALLEY) {
    orth = 2.0/(maxrange*maxrange) * (pow((grad[0]*hevec[0]+grad[1]*hevec[1]+grad[2]*hevec[2]),2) + 
                    pow((grad[0]*hevec[3]+grad[1]*hevec[4]+grad[2]*hevec[5]),2));
  }
  if (this->GetTubularType()==VTK_RIDGE) {
    orth = 2.0/(maxrange*maxrange) * (pow((grad[0]*hevec[3]+grad[1]*hevec[4]+grad[2]*hevec[5]),2) + 
                    pow((grad[0]*hevec[6]+grad[1]*hevec[7]+grad[2]*hevec[8]),2));
  }

    cout<<"Stopping criteria: 1. Orth: "<<orth<<"   2. Mode: "<<mode<<"  heval[1]: "<<heval[1]<<endl;
      
  // Conditions for being a generalized minimum
  if (featureSign*heval[1] < 0) {
    cout<<"Not a valley"<<endl;
    return STOP;
   }

  if (orth > 0.05) {
    cout<<"Gradient not orthogonal to tube frame"<<endl;
    return STOP;
    }

   // Conditions for being a strong valley
   if (featureSign*heval[1] < fabs(heval[2])) {
     cout <<"Weak valley:"<<endl;
     return STOP;
   }

  // I THINK THAT FOR "STRENGTH" OF A RIDGE LINE
  // WE SHOULD TEST FOR:
  //if (featureSign*heval[1] < fabs(heval[0])) {
  //    cout <<"Weak valley:"<<endl;
  //    return STOP;
  //  }


      
   //if (mode >= 0) {
   //  cout<<"Mode change sign: potential branch"<<endl;
   //  return STOP;
   // }

  // Using mode to check for branches is a BAD idea.
  // Mode can be positive and this could mean that this is a
  // weak tube, for example for large airway (bronchus)
  //  if (-1.0*featureSign*mode< this->ModeThreshold) {
  //    cout<<"Potenial branch"<<endl; 
  //    return STOP;
  //   }


 return CONTINUE;

}

void vtkExtractAirwayTree::PrintState(vtkTrackingState *state)
{

  const double *heval = state->heval;
  const double *hevec = state->hevec;
  const double *grad = state->grad;
  double *newS = state->Seed;
  double mode= this->Mode(heval);
      cout<<"Probing Seed: "<<newS[0]<<" "<<newS[1]<<" "<<newS[2]<<endl; 
      cout<<"Hessian eval: "<<heval[0]<<" "<<heval[1]<<" "<<heval[2]<<endl;
      cout<<"Hessian ev1: "<<hevec[0]<<" "<<hevec[1]<<" "<<hevec[2]<<endl;
      cout<<"Hessian ev2: "<<hevec[3]<<" "<<hevec[4]<<" "<<hevec[5]<<endl;
      cout<<"Hessian ev3: "<<hevec[6]<<" "<<hevec[7]<<" "<<hevec[8]<<endl;
      cout<<"Gradient: "<<grad[0]<<" "<<grad[1]<<" "<<grad[2]<<endl;
      cout<<"Mode: "<<mode<<endl;
      cout<<"Direction: "<<state->direction<<endl;

      cout<<"Tangent: "<<state->Seed[0]-state->PSeed[0]<<" "<<state->Seed[1]-state->PSeed[1]<<" "<<state->Seed[2]-state->PSeed[2]<<endl;
      cout<<"Step: "<<state->hevec[6]<<" "<<state->hevec[7]<<" "<<state->hevec[8]<<endl;
      cout<<"Inner product: "<<(state->Seed[0]-state->PSeed[0])*state->hevec[6] + 
                     (state->Seed[1]-state->PSeed[1])*state->hevec[7] +
                     (state->Seed[2]-state->PSeed[2])*state->hevec[8]<<endl;
}

int vtkExtractAirwayTree::RelocateSeed(gageContext *gtx, gagePerVolume *pvl, double Seed[3], double SeedNew[3])
{
//cout<<"RelocateSeed"<<endl;
  double m[9], m2[9];
  double gradp[3];
  double lgrad;
  double tmp[3];
  double delta = 100;
  int numIter;
  int converge = 1;
  const double *tangent1, *tangent2, *normal;
 
  const double *valu = gageAnswerPointer(gtx, pvl, gageSclValue);
  const double *grad = gageAnswerPointer(gtx, pvl, gageSclGradVec);
  const double *hess = gageAnswerPointer(gtx, pvl, gageSclHessian);
  const double *hevec = gageAnswerPointer(gtx, pvl, gageSclHessEvec);
  const double *hevec3 = gageAnswerPointer(gtx, pvl,gageSclHessEvec2);
  const double *heval = gageAnswerPointer(gtx, pvl, gageSclHessEval);


  double featureSign=1;

  // If we are working with airways:
  // Second order derivated projected in the tangent plane is positive
  if (this->GetTubularType()==VTK_VALLEY) {
    featureSign = 1;
    tangent1 = hevec;
    tangent2 = hevec+3;
    normal = hevec+6;
  }
  // If we are working with vessels:
  // Second order derivated projected in the tangent plane is negative
  if (this->GetTubularType()==VTK_RIDGE) {
    featureSign = -1;
    tangent1 = hevec+3;
    tangent2 = hevec+6;
    normal = hevec;
  }


  double x[3],xn[3];
  x[0]=Seed[0];x[1]=Seed[1];x[2]=Seed[2];
  gageProbe(gtx,x[0],x[1],x[2]);

  //project gradient in the plane tangent to the tube axis
  ELL_3MV_OUTER(m,normal,normal);
  m[0] = 1 - m[0];
  m[4] = 1 - m[4];
  m[8] = 1 - m[8];
  
    
  ELL_3MV_MUL(gradp,m,grad);
  ELL_3V_NORM(gradp,gradp,lgrad);
  double prevv = (double) valu[0];
  // First and second order derivatives along gradp
  double d1 = ELL_3V_DOT(grad, gradp); 
  ELL_3MV_MUL(tmp,hess,gradp);
  double d2=ELL_3V_DOT(gradp,tmp);

  numIter = 0;
  double step = 1;
  double hack = 1;
  while (1) {
    if (numIter >= ITERMAX) 
      {
       //We didn't find a local minima in the number of iter
       SeedNew[0]=xn[0];
       SeedNew[1]=xn[1];
       SeedNew[2]=xn[2];
       converge = 0;
       break;
      }
    if (featureSign * d2 > 0) {
      step = featureSign* (d1/d2);
    } else {
      step = hack * featureSign *lgrad;
    }
    xn[0] = x[0]-step*gradp[0];
    xn[1] = x[1]-step*gradp[1];
    xn[2] = x[2]-step*gradp[2];
    //cout<<"New Point: "<<x[0]<<" "<<x[1]<<" "<<x[2]<<endl;
    gageProbe(gtx,xn[0],xn[1],xn[2]);
    if ((featureSign*prevv - featureSign*(double) valu[0]) < 0) 
     {
      hack = hack/2;
      delta = delta*hack;
      numIter++;
      continue;
     }
    if (fabs(featureSign*prevv - featureSign*(double) valu[0]) < 0.001 || fabs(step)<0.0001) 
      {
      /*We have converge*/
      SeedNew[0]=x[0];
      SeedNew[1]=x[1];
      SeedNew[2]=x[2];
      converge = 1;
      break;
      }
    prevv = (double) valu[0];
    x[0]=xn[0]; x[1]=xn[1]; x[2]=xn[2];
    //project gradient
    ELL_3MV_OUTER(m,normal,normal);
    m[0] = 1 - m[0];
    m[4] = 1 - m[4];
    m[8] = 1 - m[8];


    ELL_3MV_MUL(gradp,m,grad);
    ELL_3V_NORM(gradp,gradp,lgrad);
    d1 = ELL_3V_DOT(grad, gradp);
    ELL_3MV_MUL(tmp,hess,gradp);
    d2=ELL_3V_DOT(gradp,tmp);
    numIter++;
  }
return converge;
}
//-------------------------------------------------------------------------
// Relocate the seed in the plane orthogonal to the axis
// axis = {0 (jk plane), 1 (ik), 2 (ij)
int vtkExtractAirwayTree::RelocateSeedInPlane(gageContext *gtx, gagePerVolume *pvl, double Seed[3], double SeedNew[3],int axis)
{

//cout<<"Relocate Seed In Plane"<<endl;
  double m[9];
  double gradp[3];
  double lgrad;
  double tmp[3];
  int numIter =0;
  int converge = 1;

  //Do not use Heassian
  //int E = 0;
  //if (!E) E |= gageQueryItemOff(gtx, pvl, gageSclHessian);
  //if (!E) E |= gageQueryItemOff(gtx, pvl, gageSclHessEvec);
  //if (!E) E |= gageQueryItemOff(gtx, pvl, gageSclHessEval);
  //if (!E) E |= gageUpdate(gtx);

  const double *valu = gageAnswerPointer(gtx, pvl, gageSclValue);
  const double *grad = gageAnswerPointer(gtx, pvl, gageSclGradVec);
  const double *hess = gageAnswerPointer(gtx, pvl, gageSclHessian);
  const double *hevec = gageAnswerPointer(gtx, pvl, gageSclHessEvec);
  const double *hevec3 = gageAnswerPointer(gtx, pvl,gageSclHessEvec2);
  const double *heval = gageAnswerPointer(gtx, pvl, gageSclHessEval);

 double featureSign=1;

  // If we are working with airways:
  // Second order derivated projected in the tangent plane is positive
  if (this->GetTubularType()==VTK_VALLEY) {
    featureSign = 1;
  }
  // If we are working with vessels:
  // Second order derivated projected in the tangent plane is negative
  if (this->GetTubularType()==VTK_RIDGE) {
    featureSign = -1;
  }

  double x[3],xn[3];
  double normal[3];
  x[0]=Seed[0];x[1]=Seed[1];x[2]=Seed[2];
  //cout<<"Init Poinit: "<<x[0]<<" "<<x[1]<<" "<<x[2]<<endl;
  gageProbe(gtx,x[0],x[1],x[2]);

  normal[0] = 0.0;
  normal[1] = 0.0;
  normal[2] = 0.0;
  // Projection matrix
  normal[axis] = -1.0;

  //project gradient
  ELL_3MV_OUTER(m,normal,normal);
  m[0] = 1 - m[0];
  m[4] = 1 - m[4];
  m[8] = 1 - m[8];

  ELL_3MV_MUL(gradp,m,grad);
  ELL_3V_NORM(gradp,gradp,lgrad);
  double prevv = (double) valu[0];
  // First and second order derivatives along gradp
  double d1 = ELL_3V_DOT(grad, gradp);
  ELL_3MV_MUL(tmp,hess,gradp);
  double d2=ELL_3V_DOT(gradp,tmp);

  double step = 1;
  double hack =1;
  while (1) {
    if (numIter >= ITERMAX) 
      {
       //We didn't find a local minima in the number of iter
       SeedNew[0]=xn[0];
       SeedNew[1]=xn[1];
       SeedNew[2]=xn[2];
       converge = 0;
       break;
      }
    if (featureSign * d2 > 0) {
      step = featureSign* (d1/d2);
    } else {
      step = hack * featureSign * lgrad;
    }
    xn[0] = x[0]-step*gradp[0];
    xn[1] = x[1]-step*gradp[1];
    xn[2] = x[2]-step*gradp[2];
    //cout<<"GRad p: "<<gradp[0]<<" "<<gradp[1]<<" "<<gradp[2]<<" lgrad: "<<lgrad<<endl;
    //cout<<"Delta: "<<delta<<"  New Point: "<<xn[0]<<" "<<xn[1]<<" "<<xn[2]<<"  Value: "<<valu[0]<<endl;
    gageProbe(gtx,xn[0],xn[1],xn[2]);
    if ((featureSign*prevv - featureSign*(double) valu[0]) < 0) {
      hack = hack/2;
      numIter++;
      //cout<<"Step update: "<<step<<endl;
      continue;
    }
    if (fabs(featureSign*prevv - featureSign*(double) valu[0]) < 0.001 || fabs(step)<0.0001) {
      SeedNew[0]=x[0];
      SeedNew[1]=x[1];
      SeedNew[2]=x[2];
      converge = 1;
      break;
    }
    // Update for next iteration
    prevv = (double) valu[0];
    x[0]=xn[0]; x[1]=xn[1]; x[2]=xn[2];
    ELL_3MV_MUL(gradp,m,grad);
    ELL_3V_NORM(gradp,gradp,lgrad);
    d1 = ELL_3V_DOT(grad, gradp);                                        \
    ELL_3MV_MUL(tmp,hess,gradp);
    d2=ELL_3V_DOT(gradp,tmp);
    numIter++;
  }

  return converge;
}

double vtkExtractAirwayTree::ScaleSelection(gageContext *gtx, gagePerVolume *pvl, double Seed[3], double initS,double maxS, double deltaS) 
{
  double kparm[3] = {3.0, 1.0, 0.0};
  //double kparm[3] = {3.0, 0.5, 0.25};
  double S = initS;
  double Sopt = -1;
  double Sopt2 = -1;
  double prevV,nextV;
  double maxval = -10000000;

  double featureSign=1;

  // If we are working with vessels:
  // Eigenvalues corresponding to airway cross section are positive
  // Mode is negative.
  if (this->GetTubularType()==VTK_VALLEY) {
    featureSign = 1;
  }
  // If we are working with vessels:
  // Eigenvalues corresponding to vessel cross section are negative
  // Mode is positive.
  if (this->GetTubularType()==VTK_RIDGE) {
    featureSign = -1;
  }

  // Allocate strength array
  int numElem = 0;
  for (double k = initS;k <=maxS;k=k+deltaS) {
    numElem++;
  }
  double *strength = new double[numElem];


  kparm[0] = (double) S;
  int E = 0;
  if (!E) E |= gageKernelSet(gtx, gageKernel00, nrrdKernelBCCubic, kparm);
  if (!E) E |= gageKernelSet(gtx, gageKernel11, nrrdKernelBCCubicD, kparm);
  if (!E) E |= gageKernelSet(gtx, gageKernel22, nrrdKernelBCCubicDD, kparm);
  if (!E) E |= gageQueryItemOn(gtx, pvl, gageSclHessEval);
  if (!E) E |= gageUpdate(gtx);
  const double *heval = gageAnswerPointer(gtx, pvl, gageSclHessEval);
  gageProbe(gtx,Seed[0],Seed[1],Seed[2]);
  // For being a valley: l1>>0, l2>>0 and l3 ~ 0
  // Our metric for valleiness is: (l1 + l2)/2 - abs(l3)
  // We want to maximize this metric.
  // Or better instead, mode: We want to minimize this metric.

  //prevV = (heval[0] + heval[1])/2 - fabs(heval[2]);

  //prevV = this->Mode(heval);
  prevV = featureSign;
  prevV = -1000;
  int idx =0;
  do {
    kparm[0] = S;
    if (!E) E |= gageKernelSet(gtx, gageKernel00, nrrdKernelBCCubic, kparm);
    if (!E) E |= gageKernelSet(gtx, gageKernel11, nrrdKernelBCCubicD, kparm);
    if (!E) E |= gageKernelSet(gtx, gageKernel22, nrrdKernelBCCubicDD, kparm);
    if (!E) E |= gageUpdate(gtx);
    //heval = gageAnswerPointer(gtx, pvl, gageSclHessEval);
    gageProbe(gtx,Seed[0],Seed[1],Seed[2]);

    //cout<<"Testing Scale: "<<S<<"  Eigenvalues: "<<heval[0]<<" "<<heval[1]<<" "<<heval[2]<<"  Mode: "<<this->Mode(heval)<<"  Disparity measure: "<< (heval[0] + heval[1])/2 - fabs(heval[2])<<endl;
    
    //Normalized strenght: multiply for scale square
    nextV = S*S*featureSign*this->Strength(heval);
    if (nextV > prevV && featureSign*heval[1]> 0 && featureSign*heval[1] > fabs(heval[2])) {
         prevV = nextV;
         Sopt = S;
     }

    // Compute strenght for locations that make sense
    if (featureSign*heval[1]> 0 && featureSign*heval[1] > fabs(heval[2])) {
        strength[idx]=S*S*(heval[0]+heval[1])/2; 
    } else {
        strength[idx]=0;
    }
   //cout<<"Scale: "<<S<<" Mode: "<<nextV<<" h[0]: "<<heval[0]<<" h[1]: "<<heval[1]<<" h[2]: "<<heval[2]<<endl;


    S = S+deltaS;
     idx++;
     } while( S <= maxS);
   //cout<<"Sopt: "<<Sopt<<"  prevV: "<<prevV<<endl;

    S = initS;
    idx = 0;
    prevV = 0;
   // Find scale with maximun strenght in the range Sinit to Sopt
    do {
      if (strength[idx] > prevV) {
         prevV = strength[idx];
         Sopt2 = S;
      }
      S = S + deltaS;
      idx++;
    } while (S<=Sopt);
    //cout<<"Sopt2: "<<Sopt2<<endl;
    // If maximun strength was positive, use Sopt2.
    if (prevV > 0)
      Sopt = Sopt2;

  delete[] strength;
  return Sopt;
}

double vtkExtractAirwayTree::Mode(const double *w)
{

  // see PhD thesis, Gordon Kindlmann
  double mean = (w[0] + w[1] + w[2])/3;
  double norm = ((w[0] - mean)*(w[0] - mean) + 
                  (w[1] - mean)*(w[1] - mean) + 
                  (w[2] - mean)*(w[2] - mean))/3;
  norm = sqrt(norm);
  norm = norm*norm*norm;
  if (norm < VTK_EPS)
     norm += VTK_EPS;
  // multiply by sqrt 2: range from -1 to 1
  return  (M_SQRT2*((w[0] + w[1] - 2*w[2]) * 
                         (2*w[0] - w[1] - w[2]) * 
                         (w[0] - 2*w[1] + w[2]))/(27*norm)); 
}

double vtkExtractAirwayTree::Strength(const double *w)
{

  return ((w[0]+w[1])/2 - fabs(w[3]));
}


void vtkExtractAirwayTree::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

}

