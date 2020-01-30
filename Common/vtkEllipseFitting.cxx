/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkEllipseFitting.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkEllipseFitting.h"

#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkMath.h"
#include "vnl/algo/vnl_real_eigensystem.h"
#include "vnl/vnl_matrix.h"

vtkStandardNewMacro(vtkEllipseFitting);

//---------------------------------------------------------------------------
// Construct object with initial Tolerance of 0.0
vtkEllipseFitting::vtkEllipseFitting()
{
  this->MajorAxisLength = 0;
  this->MinorAxisLength = 0;
  this->Angle =0;
  this->Center[0]=0;
  this->Center[1]=0;
  this->MajorAxis[0]=0;
  this->MajorAxis[1]=0;
  this->MinorAxis[0]=0;
  this->MinorAxis[1]=0;

  for (int i=0; i<6;i++) {
   this->P[i]=0;
  }
}

//--------------------------------------------------------------------------
vtkEllipseFitting::~vtkEllipseFitting()
{
}


//----------------------------------------------------------------------------
// VTK6 migration note:
// - introduced to replace ExecuteInformation()

int vtkEllipseFitting::RequestInformation (
  vtkInformation       *  vtkNotUsed(request),
  vtkInformationVector ** vtkNotUsed(inputVector),
  vtkInformationVector *  vtkNotUsed(outputVector))
{
  // This filter does not produce an polydata output
  //this->GetOutput()->SetNumberOfPoints(0);
  return 1;
}

//--------------------------------------------------------------------------
// VTK6 migration note:
// - replaced Execute()
// - replaced vtkstd::real(vMat(i,j)) to vMat(i,j).real() 

int vtkEllipseFitting::RequestData(vtkInformation* vtkNotUsed(request), 
  vtkInformationVector** inInfoVec, 
  vtkInformationVector* outInfoVec)
{  
  vtkPolyData *input = vtkPolyData::SafeDownCast(this->GetInput()); //always defined on entry into Execute
  if ( !input )
    {
    vtkErrorMacro("Input not defined");
    return 0;
    }
  vtkDebugMacro("Beginning Ellipse Fitting");
  vtkPoints   *inPts = input->GetPoints();
  vtkIdType   numPts = input->GetNumberOfPoints();
  if (numPts < 3)
    {
     vtkErrorMacro("At least 3 points are needed to fit an ellipse");
     return 0;
    }
  double p[3],x,y;
  // Define and allocate working matrices
  double S1[3][3];
  double S2[3][3];
  double S3[3][3];
  double M[3][3];
  double T[3][3];
  double S3I[3][3],S2T[3][3];

  double *m[3], *v[3];
  double m0[3], m1[3], m2[3];
  double v0[3], v1[3], v2[3];
  m[0] = m0; m[1] = m1; m[2] = m2; 
  v[0] = v0; v[1] = v1; v[2] = v2;

  for (int i=0; i<3;i++) {
    for (int j=0; j<3;j++) {
      S1[i][j]=0.0;
      S2[i][j]=0.0;
      S3[i][j]=0.0;
    }
  }
 // Build quadratic term matrix and linear term matrix
  for (int k=0;k<numPts;k++) {

    inPts->GetPoint(k,p);
    x = p[0];
    y = p[1];
    // Matrix S1
    S1[0][0] += x*x * x*x;
    S1[0][1] += x*x * x*y;
    S1[0][2] += x*x * y*y;
    S1[1][1] += x*y * x*y;
    S1[1][2] += x*y * y*y;
    S1[2][2] += y*y * y*y;

   // Matrix S2
   S2[0][0] += x*x * x;
   S2[0][1] += x*x * y;
   S2[0][2] += x*x;
   S2[1][0] += x*y *x;
   S2[1][1] += x*y * y;
   S2[1][2] += x*y;
   S2[2][0] += y*y *x;
   S2[2][1] += y*y *y;
   S2[2][2] += y*y; 

   // Matrix S3
   S3[0][0] += x*x;
   S3[0][1] += x*y;
   S3[0][2] += x;
   S3[1][1] += y*y;
   S3[1][2] += y;
   S3[2][2] += 1;

  }

  // S1 and S3 are symetric
  S1[1][0] = S1[0][1];
  S1[2][0] = S1[0][2];
  S1[2][1] = S1[1][2];
  
  S3[1][0] = S3[0][1];
  S3[2][0] = S3[0][2];
  S3[2][1] = S3[1][2];

  vtkMath::Invert3x3(S3,S3I);
  vtkMath::Transpose3x3(S2,S2T);
  for (int i=0; i<3;i++) {
    for (int j=0; j<3;j++) {
      S3I[i][j]=-S3I[i][j];
    }
  }
  //this->MultiplyMatrix(S3I,S2T,T);
  //this->MultiplyMatrix(S2,T,M);
  vtkMath::Multiply3x3(S3I,S2T,T);
  vtkMath::Multiply3x3(S2,T,M);
  for (int i=0; i<3;i++) {
    for (int j=0; j<3;j++) {
      M[i][j]=S1[i][j]+M[i][j];
    }
  }

  for (int i=0;i<3;i++) {
   m[0][i] = M[2][i]/2;
   m[1][i] = -M[1][i];
   m[2][i] = M[0][i]/2;
  }
  vnl_matrix<double> mMat(3,3);
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      mMat(i,j) = m[i][j];
    }
  }
  double cond;
  int axis =0;
  double a1[3];
  double a[6];

  vnl_real_eigensystem solver(mMat);
  vnl_matrix<std::complex<double> > vMat = solver.V;
  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      v[i][j]=vMat(i,j).real();

  for (int i=0; i<3;i++) {
   cond = 4*v[0][i] * v[2][i] - v[1][i] * v[1][i];
   if (cond >0)
     axis = i;
  }
  for (int i=0; i<3;i++) {
    a1[i] = v[i][axis];
    a[i]=a1[i];
  }

  for (int i=0; i<3; i++) {
    a[i+3] = T[i][0] * a1[0] + T[i][1] * a1[1] + T[i][2] * a1[2]; 
  }

  double mu = sqrt(1/(4*a1[0]*a1[2]  - a1[1]*a1[1]));

  int sa = 1;
  if (a[0] < 0)
    sa=-1;
  // scale a
  for (int i=0; i<6; i++) {
   a[i] = mu*sa*a[i];
  }

  // Get output ellipse params from a
  this->Angle = atan(a[1]/(a[2]-a[0]))/2.0;

  // Rotate ellipse parallel to x
  double ar[6];
  double cp = cos(this->Angle);
  double sp = sin(this->Angle);

  ar[0]=a[0]*cp*cp - a[1]*cp*sp + a[2]*sp*sp;
  ar[1]=2*(a[0]-a[2])*cp*sp + (cp*cp-sp*sp)*a[1];
  ar[2]=a[0]*sp*sp + a[1]*sp*cp + a[2]*cp*cp;
  ar[3]=a[3]*cp - a[4]*sp;
  ar[4]=a[3]*sp + a[4]*cp;
  ar[5]=a[5];

  this->Center[0] = -cp * ar[3]/(2*ar[0]) - sp * ar[4]/(2*ar[2]);
  this->Center[1] = sp * ar[3]/(2*ar[0]) - cp * ar[4]/(2*ar[2]);

  double F = -ar[5] + ar[3]*ar[3]/(4*ar[0]) + ar[4]*ar[4]/(4*ar[2]);

  this->MajorAxisLength = sqrt(F/ar[0]);
  this->MinorAxisLength = sqrt(F/ar[2]);
  this->Angle = -this->Angle;
  double tmp;
  sa = 1;
  if (this->Angle <0)
    sa = -1;
  if (this->MajorAxisLength < this->MinorAxisLength) {
    this->Angle = this->Angle - sa*vtkMath::Pi()/2.0;
    tmp =this->MajorAxisLength;
    this->MajorAxisLength = this->MinorAxisLength;
    this->MinorAxisLength = tmp;
  }

  this->MajorAxis[0] = cos(this->Angle);
  this->MajorAxis[1] = sin(this->Angle);
  this->MinorAxis[0] = cos(this->Angle+vtkMath::Pi()/2.0);
  this->MinorAxis[1] = sin(this->Angle+vtkMath::Pi()/2.0);

  for (int i=0; i<6; i++) {
   this->P[i] = a[i];
  }

  vtkDebugMacro("End Ellipse fitting");

  return 1;
} 


void vtkEllipseFitting::MultiplyMatrix(double A1[3][3], double A2[3][3], double R[3][3])
{

  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      R[i][j]=0;
      for (int k=0; k<3; k++) {
         R[i][j] += A1[i][k] * A2[k][j];
      }
    }
  } 
}
//--------------------------------------------------------------------------
void vtkEllipseFitting::PrintSelf(ostream& os, vtkIndent indent) 
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "running class: "<<this->GetClassName()<<endl;

}


