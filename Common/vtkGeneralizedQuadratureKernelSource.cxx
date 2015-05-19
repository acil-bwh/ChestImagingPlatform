#include <math.h>

#include "vtkGeneralizedQuadratureKernelSource.h"
#include "vtkObjectFactory.h"
#include "vtkMath.h"
#include "vtkImageFourierCenter.h"
#include "vtkImageFFT.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

vtkStandardNewMacro(vtkGeneralizedQuadratureKernelSource);

//----------------------------------------------------------------------------
vtkGeneralizedQuadratureKernelSource::vtkGeneralizedQuadratureKernelSource()
{
  // class members
  this->CenterFrequency = vtkMath::Pi()/4;
  this->RelativeBandwidth = 0.65;
  this->SetFilterDirection(0,0,0);
  this->AngularExponent = 1;
  this->WindowFunction = 1;
  this->WindowNExponent = 4;
  this->WindowKExponent = 2;

  this->Dimensionality = 3;

  // superclass members
  // for now this class defaults to 0 freq in center
  this->SetZeroFrequencyLocationToOrigin();
}

//----------------------------------------------------------------------------
// Centered coordinate system for calculation of voxel values in kernel.
void vtkGeneralizedQuadratureKernelSource::CalculateFourierCoordinates(double *max_x, 
						       double *max_y, 
						       double *max_z,
						       double *k_dx,  
						       double *k_dy,  
						       double *k_dz)
{
  // set up local variables, with same names as C-F's
  // original code for consistency
  int x = (this->WholeExtent[1] - this->WholeExtent[0] + 1);
  int y = (this->WholeExtent[3] - this->WholeExtent[2] + 1);
  int z = (this->WholeExtent[5] - this->WholeExtent[4] + 1);
  double dx = this->VoxelSpacing[0];
  double dy = this->VoxelSpacing[1];
  double dz = this->VoxelSpacing[2];
  double PI = vtkMath::Pi();
 
  *max_x=PI / dx;
  *max_y=PI / dy;
  *max_z=PI / dz;
  
  *k_dx=2 * (*max_x)/x;
  *k_dy=2 * (*max_y)/y;
  *k_dz=2 * (*max_z)/z; 
  // Check for dimension with 1 slice. The frenquency should be zero then
  if (x == 1) {
    *max_x = 0;
    *k_dx = 0;
  }
  if (y == 1) {
    *max_y = 0;
    *k_dy = 0;
  }
  if (z == 1) {
    *max_z = 0;
    *k_dz = 0;
  }

}

struct vtkImageQuadThreadStruct
{
  vtkGeneralizedQuadratureKernelSource *Filter;
  int CenterKernel;
  vtkImageData *Input;
  vtkImageData *Output;
};

// ---------------------------------------------------------------------------
// this mess is really a simple function. All it does is call
// the ThreadedExecute method after setting the correct
// extent for this thread. Its just a pain to calculate
// the correct extent.
//
// VTK6 migration note:
// - Replaced output->GetUpdateExtent( ext ) with outInfo->Get(...) 
//   (to get correct update extent)
// - Added vtkDebugWithObjectMacro to make sure ext value is correct
 
VTK_THREAD_RETURN_TYPE vtkImageQuadThreadedExecute( void *arg )
{
  vtkImageQuadThreadStruct *str;
  int ext[6], splitExt[6], total;
  int threadId, threadCount;
  vtkImageData *output;

  threadId = ((ThreadInfoStruct *)(arg))->ThreadID;
  threadCount = ((ThreadInfoStruct *)(arg))->NumberOfThreads;

  str = (vtkImageQuadThreadStruct *)(((ThreadInfoStruct *)(arg))->UserData);
  output = str->Output;

  vtkInformation* outInfo = output->GetInformation();
  outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), ext);

  vtkDebugWithObjectMacro(str->Filter, << "UpdateExtent: " << ext[0] << ", " << ext[1] 
    << ", " << ext[2] << ", " << ext[3] << ", " << ext[4] << ", " << ext[5]);
    
  // execute the actual method with appropriate extent
  // first find out how many pieces extent can be split into.
  total = str->Filter->SplitExtent(splitExt, ext, threadId, threadCount);
  //total = 1;
  
  if (threadId < total)
    {
    // If we don't have to move the origin, CenterKernel is zero
    if (!str->CenterKernel) {
      str->Filter->ThreadedExecute(str->Input,str->Output, splitExt,threadId);
    } else {  
      str->Filter->ThreadedFourierCenterExecute(str->Input,str->Output, splitExt, threadId);
    }
    }
  
  // else
  //   {
  //   otherwise don't use this thread. Sometimes the threads dont
  //   break up very well and it is just as efficient to leave a 
  //   few threads idle.
  //   }
  
  return VTK_THREAD_RETURN_VALUE;
}


//----------------------------------------------------------------------------
// Overload ExecuteInformation to allocate enough space for a complex vector image
// We will have 2*n components where n is the number of vector components.
// 
// VTK6 migration note:
// - introduced to replace ExecuteInformation()
// - reference: http://www.vtk.org/Wiki/VTK/VTK_6_Migration/Removal_of_SetWholeExtent
int vtkGeneralizedQuadratureKernelSource::RequestInformation (
  vtkInformation       *  request,
  vtkInformationVector ** inputVector,
  vtkInformationVector *  outputVector)
{
  vtkInformation* outInfo = outputVector->GetInformationObject(0);
  outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), this->WholeExtent, 6);
  //Create always whole extent. 
  //outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), this->WholeExtent, 6);

  // This filter always is complex, so we should set the variable
  this->ComplexOutput = 1;
  // This filter always is in the Fourier Domain
  this->SetOutputDomainToFourier();
  
  // Compute the filter dimensionality from the Extent
  if (this->WholeExtent[5] == this->WholeExtent[4]) {
    if (this->WholeExtent[3] == this->WholeExtent[2]) {
      this->Dimensionality = 1;
    } else {
      this->Dimensionality = 2;  
    } 
  } else {
      this->Dimensionality = 3;
  }    
  
  // Num components = 2 * (Dimensionality + 1)
  vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_DOUBLE, 2*(this->Dimensionality+1));
  
  return 1;
  
  // begin old implemenation before migration
  /*
  vtkImageKernel *output = this->GetOutput();
  output->SetWholeExtent(this->WholeExtent);
  //Create always whole extent. 
  //output->SetUpdateExtent(this->WholeExtent);
  output->SetExtent(this->WholeExtent);
  output->SetScalarType(VTK_DOUBLE);
  // This filter always is complex, so we should set the variable
  this->ComplexOutput = 1;
  // This filter always is in the Fourier Domain
  this->SetOutputDomainToFourier();
  
  // Compute the filter dimensionality from the Extent
  if (this->WholeExtent[5] == this->WholeExtent[4]) {
    if (this->WholeExtent[3] == this->WholeExtent[2]) {
      this->Dimensionality = 1;
    } else {
      this->Dimensionality = 2;  
    } 
  } else {
      this->Dimensionality = 3;
  }    
  
  // Num components = 2 * (Dimensionality + 1)
  output->SetNumberOfScalarComponents(2*(this->Dimensionality+1));
  */ 
  // end old implementation before migration
}



//----------------------------------------------------------------------------
//We overload this to allow further operations after threading part
//
// VTK6 migration note:
// - introduced to replace ExecuteData()
// - replaced AllocateScalars() with AllocateScalars(outInfo)
// - added outData->SetInformation(outInfo) to retrieve later

void vtkGeneralizedQuadratureKernelSource::ExecuteDataWithInformation(
  vtkDataObject *data, vtkInformation* outInfo)
{
  vtkImageKernel *outData = vtkImageKernel::SafeDownCast(data);
  outData->SetExtent(this->WholeExtent);
  outData->AllocateScalars(outInfo);
  outData->SetInformation(outInfo);
  if (outData->GetScalarType() != VTK_DOUBLE)
   {
     vtkErrorMacro("Execute: This source only outputs floats");
   }
 
  vtkIndent indent;
  vtkImageQuadThreadStruct str;
  
  str.Filter = this;
  str.CenterKernel = 0;
  str.Input = NULL;
  str.Output = outData;
  
  this->Threader->SetNumberOfThreads(this->NumberOfThreads);
  
  // setup threading and the invoke threadedExecute
  this->Threader->SetSingleMethod(vtkImageQuadThreadedExecute, &str);
  this->Threader->SingleMethodExecute();
 
 
 // see if we need to shift the volume (so DC in corner)
 if (this->ZeroFrequencyLocation == VTK_KERNEL_ZERO_FREQUENCY_ORIGIN)
   {
     vtkImageData *tmp = vtkImageData::New();
     tmp->DeepCopy(outData);
 
     str.Filter = this;
     str.CenterKernel = 1;
     str.Input = tmp;
     str.Output = outData;
     
     this->Threader->SetSingleMethod(vtkImageQuadThreadedExecute, &str);
     this->Threader->SingleMethodExecute();
     tmp->Delete();
   }
 
} 

//----------------------------------------------------------------------------
// VTK6 migration note:
// Replaced outData->GetWholeExtent(wholeExt) with outInfo->Get(...)
// Added vtkDebugMacro to make sure the replacement works correctly
void vtkGeneralizedQuadratureKernelSource::ThreadedExecute(vtkImageData *inData,
  vtkImageData *outData, 
  int outExt[6], 
  int id) 
{
  // standard vtk variables
  double *outPtr;
  int idxX, idxY, idxZ;
  int maxX, maxY, maxZ;
  vtkIdType outIncX, outIncY, outIncZ;
  int wholeExt[6];
  unsigned long count = 0;
  unsigned long target;
  // set up local variables, with same names as C-F's
  // original code for consistency
  long loop;
  double max_x, max_y, max_z;
  double k_dx, k_dy, k_dz;
  double x_pos, y_pos, z_pos;
  double dx = this->VoxelSpacing[0];
  double dy = this->VoxelSpacing[1];
  double dz = this->VoxelSpacing[2];
  double r_abs, r_dot, tmp, radial;
  double dir_x = this->FilterDirection[0];
  double dir_y = this->FilterDirection[1];
  double dir_z = this->FilterDirection[2];
  double B = this->RelativeBandwidth;
  double logB = log(B);
  double ri = this->CenterFrequency;
  double A = this->AngularExponent;
  int dimensionality = this->Dimensionality;
  // additional variables
  int complex = this->ComplexOutput;
  double w;	//Window function
  double PI = vtkMath::Pi();
  double hp_cut = 0.8*PI;
  double n = this->WindowNExponent;
  double k = this->WindowKExponent;

  vtkInformation* outInfo = outData->GetInformation();
  outInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), wholeExt);

  vtkDebugMacro(<< "wholeExt: " << wholeExt[0] << ", " << wholeExt[1] << ", " << wholeExt[2] 
    << ", " << wholeExt[3] << ", " << wholeExt[4] << ", " << wholeExt[5]);
  
  // find the region to loop over
  maxX = outExt[1] - outExt[0];
  maxY = outExt[3] - outExt[2]; 
  maxZ = outExt[5] - outExt[4];
  
  // Get increments to march through data 
  outData->GetContinuousIncrements(outExt, outIncX, outIncY, outIncZ);
  
  outPtr = (double *) outData->GetScalarPointer(outExt[0],outExt[2],outExt[4]);
  
  target = (unsigned long)((maxZ+1)*(maxY+1)/50.0);
  target++;

  // normalize filter direction
  r_abs=sqrt(dir_x*dir_x+dir_y*dir_y+dir_z*dir_z);
  if (r_abs > 0) {
    dir_x /= r_abs;
    dir_y /= r_abs;
    dir_z /= r_abs;
  }
  
  // calculate centered coordinate system 
  this->CalculateFourierCoordinates(&max_x,&max_y,&max_z, 
				    &k_dx,&k_dy,&k_dz);

  // Loop through output pixels
  z_pos=-max_z+k_dz * (outExt[4]-wholeExt[4]);
  for (idxZ = 0; idxZ <= maxZ; idxZ++)
    {
      y_pos=-max_y + k_dy * (outExt[2] - wholeExt[2]);
      for (idxY = 0; !this->AbortExecute && idxY <= maxY; idxY++)
	{
	  if (!id)
	    {
	      if (!(count%target))
	        {
	          this->UpdateProgress(count/(50.0*target));
	        }
	      count++;
            }
	  x_pos=-max_x + k_dx * (outExt[0] - wholeExt[0]);
	  for (idxX = 0; idxX <= maxX; idxX++)
	    {
	     // Pixel operation
             if (dir_x == 0 && dir_y == 0 && dir_z == 0) {
               //Spherical filter
               r_dot = 1;
             } else {  
               r_dot=dir_x*x_pos+dir_y*y_pos+dir_z*z_pos;
	     }
	     //Original code:
	     r_abs=sqrt(x_pos*x_pos+y_pos*y_pos+z_pos*z_pos);
	      
	     // need to check that the -4 and 2 in the filter
	     // fcn are ok
	    if (r_abs > 0 && r_dot >= 0 ) 
            {
                radial = 1;
                // Direction Filter
                if (!(dir_x == 0 && dir_y == 0 && dir_z == 0)) {
                    tmp=r_dot/r_abs;
                    for (loop=0; loop<A; loop++)
                        radial *= tmp*tmp;
                }
                tmp=log(r_abs/ri);
		radial *=
		    exp( -tmp*tmp*(1 / (2* logB*logB)));
		  
		    if (this->WindowFunction) {
		        // we are using a window function 
		        // to make sure that filter goes down
		        // to zero at PI.
		        //RAUL: Method uses by C-F cos^2 technique
		        //if (r_abs <= hp_cut) {
		        //  w=1.0;
		      //} else if (hp_cut<r_abs && r_abs<=PI) {
		      //tmp=cos(PI*(r_abs-hp_cut)/(2*(PI-hp_cut)));
		      //w=tmp*tmp;
		      //} else {
		      //  w=0.0;
		      //}
                        //NEW approach
                        w =  pow(cos(PI/2*pow(fabs(x_pos*dx/PI),n)),k);
                        w *= pow(cos(PI/2*pow(fabs(y_pos*dy/PI),n)),k); 
                        w *= pow(cos(PI/2*pow(fabs(z_pos*dz/PI),n)),k);
		        radial *=w;
		    }
		} 
	    else
		{
		  radial = 0;
		}
        //if (idxY == 64 && idxZ == 0) {
        //    cout<<x_pos<<" "<<y_pos<<" "<<radial<<"   "<<tmp<<" "<<logB<<" "<<r_abs<<endl;
        //}

	    // Set the complex-vector output
        if (radial == 0) {
           switch(dimensionality) {
             case 1:
               *outPtr = 0.0; outPtr++;
               *outPtr = 0.0; outPtr++;
               *outPtr = 0.0; outPtr++;
               *outPtr = 0.0;
               break;
             case 2:
               *outPtr = 0.0; outPtr++;
               *outPtr = 0.0; outPtr++;
               *outPtr = 0.0; outPtr++;
               *outPtr = 0.0; outPtr++;
               *outPtr = 0.0; outPtr++;
               *outPtr = 0.0;
               break;
             case 3:
               *outPtr = 0.0; outPtr++;
               *outPtr = 0.0; outPtr++;
               *outPtr = 0.0; outPtr++;
               *outPtr = 0.0; outPtr++;
               *outPtr = 0.0; outPtr++;
               *outPtr = 0.0; outPtr++;
               *outPtr = 0.0; outPtr++;
               *outPtr = 0.0;
               break;
            }
        } else {
            switch(dimensionality) {
              case 1:   
              //Q[0] = -j U_x/|U|
              *outPtr = 0;
              outPtr++;
              *outPtr = - x_pos/r_abs * radial;
              outPtr++;
              break;
              
              case 2:
              //Q[0] = -j U_x/|U|
              *outPtr = 0;
              outPtr++;
              *outPtr = - x_pos/r_abs * radial;
              outPtr++;
              //Q[1] = -j U_y/|U|
              *outPtr = 0;
              outPtr++;
              *outPtr = - y_pos/r_abs * radial;
              outPtr++;
              break;
              
              case 3:
              //Q[0] = -j U_x/|U|
              *outPtr = 0;
              outPtr++;
              *outPtr = - x_pos/r_abs * radial;
              outPtr++;
              //Q[1] = -j U_y/|U|
              *outPtr = 0;
              outPtr++;
              *outPtr = - y_pos/r_abs * radial;
              outPtr++;              
              //Q[2] = -j U_z/|U|
              *outPtr = 0;
              outPtr++;
              *outPtr = - z_pos/r_abs *radial;
              outPtr++;
              break;
            }
            // Q[Dimensionality+1] = 1
            *outPtr = radial;
            outPtr++;
            *outPtr = 0;
        }

	    x_pos += k_dx;	  
	    outPtr++;
	  }
	  y_pos += k_dy;
	  outPtr += outIncY;
	}
      z_pos += k_dz;
      outPtr += outIncZ;
  }
}

void vtkGeneralizedQuadratureKernelSource::PrintSelf(ostream& os, vtkIndent indent)
{
  int i;
  vtkImageKernelSource::PrintSelf(os,indent);

  os << indent << "CenterFrequency: " << this->CenterFrequency << "\n";
  os << indent << "RelativeBandwidth: " << this->RelativeBandwidth << "\n";
  os << indent << "AngularExponent: " << this->AngularExponent << "\n";
  
  os << indent << "FilterDirection: " << "\n";
  for (i=0; i<3; i++)
    {
      os << indent << this->FilterDirection[i] << "\n";      
    }
}


