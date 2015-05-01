#include <math.h>

#include "vtkImageKernelSource.h"
#include "vtkObjectFactory.h"
#include "vtkMath.h"
#include "vtkExecutive.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#define VTK_NUM_DIMENSIONS 3

vtkStandardNewMacro(vtkImageKernelSource);

//----------------------------------------------------------------------------
// VTK6 migration note:
// Added this->SetNumberOfInputPorts(0);
vtkImageKernelSource::vtkImageKernelSource()
{
  this->SetOutput(vtkImageKernel::New());
  // Releasing data for pipeline parallism.
  // Filters will know it is empty. 
  this->GetOutput()->ReleaseData();
  this->GetOutput()->Delete();
  
  //Threader
  this->Threader = vtkMultiThreader::New();
  this->NumberOfThreads = this->Threader->GetNumberOfThreads();

  // cubical voxels are default
  this->SetVoxelSpacing(1,1,1);

  // default kernel size is 64 x 64 x 32
  // this is a weird default so that kernel subclasses
  // will have to work right
  this->WholeExtent[0] = 0;  this->WholeExtent[1] = 63;
  this->WholeExtent[2] = 0;  this->WholeExtent[3] = 63;
  this->WholeExtent[4] = 0;  this->WholeExtent[5] = 31;

  // default kernel is in the fourier domain
  this->OutputDomain = VTK_KERNEL_OUTPUT_FOURIER_DOMAIN;

  // and has the zero frequency at the origin
  this->ZeroFrequencyLocation = VTK_KERNEL_ZERO_FREQUENCY_ORIGIN;

  // default is 1 scalar component output
  this->ComplexOutput = 0;
  
  this->SetNumberOfInputPorts(0);
}

vtkImageKernelSource::~vtkImageKernelSource() {

  this->Threader->Delete();

}

//----------------------------------------------------------------------------
// Specify the input data or filter.
// VTK6 migration note:
// - replaced vtkSource::SetNthOutput with GetExecutive()->SetOutputData
// - reference: Imaging/Core/vtkImageStencilAlgorithm.cxx
void vtkImageKernelSource::SetOutput(vtkImageKernel *output)
{
  this->GetExecutive()->SetOutputData(0, output);
}

//----------------------------------------------------------------------------
// Specify the input data or filter.
// VTK6 migration note:
// - replaced NumberOfOutputs with GetNumberofOutputPorts()
// - replaced Outputs[0] with GetExecutive()->GetOutputData(0)
// - reference: Imaging/Core/vtkImageStencilAlgorithm.cxx
vtkImageKernel *vtkImageKernelSource::GetOutput()
{
  if (this->GetNumberOfOutputPorts() < 1)
    {
    return NULL;
    }
  
  return (vtkImageKernel *)(this->GetExecutive()->GetOutputData(0));
}

//----------------------------------------------------------------------------
// VTK6 migration note:
// - replaced NumberOfOutputs with GetNumberofOutputPorts()
// - replaced Outputs[idx] with GetExecutive()->GetOutputData(idx)
// - reference: Imaging/Core/vtkImageStencilAlgorithm.cxx
vtkImageKernel *vtkImageKernelSource::GetOutput(int idx)
{
  if (this->GetNumberOfOutputPorts() <= idx)
    {
    return NULL;
    }
  
  return (vtkImageKernel *)(this->GetExecutive()->GetOutputData(idx));
}


//----------------------------------------------------------------------------
void vtkImageKernelSource::PrintSelf(ostream& os, vtkIndent indent)
{
  int idx;

  vtkImageAlgorithm::PrintSelf(os,indent);

  os << "VoxelSpacing:\n";
  for (idx = 0; idx < 3; ++idx)
  {
    os << indent << ", " << this->VoxelSpacing[idx];
  }
  os << ")\n";

  os << "WholeExtent:\n";
  for (idx = 0; idx < 6; ++idx)
  {
    os << indent << ", " << this->WholeExtent[idx];
  }
  os << ")\n";
}




//----------------------------------------------------------------------------
void vtkImageKernelSource::SetWholeExtent(int xMin, int xMax, 
					  int yMin, int yMax,
					  int zMin, int zMax)
{
  int modified = 0;
  
  if (this->WholeExtent[0] != xMin)
    {
      modified = 1;
      this->WholeExtent[0] = xMin ;
    }
  if (this->WholeExtent[1] != xMax)
    {
      modified = 1;
      this->WholeExtent[1] = xMax ;
    }
  if (this->WholeExtent[2] != yMin)
    {
      modified = 1;
      this->WholeExtent[2] = yMin ;
    }
  if (this->WholeExtent[3] != yMax)
    {
      modified = 1;
      this->WholeExtent[3] = yMax ;
    }
  if (this->WholeExtent[4] != zMin)
    {
      modified = 1;
      this->WholeExtent[4] = zMin ;
    }
  if (this->WholeExtent[5] != zMax)
    {
      modified = 1;
      this->WholeExtent[5] = zMax ;
    }
  if (modified)
    {
      this->Modified();
    }
}


struct vtkImageKernelSourceThreadStruct
{
  vtkImageKernelSource *Filter;
  vtkImageData *Output;
  vtkInformation *OutInfo;
};


// this mess is really a simple function. All it does is call
// the ThreadedExecute method after setting the correct
// extent for this thread. Its just a pain to calculate
// the correct extent.
//
// VTK6 migration note:
// - replaced output->GetUpdateExtent() with outInfo->Get(...)
VTK_THREAD_RETURN_TYPE vtkImageKernelSourceThreadedExecute( void *arg )
{
  vtkImageKernelSourceThreadStruct *str;
  int ext[6], splitExt[6], total;
  int threadId, threadCount;
  vtkImageData *output;
  vtkInformation *outInfo;

  threadId = ((ThreadInfoStruct *)(arg))->ThreadID;
  threadCount = ((ThreadInfoStruct *)(arg))->NumberOfThreads;

  str = (vtkImageKernelSourceThreadStruct *)(((ThreadInfoStruct *)(arg))->UserData);
  output = str->Output;
  outInfo = str->OutInfo;
  
  outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), ext);

  // execute the actual method with appropriate extent
  // first find out how many pieces extent can be split into.
  total = str->Filter->SplitExtent(splitExt, ext, threadId, threadCount);
  //total = 1;
  
  if (threadId < total)
    {
    str->Filter->ThreadedExecute(NULL, str->Output, splitExt, threadId);
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
// For streaming and threads.  Splits output update extent into num pieces.
// This method needs to be called num times.  Results must not overlap for
// consistent starting extent.  Subclass can override this method.
// This method returns the number of peices resulting from a successful split.
// This can be from 1 to "total".  
// If 1 is returned, the extent cannot be split.
int vtkImageKernelSource::SplitExtent(int splitExt[6], int startExt[6], 
				       int num, int total)
{
  int splitAxis;
  int min, max;

  vtkDebugMacro(<< "SplitExtent: ( " << startExt[0] << ", " << startExt[1] << ", "
		<< startExt[2] << ", " << startExt[3] << ", "
		<< startExt[4] << ", " << startExt[5] << "), " 
		<< num << " of " << total);

  // start with same extent
  memcpy(splitExt, startExt, 6 * sizeof(int));
  
  
  //Split the largest axis
  splitAxis = -1;
  int length_max = 0;
  int length = 0;
  for(int i=0; i < 3; i++) {
    length=startExt[2*i+1]-startExt[2*i];
    if(length>length_max){
      length_max=length;
      splitAxis=i;
    }  
  }
  
  if(splitAxis < 0) 
    {
     // cannot split
     vtkDebugMacro("  Cannot Split");
     return 1;
    }
  
  min = startExt[splitAxis*2];
  max = startExt[splitAxis*2+1];
  
  // determine the actual number of pieces that will be generated
  int range = max - min + 1;
  int valuesPerThread = (int)ceil(range/(double)total);
  int maxThreadIdUsed = (int)ceil(range/(double)valuesPerThread) - 1;
  if (num < maxThreadIdUsed)
    {
    splitExt[splitAxis*2] = splitExt[splitAxis*2] + num*valuesPerThread;
    splitExt[splitAxis*2+1] = splitExt[splitAxis*2] + valuesPerThread - 1;
    }
  if (num == maxThreadIdUsed)
    {
    splitExt[splitAxis*2] = splitExt[splitAxis*2] + num*valuesPerThread;
    }
  
  vtkDebugMacro("  Split Piece: ( " <<splitExt[0]<< ", " <<splitExt[1]<< ", "
		<< splitExt[2] << ", " << splitExt[3] << ", "
		<< splitExt[4] << ", " << splitExt[5] << ")");

  return maxThreadIdUsed + 1;
}

// ---------------------------------------------------------------------------
// VTK6 migration note:
// - introduced this method to replace ExecuteInformation()
// - reference: http://www.vtk.org/Wiki/VTK/VTK_6_Migration/Removal_of_SetWholeExtent
int vtkImageKernelSource::RequestInformation (
  vtkInformation       *  request,
  vtkInformationVector ** inputVector,
  vtkInformationVector *  outputVector)
{
  vtkInformation* outInfo = outputVector->GetInformationObject(0);
  outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), this->WholeExtent, 6);
  //Create always whole extent. 
  outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), this->WholeExtent, 6);
  if (this->ComplexOutput)
    {
    vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_FLOAT, 2);
    }
  else
    {
    vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_FLOAT, 1);
    }
    
  return 1;
    
  // begin old implementation before migration
  /*
  vtkImageKernel *output = this->GetOutput();
  output->SetWholeExtent(this->WholeExtent);
  //Create always whole extent. 
  output->SetUpdateExtent(this->WholeExtent);
  output->SetExtent(this->WholeExtent);
  output->SetScalarType(VTK_FLOAT);
  if (this->ComplexOutput)
    output->SetNumberOfScalarComponents(2);
  else
    output->SetNumberOfScalarComponents(1);    
  */ 
  // end old implementation before migration
}

//----------------------------------------------------------------------------
// VTK6 migration note:
// - introduced to replace Execute() and ExecuteData()
void vtkImageKernelSource::ExecuteDataWithInformation(vtkDataObject *data,
  vtkInformation* outInfo)
{

  vtkImageData *outData = vtkImageData::SafeDownCast(data);
  vtkImageKernelSourceThreadStruct str;
  
  str.Filter = this;
  str.Output = outData;
  str.OutInfo = outInfo;
  
  this->Threader->SetNumberOfThreads(this->NumberOfThreads);
  
  // setup threading and the invoke threadedExecute
  this->Threader->SetSingleMethod(vtkImageKernelSourceThreadedExecute, &str);
  this->Threader->SingleMethodExecute();

}

//----------------------------------------------------------------------------
void vtkImageKernelSource::ThreadedExecute(vtkImageData *inData,
                                           vtkImageData *outData,
                                           int extent[6], int threadId)
{
  extent = extent;
  vtkErrorMacro("subclass should override this method!!!");
}

//----------------------------------------------------------------------------
// The execute method for computing fftshift.
//
// VTK6 migration note:
// - Replaced outData->GetWholeExtent() with outInfo->Get(...)
void vtkImageKernelSource::ThreadedFourierCenterExecute(vtkImageData *inData,
                                                        vtkImageData *outData,
                                                        int outExt[6], int id) 
{
  double *inPtr, *outPtr;
  int idxX, idxY, idxZ,idxC;
  vtkIdType outIncX, outIncY, outIncZ;
  int wholeExtent[6];
  int maxX, maxY, maxZ;
  int numberOfComponents;
  int mid[3],inIndx[3],outIndx[3];
  unsigned long count = 0;
  unsigned long target;

// Raul: This is not anymore true for Generalized Quadrature filters
// this filter expects input to have 1 or two components
//  if (outData->GetNumberOfScalarComponents() != 1 && 
//      outData->GetNumberOfScalarComponents() != 2)
//    {
//    vtkErrorMacro(<< "Execute: Cannot handle more than 2 components");
//    return;
//    }

  // Get stuff needed to loop through the pixel
  numberOfComponents = outData->GetNumberOfScalarComponents();
  outPtr = (double *)(outData->GetScalarPointerForExtent(outExt));
  outData->GetContinuousIncrements(outExt, outIncX, outIncY, outIncZ);
  
  vtkInformation* outInfo = outData->GetInformation();
  outInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), wholeExtent); 
  
  vtkDebugMacro("wholeExtent: " << wholeExtent[0] << ", " << wholeExtent[1] << ", " << wholeExtent[2] 
      << ", " << wholeExtent[3] << ", " << wholeExtent[4] << ", " << wholeExtent[5]);
  
  for (int i = 0 ; i<VTK_NUM_DIMENSIONS ; i++) {
     //ifftshift style
     //mid[i] = (int) ceil((wholeExtent[i * 2] + wholeExtent[i * 2 + 1] + 1)/2);
     //fftshift style
     mid[i] = (int) ((wholeExtent[i * 2] + wholeExtent[i * 2 + 1] + 1)/2);
  }
  // find the region to loop over
  maxX = outExt[1] - outExt[0];
  maxY = outExt[3] - outExt[2]; 
  maxZ = outExt[5] - outExt[4];
  
  target = (unsigned long)((maxZ+1)*(maxY+1)/50.0);
  target++;
  
  for (idxZ = outExt[4]; idxZ <= outExt[5]; idxZ++)
  {
    for (idxY = outExt[2]; !this->AbortExecute && idxY <= outExt[3]; idxY++)
    { 
      if (!id)
      {
        if (!(count%target))
        {
          this->UpdateProgress(count/(50.0*target));
        }
        count++;
      }
      for (idxX = outExt[0]; idxX <= outExt[1]; idxX++)
      {
        outIndx[0]=idxX;
        outIndx[1]=idxY;
        outIndx[2]=idxZ;
        this->ComputeInputIndex(outIndx,mid,inIndx);
        inPtr = (double *) inData->GetScalarPointer(inIndx[0],inIndx[1],inIndx[2]);
        if (inPtr == NULL) {
          cout<<"Error in fftshift: index out out boundaries"<<endl;
          continue;
        }

        for (idxC = 0 ; idxC < numberOfComponents; idxC++) { 
          outPtr[idxC] = inPtr[idxC];
        }
        outPtr = outPtr+numberOfComponents;
      }
      outPtr += outIncY;
    }
    outPtr += outIncZ;
  }
}

//----------------------------------------------------------------------------
void vtkImageKernelSource::ComputeInputIndex(int outIndx[3],int mid[3],
                                             int inIndx[3])
{
  int indx;
  for(indx = 0; indx < VTK_NUM_DIMENSIONS; indx++) {
    if ( outIndx[indx] < mid[indx] ) 
       inIndx[indx] = mid[indx] + outIndx[indx];
    else
       inIndx[indx] = outIndx[indx]  - mid[indx];
  }
}  
