/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkImageStatistics.cxx,v $
  Date:      $Date: 2009/01/07 00:22:51 $
  Version:   $Revision: 1.1 $

=========================================================================auto=*/
#include "vtkImageStatistics.h"
#include "vtkObjectFactory.h"
#include "vtkImageData.h"
#include <math.h>
#include <stdlib.h>

//------------------------------------------------------------------------------
vtkImageStatistics* vtkImageStatistics::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkImageStatistics");
  if(ret)
    {
    return (vtkImageStatistics*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkImageStatistics;
}

//----------------------------------------------------------------------------
// Constructor sets default values
vtkImageStatistics::vtkImageStatistics()
{
  this->IgnoreZeroOn();

  Quartile1 = -1;
  Median    = -1;
  Quartile3 = -1;
  Quintile1 = -1;
  Quintile2 = -1;
  Quintile3 = -1;
  Quintile4 = -1;
  Average   = -1;
  Stdev     = -1;
  NumExaminedElements = -1;
}

//----------------------------------------------------------------------------
void vtkImageStatistics::ExecuteInformation(vtkImageData *input, 
                        vtkImageData *output)
{
  int min0,max0,min1,max1,min2,max2;
  input->GetWholeExtent(min0, max0, min1, max1, min2, max2);
  int numelement = (max0-min0+1)*(max1-min1+1)*(max2-min2+1);

  int ext[6];
  memset(ext, 0, 6*sizeof(int));
  ext[1] = numelement;

  vtkFloatingPointType origin[3], spacing[3];
  spacing[0] = spacing[1] = spacing[2] = 1;
  origin[0]  = origin[1] = origin[2] = 0;

  output->SetWholeExtent(input->GetWholeExtent());
  output->SetOrigin(origin);
  output->SetSpacing(spacing);
  output->SetNumberOfScalarComponents(1);
  output->SetScalarType(input->GetScalarType());
}

//----------------------------------------------------------------------------
// Get ALL of the input.
void vtkImageStatistics::ComputeInputUpdateExtent(int inExt[6], 
                                                  int outExt[6])
{
  int *wholeExtent;

  outExt = outExt;
  wholeExtent = this->GetInput()->GetWholeExtent();
  memcpy(inExt, wholeExtent, 6*sizeof(int));
}

//----------------------------------------------------------------------------

template <class T>
class vtkImageStatisticsHelper {
public:
  static int SortCompare(const void *a, const void *b)
  {
    const T *da = (const T *) a;
    const T *db = (const T *) b;
  
    return (*da > *db) - (*da < *db);
  };
};

//----------------------------------------------------------------------------

// This templated function executes the filter for any type of data.
template <class T>
static void vtkImageStatisticsExecute(vtkImageStatistics *self,
                      vtkImageData *inData, T *inPtr,
                      vtkImageData *outData, T *outPtr)
{
  int min0, max0, min1, max1, min2, max2;
  int Amin0, Amax0, Amin1, Amax1, Amin2, Amax2;

  T *dataPtr;
  // Information
  inData->GetExtent(min0, max0, min1, max1, min2, max2);
  outData->GetExtent(Amin0, Amax0, Amin1, Amax1, Amin2, Amax2);
  int numelement = (max0-min0+1)*(max1-min1+1)*(max2-min2+1);

//  cout << "in: " << min0 << " " << max0 << " " 
//       << min1 << " " << max1 << " "
//       << min2 << " " << max2 << "\n";
//  cout << "out: " << Amin0 << " " << Amax0 << " " 
//       << Amin1 << " " << Amax1 << " "
//       << Amin2 << " " << Amax2 << "\n";

  //
  // Put the zeros first if we are to ignore them.
  //

  // The offset is the number of 0's which are ignored
  int offset = 0;
  // the number of elements we actually consider
  int num_stat_element = numelement;

  if (self->GetIgnoreZero())
    {
      // Determine the number of zeros
      for(int i=0;i<numelement;i++)
        if (double(inPtr[i]) == 0.0) offset++;
      num_stat_element = numelement - offset;

      // 0 the first offset elements
      memset((void *)outPtr, 0, offset*sizeof(T));

      // fill in the rest
      dataPtr = outPtr+offset;
      for(int j=0;j<numelement;j++)
        if (double(inPtr[j]) != 0.0)
          {
            *dataPtr = inPtr[j];
            dataPtr++;
          }
      //  cout << "Pointers: " << dataPtr << " " << outPtr+numelement << '\n';
    }
  else
    {
      // Copy the input to the output data
      memcpy((void *)outPtr, (void *)inPtr, numelement*sizeof(T)); 
    }

  dataPtr = outPtr+offset;

  // sort the remaining data
  qsort((void *)dataPtr, num_stat_element, sizeof(T),
        vtkImageStatisticsHelper<T>::SortCompare);

#if 0
  int offset = 0;
  cout.setf(ios::scientific);

  if (self->GetIgnoreZero())
   { 
     if (double(*outPtr) < 0.0)
       {
         if (vtkObject::GetGlobalWarningDisplay()) 
           { 
             char *vtkmsgbuff; 
             ostrstream vtkmsg; 
             vtkmsg << "ERROR: In " __FILE__ ", line " << __LINE__ << "\n" 
                    << "vtkImageStatistics can not ignore zeros with data that includes negative numbers!" 
                    << "\n\n" << ends; 
             vtkmsgbuff = vtkmsg.str(); 
             vtkOutputWindowDisplayText(vtkmsgbuff);
             vtkmsg.rdbuf()->freeze(0); vtkObject::BreakOnError();
           }
       }

     int stillzero = 1;
     while((offset<numelement)&&(stillzero))
       {
         double num = (double) *(outPtr+offset);
         if (num == 0.0)
           {
             offset++;
             num_stat_element--;
           }
         else stillzero = 0;
       }
   }
#endif

//  cout << "num_element: " << numelement << "\n";
//  cout << "num_stat_element: " << num_stat_element << "\n";
//  cout << "offset: " << offset << "\n";

  // Now Determine the statistics
  double total=0,totalsq=0;
  double min = inData->GetScalarTypeMax();
  double max = inData->GetScalarTypeMin();

  dataPtr = outPtr+offset;

  for(int s=0;s<num_stat_element;s++)
    {
      double newnum = (double ) *dataPtr;
      dataPtr++;
      // cout << "Pointers: " << dataPtr << " " << outPtr+numelement << '\n';
      //      cout << "newnum: " << newnum << " offset: " << offset << 
      //        "s : " << s << '\n';
      total += newnum;
      totalsq += newnum*newnum;
      if (newnum < min) min = newnum;
      if (newnum > max) max = newnum;
    }

  dataPtr = outPtr+offset;

  self->SetNumExaminedElements(num_stat_element);
  if (num_stat_element != 0)
    {
      self->SetQuartile1( (double) dataPtr[(num_stat_element*1)/4]); 
      self->SetMedian   ( (double) dataPtr[(num_stat_element*1)/2]); 
      self->SetQuartile3( (double) dataPtr[(num_stat_element*3)/4]);
      self->SetQuintile1( (double) dataPtr[(num_stat_element*1)/5]);
      self->SetQuintile2( (double) dataPtr[(num_stat_element*2)/5]);
      self->SetQuintile3( (double) dataPtr[(num_stat_element*3)/5]);
      self->SetQuintile4( (double) dataPtr[(num_stat_element*4)/5]);
      self->SetAverage  ( total/num_stat_element);
      self->SetStdev    ( sqrt((totalsq - (total*total/num_stat_element))/(num_stat_element-1.0)));
      self->SetMax      ( max ); 
      self->SetMin      ( min ); 
    } else {
      self->SetQuartile1(0);
      self->SetMedian   (0);
      self->SetQuartile3(0);
      self->SetQuintile1(0);
      self->SetQuintile2(0);
      self->SetQuintile3(0);
      self->SetQuintile4(0);
      self->SetAverage  (0);
      self->SetStdev    (0); 
      self->SetMax      (0);
      self->SetMin      (0);
    }

//  cout  << "NumExaminedElements: " <<self->GetNumExaminedElements()<< "\n";
//  cout  << "Quartile1: " <<self->GetQuartile1() << "\n"; 
//  cout  << "Median: "    <<self->GetMedian()    << "\n"; 
//  cout  << "Quartile3: " <<self->GetQuartile3() << "\n";
//  cout  << "Quintile1: " <<self->GetQuintile1() << "\n";
//  cout  << "Quintile2: " <<self->GetQuintile2() << "\n";
//  cout  << "Quintile3: " <<self->GetQuintile3() << "\n";
//  cout  << "Quintile4: " <<self->GetQuintile4() << "\n";
//  cout  << "Average: "   <<self->GetAverage()   << "\n";
//  cout  << "Stdev: "     <<self->GetStdev()     << "\n";
//  cout  << "Max: "       <<self->GetMax()     << "\n";
//  cout  << "Min: "       <<self->GetMin()     << "\n";
//  cout  << "Igore Zero? "<<self->GetIgnoreZero()   << "\n";
}

    

//----------------------------------------------------------------------------
// This method is passed a input and output Data, and executes the filter
// algorithm to fill the output from the input.
// It just executes a switch statement to call the correct function for
// the Datas data types.
void vtkImageStatistics::ExecuteData(vtkDataObject *)
{
  void *inPtr;
  int *outPtr;
  int outExt[6];

  vtkImageData *inData = this->GetInput(); 
  vtkImageData *outData = this->GetOutput();
  outData->GetWholeExtent(outExt);
  outData->SetExtent(outExt);
  outData->AllocateScalars();

  inPtr  = inData->GetScalarPointer();
  outPtr = (int *)outData->GetScalarPointer();

  // this filter expects that output is type int.
  if (outData->GetScalarType() != inData->GetScalarType())
    {
    vtkErrorMacro(<< "Execute: out ScalarType " << outData->GetScalarType()
          << "must be the same as in ScalarType" 
                  << inData->GetScalarType() 
                  <<"\n");
    return;
  }

  if (inData->GetNumberOfScalarComponents() != 1)
  {
    vtkErrorMacro(<< "Execute: Number of scalar components " 
                  << outData->GetScalarType()
          << " is not 1\n");
    return;
  }

  switch (inData->GetScalarType())
    {
    vtkTemplateMacro5(vtkImageStatisticsExecute, this, inData, 
                                                (VTK_TT *)(inPtr), outData, 
                                                (VTK_TT *)(outPtr));
    default:
      vtkErrorMacro(<< "Execute: Unknown ScalarType");
      return;
    }
}

//------------------------------------------------------------------------------
void vtkImageStatistics::PrintSelf(ostream& os, vtkIndent indent)
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Igore Zero? "<<this->IgnoreZero     << "\n";
  os << indent << "NumExaminedElements: " <<this->GetNumExaminedElements()<< "\n";
  os << indent << "Quartile1: " <<this->GetQuartile1() << "\n"; 
  os << indent << "Median: "    <<this->GetMedian()    << "\n"; 
  os << indent << "Quartile3: " <<this->GetQuartile3() << "\n";
  os << indent << "Quintile1: " <<this->GetQuintile1() << "\n";
  os << indent << "Quintile2: " <<this->GetQuintile2() << "\n";
  os << indent << "Quintile3: " <<this->GetQuintile3() << "\n";
  os << indent << "Quintile4: " <<this->GetQuintile4() << "\n";
  os << indent << "Average: "   <<this->GetAverage()   << "\n";
  os << indent << "Stdev: "     <<this->GetStdev()     << "\n";
  os << indent << "Max: "       <<this->GetMax()     << "\n";
  os << indent << "Min: "       <<this->GetMin()     << "\n";
}

