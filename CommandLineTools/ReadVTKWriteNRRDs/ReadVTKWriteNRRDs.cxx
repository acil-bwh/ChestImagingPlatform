/** \file
 *  \ingroup commandLineTools
 *  \details This prgram reads a VTK polydata file containing particles 
 *  data and writes a corresponding collection of NRRD files containing 
 *  the input file's array data.
 */

#include "vtkSmartPointer.h"
#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "cipChestConventions.h"
#include "ReadVTKWriteNRRDsCLP.h"

namespace
{
  typedef itk::Image< float, 2 >                 NRRDImageType;
  typedef itk::ImageFileWriter< NRRDImageType >  WriterType;
}

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  std::cout << "Reading poly data..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > reader = vtkSmartPointer< vtkPolyDataReader >::New();
    reader->SetFileName( inFileName.c_str() );
    reader->Update();
  
  unsigned int numPoints = reader->GetOutput()->GetNumberOfPoints();

  NRRDImageType::IndexType index;
  float value;
  for ( unsigned int a=0; a<reader->GetOutput()->GetPointData()->GetNumberOfArrays(); a++ )
    {
      std::string arrayName = reader->GetOutput()->GetPointData()->GetArrayName(a);

      std::string fileName = prefix;
      if ( arrayName.compare("scale") == 0 )
	{
        fileName.append("pass");
	}
      else
	{
        fileName.append(arrayName);
	}
	fileName.append(".nrrd");
	  
      unsigned int numComponents; 
      if ( arrayName.compare("scale") == 0 )
	{	  
	  numComponents = 4;
	}
      else
	{
	  numComponents =
	    reader->GetOutput()->GetPointData()->GetArray(a)->GetNumberOfComponents();
	}

      NRRDImageType::SizeType arraySize;
        arraySize[1] = numPoints;
	arraySize[0] = numComponents;

      NRRDImageType::Pointer outArray = NRRDImageType::New();
        outArray->SetRegions( arraySize );
	outArray->Allocate();
	outArray->FillBuffer( 0 );
	
      for ( unsigned int p=0; p<numPoints; p++ )
	{
	  for ( unsigned int c=0; c<numComponents; c++ )
	    {
	      if ( arrayName.compare("scale") == 0 && c < 3 )
		{
		  value = reader->GetOutput()->GetPoint(p)[c];	
		}
	      else if ( arrayName.compare("scale") == 0 && c == 3 ) 
		{
		  value = reader->GetOutput()->GetPointData()->GetArray(a)->GetTuple(p)[0];
		}
	      else
		{
		  value = reader->GetOutput()->GetPointData()->GetArray(a)->GetTuple(p)[c];
		}

	      index[0] = c;
	      index[1] = p;
	      outArray->SetPixel( index, value );
	    }
	}
      
      std::cout << "Writing NRRD..." << std::endl;
      WriterType::Pointer writer = WriterType::New();
        writer->SetFileName( fileName );
	writer->SetInput( outArray );
      try
	{
	writer->Update();
	}
      catch ( itk::ExceptionObject &excp )
	{
	std::cerr << "Exception caught writing NRRD:";
	std::cerr << excp << std::endl;
	  
	return cip::NRRDWRITEFAILURE;
	}
    }

    std::cout << "DONE." << std::endl;
    
    return cip::EXITSUCCESS;
}

