#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "cipHelper.h"
#include "vtkSmartPointer.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkUnsignedShortArray.h"
#include "vtkUnsignedCharArray.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkFieldData.h"
#include "vtkPolyData.h"
#include "vtkCellArray.h"
#include "vtkVertex.h"
#include "UpdateParticlesDataCLP.h"

int main(int argc, char *argv[])
{
  PARSE_ARGS;

  cip::ChestConventions conventions;

  // Read the poly data
  std::cout << "Reading VTK polydata..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > reader = vtkSmartPointer< vtkPolyDataReader >::New();
    reader->SetFileName(inFileName.c_str());
    reader->Update();
  
  std::cout << "Updating..." << std::endl;
  unsigned int numParticles = reader->GetOutput()->GetNumberOfPoints();
  
  vtkSmartPointer< vtkPolyData > outPolyData = vtkSmartPointer< vtkPolyData >::New();
    outPolyData->SetPoints(reader->GetOutput()->GetPoints());
  
  // As we populate outPolyData, we'll keep track of whether or not these
  // arrays are present.
  bool chestRegionChestTypeArrayPresent = false;
  bool chestRegionArrayPresent = false;
  bool chestTypeArrayPresent = false;
  
  // First transfer all point data arrays from the input to the output
  for (unsigned int i = 0; i<reader->GetOutput()->GetPointData()->GetNumberOfArrays(); i++)
    {
      outPolyData->GetPointData()->AddArray(reader->GetOutput()->GetPointData()->GetArray(i));
      if ((std::string(reader->GetOutput()->GetPointData()->GetArray(i)->GetName())).compare("ChestRegionChestType") == 0)
	{
	  chestRegionChestTypeArrayPresent = true;
	  vtkSmartPointer< vtkUnsignedShortArray > chestRegionChestTypeArray = vtkSmartPointer< vtkUnsignedShortArray >::New();
	    chestRegionChestTypeArray->SetNumberOfComponents(1);
	    chestRegionChestTypeArray->SetName("ChestRegionChestType");
	  
	  unsigned short value;
	  for (unsigned int n = 0; n < numParticles; n++)
	    {
	      value = (unsigned short)(outPolyData->GetPointData()->GetArray("ChestRegionChestType")->GetTuple(n)[0]);
	      chestRegionChestTypeArray->InsertTypedTuple(n, &value);
	    }
	  outPolyData->GetPointData()->RemoveArray(i);
	  outPolyData->GetPointData()->AddArray(chestRegionChestTypeArray);
	}
      if ((std::string(reader->GetOutput()->GetPointData()->GetArray(i)->GetName())).compare("ChestRegion") == 0)
	{  
	  chestRegionArrayPresent = true;
	  vtkSmartPointer< vtkUnsignedShortArray > chestRegionArray = vtkSmartPointer< vtkUnsignedShortArray >::New();
	    chestRegionArray->SetNumberOfComponents(1);
	    chestRegionArray->SetName("ChestRegion");
	    
	  unsigned short value;
	  for (unsigned int n = 0; n < numParticles; n++)
	    {
	      value = (unsigned short)(outPolyData->GetPointData()->GetArray("ChestRegion")->GetTuple(n)[0]);
	      chestRegionArray->InsertTypedTuple(n, &value);
	    }
	  outPolyData->GetPointData()->RemoveArray(i);
	  outPolyData->GetPointData()->AddArray(chestRegionArray);
	}
      if ((std::string(reader->GetOutput()->GetPointData()->GetArray(i)->GetName())).compare("ChestType") == 0)
	{
	  chestTypeArrayPresent = true;
	  vtkSmartPointer< vtkUnsignedShortArray > chestTypeArray = vtkSmartPointer< vtkUnsignedShortArray >::New();
  	    chestTypeArray->SetNumberOfComponents(1);
	    chestTypeArray->SetName("ChestType");
	    
	  unsigned short value;
	  for (unsigned int n = 0; n < numParticles; n++)
	    {
	      value = (unsigned short)(outPolyData->GetPointData()->GetArray("ChestType")->GetTuple(n)[0]);
	      chestTypeArray->InsertTypedTuple(n, &value);
	    }
	  outPolyData->GetPointData()->RemoveArray(i);
	  outPolyData->GetPointData()->AddArray(chestTypeArray);
	}
    }
  
  // Now go through the reader's field data arrays. If we see a field data array
  // with the same number of tuples as there are particles, we'll assume that
  // this array should instead be a point data array. Otherwise, we'll just
  // transfer the field data to the output.
  for (unsigned int i = 0; i<reader->GetOutput()->GetFieldData()->GetNumberOfArrays(); i++)
    {
      if (reader->GetOutput()->GetFieldData()->GetArray(i)->GetNumberOfTuples() ==
	  numParticles)
	{
	  // Make sure this array is not already there
	  bool alreadyPresent = false;
	  for (unsigned int j = 0; j<outPolyData->GetPointData()->GetNumberOfArrays(); j++)
	    {
	      if ((std::string(outPolyData->GetPointData()->GetArray(j)->GetName())).
		  compare(reader->GetOutput()->GetFieldData()->GetArray(i)->GetName()) == 0)
		{
		  alreadyPresent = true;
		  break;
		}
	    }
	  if (!alreadyPresent)
	    {
	      outPolyData->GetPointData()->AddArray(reader->GetOutput()->GetFieldData()->GetArray(i));
	      if ((std::string(reader->GetOutput()->GetFieldData()->GetArray(i)->GetName())).compare("ChestRegionChestType") == 0)
		{
		  chestRegionChestTypeArrayPresent = true;
		}
	      if ((std::string(reader->GetOutput()->GetFieldData()->GetArray(i)->GetName())).compare("ChestRegion") == 0)
		{
		  chestRegionArrayPresent = true;
		}
	      if ((std::string(reader->GetOutput()->GetFieldData()->GetArray(i)->GetName())).compare("ChestType") == 0)
		{
		  chestTypeArrayPresent = true;
		}
	    }
	}
      else
	{
	  outPolyData->GetFieldData()->AddArray(reader->GetOutput()->GetFieldData()->GetArray(i));
	}
    }
  
  // Make sure the ChestRegionChestType array is present and populated.
  if (!chestRegionChestTypeArrayPresent)
    {
      vtkSmartPointer< vtkUnsignedShortArray > chestRegionChestTypeArray = vtkSmartPointer< vtkUnsignedShortArray >::New();
        chestRegionChestTypeArray->SetNumberOfComponents(1);
	chestRegionChestTypeArray->SetName("ChestRegionChestType");
      
      outPolyData->GetPointData()->AddArray(chestRegionChestTypeArray);
      
      unsigned char chestRegion;
      unsigned char chestType;
      for (unsigned int n = 0; n<numParticles; n++)
	{
	  if (chestRegionArrayPresent)
	    {
	      chestRegion = (unsigned char)(outPolyData->GetPointData()->GetArray("ChestRegion")->GetTuple(n)[0]);
	    }
	  else
	    {
	      chestRegion = (unsigned char)(cip::UNDEFINEDREGION);
	    }
	  
	  if (chestTypeArrayPresent)
	    {
	      chestType = (unsigned char)(outPolyData->GetPointData()->GetArray("ChestType")->GetTuple(n)[0]);
	    }
	  else
	    {
	      chestRegion = (unsigned char)(cip::UNDEFINEDREGION);
	    }
	  
	  unsigned short value = (unsigned short)(conventions.GetValueFromChestRegionAndType(chestRegion, chestType));
	  chestRegionChestTypeArray->InsertTypedTuple(n, &value);
	}
      outPolyData->GetPointData()->AddArray(chestRegionChestTypeArray);
    }

  for ( unsigned int n=0; n<numParticles; n++ )
    {
      if ( scale > 0 )
	{
	  outPolyData->GetPointData()->GetArray( "scale" )->SetTuple( n, &scale );
	}
      else if ( outPolyData->GetPointData()->GetArray( "scale" )->GetTuple(n)[0] <= 0)
	{
	  std::cerr << "WARNING: Particle has scale value <= 0" << std::endl;
	  break;
	}
    }

  // If not present, add Vertices to the polydata file
  if ( outPolyData->GetNumberOfVerts() == 0 )
    {
  	vtkSmartPointer< vtkCellArray > cellArray = vtkSmartPointer< vtkCellArray >::New();
   	for ( unsigned int pid = 0; pid < outPolyData->GetNumberOfPoints(); pid++ )
   	   	{
	   		vtkSmartPointer< vtkVertex > Vertex = vtkSmartPointer< vtkVertex >::New();
	   		Vertex->GetPointIds()->SetId(0, pid);
	   		cellArray->InsertNextCell(Vertex);
	   	}
	outPolyData->SetVerts(cellArray);
	}


  // Write the poly data
  std::cout << "Writing VTK polydata..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > writer = vtkSmartPointer< vtkPolyDataWriter >::New();
    writer->SetFileName( outFileName.c_str() );
    writer->SetInputData( outPolyData );
    writer->SetFileTypeToBinary();
    writer->Update();
  
  std::cout << "DONE." << std::endl;
  
  return cip::EXITSUCCESS;
}

#endif
