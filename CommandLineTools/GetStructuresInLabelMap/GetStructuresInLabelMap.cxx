/** \file
 *  \ingroup commandLineTools 
 *  \details This 
 *
 *  $Date$
 *  $Revision$
 *  $Author$
 *
 */


#include "cipChestConventions.h"
#include "cipHelper.h"
#include "GetStructuresInLabelMapCLP.h"

typedef itk::ImageRegionIterator< cip::LabelMapType > IteratorType;

int main( int argc, char *argv[] )
{

  PARSE_ARGS;
    
  cip::ChestConventions conventions;

  std::cout << "Reading label map..." << std::endl;
  cip::LabelMapReaderType::Pointer reader = cip::LabelMapReaderType::New();
    reader->SetFileName(labelMapFileName.c_str());
  try
    {
    reader->Update();
    }
  catch (itk::ExceptionObject &excp)
    {
    std::cerr << "Exception caught reading label mape:";
    std::cerr << excp << std::endl;

    return cip::LABELMAPREADFAILURE;
    }

  std::list< unsigned short > labelsList;

  IteratorType it(reader->GetOutput(), reader->GetOutput()->GetBufferedRegion());

  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
    if ( it.Get() != 0 )
      {
      labelsList.push_back( it.Get() );
      }

    ++it;
    }

  labelsList.unique();
  labelsList.sort();
  labelsList.unique();

  std::list< unsigned short >::iterator listIt = labelsList.begin();

  while ( listIt != labelsList.end() )
    {
    std::cout << conventions.GetChestRegionNameFromValue(*listIt) << "\t";
    std::cout << conventions.GetChestTypeNameFromValue(*listIt) << std::endl;

    ++listIt;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

