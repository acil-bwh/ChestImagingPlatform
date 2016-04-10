#include "ShapeModelImageFactory.h"
#include "ShapeModelImageShort.h"
#include "ShapeModelImageFloat.h"
#include <string>

ShapeModelImage*
ShapeModelImageFactory::create( const std::string& imageType )
{
  if (imageType == "CT")
  {
    return new ShapeModelImageShort();
  }
  else if (imageType == "DT")
  {
    return new ShapeModelImageFloat();
  }
  else
  {
    throw std::runtime_error( "Unsupported input image type: " + imageType );
  }
}