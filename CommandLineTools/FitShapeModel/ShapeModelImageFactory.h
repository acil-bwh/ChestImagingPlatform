#ifndef _ShapeModelImageFactory_h_
#define _ShapeModelImageFactory_h_

#include <string>

class ShapeModelImage;

class ShapeModelImageFactory
{
public:
  static ShapeModelImage* create( const std::string& imageType );  
};

#endif // _ShapeModelImageFactory_h_
