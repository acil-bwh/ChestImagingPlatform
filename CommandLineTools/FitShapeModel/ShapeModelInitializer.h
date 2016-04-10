#ifndef _ShapeModelInitializer_h_
#define _ShapeModelInitializer_h_

#include "FitShapeModelTypes.h"
#include "ShapeModelObject.h"

class ShapeModel;
class ShapeModelImage;

// very simple initializer (mostly depends on hard-coding and manual input)
class ShapeModelInitializer : public ShapeModelObject
{
public:
  ShapeModelInitializer( ShapeModel& shapeModel,
                         ShapeModelImage& image );
  ~ShapeModelInitializer();
  void run( double offsetX = 0,
            double offsetY = 0,
            double offsetZ = 0 );
  void run( const std::string& transformFileName );

private:
  ShapeModel& _shapeModel;
  ShapeModelImage& _image;
};

#endif
