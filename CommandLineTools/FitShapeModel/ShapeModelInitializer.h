#ifndef _ShapeModelInitializer_h_
#define _ShapeModelInitializer_h_

#include "FitShapeModelTypes.h"

class ShapeModel;

// very simple initializer (mostly depends on hard-coding and manual input)
class ShapeModelInitializer
{
public:
  ShapeModelInitializer( ShapeModel& shapeModel,
                         ImageType::Pointer image );
  ~ShapeModelInitializer();
  void run( double offsetX = 0,
            double offsetY = 0,
            double offsetZ = 0 );

private:
  ShapeModel& _shapeModel;
  ImageType::Pointer _image;
};

#endif
