#include "cipExceptionObject.h"

namespace cip
{

ExceptionObject::ExceptionObject( const char* file, unsigned int line, std::string method, std::string msg )
{
  std::string tmp(file);

  this->File    = tmp;
  this->Line    = line;
  this->Method  = method;
  this->Message = msg;
}

ExceptionObject::~ExceptionObject() throw()
{
}

void ExceptionObject::Print( std::ostream & os ) const
{
  os << std::endl << "CIP Exception:" << std::endl;
  os << "Message:\t" << this->Message << std::endl;
  os << "In file:\t" << this->File << std::endl;
  os << "On line:\t" << this->Line << std::endl;
  os << "In method:\t" << this->Method << std::endl;
}

} // end namespace cip
