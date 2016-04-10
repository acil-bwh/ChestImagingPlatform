#include "ShapeModelUtils.h"
#include <sstream>

std::string int2str( int i )
{
  std::stringstream sstr;
  sstr << i;
  return sstr.str();
}

bool iequals( const std::string& a, const std::string& b )
{
  unsigned int sz = a.size();
  if (b.size() != sz)
  {
    return false;
  }
  for (unsigned int i = 0; i < sz; ++i)
  {
    if (tolower(a[i]) != tolower(b[i]))
    {
      return false;
    }
  }
  return true;
}
