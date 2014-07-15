/*========================================================
 *
 *========================================================*/

#ifndef __cipExceptionObject_h
#define __cipExceptionObject_h

#include <iostream>
#include <stdexcept>
#include <string>

namespace cip
{
class ExceptionObject : public std::exception
  {
  public:
    ExceptionObject();
    ExceptionObject( const char* file, unsigned int line, std::string method, std::string msg );

    ~ExceptionObject() throw();

    /** Print out the exception information */
    void Print(std::ostream & os) const;

    /** Provide std::exception::what() implementation. */
    virtual const char* what() const throw()
    {
      return "CIP Exception";
    }

  private:
    std::string  File;
    std::string  Method;
    std::string  Message;
    unsigned int Line;
  };

inline std::ostream & operator<<(std::ostream & os, ExceptionObject & e)
{
  ( &e )->Print(os);
  return os;
}

} // end namespace cip

#endif // __cipExceptionObject_h
