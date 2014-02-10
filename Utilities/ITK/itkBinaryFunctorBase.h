/*=========================================================================
*
* Copyright Marius Staring, Stefan Klein, David Doria. 2011.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0.txt
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
*=========================================================================*/
#ifndef __itkBinaryFunctorBase_h
#define __itkBinaryFunctorBase_h

#include "itkObject.h"

namespace itk
{
/** \class BinaryFunctorBase
* \brief Base functor
*
* \sa BinaryFunctorImageFilter
* \ingroup IntensityImageFilters
* \authors Changyan Xiao, Marius Staring, Denis Shamonin,
* Johan H.C. Reiber, Jan Stolk, Berend C. Stoel
*/

namespace Functor
{

template<class TInput1, class TInput2, class TOutput>
class BinaryFunctorBase : public Object
{
public:
  /** Standard class typedefs. */
  typedef BinaryFunctorBase          Self;
  typedef Object                     Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** New macro for creation of through a smart pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( BinaryFunctorBase, Object );

  /** Typedefs. */
  typedef TInput1 Input1Type;
  typedef TInput2 Input2Type;
  typedef TOutput OutputType;

  /** This does the real computation */
  virtual TOutput Evaluate( const TInput1 & value1, const TInput2 & value2 ) const
  {
    return NumericTraits<TOutput>::Zero;
  }

protected:
  BinaryFunctorBase(){};
  virtual ~BinaryFunctorBase(){};

private:
  BinaryFunctorBase(const Self &); // purposely not implemented
  void operator=(const Self &);    // purposely not implemented

}; // end class BinaryFunctorBase

} // end namespace itk::Functor
} // end namespace itk

#endif // end #ifndef __itkBinaryFunctorBase_h
