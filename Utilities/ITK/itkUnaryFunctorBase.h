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
#ifndef __itkUnaryFunctorBase_h
#define __itkUnaryFunctorBase_h

#include "itkObject.h"

namespace itk
{
/** \class UnaryFunctorBase
* \brief Base functor
*
* \sa UnaryFunctorImageFilter
* \ingroup IntensityImageFilters
* \authors Changyan Xiao, Marius Staring, Denis Shamonin,
* Johan H.C. Reiber, Jan Stolk, Berend C. Stoel
*/

namespace Functor
{

template<class TInput, class TOutput>
class UnaryFunctorBase : public Object
{
public:
  /** Standard class typedefs. */
  typedef UnaryFunctorBase           Self;
  typedef Object                     Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** New macro for creation of through a smart pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( UnaryFunctorBase, Object );

  /** This does the real computation */
  virtual TOutput Evaluate( const TInput & value ) const
  {
    return NumericTraits<TOutput>::Zero;
  }

protected:
  UnaryFunctorBase(){};
  virtual ~UnaryFunctorBase(){};

private:
  UnaryFunctorBase(const Self &); // purposely not implemented
  void operator=(const Self &);   // purposely not implemented

}; // end class UnaryFunctorBase

} // end namespace itk::Functor
} // end namespace itk

#endif // end #ifndef __itkUnaryFunctorBase_h
