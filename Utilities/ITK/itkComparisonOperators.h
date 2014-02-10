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
#ifndef __itkComparisonOperators_h
#define __itkComparisonOperators_h

#include <algorithm>
#include "vnl/vnl_math.h"


namespace itk
{
namespace Functor
{
// Define functors used to sort

/** \class AbsLessEqualCompare
 * \brief Returns ( abs(a) <= abs(b) )
 *
 * |e1|<=|e2|<=...<=|eN|
 *
 * \ingroup ITKReview
 */
template< class T >
class AbsLessEqualCompare
{
public:
  bool operator()( T a, T b )
  {
    return vnl_math_abs( a ) <= vnl_math_abs( b );
  }
};

/** \class AbsLessCompare
 * \brief Returns ( abs(a) < abs(b) )
 *
 * |e1|<|e2|<...<|eN|
 *
 * \ingroup ITKReview
 */
template< class T >
class AbsLessCompare
{
public:
  bool operator()( T a, T b )
  {
    return vnl_math_abs( a ) < vnl_math_abs( b );
  }
};

} // end namespace itk::Functor
} // end namespace itk

#endif // end #ifndef __itkComparisonOperators_h
