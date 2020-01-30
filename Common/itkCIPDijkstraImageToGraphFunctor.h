/**
 *  \class itkCIPDijkstraGraphTraits.h
 *  \ingroup common
 *  \brief Class which defines node/edge weighting in constructing a
 *  graph from an image. 
 *
 *  \par
 *  Using the ImageToGraphFilter class, one wishes to construct a
 *  graph from a given input image.  This functor determines whether
 *  or not a given pixel constitutes a node, what weight that node
 *  should have, and what edge weight should be assigned an edge
 *  between two nodes. 
 *
 *  The weight assigned to a node is an exponential or sigmoid function
 *  of the image index gray level.  The user has control over two parameters
 *  in the exponential: the coefficient and the time constant.  Note
 *  that dark paths can be made optimal by choosing a positive time
 *  constant, and bright paths can be made optimal by choosing a
 *  negative time constant.  The onus is on the user to select proper
 *  values for these parameters, but as an example, a coefficient of 50
 *  and a time constant of -700 would produce a reasonable weighting
 *  function for a grayscale range [-1024, 1024] (these parameters
 *  would yield bright path preferences). If the user chooses to use a
 *  sigmoid function to assign costs, there are three parameters that
 *  can be chosen: SigmoidShift (controls the domain point
 *  corresponding to the 50% point of the function), SigmoidSteepness
 *  (controls how steep the sigmoid is -- larger values result in
 *  steeper functions), and SigmoidScale (controls the scale of the
 *  sigmoid function).  By default, these parameters are, respectively,
 *  -800, 0.05, and 5.0.  These parameters are reasonable for finding
 *  dark pathways (e.g. airways). Note that by making the steepness
 *  value negative, bright paths can be favored instead of dark paths.
 *
 *  $Date: 2012-04-24 17:06:09 -0700 (Tue, 24 Apr 2012) $
 *  $Revision: 93 $
 *  $Author: jross $
 *
 */
#ifndef __itkCIPDijkstraImageToGraphFunctor_h
#define __itkCIPDijkstraImageToGraphFunctor_h

#include "itkDefaultImageToGraphFunctor.h"

namespace itk
{

template<typename TInputImage, typename TOutputGraph>
class CIPDijkstraImageToGraphFunctor
: public DefaultImageToGraphFunctor<TInputImage, TOutputGraph>
{
public:
  /** Standard class typedefs. */
  typedef CIPDijkstraImageToGraphFunctor                            Self;
  typedef DefaultImageToGraphFunctor<TInputImage, TOutputGraph>  Superclass;
  typedef SmartPointer<Self>                                     Pointer;
  typedef SmartPointer<const Self>                               ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  typedef TInputImage                               InputImageType;
  typedef TOutputGraph                              OutputGraphType;

  typedef typename Superclass::IndexType            IndexType;
  typedef typename Superclass::PixelType            PixelType;
  typedef typename Superclass::NodeType             NodeType;
  typedef typename Superclass::EdgeType             EdgeType;
  typedef typename Superclass::NodeIteratorType     NodeIteratorType;
  typedef typename Superclass::EdgeIteratorType     EdgeIteratorType;
  typedef typename Superclass::NodePointerType      NodePointerType;
  typedef typename Superclass::EdgePointerType      EdgePointerType;
  typedef typename Superclass::NodeWeightType       NodeWeightType;
  typedef typename Superclass::EdgeWeightType       EdgeWeightType;
  typedef typename Superclass::NodeImageType        NodeImageType;
  typedef typename Superclass::EdgeIdentifierContainerType
                                                    EdgeIdentifierContainerType;

  typedef double                 RealType;
  typedef std::vector<IndexType> IndexContainerType;

  /** define virtual functions */
  virtual EdgeWeightType GetEdgeWeight( IndexType, IndexType ) override;
  virtual NodeWeightType GetNodeWeight( IndexType idx ) override;
  virtual bool IsPixelANode( IndexType idx ) override;
  virtual void NormalizeGraph( NodeImageType *, OutputGraphType * ) override;

  bool IsAnEdge( IndexType, IndexType );

  itkSetMacro( LowerThreshold, PixelType );
  itkGetMacro( LowerThreshold, PixelType );

  itkSetMacro( UpperThreshold, PixelType );
  itkGetMacro( UpperThreshold, PixelType );

  itkSetMacro( StartIndex, IndexType );
  itkGetMacro( StartIndex, IndexType );

  itkSetMacro( EndIndex, IndexType );
  itkGetMacro( EndIndex, IndexType );

  itkSetMacro( ExponentialCoefficient, double );
  itkGetMacro( ExponentialCoefficient, double );

  itkSetMacro( ExponentialTimeConstant, double );
  itkGetMacro( ExponentialTimeConstant, double );

  itkSetMacro( SigmoidScale, double );
  itkGetMacro( SigmoidScale, double );

  itkSetMacro( SigmoidShift, double );
  itkGetMacro( SigmoidShift, double );

  itkSetMacro( SigmoidSteepness, double );
  itkGetMacro( SigmoidSteepness, double );

  inline void SetExponentialBasedCostAssignment( bool use )
    {
      m_ExponentialBasedCostAssignment =  use;
      m_SigmoidBasedCostAssignment     = !use;
      m_LinearBasedCostAssignment      = !use;
    }
  inline bool GetExponentialBasedCostAssignment()
    {
      return m_ExponentialBasedCostAssignment;
    }

  inline void SetSigmoidBasedCostAssignment( bool use )
    {
      m_SigmoidBasedCostAssignment     =  use;
      m_ExponentialBasedCostAssignment = !use;
      m_LinearBasedCostAssignment      = !use;
    }
  inline bool GetSigmoidBasedCostAssignment()
    {
      return m_SigmoidBasedCostAssignment;
    }

  inline void SetLinearBasedCostAssignment( bool use )
    {
      m_LinearBasedCostAssignment      =  use;
      m_SigmoidBasedCostAssignment     = !use;
      m_ExponentialBasedCostAssignment = !use;
    }
  inline bool GetLinearBasedCostAssignment()
    {
      return m_LinearBasedCostAssignment;
    }


protected:
  CIPDijkstraImageToGraphFunctor();
  ~CIPDijkstraImageToGraphFunctor() {}
  void PrintSelf( std::ostream& os, Indent indent ) const override;

private:
  CIPDijkstraImageToGraphFunctor( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented

  PixelType m_LowerThreshold;
  PixelType m_UpperThreshold;
  IndexType m_StartIndex;
  IndexType m_EndIndex;
  double    m_ExponentialCoefficient;
  double    m_ExponentialTimeConstant;
  double    m_SigmoidScale;
  double    m_SigmoidShift;
  double    m_SigmoidSteepness;
  bool      m_SigmoidBasedCostAssignment;
  bool      m_ExponentialBasedCostAssignment;
  bool      m_LinearBasedCostAssignment;
};


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCIPDijkstraImageToGraphFunctor.txx"
#endif

#endif
