/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkMSTGraphAlphaMutualInformationImageToImageMetric.h,v $
  Language:  C++
  Date:      $Date: 2012/07/17 11:23:34 $
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkMSTGraphAlphaMutualInformationImageToImageMetric_h
#define __itkMSTGraphAlphaMutualInformationImageToImageMetric_h

/** Includes for the Superclass. */
#include "itkMultiInputImageToImageMetricBase.h"

/** Includes for the graph algorithm classes. */
#include "itkGraph.h"
#include "itkMinimumSpanningTreeGraphTraits.h"
#include "itkFastPrimMinimumSpanningTreeGraphFilter.h"
#include "itkMultiInputImageRandomCoordinateMoranSampler.h"

#include "itkArray.h"
#include "itkListSampleCArray.h"

/** Include for the spatial derivatives. */
#include "itkArray2D.h"

#include <iostream>

#define MIN_DISTANCE 1e-10

namespace itk
{
/**
 * \class MSTGraphAlphaMutualInformationImageToImageMetric
 *
 * \brief Computes similarity between two images to be registered.
 *
 * This metric computes the alpha-Mutual Information (aMI) between
 * two multi-channeled data sets. Said otherwise, given two sets of
 * features, the aMI between them is calculated.
 * Since for higher dimensional aMI it is infeasible to compute high
 * dimensional joint histograms, here we adopt a framework based on
 * the length of certain graphs, see Neemuchwala. Specifically, we use
 * the Minimum Spanning Tree (MST) graph, which can be implemented 
 * by the Kruskal's algorithm and the Prim's algorithm.
 *
 * Note that the feature image are given beforehand, and that values
 * are calculated by interpolation on the transformed point. For some
 * features, it would be better (but slower) to first apply the transform
 * on the image and then recalculate the feature.
 *
 * \ingroup RegistrationMetrics
 */

template <class TFixedImage, class TMovingImage>
class MSTGraphAlphaMutualInformationImageToImageMetric :
  public MultiInputImageToImageMetricBase<TFixedImage, TMovingImage>
{
public:
  /** Standard itk. */
  typedef MSTGraphAlphaMutualInformationImageToImageMetric  Self;
  typedef MultiInputImageToImageMetricBase<
    TFixedImage, TMovingImage>                              Superclass;
  typedef SmartPointer<Self>                                Pointer;
  typedef SmartPointer<const Self>                          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( MSTGraphAlphaMutualInformationImageToImageMetric,
    MultiInputImageToImageMetricBase );

  /** Typedefs from the superclass. */
  typedef typename
    Superclass::CoordinateRepresentationType              CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType            MovingImageType;
  typedef typename Superclass::MovingImagePixelType       MovingImagePixelType;
  typedef typename Superclass::MovingImageConstPointer    MovingImageConstPointer;
  typedef typename Superclass::FixedImageType             FixedImageType;
  typedef typename Superclass::FixedImageConstPointer     FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType       FixedImageRegionType;
  typedef typename Superclass::TransformType              TransformType;
  typedef typename Superclass::TransformPointer           TransformPointer;
  typedef typename Superclass::InputPointType             InputPointType;
  typedef typename Superclass::OutputPointType            OutputPointType;
  typedef typename Superclass::TransformParametersType    TransformParametersType;
  typedef typename Superclass::TransformJacobianType      TransformJacobianType;
  typedef typename Superclass::InterpolatorType           InterpolatorType;
  typedef typename Superclass::InterpolatorPointer        InterpolatorPointer;
  typedef typename Superclass::RealType                   RealType;
  typedef typename Superclass::GradientPixelType          GradientPixelType;
  typedef typename Superclass::GradientImageType          GradientImageType;
  typedef typename Superclass::GradientImagePointer       GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType    GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType         FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer      FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType        MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer     MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                MeasureType;
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename Superclass::DerivativeValueType        DerivativeValueType;
  typedef typename Superclass::ParametersType             ParametersType;
  typedef typename Superclass::FixedImagePixelType        FixedImagePixelType;
  typedef typename Superclass::MovingImageRegionType      MovingImageRegionType;
  typedef typename Superclass::ImageSamplerType           ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer        ImageSamplerPointer;
  typedef typename Superclass::ImageSampleContainerType   ImageSampleContainerType;
  typedef typename
    Superclass::ImageSampleContainerPointer               ImageSampleContainerPointer;
  typedef typename Superclass::FixedImageLimiterType      FixedImageLimiterType;
  typedef typename Superclass::MovingImageLimiterType     MovingImageLimiterType;
  typedef typename
    Superclass::FixedImageLimiterOutputType               FixedImageLimiterOutputType;
  typedef typename
    Superclass::MovingImageLimiterOutputType              MovingImageLimiterOutputType;
  typedef typename Superclass::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;

  /** Typedefs for storing multiple inputs. */
  typedef typename Superclass::FixedImageVectorType       FixedImageVectorType;
  typedef typename Superclass::FixedImageMaskVectorType   FixedImageMaskVectorType;
  typedef typename Superclass::FixedImageRegionVectorType FixedImageRegionVectorType;
  typedef typename Superclass::MovingImageVectorType      MovingImageVectorType;
  typedef typename Superclass::MovingImageMaskVectorType  MovingImageMaskVectorType;
  typedef typename Superclass::InterpolatorVectorType     InterpolatorVectorType;
  typedef typename 
    Superclass::FixedImageInterpolatorVectorType          FixedImageInterpolatorVectorType;

  /** The fixed image dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );
  itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );

  /** Typedefs for the features of sample points. */
  typedef Array< double >                                                      MeasurementVectorType;
  typedef typename MeasurementVectorType::ValueType                            MeasurementVectorValueType;
  typedef typename Statistics::ListSampleCArray< 
                         MeasurementVectorType, double >                       ListSampleType;
  typedef typename ListSampleType::Pointer                                     ListSamplePointer;

  /** Typedefs for the minimum spanning tree. */
  typedef itk::MinimumSpanningTreeGraphTraits<
                         double, FixedImageType::ImageDimension>               GraphTraitsType;
  typedef itk::Graph<GraphTraitsType>                                          GraphType;
  typedef itk::FastPrimMinimumSpanningTreeGraphFilter<GraphType>               SampleToMinimumSpanningTreeFilterType;
  typedef typename GraphType::Pointer                                          GraphPointer;
  typedef typename GraphType::NodeIdentifierType                               NodeIdentifierType;
  typedef typename GraphType::EdgeWeightType                                   EdgeWeightType;
  typedef typename GraphType::EdgePointerType                                  EdgePointerType;
  typedef typename GraphType::NodeIterator                                     NodeIteratorType;
  typedef typename GraphType::EdgeIterator                                     EdgeIteratorType;

  /** Typedefs for multi-threading. */
  typedef typename Superclass::ThreaderType                                    MSTThreaderType;
  typedef typename Superclass::ThreadInfoType                                  MSTThreadInfoType;
  typedef std::vector<NodeIdentifierType>                                      NodeIdentifierContainerType;
  typedef std::vector<EdgeWeightType>                                          EdgeWeightContainerType;
  
  /** Typedefs for the computation of the derivative. */
  typedef typename Superclass::FixedImagePointType                             FixedImagePointType;
  typedef typename Superclass::MovingImagePointType                            MovingImagePointType;
  typedef typename Superclass::MovingImageDerivativeType                       MovingImageDerivativeType;
  typedef typename Superclass::MovingImageContinuousIndexType                  MovingImageContinuousIndexType;
  typedef std::vector<TransformJacobianType>                                   TransformJacobianContainerType;
  typedef std::vector<NonZeroJacobianIndicesType>                              TransformJacobianIndicesContainerType;
  typedef Array2D<double>                                                      SpatialDerivativeType;
  typedef std::vector<SpatialDerivativeType>                                   SpatialDerivativeContainerType;

  /**
   * *** Standard metric stuff: ***
   */

  /** Initialize the metric. */
  virtual void Initialize( void ) throw ( ExceptionObject );

  /** Get the value for single valued optimizers. */
  MeasureType GetValue( const TransformParametersType & parameters ) const;

  /** Get the derivatives of the match measure. */
  void GetDerivative( const TransformParametersType & parameters,
    DerivativeType & derivative ) const;

  /** Get value and derivatives for multiple valued optimizers. */
  void GetValueAndDerivativeSingleThreaded( const TransformParametersType & parameters,
	MeasureType & value, DerivativeType & derivative ) const;

  void GetValueAndDerivative( const TransformParametersType & parameters,
    MeasureType & value, DerivativeType & derivative ) const;

  /** Set alpha from alpha - mutual information. */
  itkSetClampMacro( Alpha, double, 0.0, 1.0 );

  /** Get alpha from alpha - mutual information. */
  itkGetConstReferenceMacro( Alpha, double );

  /** Set the bucket size. */
  itkSetMacro( BucketSize, unsigned int );

  /** Set the splitting rule. */
  itkSetMacro( SplittingRule, std::string );

  /** Set the k nearest neighbours. */
  itkSetMacro( KNearestNeighbours, unsigned int );

  /** Set the error bound. */
  itkSetMacro( ErrorBound, double );

  /** Set the use of penalty weight. */
  itkSetMacro( UsePenaltyWeight, bool );

  /** Set the number of penalty images. */
  itkSetMacro( NumberOfPenaltyImages, unsigned int );

  /** Set the penalty images. */
  itkSetConstObjectMacro( PenaltyImages, std::vector<FixedImageConstPointer> );

protected:
  /** Constructor. */
  MSTGraphAlphaMutualInformationImageToImageMetric();

  /** Destructor. */
  virtual ~MSTGraphAlphaMutualInformationImageToImageMetric();

  /** PrintSelf. */
  virtual void PrintSelf( std::ostream& os, Indent indent ) const;

  /** Member variables. */
  double                      m_Alpha;
  unsigned int                m_BucketSize;
  std::string                 m_SplittingRule;
  unsigned int                m_KNearestNeighbours;
  double                      m_ErrorBound;
  bool                        m_UsePenaltyWeight;
  unsigned int                m_NumberOfPenaltyImages;
  const std::vector<
	FixedImageConstPointer> * m_PenaltyImages;

  /** Helper structs that multi-threads the computation of
  * the metric derivative using ITK threads.
  */
  struct MSTMultiThreaderParameterType
  {
	// To give the threads access to all members.
	MSTGraphAlphaMutualInformationImageToImageMetric * st_Metric;

	// Used for some parameters of accumulating derivatives
	ListSamplePointer st_FixedListSample;
	ListSamplePointer st_MovingListSample;
	ListSamplePointer st_JointListSample;
	ListSamplePointer st_PenaltyListSample;
	TransformJacobianContainerType st_JacobianContainer;
	TransformJacobianIndicesContainerType st_JacobianIndicesContainer;
	SpatialDerivativeContainerType st_SpatialDerivativesContainer;

	NodeIdentifierContainerType st_MovingSRC;
	NodeIdentifierContainerType st_MovingTAR;
	EdgeWeightContainerType st_MovingEW;
	NodeIdentifierContainerType st_JointSRC;
	NodeIdentifierContainerType st_JointTAR;
	EdgeWeightContainerType st_JointEW;
	EdgeWeightContainerType st_JointPEW;

	DerivativeType        st_MSTMDerivative;
	DerivativeType        st_MSTJDerivative;
  };
  mutable MSTMultiThreaderParameterType m_MSTThreaderMetricParameters;

  /** Most metrics will perform multi-threading by letting
  * each thread compute a part of the value and derivative.
  *
  * These parameters are initialized at every call of GetValueAndDerivative
  * in the function InitializeThreadingParameters(). Since GetValueAndDerivative
  * is const, also InitializeThreadingParameters should be const, and therefore
  * these member variables are mutable.
  */

  // test per thread struct with padding and alignment
  struct MSTGetValueAndDerivativePerThreadStruct
  {
    unsigned long st_numberOfPixelsCounted;

	ListSamplePointer st_fixedListSample;
	ListSamplePointer st_movingListSample;
	ListSamplePointer st_jointListSample;
	ListSamplePointer st_penaltyListSample;
	TransformJacobianContainerType st_jacobianContainer;
	TransformJacobianIndicesContainerType st_jacobianIndicesContainer;
	SpatialDerivativeContainerType st_spatialDerivativesContainer;
		
	MeasureType           st_MValue;
	MeasureType           st_JValue;
	DerivativeType        st_MDerivative;
	DerivativeType        st_JDerivative;
  };
  itkPadStruct( ITK_CACHE_LINE_ALIGNMENT, MSTGetValueAndDerivativePerThreadStruct,
	  PaddedMSTGetValueAndDerivativePerThreadStruct );
  itkAlignedTypedef( ITK_CACHE_LINE_ALIGNMENT, PaddedMSTGetValueAndDerivativePerThreadStruct,
	  AlignedMSTGetValueAndDerivativePerThreadStruct );
  mutable AlignedMSTGetValueAndDerivativePerThreadStruct * m_MSTGetValueAndDerivativePerThreadVariables;
  mutable ThreadIdType                                     m_MSTGetValueAndDerivativePerThreadVariablesSize;

  /** Initialize some multi-threading related parameters. */
  void InitializeMSTThreadingParameters( void ) const;

  /** MSTListSamplesAndDerivativePlusJacobian threader callback function. */
  static ITK_THREAD_RETURN_TYPE MSTListSamplesAndDerivativePlusJacobianThreaderCallback( void * arg );

  /** MSTGetValueAndDerivative threader callback function. */
  static ITK_THREAD_RETURN_TYPE MSTGetValueAndDerivativeThreaderCallback( void * arg );

  /** MSTMergeDerivative threader callback function. */
  static ITK_THREAD_RETURN_TYPE MSTMergeDerivativeThreaderCallback( void * arg );

  /** Get listsamples and derivatives plus jacobian for each thread. */
  void MSTThreadedListSamplesAndDerivativePlusJacobian( ThreadIdType threadId );

  /** Get value and derivatives for each thread. */
  void MSTThreadedGetValueAndDerivative( ThreadIdType threadId );

  /** Gather derivatives from all threads. */
  void MSTThreadedMergeDerivative( ThreadIdType threadId );

  /** Gather listsamples and derivatives plus jacobian from all threads. */
  void AfterMSTThreadedListSamplesAndDerivativePlusJacobian( void ) const;

private:
  MSTGraphAlphaMutualInformationImageToImageMetric(const Self&);  //purposely not implemented
  void operator=(const Self&);                                  //purposely not implemented

  /** This function takes the fixed image samples from the ImageSampler
   * and puts them in the imageFeatureFixed, together with the fixed feature
   * image samples. Also the corresponding moving image values and moving
   * feature values are computed and put into imageFeatureMoving. The
   * concatenation is put into imageFeatureJoint.
   * If desired, i.e. if doDerivative is true, then also things needed to
   * compute the derivative of the cost function to the transform parameters
   * are computed:
   * - The sparse Jacobian of the transformation (dT/dmu).
   * - The spatial derivatives of the moving (feature) images (dm/dx).
   */
  virtual void ComputeListSampleValuesAndDerivativePlusJacobian(
    const ListSamplePointer & fixedListSample,
    const ListSamplePointer & movingListSample,
    const ListSamplePointer & jointListSample,
	const ListSamplePointer & penaltyListSample,
    const bool & doDerivative,
    TransformJacobianContainerType & jacobians,
    TransformJacobianIndicesContainerType & jacobiansIndices,
    SpatialDerivativeContainerType & spatialDerivatives ) const;
 
  /** This function calculates the spatial derivative of the
   * featureNr feature image at the point mappedPoint.
   * \todo move this to base class.
   */
  virtual void EvaluateMovingFeatureImageDerivatives(
    const MovingImagePointType & mappedPoint,
    SpatialDerivativeType & featureGradients ) const;

 }; // end class MSTGraphAlphaMutualInformationImageToImageMetric

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMSTGraphAlphaMutualInformationImageToImageMetric.txx"
#endif

#endif // end #ifndef __itkMSTGraphAlphaMutualInformationImageToImageMetric_h

