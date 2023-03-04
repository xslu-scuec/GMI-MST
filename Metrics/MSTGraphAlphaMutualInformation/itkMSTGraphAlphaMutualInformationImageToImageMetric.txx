/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkMSTGraphAlphaMutualInformationImageToImageMetric.txx,v $
  Language:  C++
  Date:      $Date: 2012/07/17 11:23:34 $
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkMSTGraphAlphaMutualInformationImageToImageMetric_txx
#define _itkMSTGraphAlphaMutualInformationImageToImageMetric_txx

#include "itkMSTGraphAlphaMutualInformationImageToImageMetric.h"

namespace itk
{

/**
 * ************************ Constructor *************************
 */

template <class TFixedImage, class TMovingImage>
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::MSTGraphAlphaMutualInformationImageToImageMetric()
{
  this->SetComputeGradient( false ); // don't use the default gradient
  this->SetUseImageSampler( true );

  this->m_Alpha = 0.5;

  this->m_UsePenaltyWeight = false;
  this->m_NumberOfPenaltyImages = 0;
  this->m_PenaltyImages = NULL;

  /** Initialize the m_MSTThreaderMetricParameters. */
  this->m_MSTThreaderMetricParameters.st_Metric = this;

  // Multi-threading structs
  this->m_MSTGetValueAndDerivativePerThreadVariables     = NULL;
  this->m_MSTGetValueAndDerivativePerThreadVariablesSize = 0;

} // end Constructor()

/**
 * ********************* Destructor ****************************
 */

template<class TFixedImage, class TMovingImage>
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::~MSTGraphAlphaMutualInformationImageToImageMetric()
{
  if ( this->m_PenaltyImages != NULL )
    {
	delete []this->m_PenaltyImages;
	this->m_PenaltyImages = NULL;
    }
  delete []this->m_MSTGetValueAndDerivativePerThreadVariables;

} // end Destructor

/**
 * ********************* Initialize *****************************
 */

template <class TFixedImage, class TMovingImage>
void
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::Initialize( void ) throw ( ExceptionObject )
{
  /** Call the superclass. */
  this->Superclass::Initialize();

} // end Initialize()

/**
 * ********************* InitializeMSTThreadingParameters ****************************
 */

template<class TFixedImage, class TMovingImage>
void
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::InitializeMSTThreadingParameters( void ) const
{
  /** Resize and initialize the threading related parameters.
   * The SetSize() functions do not resize the data when this is not
   * needed, which saves valuable re-allocation time.
   *
   * This function is only to be called at the start of each resolution.
   * Re-initialization of the potentially large vectors is performed after
   * each iteration, in the accumulate functions, in a multi-threaded fashion.
   * This has performance benefits for larger vector sizes.
   */

  /** Only resize the array of structs when needed. */
  if ( this->m_MSTGetValueAndDerivativePerThreadVariablesSize != this->m_NumberOfThreads )
    {
    delete []this->m_MSTGetValueAndDerivativePerThreadVariables;
    this->m_MSTGetValueAndDerivativePerThreadVariables     = new AlignedMSTGetValueAndDerivativePerThreadStruct[ this->m_NumberOfThreads ];
    this->m_MSTGetValueAndDerivativePerThreadVariablesSize = this->m_NumberOfThreads;
    }

  /** Some initialization. */
  this->m_MSTThreaderMetricParameters.st_FixedListSample   = ListSampleType::New();
  this->m_MSTThreaderMetricParameters.st_MovingListSample  = ListSampleType::New();
  this->m_MSTThreaderMetricParameters.st_JointListSample   = ListSampleType::New();
  this->m_MSTThreaderMetricParameters.st_PenaltyListSample = ListSampleType::New();
  this->m_MSTThreaderMetricParameters.st_MSTMDerivative.SetSize( this->GetNumberOfParameters() );
  this->m_MSTThreaderMetricParameters.st_MSTMDerivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );
  this->m_MSTThreaderMetricParameters.st_MSTJDerivative.SetSize( this->GetNumberOfParameters() );
  this->m_MSTThreaderMetricParameters.st_MSTJDerivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

  for ( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
    {
    this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_fixedListSample   = ListSampleType::New();
    this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_movingListSample  = ListSampleType::New();
    this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_jointListSample   = ListSampleType::New();
    this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_penaltyListSample = ListSampleType::New();
  
    this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_MValue = NumericTraits< MeasureType >::Zero;
    this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_JValue = NumericTraits< MeasureType >::Zero;
    this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_MDerivative.SetSize( this->GetNumberOfParameters() );
    this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_MDerivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );
    this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_JDerivative.SetSize( this->GetNumberOfParameters() );
    this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_JDerivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );
    }  

} // end InitializeMSTThreadingParameters()

/**
 * ************************ GetValue *************************
 */

template <class TFixedImage, class TMovingImage>
typename MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::GetValue( const TransformParametersType & parameters ) const
{
  /** Initialize some variables. */
  MeasureType measure = NumericTraits< MeasureType >::Zero;
  
  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  /**
   * *************** Create the three list samples ******************
   */

  /** Create list samples. */
  ListSamplePointer fixedListSample  = ListSampleType::New();
  ListSamplePointer movingListSample = ListSampleType::New();
  ListSamplePointer jointListSample  = ListSampleType::New();
  
  ListSamplePointer penaltyListSample = ListSampleType::New();

  /** Compute the three list samples and the derivatives. */
  TransformJacobianContainerType dummyJacobianContainer;
  TransformJacobianIndicesContainerType dummyJacobianIndicesContainer;
  SpatialDerivativeContainerType dummySpatialDerivativesContainer;
  this->ComputeListSampleValuesAndDerivativePlusJacobian(
    fixedListSample, movingListSample, jointListSample, penaltyListSample, false, 
    dummyJacobianContainer, dummyJacobianIndicesContainer, dummySpatialDerivativesContainer );

  /** Check if enough samples were valid. */
  unsigned long numSamplePoints = this->GetImageSampler()->GetOutput()->Size();
  this->CheckNumberOfSamples( numSamplePoints, this->m_NumberOfPixelsCounted );
  
  /** Get the size of the feature vectors. */
  const unsigned int fixedSize  = this->GetNumberOfFixedImages();
  const unsigned int movingSize = this->GetNumberOfMovingImages();
  const unsigned int jointSize  = fixedSize + movingSize;

  /**
   * *************** Generate the spanning graphs ******************
   *
   * and get the minimum spanning trees.
   */

  /** Generate the fixed, moving and joint minimum spanning trees from sample points. */
  typename SampleToMinimumSpanningTreeFilterType::Pointer fixedMSTFilter = SampleToMinimumSpanningTreeFilterType::New();
  fixedMSTFilter->SetListSample( fixedListSample );
  fixedMSTFilter->SetBucketSize( this->m_BucketSize );
  fixedMSTFilter->SetSplittingRule( this->m_SplittingRule );
  fixedMSTFilter->SetKNearestNeighbours( this->m_KNearestNeighbours );
  fixedMSTFilter->SetErrorBound( this->m_ErrorBound );
  fixedMSTFilter->Update();

  typename SampleToMinimumSpanningTreeFilterType::Pointer movingMSTFilter = SampleToMinimumSpanningTreeFilterType::New();
  movingMSTFilter->SetListSample( movingListSample );
  movingMSTFilter->SetBucketSize( this->m_BucketSize );
  movingMSTFilter->SetSplittingRule( this->m_SplittingRule );
  movingMSTFilter->SetKNearestNeighbours( this->m_KNearestNeighbours );
  movingMSTFilter->SetErrorBound( this->m_ErrorBound );
  movingMSTFilter->Update();

  typename SampleToMinimumSpanningTreeFilterType::Pointer jointMSTFilter = SampleToMinimumSpanningTreeFilterType::New();
  jointMSTFilter->SetListSample( jointListSample );
  if ( this->m_UsePenaltyWeight )
    jointMSTFilter->SetPenaltyListSample( penaltyListSample );
  else
    {
	jointMSTFilter->SetBucketSize( this->m_BucketSize );
    jointMSTFilter->SetSplittingRule( this->m_SplittingRule );
    jointMSTFilter->SetKNearestNeighbours( this->m_KNearestNeighbours );
    jointMSTFilter->SetErrorBound( this->m_ErrorBound );
    }
  jointMSTFilter->Update();
  
  /**
   * *************** Estimate the \alpha MI and its derivatives ******************
   *
   * This is done by searching for the minimum spanning tree and calculating the length
   * in order to estimate the value of the alpha - mutual information.
   *
   * The estimate for the alpha - mutual information is given by:
   *
   *  \alpha MI = H<alpha>(F) + H<alpha>(M) - H<alpha>(F, M)
   *            = 1 / ( 1 - \alpha ) * ( \log( fixedLength / ( n^\alpha ) ) +
   *              \log( movingLength / ( n^\alpha ) ) - \log( jointLength / ( n^\alpha ) ) ),
   *
   * where
   *   - \alpha is set by the user and refers to \alpha - mutual information
   *   - n is the number of samples
   *   - fixedLength  is the length of minimum spanning tree in fixed image
   *   - movingLength  is the length of minimum spanning tree in moving image
   *   - jointLength is the length of minimum spanning tree in joint image
   *   - \gamma relates to the distance metric and relates to \alpha as:
   *
   *        \gamma = d * ( 1 - \alpha ),
   *
   *     where d is the dimension of the feature space.
   *
   * In the original paper it is assumed that the mutual information of
   * two feature sets of equal dimension is calculated. If not this is not
   * true, then
   *
   *        \gamma = ( ( d1 + d2 ) / 2 ) * ( 1 - alpha ),
   *
   * where d1 and d2 are the possibly different dimensions of the two feature sets.
   */

  /** Temporary variables. */
  MeasureType mstLength_F, mstLength_M, mstLength_J;
  MeasureType distance_M, distance_J;
  
  /** Get the value of \gamma. */
  double gamma = fixedSize * ( 1.0 - this->m_Alpha );

  /** Create an iterator over the fixed minimum spanning tree. */
  EdgeIteratorType fiter( fixedMSTFilter->GetOutput() );
  
  /** Loop over all edges in the fixed minimum spanning tree. */
  mstLength_F = 0.0;
  for ( fiter.GoToBegin(); !fiter.IsAtEnd(); ++fiter )
    {
    EdgePointerType edge = fiter.GetPointer();
    mstLength_F += vcl_pow( edge->Weight, gamma );
    } // end looping over all edges

  /** Create an iterator over the moving minimum spanning tree. */
  EdgeIteratorType miter( movingMSTFilter->GetOutput() );
  
  /** Loop over all edges in the moving minimum spanning tree. */
  mstLength_M = 0.0;
  for ( miter.GoToBegin(); !miter.IsAtEnd(); ++miter )
    {
    EdgePointerType edge = miter.GetPointer();
   
    distance_M = vcl_pow( edge->Weight, 2 );
    if ( distance_M < MIN_DISTANCE ) 
      continue;
    
    mstLength_M += vcl_pow( edge->Weight, gamma );
    } // end looping over all edges

  /** Create an iterator over the joint minimum spanning tree. */
  EdgeIteratorType jiter( jointMSTFilter->GetOutput() );
  
  /** Loop over all edges in the joint minimum spanning tree. */
  mstLength_J = 0.0;
  for ( jiter.GoToBegin(); !jiter.IsAtEnd(); ++jiter )
    {
    EdgePointerType edge = jiter.GetPointer();
   
    if ( edge->Weight < MIN_DISTANCE ) 
      continue;
    
    distance_J = vcl_pow( edge->Weight, 2 );
    mstLength_J += vcl_pow( edge->Weight, 2.0 * gamma );
    } // end looping over all edges
  
  /**
   * *************** Finally, calculate the metric value \alpha MI ******************
   */

  /** Compute the value. */
  double n, number;
  if ( (mstLength_F > MIN_DISTANCE) && (mstLength_M > MIN_DISTANCE) && (mstLength_J > MIN_DISTANCE) )
    {
    /** Compute the measure. */
    n = static_cast<double>( this->m_NumberOfPixelsCounted );
    number = vcl_pow( n, this->m_Alpha );
    measure = ( vcl_log( mstLength_F / number ) + vcl_log( mstLength_M / number ) -
                vcl_log( mstLength_J / number ) ) / ( 1.0 - this->m_Alpha );
    }

  /** Return the negative alpha - mutual information. */
  return -measure;

} // end GetValue()

/**
 * ************************ GetDerivative *************************
 */

template <class TFixedImage, class TMovingImage>
void
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::GetDerivative(
  const TransformParametersType & parameters,
  DerivativeType & derivative ) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable.
   */
  MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
  this->GetValueAndDerivative( parameters, dummyvalue, derivative );

} // end GetDerivative()

/**
 * ************************ GetValueAndDerivativeSingleThreaded *************************
 */

template <class TFixedImage, class TMovingImage>
void
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivativeSingleThreaded(
  const TransformParametersType & parameters,
  MeasureType & value,
  DerivativeType & derivative ) const
{
  /** Initialize some variables. */
  MeasureType measure = NumericTraits< MeasureType >::Zero;
  DerivativeType contribution( this->GetNumberOfParameters() );
  derivative = DerivativeType( this->GetNumberOfParameters() );
  contribution.Fill( NumericTraits< DerivativeValueType >::Zero );
  derivative.Fill( NumericTraits< DerivativeValueType >::Zero );
  
  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  /**
   * *************** Create the three list samples ******************
   */

  /** Create list samples. */
  ListSamplePointer fixedListSample  = ListSampleType::New();
  ListSamplePointer movingListSample = ListSampleType::New();
  ListSamplePointer jointListSample  = ListSampleType::New();
  
  ListSamplePointer penaltyListSample = ListSampleType::New();

  /** Compute the three list samples and the derivatives. */
  TransformJacobianContainerType jacobianContainer;
  TransformJacobianIndicesContainerType jacobianIndicesContainer;
  SpatialDerivativeContainerType spatialDerivativesContainer;
  this->ComputeListSampleValuesAndDerivativePlusJacobian(
    fixedListSample, movingListSample, jointListSample, penaltyListSample, true, 
    jacobianContainer, jacobianIndicesContainer, spatialDerivativesContainer );

  /** Check if enough samples were valid. */
  unsigned long numSamplePoints = this->GetImageSampler()->GetOutput()->Size();
  this->CheckNumberOfSamples( numSamplePoints, this->m_NumberOfPixelsCounted );
  
  /** Get the size of the feature vectors. */
  const unsigned int fixedSize  = this->GetNumberOfFixedImages();
  const unsigned int movingSize = this->GetNumberOfMovingImages();
  const unsigned int jointSize  = fixedSize + movingSize;

  /**
   * *************** Generate the spanning graphs ******************
   *
   * and get the minimum spanning trees.
   */

  /** Generate the fixed, moving and joint minimum spanning trees from sample points. */
  typename SampleToMinimumSpanningTreeFilterType::Pointer fixedMSTFilter = SampleToMinimumSpanningTreeFilterType::New();
  fixedMSTFilter->SetListSample( fixedListSample );
  fixedMSTFilter->SetBucketSize( this->m_BucketSize );
  fixedMSTFilter->SetSplittingRule( this->m_SplittingRule );
  fixedMSTFilter->SetKNearestNeighbours( this->m_KNearestNeighbours );
  fixedMSTFilter->SetErrorBound( this->m_ErrorBound );
  fixedMSTFilter->Update();

  typename SampleToMinimumSpanningTreeFilterType::Pointer movingMSTFilter = SampleToMinimumSpanningTreeFilterType::New();
  movingMSTFilter->SetListSample( movingListSample );
  movingMSTFilter->SetBucketSize( this->m_BucketSize );
  movingMSTFilter->SetSplittingRule( this->m_SplittingRule );
  movingMSTFilter->SetKNearestNeighbours( this->m_KNearestNeighbours );
  movingMSTFilter->SetErrorBound( this->m_ErrorBound );
  movingMSTFilter->Update();

  typename SampleToMinimumSpanningTreeFilterType::Pointer jointMSTFilter = SampleToMinimumSpanningTreeFilterType::New();
  jointMSTFilter->SetListSample( jointListSample );
  if ( this->m_UsePenaltyWeight )
    jointMSTFilter->SetPenaltyListSample( penaltyListSample );
  else
    {
	jointMSTFilter->SetBucketSize( this->m_BucketSize );
    jointMSTFilter->SetSplittingRule( this->m_SplittingRule );
    jointMSTFilter->SetKNearestNeighbours( this->m_KNearestNeighbours );
    jointMSTFilter->SetErrorBound( this->m_ErrorBound );
    }
  jointMSTFilter->Update();
  
  /**
   * *************** Estimate the \alpha MI and its derivatives ******************
   *
   * This is done by searching for the minimum spanning tree and calculating the length
   * in order to estimate the value of the alpha - mutual information.
   *
   * The estimate for the alpha - mutual information is given by:
   *
   *  \alpha MI = H<alpha>(F) + H<alpha>(M) - H<alpha>(F, M)
   *            = 1 / ( 1 - \alpha ) * ( \log( fixedLength / ( n^\alpha ) ) +
   *              \log( movingLength / ( n^\alpha ) ) - \log( jointLength / ( n^\alpha ) ) ),
   *
   * where
   *   - \alpha is set by the user and refers to \alpha - mutual information
   *   - n is the number of samples
   *   - fixedLength  is the length of minimum spanning tree in fixed image
   *   - movingLength  is the length of minimum spanning tree in moving image
   *   - jointLength is the length of minimum spanning tree in joint image
   *   - \gamma relates to the distance metric and relates to \alpha as:
   *
   *        \gamma = d * ( 1 - \alpha ),
   *
   *     where d is the dimension of the feature space.
   *
   * In the original paper it is assumed that the mutual information of
   * two feature sets of equal dimension is calculated. If not this is not
   * true, then
   *
   *        \gamma = ( ( d1 + d2 ) / 2 ) * ( 1 - alpha ),
   *
   * where d1 and d2 are the possibly different dimensions of the two feature sets.
   */

  /** Temporary variables. */
  MeasurementVectorType movingSrcArray, movingTargetArray; 
  MeasurementVectorType jointSrcArray, jointTargetArray; 
  MeasurementVectorType srcPenalty, targetPenalty; 

  MeasureType mstLength_F, mstLength_M, mstLength_J;
  MeasureType distance_M, distance_J;
  MeasureType diff_M, diff_J;
  MeasureType coef_M, coef_J;
  MeasureType spw_J = 1.0;

  DerivativeType dMST_M( this->GetNumberOfParameters() );
  DerivativeType dMST_J( this->GetNumberOfParameters() );
  dMST_M.Fill( NumericTraits< DerivativeValueType >::Zero );
  dMST_J.Fill( NumericTraits< DerivativeValueType >::Zero );
  
  /** Get the value of \gamma. */
  double gamma = fixedSize * ( 1.0 - this->m_Alpha );

  /** Create an iterator over the fixed minimum spanning tree. */
  EdgeIteratorType fiter( fixedMSTFilter->GetOutput() );
  
  /** Loop over all edges in the fixed minimum spanning tree. */
  mstLength_F = 0.0;
  for ( fiter.GoToBegin(); !fiter.IsAtEnd(); ++fiter )
    {
    EdgePointerType edge = fiter.GetPointer();
    mstLength_F += vcl_pow( edge->Weight, gamma );
    } // end looping over all edges

  /** Create an iterator over the moving minimum spanning tree. */
  EdgeIteratorType miter( movingMSTFilter->GetOutput() );
  
  /** Loop over all edges in the moving minimum spanning tree. */
  mstLength_M = 0.0;
  for ( miter.GoToBegin(); !miter.IsAtEnd(); ++miter )
    {
    EdgePointerType edge = miter.GetPointer();
    NodeIdentifierType src = edge->SourceIdentifier;
    NodeIdentifierType target = edge->TargetIdentifier;

    distance_M = vcl_pow( edge->Weight, 2 );
    if ( distance_M < MIN_DISTANCE ) 
      continue;
    
    mstLength_M += vcl_pow( edge->Weight, gamma );

    SpatialDerivativeType movingSrcSparse, movingTargetSparse;
    movingSrcSparse = spatialDerivativesContainer[ src ] * jacobianContainer[ src ];
    movingTargetSparse = spatialDerivativesContainer[ target ] * jacobianContainer[ target ];

    movingListSample->GetMeasurementVector( src, movingSrcArray );
    movingListSample->GetMeasurementVector( target, movingTargetArray );

    for ( unsigned int i = 0; i < movingSize; ++i )
      {
      diff_M = movingSrcArray[i] - movingTargetArray[i];
      coef_M = gamma * vcl_pow( distance_M, gamma/2.0-1.0 ) * diff_M;
      
      for ( unsigned int j = 0; j < jacobianIndicesContainer[ src ].size(); ++j )
        {
        dMST_M[jacobianIndicesContainer[src][j]] += ( coef_M * movingSrcSparse[i][j] );
        }

      for ( unsigned int j = 0; j < jacobianIndicesContainer[ target ].size(); ++j )
        {
        dMST_M[jacobianIndicesContainer[target][j]] -= ( coef_M * movingTargetSparse[i][j] );
        }
      }
    } // end looping over all edges

  /** Create an iterator over the joint minimum spanning tree. */
  EdgeIteratorType jiter( jointMSTFilter->GetOutput() );
  
  /** Loop over all edges in the joint minimum spanning tree. */
  mstLength_J = 0.0;
  for ( jiter.GoToBegin(); !jiter.IsAtEnd(); ++jiter )
    {
    EdgePointerType edge = jiter.GetPointer();
    NodeIdentifierType src = edge->SourceIdentifier;
    NodeIdentifierType target = edge->TargetIdentifier;

    if ( edge->Weight < MIN_DISTANCE ) 
      continue;
    
    mstLength_J += vcl_pow( edge->Weight, 2.0 * gamma );
    distance_J = vcl_pow( edge->Weight, 2 );

    SpatialDerivativeType jointSrcSparse, jointTargetSparse;
    jointSrcSparse = spatialDerivativesContainer[ src ] * jacobianContainer[ src ];
    jointTargetSparse = spatialDerivativesContainer[ target ] * jacobianContainer[ target ];

    jointListSample->GetMeasurementVector( src, jointSrcArray );
    jointListSample->GetMeasurementVector( target, jointTargetArray );

    if ( this->m_UsePenaltyWeight )   
	  spw_J = edge->PenaltyWeight;
      
    for ( unsigned int i = 0; i < movingSize; ++i )
      {
      diff_J = jointSrcArray[i+fixedSize] - jointTargetArray[i+fixedSize];
      coef_J = 2.0 * gamma * vcl_pow( distance_J, gamma-1.0 ) * spw_J * spw_J * diff_J;
      
      for ( unsigned int j = 0; j < jacobianIndicesContainer[ src ].size(); ++j )
        {
        dMST_J[jacobianIndicesContainer[src][j]] += ( coef_J * jointSrcSparse[i][j] );
        }

      for ( unsigned int j = 0; j < jacobianIndicesContainer[ target ].size(); ++j )
        {
        dMST_J[jacobianIndicesContainer[target][j]] -= ( coef_J * jointTargetSparse[i][j] );
        }
      }
    } // end looping over all edges
    
  /**
   * *************** Finally, calculate the metric value and derivative ******************
   */

  /** Compute the value. */
  double n, number;
  if ( (mstLength_F > MIN_DISTANCE) && (mstLength_M > MIN_DISTANCE) && (mstLength_J > MIN_DISTANCE) )
    {
    /** Compute the measure. */
    n = static_cast<double>( this->m_NumberOfPixelsCounted );
    number = vcl_pow( n, this->m_Alpha );
    measure = ( vcl_log( mstLength_F / number ) + vcl_log( mstLength_M / number ) -
                vcl_log( mstLength_J / number ) ) / ( 1.0 - this->m_Alpha );
  
    /** Compute the derivative. */
    contribution = dMST_M / mstLength_M - dMST_J / mstLength_J;
    derivative = -contribution / ( 1.0 - this->m_Alpha );
    }

  value = -measure;

} // end GetValueAndDerivativeSingleThreaded()

/**
 * ************************ GetValueAndDerivative *************************
 */

template <class TFixedImage, class TMovingImage>
void
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType & value,
  DerivativeType & derivative ) const 
{
  /** Option for now to still use the single threaded code. */
  if ( !this->m_UseMultiThread )
    {
    return this->GetValueAndDerivativeSingleThreaded(
      parameters, value, derivative );
    }
  
  /** Initialize some variables. */
  MeasureType measure = NumericTraits< MeasureType >::Zero;
  DerivativeType contribution( this->GetNumberOfParameters() );
  derivative = DerivativeType( this->GetNumberOfParameters() );
  contribution.Fill( NumericTraits< DerivativeValueType >::Zero );
  derivative.Fill( NumericTraits< DerivativeValueType >::Zero );
  
  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  /** Update the imageSampler. */
  this->GetImageSampler()->Update();

  /** Get the size of the feature vectors. */
  const unsigned int fixedSize  = this->GetNumberOfFixedImages();
  const unsigned int movingSize = this->GetNumberOfMovingImages();
  const unsigned int jointSize  = fixedSize + movingSize;
  
  /** Initialize some threading related parameters. */
  this->InitializeMSTThreadingParameters();
 
  /**
   * *************** Evoking the multi-threads for the list samples ******************
   *
   * and the derivatives plus jacobian.
   */

  this->m_Threader->SetNumberOfThreads( this->m_NumberOfThreads );
  this->m_Threader->SetSingleMethod( MSTListSamplesAndDerivativePlusJacobianThreaderCallback, 
    const_cast<void *>(static_cast<const void *>(&this->m_MSTThreaderMetricParameters)) );
  this->m_Threader->SingleMethodExecute();

  /** Collect listsamples and derivatives plus jacobian from all threads. */
  this->AfterMSTThreadedListSamplesAndDerivativePlusJacobian();

  /**
   * *************** Generate the spanning graphs ******************
   *
   * and get the minimum spanning trees.
   */

  /** Generate the fixed, moving and joint minimum spanning trees from sample points. */
  typename SampleToMinimumSpanningTreeFilterType::Pointer fixedMSTFilter = SampleToMinimumSpanningTreeFilterType::New();
  fixedMSTFilter->SetListSample( this->m_MSTThreaderMetricParameters.st_FixedListSample );
  fixedMSTFilter->SetBucketSize( this->m_BucketSize );
  fixedMSTFilter->SetSplittingRule( this->m_SplittingRule );
  fixedMSTFilter->SetKNearestNeighbours( this->m_KNearestNeighbours );
  fixedMSTFilter->SetErrorBound( this->m_ErrorBound );
  fixedMSTFilter->Update();

  typename SampleToMinimumSpanningTreeFilterType::Pointer movingMSTFilter = SampleToMinimumSpanningTreeFilterType::New();
  movingMSTFilter->SetListSample( this->m_MSTThreaderMetricParameters.st_MovingListSample );
  movingMSTFilter->SetBucketSize( this->m_BucketSize );
  movingMSTFilter->SetSplittingRule( this->m_SplittingRule );
  movingMSTFilter->SetKNearestNeighbours( this->m_KNearestNeighbours );
  movingMSTFilter->SetErrorBound( this->m_ErrorBound );
  movingMSTFilter->Update();
  GraphPointer movingMST = movingMSTFilter->GetOutput();

  typename SampleToMinimumSpanningTreeFilterType::Pointer jointMSTFilter = SampleToMinimumSpanningTreeFilterType::New();
  jointMSTFilter->SetListSample( this->m_MSTThreaderMetricParameters.st_JointListSample );
  if ( this->m_UsePenaltyWeight )
    {
    jointMSTFilter->SetPenaltyListSample( this->m_MSTThreaderMetricParameters.st_PenaltyListSample );
	jointMSTFilter->m_FPGThreader = this->m_Threader;
	jointMSTFilter->SetNumberOfThreads( this->m_NumberOfThreads );
    }
  else
    {
	jointMSTFilter->SetBucketSize( this->m_BucketSize );
    jointMSTFilter->SetSplittingRule( this->m_SplittingRule );
    jointMSTFilter->SetKNearestNeighbours( this->m_KNearestNeighbours );
    jointMSTFilter->SetErrorBound( this->m_ErrorBound );
    }
  jointMSTFilter->Update();
  GraphPointer jointMST = jointMSTFilter->GetOutput();

  /** Potential speedup: it avoids re-allocations. I noticed performance
   * gains when nrOfRequestedSamples is about 10000 or higher.
   */
  this->m_MSTThreaderMetricParameters.st_MovingSRC.reserve( movingMST->GetTotalNumberOfEdges() );
  this->m_MSTThreaderMetricParameters.st_MovingTAR.reserve( movingMST->GetTotalNumberOfEdges() );
  this->m_MSTThreaderMetricParameters.st_MovingEW.reserve( movingMST->GetTotalNumberOfEdges() );

  this->m_MSTThreaderMetricParameters.st_JointSRC.reserve( jointMST->GetTotalNumberOfEdges() );
  this->m_MSTThreaderMetricParameters.st_JointTAR.reserve( jointMST->GetTotalNumberOfEdges() );
  this->m_MSTThreaderMetricParameters.st_JointEW.reserve( jointMST->GetTotalNumberOfEdges() );
  if ( this->m_UsePenaltyWeight )
    this->m_MSTThreaderMetricParameters.st_JointPEW.reserve( jointMST->GetTotalNumberOfEdges() );
  
  /** Transfer the edge elements of minimum spanning tree for multi-threading. */
  EdgeIteratorType miter( movingMST );
  for ( miter.GoToBegin(); !miter.IsAtEnd(); ++miter )
    {
	EdgePointerType edge = miter.GetPointer();
	this->m_MSTThreaderMetricParameters.st_MovingSRC.push_back( edge->SourceIdentifier );
    this->m_MSTThreaderMetricParameters.st_MovingTAR.push_back( edge->TargetIdentifier );
	this->m_MSTThreaderMetricParameters.st_MovingEW.push_back( edge->Weight );
    }

  EdgeIteratorType jiter( jointMST );
  for ( jiter.GoToBegin(); !jiter.IsAtEnd(); ++jiter )
    {
	EdgePointerType edge = jiter.GetPointer();
    this->m_MSTThreaderMetricParameters.st_JointSRC.push_back( edge->SourceIdentifier );
    this->m_MSTThreaderMetricParameters.st_JointTAR.push_back( edge->TargetIdentifier );
	this->m_MSTThreaderMetricParameters.st_JointEW.push_back( edge->Weight );
    }
  if ( this->m_UsePenaltyWeight )
    {
	for ( jiter.GoToBegin(); !jiter.IsAtEnd(); ++jiter )
	  this->m_MSTThreaderMetricParameters.st_JointPEW.push_back( jiter.GetPointer()->PenaltyWeight );
    }
  
  /**
   * *************** Estimate the \alpha MI and its derivatives ******************
   *
   * This is done by searching for the minimum spanning tree and calculating the length
   * in order to estimate the value of the alpha - mutual information.
   *
   * The estimate for the alpha - mutual information is given by:
   *
   *  \alpha MI = H<alpha>(F) + H<alpha>(M) - H<alpha>(F, M)
   *            = 1 / ( 1 - \alpha ) * ( \log( fixedLength / ( n^\alpha ) ) +
   *              \log( movingLength / ( n^\alpha ) ) - \log( jointLength / ( n^\alpha ) ) ),
   *
   * where
   *   - \alpha is set by the user and refers to \alpha - mutual information
   *   - n is the number of samples
   *   - fixedLength  is the length of minimum spanning tree in fixed image
   *   - movingLength  is the length of minimum spanning tree in moving image
   *   - jointLength is the length of minimum spanning tree in joint image
   *   - \gamma relates to the distance metric and relates to \alpha as:
   *
   *        \gamma = d * ( 1 - \alpha ),
   *
   *     where d is the dimension of the feature space.
   *
   * In the original paper it is assumed that the mutual information of
   * two feature sets of equal dimension is calculated. If not this is not
   * true, then
   *
   *        \gamma = ( ( d1 + d2 ) / 2 ) * ( 1 - alpha ),
   *
   * where d1 and d2 are the possibly different dimensions of the two feature sets.
   */

  /** Temporary variables. */
  MeasureType mstLength_F, mstLength_M, mstLength_J;
  
  /** Get the value of \gamma. */
  double gamma = fixedSize * ( 1.0 - this->m_Alpha );

  /** Create an iterator over the fixed minimum spanning tree. */
  EdgeIteratorType fiter( fixedMSTFilter->GetOutput() );
  
  /** Loop over all edges in the fixed minimum spanning tree. */
  mstLength_F = 0.0;
  for ( fiter.GoToBegin(); !fiter.IsAtEnd(); ++fiter )
    {
    EdgePointerType edge = fiter.GetPointer();
    mstLength_F += vcl_pow( edge->Weight, gamma );
    } // end looping over all edges

  /**
   * *************** Evoking the multi-threads for the value and derivative ******************
   */

  this->m_Threader->SetNumberOfThreads( this->m_NumberOfThreads );
  this->m_Threader->SetSingleMethod( MSTGetValueAndDerivativeThreaderCallback, 
    const_cast<void *>(static_cast<const void *>(&this->m_MSTThreaderMetricParameters)) );
  this->m_Threader->SingleMethodExecute();

  /**
   * *************** Finally, calculate the metric value and derivative ******************
   */ 

  /** Collect the value from all threads. */
  mstLength_M = 0.0;
  mstLength_J = 0.0;
  for ( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
   {
   mstLength_M += this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_MValue;
   mstLength_J += this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_JValue;

   this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_jacobianContainer.swap( TransformJacobianContainerType() );
   this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_jacobianIndicesContainer.swap( TransformJacobianIndicesContainerType() );
   this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_spatialDerivativesContainer.swap( SpatialDerivativeContainerType() );   
   }

  /** Collect the derivative using multi-threading. */
  this->m_Threader->SetNumberOfThreads( this->m_NumberOfThreads );
  this->m_Threader->SetSingleMethod( MSTMergeDerivativeThreaderCallback, 
    const_cast<void *>(static_cast<const void *>(&this->m_MSTThreaderMetricParameters)) );
  this->m_Threader->SingleMethodExecute();
  
  /** Compute the value. */
  double n, number;
  if ( (mstLength_F > MIN_DISTANCE) && (mstLength_M > MIN_DISTANCE) && (mstLength_J > MIN_DISTANCE) )
    {
    /** Compute the measure. */
    n = static_cast<double>( this->m_NumberOfPixelsCounted );
    number = vcl_pow( n, this->m_Alpha );
    measure = ( vcl_log( mstLength_F / number ) + vcl_log( mstLength_M / number ) -
                vcl_log( mstLength_J / number ) ) / ( 1.0 - this->m_Alpha );
  
    /** Compute the derivative. */
    contribution = this->m_MSTThreaderMetricParameters.st_MSTMDerivative / mstLength_M - 
      this->m_MSTThreaderMetricParameters.st_MSTJDerivative / mstLength_J;
    derivative = -contribution / ( 1.0 - this->m_Alpha );
    }

  value = -measure;

  /** Release some space for next iteration. */
  this->m_MSTThreaderMetricParameters.st_JacobianContainer.swap( TransformJacobianContainerType() );
  this->m_MSTThreaderMetricParameters.st_JacobianIndicesContainer.swap( TransformJacobianIndicesContainerType() );
  this->m_MSTThreaderMetricParameters.st_SpatialDerivativesContainer.swap( SpatialDerivativeContainerType() ); 
  this->m_MSTThreaderMetricParameters.st_MovingSRC.swap( NodeIdentifierContainerType() );
  this->m_MSTThreaderMetricParameters.st_MovingTAR.swap( NodeIdentifierContainerType() );
  this->m_MSTThreaderMetricParameters.st_MovingEW.swap( EdgeWeightContainerType() );
  this->m_MSTThreaderMetricParameters.st_JointSRC.swap( NodeIdentifierContainerType() );
  this->m_MSTThreaderMetricParameters.st_JointTAR.swap( NodeIdentifierContainerType() );
  this->m_MSTThreaderMetricParameters.st_JointEW.swap( EdgeWeightContainerType() );
  if ( this->m_UsePenaltyWeight )
    this->m_MSTThreaderMetricParameters.st_JointPEW.swap( EdgeWeightContainerType() );

} // end GetValueAndDerivative()

/**
 * ************************ ComputeListSampleValuesAndDerivativePlusJacobian *************************
 */

template <class TFixedImage, class TMovingImage>
void
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::ComputeListSampleValuesAndDerivativePlusJacobian(
  const ListSamplePointer & fixedListSample,
  const ListSamplePointer & movingListSample,
  const ListSamplePointer & jointListSample,
  const ListSamplePointer & penaltyListSample,
  const bool & doDerivative,
  TransformJacobianContainerType & jacobianContainer,
  TransformJacobianIndicesContainerType & jacobianIndicesContainer,
  SpatialDerivativeContainerType & spatialDerivativesContainer ) const
{
  /** Initialize. */
  this->m_NumberOfPixelsCounted = 0;
  jacobianContainer.resize( 0 );
  jacobianIndicesContainer.resize( 0 );
  spatialDerivativesContainer.resize( 0 );

  /** Update the imageSampler and get a handle to the sample container. */
  this->GetImageSampler()->Update();
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long nrOfRequestedSamples = sampleContainer->Size();

  /** Get the size of the feature vectors. */
  const unsigned int fixedSize  = this->GetNumberOfFixedImages();
  const unsigned int movingSize = this->GetNumberOfMovingImages();
  const unsigned int jointSize  = fixedSize + movingSize;

  /** Resize the list samples so that enough memory is allocated. */
  fixedListSample->SetMeasurementVectorSize( fixedSize );
  fixedListSample->Resize( nrOfRequestedSamples );
  movingListSample->SetMeasurementVectorSize( movingSize );
  movingListSample->Resize( nrOfRequestedSamples );
  jointListSample->SetMeasurementVectorSize( jointSize );
  jointListSample->Resize( nrOfRequestedSamples );
  if ( this->m_UsePenaltyWeight )
    {   
    penaltyListSample->SetMeasurementVectorSize( this->m_NumberOfPenaltyImages );
    penaltyListSample->Resize( nrOfRequestedSamples );
    }

  /** Potential speedup: it avoids re-allocations. I noticed performance
   * gains when nrOfRequestedSamples is about 10000 or higher.
   */
  jacobianContainer.reserve( nrOfRequestedSamples );
  jacobianIndicesContainer.reserve( nrOfRequestedSamples );
  spatialDerivativesContainer.reserve( nrOfRequestedSamples );
  
  /** Create variables to store intermediate results. */
  RealType movingImageValue;
  MovingImagePointType mappedPoint;
  FixedImageType::IndexType pindex;
  double fixedFeatureValue = 0.0;
  double movingFeatureValue = 0.0;
  double penaltyFeatureValue = 0.0;
  NonZeroJacobianIndicesType nzji(
    this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  TransformJacobianType jacobian;

  /** Create an iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();
  
  /** Loop over the fixed image samples to calculate the list samples. */
  for ( fiter = fbegin; fiter != fend; ++fiter )
    {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
    
    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside all moving masks. */
    if ( sampleOk )
      {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
      }

    /** Compute the moving image value M(T(x)) and possibly the
     * derivative dM/dx and check if the point is inside all
     * moving images buffers.
     */
    MovingImageDerivativeType movingImageDerivative;
    if ( sampleOk )
      {
      if ( doDerivative )
        {
        sampleOk = this->EvaluateMovingImageValueAndDerivative(
          mappedPoint, movingImageValue, &movingImageDerivative );
        }
      else
        {
        sampleOk = this->EvaluateMovingImageValueAndDerivative(
          mappedPoint, movingImageValue, 0 );
        }
      }

    /** This is a valid sample: in this if-statement the actual
     * addition to the list samples is done.
     */
    if ( sampleOk )
      {
      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast<RealType>(
        (*fiter).Value().m_ImageValue );

      /** Add the samples to the ListSamplearrays. */
      fixedListSample->SetMeasurement( this->m_NumberOfPixelsCounted, 0, fixedImageValue );
      movingListSample->SetMeasurement( this->m_NumberOfPixelsCounted, 0, movingImageValue );
      jointListSample->SetMeasurement( this->m_NumberOfPixelsCounted, 0, fixedImageValue );
      jointListSample->SetMeasurement( this->m_NumberOfPixelsCounted, fixedSize, movingImageValue );

      /** Get and set the values of the fixed feature images. */
      for ( unsigned int j = 1; j < fixedSize; ++j )
        {
        fixedFeatureValue = this->m_FixedImageInterpolatorVector[ j ]->Evaluate( fixedPoint );
        fixedListSample->SetMeasurement( this->m_NumberOfPixelsCounted, j, fixedFeatureValue );
        jointListSample->SetMeasurement( this->m_NumberOfPixelsCounted, j, fixedFeatureValue );
        }
      if ( this->m_UsePenaltyWeight )
        {
        for ( unsigned int j = 0; j < this->m_NumberOfPenaltyImages; ++j )
          {
		  (*this->m_PenaltyImages)[ j ]->TransformPhysicalPointToIndex( fixedPoint, pindex );
          penaltyFeatureValue = (*this->m_PenaltyImages)[ j ]->GetPixel( pindex );
          penaltyListSample->SetMeasurement( this->m_NumberOfPixelsCounted, j, penaltyFeatureValue );
          }
        }

      /** Get and set the values of the moving feature images. */
      for ( unsigned int j = 1; j < movingSize; ++j )
        {
        movingFeatureValue = this->m_InterpolatorVector[ j ]->Evaluate( mappedPoint );
        movingListSample->SetMeasurement( this->m_NumberOfPixelsCounted, j, movingFeatureValue );
        jointListSample->SetMeasurement( this->m_NumberOfPixelsCounted, j + fixedSize, movingFeatureValue );
        }

      /** Compute additional stuff for the computation of the derivative, if necessary.
       * - the Jacobian of the transform: dT/dmu(x_i).
       * - the spatial derivative of all moving feature images: dz_q^m/dx(T(x_i)).
       */
      if ( doDerivative )
        {
        /** Get the TransformJacobian dT/dmu. */
        this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );
        jacobianContainer.push_back( jacobian );
        jacobianIndicesContainer.push_back( nzji );

        /** Get the spatial derivative of the moving image. */
        SpatialDerivativeType spatialDerivatives(
          this->GetNumberOfMovingImages(),
          this->FixedImageDimension );
        spatialDerivatives.set_row( 0, movingImageDerivative.GetDataPointer() );

        /** Get the spatial derivatives of the moving feature images. */
        SpatialDerivativeType movingFeatureImageDerivatives(
          this->GetNumberOfMovingImages() - 1,
          this->FixedImageDimension );
        this->EvaluateMovingFeatureImageDerivatives(
          mappedPoint, movingFeatureImageDerivatives );
        spatialDerivatives.update( movingFeatureImageDerivatives, 1, 0 );

        /** Put the spatial derivatives of this sample into the container. */
        spatialDerivativesContainer.push_back( spatialDerivatives );

        } // end if doDerivative

      /** Update the NumberOfPixelsCounted. */
      this->m_NumberOfPixelsCounted++;

      } // end if sampleOk

    } // end for loop over the image sample container
     
  /** The listSamples are of size sampleContainer->Size(). However, not all of
   * those points made it to the respective list samples. Therefore, we set
   * the actual number of pixels in the sample container, so that the binary
   * trees know where to loop over. This must not be forgotten!
   */
  fixedListSample->SetActualSize( this->m_NumberOfPixelsCounted );
  movingListSample->SetActualSize( this->m_NumberOfPixelsCounted );
  jointListSample->SetActualSize( this->m_NumberOfPixelsCounted );
  if ( this->m_UsePenaltyWeight )
    {
    penaltyListSample->SetActualSize( this->m_NumberOfPixelsCounted );
    }

} // end ComputeListSampleValuesAndDerivativePlusJacobian()

/**
 * ************************ EvaluateMovingFeatureImageDerivatives *************************
 */

template <class TFixedImage, class TMovingImage>
void
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::EvaluateMovingFeatureImageDerivatives(
  const MovingImagePointType & mappedPoint,
  SpatialDerivativeType & featureGradients ) const
{
  /** Convert point to a continous index. */
  MovingImageContinuousIndexType cindex;
  this->m_Interpolator->ConvertPointToContinuousIndex( mappedPoint, cindex );

  /** Compute the spatial derivative for all feature images:
   * - either by calling a special function that only B-spline
   *   interpolators have,
   * - or by using a finite difference approximation of the
   *   pre-computed gradient images.
   * \todo: for now we only implement the first option.
   */
  if ( this->m_InterpolatorsAreBSpline && !this->GetComputeGradient() )
    {
    /** Computed moving image gradient using derivative B-spline kernel. */
    MovingImageDerivativeType gradient;
    for ( unsigned int i = 1; i < this->GetNumberOfMovingImages(); ++i )
      {
      /** Compute the gradient at feature image i. */
      gradient = this->m_BSplineInterpolatorVector[ i ]
      ->EvaluateDerivativeAtContinuousIndex( cindex );

      /** Set the gradient into the Array2D. */
      featureGradients.set_row( i - 1, gradient.GetDataPointer() );
      } // end for-loop
    } // end if

} // end EvaluateMovingFeatureImageDerivatives()

/**
 * **************** MSTListSamplesAndDerivativePlusJacobianThreaderCallback *******
 */

template<class TFixedImage, class TMovingImage>
ITK_THREAD_RETURN_TYPE
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::MSTListSamplesAndDerivativePlusJacobianThreaderCallback( void * arg )
{
  MSTThreadInfoType * infoStruct = static_cast< MSTThreadInfoType * >( arg );
  ThreadIdType        threadID   = infoStruct->ThreadID;

  MSTMultiThreaderParameterType * temp
    = static_cast< MSTMultiThreaderParameterType * >( infoStruct->UserData );

  temp->st_Metric->MSTThreadedListSamplesAndDerivativePlusJacobian( threadID );
  
  return ITK_THREAD_RETURN_VALUE;

} // end MSTListSamplesAndDerivativePlusJacobianThreaderCallback()

/**
 * ******************* MSTThreadedListSamplesAndDerivativePlusJacobian *******************
 */

template<class TFixedImage, class TMovingImage>
void
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::MSTThreadedListSamplesAndDerivativePlusJacobian( ThreadIdType threadId )
{
  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long sampleContainerSize     = sampleContainer->Size();

  /** Get the size of the feature vectors. */
  const unsigned int fixedSize  = this->GetNumberOfFixedImages();
  const unsigned int movingSize = this->GetNumberOfMovingImages();
  const unsigned int jointSize  = fixedSize + movingSize;

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads
    = static_cast< unsigned long >( vcl_ceil( static_cast< double >( sampleContainerSize )
    / static_cast< double >( this->m_NumberOfThreads  ) ) );

  unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
  unsigned long pos_end   = nrOfSamplesPerThreads * ( threadId + 1 );
  pos_begin = ( pos_begin > sampleContainerSize ) ? sampleContainerSize : pos_begin;
  pos_end   = ( pos_end > sampleContainerSize ) ? sampleContainerSize : pos_end;

  /** Resize the list samples so that enough memory is allocated. */
  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_fixedListSample->SetMeasurementVectorSize( fixedSize );
  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_fixedListSample->Resize( nrOfSamplesPerThreads );
  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_movingListSample->SetMeasurementVectorSize( movingSize );
  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_movingListSample->Resize( nrOfSamplesPerThreads );
  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_jointListSample->SetMeasurementVectorSize( jointSize );
  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_jointListSample->Resize( nrOfSamplesPerThreads );
  if ( this->m_UsePenaltyWeight )
    {   
    this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_penaltyListSample->SetMeasurementVectorSize( this->m_NumberOfPenaltyImages );
    this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_penaltyListSample->Resize( nrOfSamplesPerThreads );
    }

  /** Potential speedup: it avoids re-allocations. I noticed performance
   * gains when nrOfRequestedSamples is about 10000 or higher.
   */
  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_jacobianContainer.reserve( nrOfSamplesPerThreads );
  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_jacobianIndicesContainer.reserve( nrOfSamplesPerThreads );
  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_spatialDerivativesContainer.reserve( nrOfSamplesPerThreads ); 

  /** Create variables to store intermediate results. */
  RealType movingImageValue;
  MovingImagePointType mappedPoint;
  FixedImageType::IndexType pindex;
  double fixedFeatureValue = 0.0;
  double movingFeatureValue = 0.0;
  double penaltyFeatureValue = 0.0;
  NonZeroJacobianIndicesType nzji(
    this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  TransformJacobianType jacobian;

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator threader_fiter;
  typename ImageSampleContainerType::ConstIterator threader_fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator threader_fend   = sampleContainer->Begin();

  threader_fbegin += (int)pos_begin;
  threader_fend   += (int)pos_end;
 
  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted = 0;
  /** Loop over the fixed image samples to calculate the list samples. */
  for( threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter )
    {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = (*threader_fiter).Value().m_ImageCoordinates;
    
    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside all moving masks. */
    if ( sampleOk )
      {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
      }

    /** Compute the moving image value M(T(x)) and possibly the
     * derivative dM/dx and check if the point is inside all
     * moving images buffers.
     */
    MovingImageDerivativeType movingImageDerivative;
    if ( sampleOk )
      {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, &movingImageDerivative );
      }

    /** This is a valid sample: in this if-statement the actual
     * addition to the list samples is done.
     */
    if ( sampleOk )
      {
      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast<RealType>(
        (*threader_fiter).Value().m_ImageValue );

      /** Add the samples to the ListSamplearrays. */
      this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_fixedListSample->SetMeasurement( 
		this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, 0, fixedImageValue );
      this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_movingListSample->SetMeasurement( 
		this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, 0, movingImageValue );
      this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_jointListSample->SetMeasurement( 
		this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, 0, fixedImageValue );
      this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_jointListSample->SetMeasurement( 
		this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, fixedSize, movingImageValue );

      /** Get and set the values of the fixed feature images. */
      for ( unsigned int j = 1; j < fixedSize; ++j )
        {
        fixedFeatureValue = this->m_FixedImageInterpolatorVector[ j ]->Evaluate( fixedPoint );
        this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_fixedListSample->SetMeasurement( 
	      this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, j, fixedFeatureValue );
        this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_jointListSample->SetMeasurement( 
		  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, j, fixedFeatureValue );
        }
      if ( this->m_UsePenaltyWeight )
        {
        for ( unsigned int j = 0; j < this->m_NumberOfPenaltyImages; ++j )
          {
		  (*this->m_PenaltyImages)[ j ]->TransformPhysicalPointToIndex( fixedPoint, pindex );
          penaltyFeatureValue = (*this->m_PenaltyImages)[ j ]->GetPixel( pindex );
          this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_penaltyListSample->SetMeasurement( 
			this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, j, penaltyFeatureValue );
          }
        }

      /** Get and set the values of the moving feature images. */
      for ( unsigned int j = 1; j < movingSize; ++j )
        {
        movingFeatureValue = this->m_InterpolatorVector[ j ]->Evaluate( mappedPoint );
        this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_movingListSample->SetMeasurement( 
		  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, j, movingFeatureValue );
        this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_jointListSample->SetMeasurement( 
		  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, j + fixedSize, movingFeatureValue );
        }

      /** Compute additional stuff for the computation of the derivative, if necessary.
       * - the Jacobian of the transform: dT/dmu(x_i).
       * - the spatial derivative of all moving feature images: dz_q^m/dx(T(x_i)).
       */
      
      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );
      this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_jacobianContainer.push_back( jacobian );
      this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_jacobianIndicesContainer.push_back( nzji );

      /** Get the spatial derivative of the moving image. */
      SpatialDerivativeType spatialDerivatives(
        this->GetNumberOfMovingImages(),
        this->FixedImageDimension );
      spatialDerivatives.set_row( 0, movingImageDerivative.GetDataPointer() );

      /** Get the spatial derivatives of the moving feature images. */
      SpatialDerivativeType movingFeatureImageDerivatives(
        this->GetNumberOfMovingImages() - 1,
        this->FixedImageDimension );
      this->EvaluateMovingFeatureImageDerivatives(
        mappedPoint, movingFeatureImageDerivatives );
      spatialDerivatives.update( movingFeatureImageDerivatives, 1, 0 );

      /** Put the spatial derivatives of this sample into the container. */
      this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_spatialDerivativesContainer.push_back( spatialDerivatives ); 

      /** Update the NumberOfPixelsCounted for this thread. */
      this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted++;

      } // end if sampleOk

    } // end for loop over the image sample container
     
  /** The listSamples are of size sampleContainer->Size(). However, not all of
   * those points made it to the respective list samples. Therefore, we set
   * the actual number of pixels in the sample container, so that the binary
   * trees know where to loop over. This must not be forgotten!
   */
  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_fixedListSample->SetActualSize( 
	this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted );
  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_movingListSample->SetActualSize( 
	this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted );
  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_jointListSample->SetActualSize( 
	this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted );
  if ( this->m_UsePenaltyWeight )
    {
    this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_penaltyListSample->SetActualSize( 
	  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted );
    }
  
} // end MSTThreadedListSamplesAndDerivativePlusJacobian()

/**
 * ******************* AfterMSTThreadedListSamplesAndDerivativePlusJacobian *******************
 */

template<class TFixedImage, class TMovingImage>
void
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::AfterMSTThreadedListSamplesAndDerivativePlusJacobian( void ) const
{
  /** Gather the NumberOfPixelsCounted from all threads. */
  this->m_NumberOfPixelsCounted = 0;
  for ( ThreadIdType i = 0; i < this->m_NumberOfThreads ; ++i )
    this->m_NumberOfPixelsCounted += this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_numberOfPixelsCounted;

  /** Check if enough samples were valid. */
  unsigned long numSamplePoints = this->GetImageSampler()->GetOutput()->Size();
  this->CheckNumberOfSamples( numSamplePoints, this->m_NumberOfPixelsCounted );
  
  /** Get the size of the feature vectors. */
  const unsigned int fixedSize  = this->GetNumberOfFixedImages();
  const unsigned int movingSize = this->GetNumberOfMovingImages();
  const unsigned int jointSize  = fixedSize + movingSize;

  /** Resize the list samples so that enough memory is allocated. */
  this->m_MSTThreaderMetricParameters.st_FixedListSample->SetMeasurementVectorSize( fixedSize );
  this->m_MSTThreaderMetricParameters.st_FixedListSample->Resize( this->m_NumberOfPixelsCounted );
  this->m_MSTThreaderMetricParameters.st_FixedListSample->SetActualSize( this->m_NumberOfPixelsCounted );
  this->m_MSTThreaderMetricParameters.st_MovingListSample->SetMeasurementVectorSize( movingSize );
  this->m_MSTThreaderMetricParameters.st_MovingListSample->Resize( this->m_NumberOfPixelsCounted );
  this->m_MSTThreaderMetricParameters.st_MovingListSample->SetActualSize( this->m_NumberOfPixelsCounted );
  this->m_MSTThreaderMetricParameters.st_JointListSample->SetMeasurementVectorSize( jointSize );
  this->m_MSTThreaderMetricParameters.st_JointListSample->Resize( this->m_NumberOfPixelsCounted );
  this->m_MSTThreaderMetricParameters.st_JointListSample->SetActualSize( this->m_NumberOfPixelsCounted );
  if ( this->m_UsePenaltyWeight )
    {   
    this->m_MSTThreaderMetricParameters.st_PenaltyListSample->SetMeasurementVectorSize( this->m_NumberOfPenaltyImages );
	this->m_MSTThreaderMetricParameters.st_PenaltyListSample->Resize( this->m_NumberOfPixelsCounted );
    this->m_MSTThreaderMetricParameters.st_PenaltyListSample->SetActualSize( this->m_NumberOfPixelsCounted );
    } 

  /** Potential speedup: it avoids re-allocations. I noticed performance
   * gains when nrOfRequestedSamples is about 10000 or higher.
   */
  this->m_MSTThreaderMetricParameters.st_JacobianContainer.reserve( this->m_NumberOfPixelsCounted );
  this->m_MSTThreaderMetricParameters.st_JacobianIndicesContainer.reserve( this->m_NumberOfPixelsCounted );
  this->m_MSTThreaderMetricParameters.st_SpatialDerivativesContainer.reserve( this->m_NumberOfPixelsCounted );
   
  /** Gather listsamples and derivatives plus jacobian from all threads. */
  unsigned long nOPC = 0;
  for ( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
    {
	unsigned long nOPCThread = this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_numberOfPixelsCounted;
	for ( unsigned long j = 0; j < nOPCThread; ++j )
	  {
	  this->m_MSTThreaderMetricParameters.st_FixedListSample->SetMeasurementVector(
		nOPC, this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_fixedListSample->GetMeasurementVector( j ) );
	  this->m_MSTThreaderMetricParameters.st_MovingListSample->SetMeasurementVector(
		nOPC, this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_movingListSample->GetMeasurementVector( j ) );
	  this->m_MSTThreaderMetricParameters.st_JointListSample->SetMeasurementVector(
		nOPC, this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_jointListSample->GetMeasurementVector( j ) );
	  if ( this->m_UsePenaltyWeight )
	    {
		this->m_MSTThreaderMetricParameters.st_PenaltyListSample->SetMeasurementVector( 
		  nOPC, this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_penaltyListSample->GetMeasurementVector( j ) );
	    }

	  this->m_MSTThreaderMetricParameters.st_JacobianContainer.push_back( this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_jacobianContainer[ j ] );
	  this->m_MSTThreaderMetricParameters.st_JacobianIndicesContainer.push_back( this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_jacobianIndicesContainer[ j ] );
	  this->m_MSTThreaderMetricParameters.st_SpatialDerivativesContainer.push_back( this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_spatialDerivativesContainer[ j ] );
	  nOPC++;
	  }
    }

} // end AfterMSTThreadedListSamplesAndDerivativePlusJacobian()

/**
 * **************** MSTGetValueAndDerivativeThreaderCallback *******
 */

template<class TFixedImage, class TMovingImage>
ITK_THREAD_RETURN_TYPE
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::MSTGetValueAndDerivativeThreaderCallback( void * arg )
{
  MSTThreadInfoType * infoStruct = static_cast< MSTThreadInfoType * >( arg );
  ThreadIdType        threadID   = infoStruct->ThreadID;

  MSTMultiThreaderParameterType * temp
    = static_cast< MSTMultiThreaderParameterType * >( infoStruct->UserData );

  temp->st_Metric->MSTThreadedGetValueAndDerivative( threadID );
  
  return ITK_THREAD_RETURN_VALUE;

} // end MSTGetValueAndDerivativeThreaderCallback()

/**
 * ******************* MSTThreadedGetValueAndDerivative *******************
 */

template<class TFixedImage, class TMovingImage>
void
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::MSTThreadedGetValueAndDerivative( ThreadIdType threadId ) 
{
  /** Get the size of the feature vectors. */
  const unsigned int fixedSize  = this->GetNumberOfFixedImages();
  const unsigned int movingSize = this->GetNumberOfMovingImages();
  const unsigned int jointSize  = fixedSize + movingSize;

  /** Temporary variables. */
  MeasurementVectorType movingSrcArray, movingTargetArray; 
  MeasurementVectorType jointSrcArray, jointTargetArray; 
  MeasurementVectorType srcPenalty, targetPenalty; 

  MeasureType mstLength_M, mstLength_J;
  MeasureType distance_M, distance_J;
  MeasureType diff_M, diff_J;
  MeasureType coef_M, coef_J;
  MeasureType spw_J = 1.0;

  /** Get the value of \gamma. */
  double gamma = fixedSize * ( 1.0 - this->m_Alpha );

  /** Get the number of edges in moving or joint minimal spanning tree. */
  const unsigned long edgeNumbers = this->m_MSTThreaderMetricParameters.st_MovingSRC.size();

  /** Get the moving or joint edges for this thread. */
  const unsigned long nrOfEdgesPerThreads
    = static_cast< unsigned long >( vcl_ceil( static_cast< double >( edgeNumbers )
    / static_cast< double >( this->m_NumberOfThreads ) ) );

  unsigned long pos_begin = nrOfEdgesPerThreads * threadId;
  unsigned long pos_end   = nrOfEdgesPerThreads * ( threadId + 1 );
  pos_begin = ( pos_begin > edgeNumbers ) ? edgeNumbers : pos_begin;
  pos_end   = ( pos_end > edgeNumbers ) ? edgeNumbers : pos_end;

  /** Get a handle to the pre-allocated derivative for the current thread.
   * The initialization is performed at the beginning of each resolution in
   * InitializeThreadingParameters(), and at the end of each iteration in
   * AfterThreadedGetValueAndDerivative() and the accumulate functions.
   */
  DerivativeType & dMST_M = this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_MDerivative;

  /** Loop over all edges in moving minimum spanning tree. */
  mstLength_M = 0.0;
  for ( unsigned long i = pos_begin; i < pos_end; ++i )
    {
	NodeIdentifierType src = this->m_MSTThreaderMetricParameters.st_MovingSRC[ i ];
    NodeIdentifierType target = this->m_MSTThreaderMetricParameters.st_MovingTAR[ i ];
	EdgeWeightType weight = this->m_MSTThreaderMetricParameters.st_MovingEW[ i ];

    distance_M = vcl_pow( weight, 2 );
    if ( distance_M < MIN_DISTANCE ) 
      continue;
    
    mstLength_M += vcl_pow( weight, gamma );

	SpatialDerivativeType movingSrcSparse, movingTargetSparse;
    movingSrcSparse = this->m_MSTThreaderMetricParameters.st_SpatialDerivativesContainer[ src ] * 
	  this->m_MSTThreaderMetricParameters.st_JacobianContainer[ src ];
    movingTargetSparse = this->m_MSTThreaderMetricParameters.st_SpatialDerivativesContainer[ target ] * 
	  this->m_MSTThreaderMetricParameters.st_JacobianContainer[ target ];

    this->m_MSTThreaderMetricParameters.st_MovingListSample->GetMeasurementVector( src, movingSrcArray );
    this->m_MSTThreaderMetricParameters.st_MovingListSample->GetMeasurementVector( target, movingTargetArray );

    for ( unsigned int j = 0; j < movingSize; ++j )
      {
      diff_M = movingSrcArray[ j ] - movingTargetArray[ j ];
      coef_M = gamma * vcl_pow( distance_M, gamma/2.0-1.0 ) * diff_M;
      
      for ( unsigned int k = 0; k < this->m_MSTThreaderMetricParameters.st_JacobianIndicesContainer[ src ].size(); ++k )
        {
		dMST_M[this->m_MSTThreaderMetricParameters.st_JacobianIndicesContainer[src][k]] += ( coef_M * movingSrcSparse[j][k] );
        }

      for ( unsigned int k = 0; k < this->m_MSTThreaderMetricParameters.st_JacobianIndicesContainer[ target ].size(); ++k )
        {
	    dMST_M[this->m_MSTThreaderMetricParameters.st_JacobianIndicesContainer[target][k]] -= ( coef_M * movingTargetSparse[j][k] );
        }
      }
    } // end looping over all edges
  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_MValue = mstLength_M;
  
  /** Get a handle to the pre-allocated derivative for the current thread.
   * The initialization is performed at the beginning of each resolution in
   * InitializeThreadingParameters(), and at the end of each iteration in
   * AfterThreadedGetValueAndDerivative() and the accumulate functions.
   */
  DerivativeType & dMST_J = this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_JDerivative;

  /** Loop over all edges in joint minimum spanning tree. */
  mstLength_J = 0.0;
  for ( unsigned long i = pos_begin; i < pos_end; ++i )
    {
    NodeIdentifierType src = this->m_MSTThreaderMetricParameters.st_JointSRC[ i ];
    NodeIdentifierType target = this->m_MSTThreaderMetricParameters.st_JointTAR[ i ];
	EdgeWeightType weight = this->m_MSTThreaderMetricParameters.st_JointEW[ i ];

    if ( weight < MIN_DISTANCE ) 
      continue;
    
    mstLength_J += vcl_pow( weight, 2.0 * gamma );
    distance_J = vcl_pow( weight, 2 );

	SpatialDerivativeType jointSrcSparse, jointTargetSparse;
    jointSrcSparse = this->m_MSTThreaderMetricParameters.st_SpatialDerivativesContainer[ src ] * 
	  this->m_MSTThreaderMetricParameters.st_JacobianContainer[ src ];
    jointTargetSparse = this->m_MSTThreaderMetricParameters.st_SpatialDerivativesContainer[ target ] * 
	  this->m_MSTThreaderMetricParameters.st_JacobianContainer[ target ];

    this->m_MSTThreaderMetricParameters.st_JointListSample->GetMeasurementVector( src, jointSrcArray );
    this->m_MSTThreaderMetricParameters.st_JointListSample->GetMeasurementVector( target, jointTargetArray );

    if ( this->m_UsePenaltyWeight )   
	  spw_J = this->m_MSTThreaderMetricParameters.st_JointPEW[ i ];
      
    for ( unsigned int j = 0; j < movingSize; ++j )
      {
      diff_J = jointSrcArray[ j+fixedSize ] - jointTargetArray[ j+fixedSize ];
      coef_J = 2.0 * gamma * vcl_pow( distance_J, gamma-1.0 ) * spw_J * spw_J * diff_J;
      
      for ( unsigned int k = 0; k < this->m_MSTThreaderMetricParameters.st_JacobianIndicesContainer[ src ].size(); ++k )
        {
        dMST_J[this->m_MSTThreaderMetricParameters.st_JacobianIndicesContainer[src][k]] += ( coef_J * jointSrcSparse[j][k] );
        }

      for ( unsigned int k = 0; k < this->m_MSTThreaderMetricParameters.st_JacobianIndicesContainer[ target ].size(); ++k )
        {
        dMST_J[this->m_MSTThreaderMetricParameters.st_JacobianIndicesContainer[target][k]] -= ( coef_J * jointTargetSparse[j][k] );
        }
      }
    } // end looping over all edges
  this->m_MSTGetValueAndDerivativePerThreadVariables[ threadId ].st_JValue = mstLength_J; 

} // end MSTThreadedGetValueAndDerivative()

/**
 * **************** MSTMergeDerivativeThreaderCallback *******
 */

template<class TFixedImage, class TMovingImage>
ITK_THREAD_RETURN_TYPE
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::MSTMergeDerivativeThreaderCallback( void * arg )
{
  MSTThreadInfoType * infoStruct  = static_cast< MSTThreadInfoType * >( arg );
  ThreadIdType        threadID    = infoStruct->ThreadID;

  MSTMultiThreaderParameterType * temp
    = static_cast< MSTMultiThreaderParameterType * >( infoStruct->UserData );

  temp->st_Metric->MSTThreadedMergeDerivative( threadID );
  
  return ITK_THREAD_RETURN_VALUE;

} // end MSTMergeDerivativeThreaderCallback()

/**
 * ******************* MSTThreadedMergeDerivative *******************
 */

template<class TFixedImage, class TMovingImage>
void
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::MSTThreadedMergeDerivative( ThreadIdType threadId ) 
{
  const unsigned long numPar  = this->GetNumberOfParameters();
  const unsigned long subSize = static_cast< unsigned int >(
    vcl_ceil( static_cast< double >( numPar )
    / static_cast< double >( this->m_NumberOfThreads ) ) );
  unsigned long jmin = threadId * subSize;
  unsigned long jmax = ( threadId + 1 ) * subSize;
  jmin = ( jmin > numPar ) ? numPar : jmin;
  jmax = ( jmax > numPar ) ? numPar : jmax;

  /** This thread accumulates all sub-derivatives into a single one, for the
   * range [ jmin, jmax [. Additionally, the sub-derivatives are reset.
   */
  for( unsigned long j = jmin; j < jmax; ++j )
    {
    for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
      {
      this->m_MSTThreaderMetricParameters.st_MSTMDerivative[ j ] += this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_MDerivative[ j ];
      this->m_MSTThreaderMetricParameters.st_MSTJDerivative[ j ] += this->m_MSTGetValueAndDerivativePerThreadVariables[ i ].st_JDerivative[ j ];
      }
    }

} // end MSTThreadedMergeDerivative()

/**
 * ************************ PrintSelf *************************
 */

template <class TFixedImage, class TMovingImage>
void
MSTGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "Alpha: " << this->m_Alpha << std::endl;

} // end PrintSelf()


} // end namespace itk


#endif // end #ifndef _itkMSTGraphAlphaMutualInformationImageToImageMetric_txx

