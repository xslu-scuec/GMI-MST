/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef __MultiInputImageRandomCoordinateMoranSampler_txx
#define __MultiInputImageRandomCoordinateMoranSampler_txx

#include "itkMultiInputImageRandomCoordinateMoranSampler.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "vnl/vnl_inverse.h"
#include "itkConfigure.h"

namespace itk
{

/**
 * ******************* Constructor ********************
 */

template< class TInputImage >
MultiInputImageRandomCoordinateMoranSampler< TInputImage >
::MultiInputImageRandomCoordinateMoranSampler()
{
  /** Set the default interpolator. */
  typename DefaultInterpolatorType::Pointer bsplineInterpolator
    = DefaultInterpolatorType::New();
  bsplineInterpolator->SetSplineOrder( 3 );
  this->m_Interpolator = bsplineInterpolator;

  /** Setup the random generator. */
  this->m_RandomGenerator = RandomGeneratorType::GetInstance();

  this->m_UseRandomSampleRegion = false;
  this->m_SampleRegionSize.Fill( 1.0 );

  this->m_MoranCoefImage = NULL;

}   // end Constructor()


/**
 * ******************* GenerateData *******************
 */

template< class TInputImage >
void
MultiInputImageRandomCoordinateMoranSampler< TInputImage >
::GenerateData( void )
{
  /** Check. */
  if( !this->CheckInputImageRegions() )
  {
    itkExceptionMacro( << "ERROR: at least one of the InputImageRegions "
                       << "is not a subregion of the LargestPossibleRegion" );
  }

  /** Get handles to the input image, output sample container. */
  InputImageConstPointer inputImage = this->GetInput();
  typename ImageSampleContainerType::Pointer sampleContainer = this->GetOutput();
  typename InterpolatorType::Pointer interpolator            = this->GetInterpolator();

  /** Set up the interpolator. */
  interpolator->SetInputImage( inputImage );

  /** Get the intersection of all sample regions. */
  InputImageContinuousIndexType smallestContIndex;
  InputImageContinuousIndexType largestContIndex;
  this->GenerateSampleRegion(
    smallestContIndex, largestContIndex );

  /** Reserve memory for the output. */
  sampleContainer->Reserve( this->GetNumberOfSamples() );

  /** Setup an iterator over the output, which is of ImageSampleContainerType. */
  typename ImageSampleContainerType::Iterator iter;
  typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

  InputImageContinuousIndexType sampleContIndex;
  /** Fill the sample container. */
  if( !this->m_MoranCoefImage )
  {
	//elxout << "Could not find the moran coefficient image! " << std::endl;

	/** Start looping over the sample container. */
	for( iter = sampleContainer->Begin(); iter != end; ++iter )
	{
	  /** Make a reference to the current sample in the container. */
	  InputImagePointType &  samplePoint = ( *iter ).Value().m_ImageCoordinates;
	  ImageSampleValueType & sampleValue = ( *iter ).Value().m_ImageValue;

	  /** Generate a point in the input image region. */
	  this->GenerateRandomCoordinate( smallestContIndex, largestContIndex, sampleContIndex );

	  /** Convert to point */
	  inputImage->TransformContinuousIndexToPhysicalPoint( sampleContIndex, samplePoint );

	  /** Compute the value at the contindex. */
	  sampleValue = static_cast< ImageSampleValueType >(
	    this->m_Interpolator->EvaluateAtContinuousIndex( sampleContIndex ) );

	} // end for loop
  }   // end if no mask
  else
  {
    float moranCoefThreshold = this->CalculateMoranThreshold( smallestContIndex, largestContIndex );

	//elxout << "The moran coefficient threshold is: " << moranCoefThreshold << std::endl;

    /** Set up some variable that are used to make sure we are not forever
     * walking around on this image, trying to look for valid samples.
     */
    unsigned long numberOfSamplesTried        = 0;
    unsigned long maximumNumberOfSamplesToTry = 10 * this->GetNumberOfSamples();

    /** Start looping over the sample container. */
    for( iter = sampleContainer->Begin(); iter != end; ++iter )
    {
      /** Make a reference to the current sample in the container. */
      InputImagePointType &  samplePoint = ( *iter ).Value().m_ImageCoordinates;
      ImageSampleValueType & sampleValue = ( *iter ).Value().m_ImageValue;

      /** Walk over the image until we find a valid point. */
      do
      {
        /** Check if we are not trying eternally to find a valid point. */
        ++numberOfSamplesTried;
        if( numberOfSamplesTried > maximumNumberOfSamplesToTry )
        {
          /** Squeeze the sample container to the size that is still valid. */
          typename ImageSampleContainerType::iterator stlnow = sampleContainer->begin();
          typename ImageSampleContainerType::iterator stlend = sampleContainer->end();
          stlnow                                            += iter.Index();
          sampleContainer->erase( stlnow, stlend );
          itkExceptionMacro( << "Could not find enough image samples within "
                             << "reasonable time. Probably the region is too small" );
        }

        /** Generate a point in the input image region. */
        this->GenerateRandomCoordinate( smallestContIndex, largestContIndex, sampleContIndex );
        inputImage->TransformContinuousIndexToPhysicalPoint( sampleContIndex, samplePoint );
      }
      while( this->IsInsideStructureRegion( samplePoint, moranCoefThreshold ) );

      /** Compute the value at the contindex. */
      sampleValue = static_cast< ImageSampleValueType >(
        this->m_Interpolator->EvaluateAtContinuousIndex( sampleContIndex ) );

    } // end for loop
  }   // end if mask

}   // end GenerateData()


/**
 * ******************* GenerateSampleRegion *******************
 */

template< class TInputImage >
void
MultiInputImageRandomCoordinateMoranSampler< TInputImage >::GenerateSampleRegion(
  InputImageContinuousIndexType & smallestContIndex,
  InputImageContinuousIndexType & largestContIndex )
{
  /** Get handles to the number of inputs and regions. */
  const unsigned int numberOfInputs  = this->GetNumberOfInputs();
  const unsigned int numberOfRegions = this->GetNumberOfInputImageRegions();

  /** Check. */
  if( numberOfRegions != numberOfInputs && numberOfRegions != 1 )
  {
    itkExceptionMacro( << "ERROR: The number of regions should be 1 or the number of inputs." );
  }

  typedef typename InputImageType::DirectionType DirectionType;
  DirectionType dir0 = this->GetInput( 0 )->GetDirection();
  typename DirectionType::InternalMatrixType dir0invtemp
    = vnl_inverse( dir0.GetVnlMatrix() );
  DirectionType dir0inv( dir0invtemp );
  for( unsigned int i = 1; i < numberOfInputs; ++i )
  {
    DirectionType diri = this->GetInput( i )->GetDirection();
    if( diri != dir0 )
    {
      itkExceptionMacro( << "ERROR: All input images should have the same direction cosines matrix." );
    }
  }

  /** Initialize the smallest and largest point. */
  InputImagePointType smallestPoint;
  InputImagePointType largestPoint;
  smallestPoint.Fill( NumericTraits< InputImagePointValueType >::NonpositiveMin() );
  largestPoint.Fill( NumericTraits< InputImagePointValueType >::max() );

  /** Determine the intersection of all regions, assuming identical direction cosines,
   * but possibly different origin/spacing.
   * \todo: test this really carefully!
   */
  InputImageSizeType unitSize;
  unitSize.Fill( 1 );
  for( unsigned int i = 0; i < numberOfRegions; ++i )
  {
    /** Get the outer indices. */
    InputImageIndexType smallestIndex
      = this->GetInputImageRegion( i ).GetIndex();
    InputImageIndexType largestIndex
      = smallestIndex + this->GetInputImageRegion( i ).GetSize() - unitSize;

    /** Convert to points */
    InputImagePointType smallestImagePoint;
    InputImagePointType largestImagePoint;
    this->GetInput( i )->TransformIndexToPhysicalPoint(
      smallestIndex, smallestImagePoint );
    this->GetInput( i )->TransformIndexToPhysicalPoint(
      largestIndex, largestImagePoint );

    /** apply inverse direction, so that next max-operation makes sense. */
    smallestImagePoint = dir0inv * smallestImagePoint;
    largestImagePoint  = dir0inv * largestImagePoint;

    /** Determine intersection. */
    for( unsigned int j = 0; j < InputImageDimension; ++j )
    {
      /** Get the largest smallest point. */
      smallestPoint[ j ] = vnl_math_max( smallestPoint[ j ], smallestImagePoint[ j ] );

      /** Get the smallest largest point. */
      largestPoint[ j ] = vnl_math_min( largestPoint[ j ], largestImagePoint[ j ] );
    }
  }

  /** Convert to continuous index in input image 0. */
  smallestPoint = dir0 * smallestPoint;
  largestPoint  = dir0 * largestPoint;
  this->GetInput( 0 )->TransformPhysicalPointToContinuousIndex( smallestPoint, smallestContIndex );
  this->GetInput( 0 )->TransformPhysicalPointToContinuousIndex( largestPoint, largestContIndex );

  /** Support for localised version of metric. */
  if( this->GetUseRandomSampleRegion() )
  {
    /** Convert sampleRegionSize to continuous index space */
    typedef typename InputImageContinuousIndexType::VectorType CIndexVectorType;
    CIndexVectorType sampleRegionSize;
    for( unsigned int i = 0; i < InputImageDimension; ++i )
    {
      sampleRegionSize[ i ] = this->GetSampleRegionSize()[ i ]
        / this->GetInput()->GetSpacing()[ i ];
    }
    InputImageContinuousIndexType maxSmallestContIndex = largestContIndex;
    maxSmallestContIndex -= sampleRegionSize;
    this->GenerateRandomCoordinate( smallestContIndex, maxSmallestContIndex, smallestContIndex );
    largestContIndex  = smallestContIndex;
    largestContIndex += sampleRegionSize;
  }
 
}   // end GenerateSampleRegion()


/**
 * ******************* GenerateRandomCoordinate *******************
 */

template< class TInputImage >
void
MultiInputImageRandomCoordinateMoranSampler< TInputImage >::GenerateRandomCoordinate(
  const InputImageContinuousIndexType & smallestContIndex,
  const InputImageContinuousIndexType & largestContIndex,
  InputImageContinuousIndexType &       randomContIndex )
{
  for( unsigned int i = 0; i < InputImageDimension; ++i )
  {
    randomContIndex[ i ] = static_cast< InputImagePointValueType >(
      this->m_RandomGenerator->GetUniformVariate(
      smallestContIndex[ i ], largestContIndex[ i ] ) );
  }
}   // end GenerateRandomCoordinate()


/**
* ******************* CalculateMoranThreshold *******************
*/

template< class TInputImage >
float
MultiInputImageRandomCoordinateMoranSampler< TInputImage >::CalculateMoranThreshold(
	const InputImageContinuousIndexType & smallestContIndex,
	const InputImageContinuousIndexType & largestContIndex )
{
	/** Prepare for the sample region. */
	InputImageType::IndexType regionIndex;
	for( unsigned int i = 0; i < InputImageDimension; ++i )
	{
		if( vnl_math_floor(smallestContIndex[i]) < 0 )
			regionIndex[ i ] = 0;
		else
			regionIndex[ i ] = vnl_math_floor( smallestContIndex[i] );
	}

	unsigned long regionLen = 1;
	InputImageType::SizeType regionSize;
	InputImageType::SizeType inputImageSize = this->GetInput()->GetLargestPossibleRegion().GetSize();
	for( unsigned int i = 0; i < InputImageDimension; ++i )
	{
		regionSize[ i ] = vnl_math_floor( this->GetSampleRegionSize()[ i ]
		/ this->GetInput()->GetSpacing()[ i ] + 1 );
		if( (regionIndex[i]+regionSize[i]-1) > (inputImageSize[i]-1) )
			regionSize[ i ] = inputImageSize[ i ] - regionIndex[ i ];
		regionLen *= regionSize[ i ];
	}

	InputImageType::RegionType sampleRegion;
	sampleRegion.SetIndex( regionIndex );
	sampleRegion.SetSize( regionSize );

	ImageIteratorType moranIt( this->m_MoranCoefImage, sampleRegion );
	vnl_vector< double > sampleRegionVec( regionLen, 0.0 );
	unsigned long sampleNum = 0;

	/** Start looping over the sample region. */
	for( moranIt.GoToBegin(); !moranIt.IsAtEnd(); ++moranIt )
	{
		sampleRegionVec[ sampleNum ] = moranIt.Get();
		sampleNum++;
	}

	/** Generate the threshold of the Moran's I test. */
	double sampleVecMean = sampleRegionVec.mean();
	for( unsigned long i = 0; i < sampleNum; ++i )
	{
		sampleRegionVec[ i ] -= sampleVecMean;
	}
	double sampleVar = dot_product( sampleRegionVec, sampleRegionVec );

	return vcl_sqrt( sampleVar / sampleNum );

}   // end CalculateMoranThreshold()


/**
 * ******************* IsInsideStructureRegion *******************
 */

template< class TInputImage >
bool
MultiInputImageRandomCoordinateMoranSampler< TInputImage >::IsInsideStructureRegion(
  const InputImagePointType & samplePoint, float moranThreshold )
{
  InputImageType::IndexType sampleIndex;

  this->m_MoranCoefImage->TransformPhysicalPointToIndex( samplePoint, sampleIndex );
  
  const bool insideStrRegion
		= ( vnl_math_abs( this->m_MoranCoefImage->GetPixel( sampleIndex ) ) <= moranThreshold );

  return insideStrRegion;
	 
}   // end IsInsideStructureRegion()


/**
 * ******************* PrintSelf *******************
 */

template< class TInputImage >
void
MultiInputImageRandomCoordinateMoranSampler< TInputImage >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "Interpolator: " << this->m_Interpolator.GetPointer() << std::endl;
  os << indent << "RandomGenerator: " << this->m_RandomGenerator.GetPointer() << std::endl;

}   // end PrintSelf


} // end namespace itk

#endif // end #ifndef __MultiInputImageRandomCoordinateMoranSampler_txx
