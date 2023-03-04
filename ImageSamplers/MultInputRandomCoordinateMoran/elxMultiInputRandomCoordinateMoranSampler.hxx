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

#ifndef __elxMultiInputRandomCoordinateMoranSampler_hxx
#define __elxMultiInputRandomCoordinateMoranSampler_hxx

#include "elxMultiInputRandomCoordinateMoranSampler.h"

namespace elastix
{

/**
* ******************* BeforeEachResolution ******************
*/

template< class TElastix >
void
MultiInputRandomCoordinateMoranSampler< TElastix >
::BeforeEachResolution( void )
{
  const unsigned int level
    = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

  /** Set the NumberOfSpatialSamples. */
  unsigned long numberOfSpatialSamples = 5000;
  this->GetConfiguration()->ReadParameter( numberOfSpatialSamples,
    "NumberOfSpatialSamples", this->GetComponentLabel(), level, 0 );
  this->SetNumberOfSamples( numberOfSpatialSamples );

  /** Set up the fixed image interpolator and set the SplineOrder, default value = 1. */
  typename DefaultInterpolatorType::Pointer fixedImageInterpolator
    = DefaultInterpolatorType::New();
  unsigned int splineOrder = 1;
  this->GetConfiguration()->ReadParameter( splineOrder,
    "FixedImageBSplineInterpolationOrder", this->GetComponentLabel(), level, 0 );
  fixedImageInterpolator->SetSplineOrder( splineOrder );
  this->SetInterpolator( fixedImageInterpolator );

  /** Set the UseRandomSampleRegion bool. */
  bool useRandomSampleRegion = false;
  this->GetConfiguration()->ReadParameter( useRandomSampleRegion,
    "UseRandomSampleRegion", this->GetComponentLabel(), level, 0 );
  this->SetUseRandomSampleRegion( useRandomSampleRegion );

  /** Set the SampleRegionSize. */
  if( useRandomSampleRegion )
  {
    InputImageSpacingType sampleRegionSize;
    InputImageSpacingType fixedImageSpacing
      = this->GetElastix()->GetFixedImage()->GetSpacing();
    InputImageSizeType fixedImageSize
      = this->GetElastix()->GetFixedImage()->GetLargestPossibleRegion().GetSize();

    /** Estimate default:
     * sampleRegionSize[i] = min ( fixedImageSizeInMM[i], max_i ( fixedImageSizeInMM[i]/3 ) )
     */
    double maxthird = 0.0;
    for( unsigned int i = 0; i < InputImageDimension; ++i )
    {
      sampleRegionSize[ i ] = ( fixedImageSize[ i ] - 1 ) * fixedImageSpacing[ i ];
      maxthird              = vnl_math_max( maxthird, sampleRegionSize[ i ] / 3.0 );
    }
    for( unsigned int i = 0; i < InputImageDimension; ++i )
    {
      sampleRegionSize[ i ] = vnl_math_min( maxthird, sampleRegionSize[ i ] );
    }

    /** Read user's choice. */
    for( unsigned int i = 0; i < InputImageDimension; ++i )
    {
      this->GetConfiguration()->ReadParameter(
        sampleRegionSize[ i ], "SampleRegionSize",
        this->GetComponentLabel(), level * InputImageDimension + i, 0 );
    }
    this->SetSampleRegionSize( sampleRegionSize );
  }

  /** Read and set the Moran's coefficient image. */
  std::stringstream fmoranNum;
  fmoranNum << level;
  std::string moranImageName = this->GetConfiguration()->GetCommandLineArgument( "-fmoran" + fmoranNum.str() );
  if( !moranImageName.empty() )
  {
	typedef itk::ImageFileReader< InputImageType >    ImageReaderType;
	ImageReaderType::Pointer imageReader = ImageReaderType::New();
    imageReader->SetFileName( moranImageName );

	try
	  {
	  imageReader->Update();
	  }
	catch ( itk::ExceptionObject )
	  {
	  itkExceptionMacro( << "Unable to open the moran coefficient image! " << std::endl );
	  }
	
	InputImageType::Pointer moranCoefImage = imageReader->GetOutput();
	this->SetMoranCoefImage( moranCoefImage );

	elxout << "Reading moran coefficient image has finished! " << std::endl;
  }

}   // end BeforeEachResolution


} // end namespace elastix

#endif // end #ifndef __elxMultiInputRandomCoordinateMoranSampler_hxx
