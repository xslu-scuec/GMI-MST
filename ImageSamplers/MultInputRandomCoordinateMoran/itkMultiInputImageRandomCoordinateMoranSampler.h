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
#ifndef __MultiInputImageRandomCoordinateMoranSampler_h
#define __MultiInputImageRandomCoordinateMoranSampler_h

#include "itkImageRandomSamplerBase.h"
#include "itkInterpolateImageFunction.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"

namespace itk
{

/** \class MultiInputImageRandomCoordinateMoranSampler
 *
 * \brief Samples an image by randomly composing a set of physical coordinates, 
 * which selected by an image after Moran's I test.
 *
 * This image sampler generates not only samples that correspond with
 * pixel locations, but selects points in physical space.
 *
 * \ingroup ImageSamplers
 */

template< class TInputImage >
class MultiInputImageRandomCoordinateMoranSampler :
  public ImageRandomSamplerBase< TInputImage >
{
public:

  /** Standard ITK-stuff. */
  typedef MultiInputImageRandomCoordinateMoranSampler Self;
  typedef ImageRandomSamplerBase< TInputImage >       Superclass;
  typedef SmartPointer< Self >                        Pointer;
  typedef SmartPointer< const Self >                  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( MultiInputImageRandomCoordinateMoranSampler, ImageRandomSamplerBase );

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass::DataObjectPointer            DataObjectPointer;
  typedef typename Superclass::OutputVectorContainerType    OutputVectorContainerType;
  typedef typename Superclass::OutputVectorContainerPointer OutputVectorContainerPointer;
  typedef typename Superclass::InputImageType               InputImageType;
  typedef typename Superclass::InputImagePointer            InputImagePointer;
  typedef typename Superclass::InputImageConstPointer       InputImageConstPointer;
  typedef typename Superclass::InputImageRegionType         InputImageRegionType;
  typedef typename Superclass::InputImagePixelType          InputImagePixelType;
  typedef typename Superclass::ImageSampleType              ImageSampleType;
  typedef typename Superclass::ImageSampleContainerType     ImageSampleContainerType;
  typedef typename Superclass::MaskType                     MaskType;
  typedef typename Superclass::InputImageSizeType           InputImageSizeType;
  typedef typename InputImageType::SpacingType              InputImageSpacingType;
  typedef typename Superclass::InputImageIndexType          InputImageIndexType;
  typedef typename Superclass::InputImagePointType          InputImagePointType;
  typedef typename Superclass::InputImagePointValueType     InputImagePointValueType;
  typedef typename Superclass::ImageSampleValueType         ImageSampleValueType;
  
  /** The input image dimension. */
  itkStaticConstMacro( InputImageDimension, unsigned int,
    Superclass::InputImageDimension );

  /** This image sampler samples the image on physical coordinates and thus
   * needs an interpolator.
   */
  typedef double                                                                  CoordRepType;
  typedef InterpolateImageFunction< InputImageType, CoordRepType >                InterpolatorType;
  typedef typename InterpolatorType::Pointer                                      InterpolatorPointer;
  typedef BSplineInterpolateImageFunction< InputImageType, CoordRepType, double > DefaultInterpolatorType;

  typedef itk::ImageRegionIteratorWithIndex< InputImageType >                     ImageIteratorType;

  /** The random number generator used to generate random coordinates. */
  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator RandomGeneratorType;
  typedef typename RandomGeneratorType::Pointer                  RandomGeneratorPointer;

  /** Set/Get the interpolator. A 3rd order B-spline interpolator is used by default. */
  itkSetObjectMacro( Interpolator, InterpolatorType );
  itkGetObjectMacro( Interpolator, InterpolatorType );

  /** Set/Get the sample region size (in mm). Only needed when UseRandomSampleRegion==true;
   * default: filled with ones.
   */
  itkSetMacro( SampleRegionSize, InputImageSpacingType );
  itkGetConstReferenceMacro( SampleRegionSize, InputImageSpacingType );

  /** Set/Get the image through Moran's I detection. */
  itkSetObjectMacro( MoranCoefImage, InputImageType );
  itkGetConstObjectMacro( MoranCoefImage, InputImageType );

  /** Set/Get whether to use randomly selected sample regions, or just the whole image
   * Default: false. */
  itkGetConstMacro( UseRandomSampleRegion, bool );
  itkSetMacro( UseRandomSampleRegion, bool );

protected:

  typedef typename InterpolatorType::ContinuousIndexType InputImageContinuousIndexType;

  /** The constructor. */
  MultiInputImageRandomCoordinateMoranSampler();

  /** The destructor. */
  virtual ~MultiInputImageRandomCoordinateMoranSampler() {}

  /** PrintSelf. */
  void PrintSelf( std::ostream & os, Indent indent ) const;

  /** Function that does the work. */
  virtual void GenerateData( void );

  /** Generate a point randomly in a bounding box.
   * This method can be overwritten in subclasses if a different distribution is desired. */
  virtual void GenerateRandomCoordinate(
    const InputImageContinuousIndexType & smallestContIndex,
    const InputImageContinuousIndexType & largestContIndex,
    InputImageContinuousIndexType &       randomContIndex );

  InterpolatorPointer    m_Interpolator;
  RandomGeneratorPointer m_RandomGenerator;
  InputImageSpacingType  m_SampleRegionSize;
  InputImagePointer      m_MoranCoefImage;

  /** Generate the two corners of a sampling region. */
  virtual void GenerateSampleRegion(
    InputImageContinuousIndexType & smallestContIndex,
    InputImageContinuousIndexType & largestContIndex );

  /** Calculate the threshold of the Moran's I test within a sampling region. */
  float CalculateMoranThreshold(
	const InputImageContinuousIndexType & smallestContIndex,
	const InputImageContinuousIndexType & largestContIndex );

  /** Whether the sampleContIndex is inside the structure region or not. */
  bool IsInsideStructureRegion(
    const InputImagePointType & samplePoint,
	float moranThreshold );

private:

  /** The private constructor. */
  MultiInputImageRandomCoordinateMoranSampler( const Self & );           // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );             // purposely not implemented

  /** Member variables. */
  bool m_UseRandomSampleRegion;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiInputImageRandomCoordinateMoranSampler.hxx"
#endif

#endif // end #ifndef __MultiInputImageRandomCoordinateMoranSampler_h
