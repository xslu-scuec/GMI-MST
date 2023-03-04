/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxMSTGraphAlphaMutualInformationMetric_HXX__
#define __elxMSTGraphAlphaMutualInformationMetric_HXX__

#include "elxMSTGraphAlphaMutualInformationMetric.h"

#include "itkBSplineInterpolateImageFunction.h"
#include <string>


namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void
MSTGraphAlphaMutualInformationMetric<TElastix>
::Initialize( void ) throw (itk::ExceptionObject)
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of MSTGraphAlphaMutualInformation metric took: "
    << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void MSTGraphAlphaMutualInformationMetric<TElastix>
::BeforeRegistration( void )
{
  /** Get and set alpha, from alpha - MI. */
  double alpha = 0.5;
  this->m_Configuration->ReadParameter( alpha, "Alpha", 0 );
  this->SetAlpha( alpha );

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void MSTGraphAlphaMutualInformationMetric<TElastix>
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level
    = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Get and set the parameters for the fast MST algorithm. */

  /** Get and set the bucket size. */
  unsigned int bucketSize = 50;
  this->m_Configuration->ReadParameter( bucketSize, "BucketSize", 0 );
  this->m_Configuration->ReadParameter( bucketSize, "BucketSize", level, true );
  this->SetBucketSize( bucketSize );

  /** Get and set the splitting rule for the trees. */
  std::string splittingRule = "ANN_KD_STD";
  this->m_Configuration->ReadParameter( splittingRule, "SplittingRule", 0 );
  this->m_Configuration->ReadParameter( splittingRule, "SplittingRule", level, true );
  this->SetSplittingRule( splittingRule );

  /** Get and set the k nearest neighbours. */
  unsigned int kNearestNeighbours = 100;
  this->m_Configuration->ReadParameter( kNearestNeighbours,"KNearestNeighbours", 0 );
  this->m_Configuration->ReadParameter( kNearestNeighbours,"KNearestNeighbours", level, true );
  this->SetKNearestNeighbours( kNearestNeighbours );

  /** Get and set the error bound. */
  double errorBound = 0.0;
  this->m_Configuration->ReadParameter( errorBound, "ErrorBound", 0 );
  this->m_Configuration->ReadParameter( errorBound, "ErrorBound", level, true );
  this->SetErrorBound( errorBound );

  /** Get and set bool variable which decides to use penalty weight or not. */
  bool usePenaltyWeight = false;
  this->m_Configuration->ReadParameter( usePenaltyWeight, "UsePenaltyWeight", 0 );
  this->m_Configuration->ReadParameter( usePenaltyWeight, "UsePenaltyWeight", level, true );
  this->SetUsePenaltyWeight( usePenaltyWeight );

  /** Get and set the number of penalty images. */
  unsigned int numberOfPenaltyImages = 0;
  this->m_Configuration->ReadParameter( numberOfPenaltyImages, "NumberOfPenaltyImages", 0 );
  this->m_Configuration->ReadParameter( numberOfPenaltyImages, "NumberOfPenaltyImages", level, true );
  this->SetNumberOfPenaltyImages( numberOfPenaltyImages );

  /** Read and set the penalty weight images. */
  if ( usePenaltyWeight )
    {
	if ( !numberOfPenaltyImages )
	  {
	  itkExceptionMacro( << "The number of penalty images is 0! " << std::endl );
	  }
	else
	  {
	  std::vector< FixedImageConstPointer > * const penaltyImgs = new std::vector< FixedImageConstPointer >();
	  typedef itk::ImageFileReader< FixedImageType > ImageReaderType;  
	  for ( unsigned int i = 0; i < numberOfPenaltyImages; ++i )
		{
		std::stringstream pwNum;
		pwNum << ( numberOfPenaltyImages * level + i );
		std::string penaltyImageName = this->m_Configuration->GetCommandLineArgument( "-pw" + pwNum.str() );
		ImageReaderType::Pointer imageReader = ImageReaderType::New();
		imageReader->SetFileName( penaltyImageName );

		try
		  {
		  imageReader->Update();
		  }
		catch ( itk::ExceptionObject )
		  {
		  itkExceptionMacro( << "Unable to open the penalty image! " << std::endl );
		  }

		FixedImageConstPointer pImgElem = imageReader->GetOutput();
		penaltyImgs->push_back( pImgElem );
		}
	  this->SetPenaltyImages( penaltyImgs );
	  elxout << "Reading the penalty images has finished!  " << std::endl;
	  }
    }

} // end BeforeEachResolution()


} // end namespace elastix


#endif // end #ifndef __elxMSTGraphAlphaMutualInformationMetric_HXX__
