/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkFastPrimMinimumSpanningTreeGraphFilter.h,v $
  Language:  C++
  Date:      $Date: 2021/08/16 20:30:00 $
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notices for more information.

  =========================================================================*/
#ifndef __itkFastPrimMinimumSpanningTreeGraphFilter_h
#define __itkFastPrimMinimumSpanningTreeGraphFilter_h

#include "itkGraphSource.h"
#include "emdL1.h"
#include "itkMultiThreader.h"

#include "itkArray.h"
#include "itkListSampleCArray.h"

#include "itkBinaryTreeBase.h"
#include "itkBinaryTreeSearchBase.h"
#include "itkANNkDTree.h"
#include "itkANNStandardTreeSearch.h"

namespace itk
{

/** \class FastPrimMinimumSpanningTreeGraphFilter
 * \brief
 *
 * FastPrimMinimumSpanningTreeGraphFilter is the class for image samples
 * that output Graph data for minimum spanning tree. Specifically, this
 * class regards the Euclidean distance of feature vector of two nodes
 * as weight of edge, and uses the Prim's algorithm in multi-threading.
 *
 * \ingroup GraphSources
 **/

template <class TOutputGraph>
class ITK_EXPORT FastPrimMinimumSpanningTreeGraphFilter
  : public GraphSource<TOutputGraph>
{
public:
  /** Standard class typedefs. */
  typedef FastPrimMinimumSpanningTreeGraphFilter     Self;
  typedef GraphSource<TOutputGraph>                  Superclass;
  typedef SmartPointer<Self>                         Pointer;
  typedef SmartPointer<const Self>                   ConstPointer;

  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( FastPrimMinimumSpanningTreeGraphFilter, GraphSource );

  /** Some Graph related typedefs. */
  typedef TOutputGraph                                          GraphType;
  typedef typename GraphType::Pointer                           GraphPointer;
  typedef typename GraphType::NodeIterator                      NodeIteratorType;
  typedef typename GraphType::EdgeIterator                      EdgeIteratorType;
  typedef typename GraphType::GraphTraitsType                   GraphTraitsType;
  typedef typename GraphTraitsType::NodeType                    NodeType;
  typedef typename GraphTraitsType::EdgeType                    EdgeType;
  typedef typename GraphTraitsType::NodePointerType             NodePointerType;
  typedef typename GraphTraitsType::EdgePointerType             EdgePointerType;
  typedef typename GraphTraitsType::NodeIdentifierType          NodeIdentifierType;
  typedef typename GraphTraitsType::NodeWeightType              NodeWeightType;
  typedef typename GraphTraitsType::EdgeWeightType              EdgeWeightType;

  /** Typedefs for the list sample of three images. */
  typedef Array< double >                                       MeasurementVectorType;
  typedef typename Statistics::ListSampleCArray<
	  MeasurementVectorType, double >                           ListSampleType;
  typedef typename ListSampleType::Pointer                      ListSamplePointer;
		
  /** Typedefs for trees and searchers. */
  typedef BinaryTreeBase< ListSampleType >                      BinaryKNNTreeType;
  typedef ANNkDTree< ListSampleType >                           ANNkDTreeType;
  typedef BinaryTreeSearchBase< ListSampleType >                BinaryKNNTreeSearchType;
  typedef ANNStandardTreeSearch< ListSampleType >               ANNStandardTreeSearchType;

  typedef typename BinaryKNNTreeSearchType::IndexArrayType      IndexArrayType;
  typedef typename BinaryKNNTreeSearchType::DistanceArrayType   DistanceArrayType;
  
  /** Define other types. */
  typedef itk::MultiThreader                                    ThreaderType;
  typedef typename ThreaderType::ThreadInfoStruct               ThreadInfoType;

  /** The length of Minimum Spanning Tree. */
  float m_MSTLength;

  /** Set the multi-threading handle. */
  ThreaderType::Pointer m_FPGThreader;

  /** Set the bucket size. */
  itkSetMacro( BucketSize, unsigned int );

  /** Set the splitting rule. */
  itkSetMacro( SplittingRule, std::string );

  /** Set the k nearest neighbours. */
  itkSetMacro( KNearestNeighbours, unsigned int );

  /** Set the error bound. */
  itkSetMacro( ErrorBound, double );
  
  /** Set the list sample. */
  void SetListSample( ListSamplePointer & listSample );

  /** Set the penalty list sample. */
  void SetPenaltyListSample( ListSamplePointer & listSample );
		
  /** Set the multi-threading number. */
  void SetNumberOfThreads( unsigned int nOfThreads );

protected:
  FastPrimMinimumSpanningTreeGraphFilter();
  ~FastPrimMinimumSpanningTreeGraphFilter();

  /** Helper structs that multi-threads the computation of
  * the metric derivative using ITK threads.
  */
  struct FPGMultiThreaderParameterType
  {
	// To give the threads access to all members.
	FastPrimMinimumSpanningTreeGraphFilter * st_Filter;

	// Used for some parameters of accumulating derivatives
	ListSamplePointer st_ListSample;
	ListSamplePointer st_PenaltyListSample;

	EdgeWeightType ** st_PenaltyCoeff;
	EdgeWeightType ** st_AdjacencyWeight;
  };
  mutable FPGMultiThreaderParameterType m_FPGThreaderFilterParameters;

  // test per thread struct with padding and alignment
  struct FPGPenaltyWeightPerThreadStruct
  {
	double * st_srcHist;
	double * st_tarHist;
  };
  itkPadStruct( ITK_CACHE_LINE_ALIGNMENT, FPGPenaltyWeightPerThreadStruct,
	  PaddedFPGPenaltyWeightPerThreadStruct );
  itkAlignedTypedef( ITK_CACHE_LINE_ALIGNMENT, PaddedFPGPenaltyWeightPerThreadStruct,
	  AlignedFPGPenaltyWeightPerThreadStruct );
  mutable AlignedFPGPenaltyWeightPerThreadStruct * m_FPGPenaltyWeightPerThreadVariables;
  mutable ThreadIdType                             m_FPGPenaltyWeightPerThreadVariablesSize;

  void PrintSelf( std::ostream& os, Indent indent ) const;

  void GenerateData();
  void GenerateMinimumSpanningTree();
  void GeneratePenaltyMinimumSpanningTree();
  void GeneratePenaltyMinimumSpanningTreeSingleThreaded();

  void InitializeFPGThreadingParameters();

  static ITK_THREAD_RETURN_TYPE FPGPenaltyWeightThreaderCallback( void * arg );
  void FPGThreadedPenaltyWeight( ThreadIdType threadId );

private:
  FastPrimMinimumSpanningTreeGraphFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
 
  void UpdatePriorityQueue( EdgeWeightType**, NodeIdentifierType, NodeIdentifierType );

  typename BinaryKNNTreeType::Pointer       m_BinaryKNNTree;
  typename BinaryKNNTreeSearchType::Pointer m_BinaryKNNTreeSearcher;

  ListSamplePointer     m_ListSample;
  ListSamplePointer     m_PenaltyListSample;

  unsigned int          m_BucketSize;
  std::string           m_SplittingRule;
  unsigned int          m_KNearestNeighbours;
  double                m_ErrorBound;

  GraphPointer          m_MinimumSpanningTree;

  /** Threading related parameters. */
  unsigned int          m_NumberOfThreads;
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFastPrimMinimumSpanningTreeGraphFilter.txx"
#endif

#endif
