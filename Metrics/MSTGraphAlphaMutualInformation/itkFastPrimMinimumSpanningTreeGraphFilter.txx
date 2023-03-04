/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkFastPrimMinimumSpanningTreeGraphFilter.txx,v $
  Language:  C++
  Date:      $Date: 2020/08/16 20:30:00 $ 
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkFastPrimMinimumSpanningTreeGraphFilter_txx
#define __itkFastPrimMinimumSpanningTreeGraphFilter_txx

#include "itkFastPrimMinimumSpanningTreeGraphFilter.h"
#include "itkTimeProbe.h"

namespace itk
{

/**
 *
 */
template <class TOutputGraph>
FastPrimMinimumSpanningTreeGraphFilter<TOutputGraph>
::FastPrimMinimumSpanningTreeGraphFilter()
{
  this->m_ListSample = NULL;
  this->m_PenaltyListSample = NULL;

  this->m_BinaryKNNTree = NULL;
  this->m_BinaryKNNTreeSearcher = NULL;

  this->m_MSTLength = 0.0;
  this->m_MinimumSpanningTree = NULL;

  this->m_FPGThreader = NULL;
  this->m_NumberOfThreads = 1;

  this->m_FPGThreaderFilterParameters.st_Filter = this;

  this->m_FPGPenaltyWeightPerThreadVariables     = NULL;
  this->m_FPGPenaltyWeightPerThreadVariablesSize = 0;

}
   
/**
 *
 */
template <class TOutputGraph>
FastPrimMinimumSpanningTreeGraphFilter<TOutputGraph>
::~FastPrimMinimumSpanningTreeGraphFilter()
{
  delete[] this->m_FPGPenaltyWeightPerThreadVariables;

}

/**
 *
 */
template <class TOutputGraph>
void
FastPrimMinimumSpanningTreeGraphFilter<TOutputGraph>
::SetListSample( ListSamplePointer & listSample )
{
  this->m_ListSample = listSample;

}

/**
 *
 */
template <class TOutputGraph>
void
FastPrimMinimumSpanningTreeGraphFilter<TOutputGraph>
::SetPenaltyListSample( ListSamplePointer & listSample )
{
  this->m_PenaltyListSample = listSample;

}

/**
 *
 */
template <class TOutputGraph>
void
FastPrimMinimumSpanningTreeGraphFilter<TOutputGraph>
::SetNumberOfThreads( unsigned int nOfThreads )
{
  this->m_NumberOfThreads = nOfThreads;

}

/**
 *
 */
template <class TOutputGraph>
void
FastPrimMinimumSpanningTreeGraphFilter<TOutputGraph>
::GenerateData( void )
{
  if ( this->m_PenaltyListSample )
    {
    this->GeneratePenaltyMinimumSpanningTree();
    }
  else
    {
    this->GenerateMinimumSpanningTree();
    }

}

/**
 *
 */
template <class TOutputGraph>
void
FastPrimMinimumSpanningTreeGraphFilter<TOutputGraph>
::GenerateMinimumSpanningTree( void )
{
  /** Prepare thw weighted adjacency matrix from listsample. */
  NodeIdentifierType i, j;
  MeasurementVectorType srcArray;
  IndexArrayType       indices;
  DistanceArrayType    distances;
	  
  /** Get size of the list sample. */
  unsigned long numSamplePoints = this->m_ListSample->GetActualSize();

  /** Creat memory space for adjacencyWeight. */
  EdgeWeightType ** adjacencyWeight = new EdgeWeightType * [ numSamplePoints ];
  if ( adjacencyWeight == NULL )
    {
    itkExceptionMacro( "Have not enough space. " );
    }
  for ( i = 0; i < numSamplePoints; ++i )
    {
    adjacencyWeight[ i ] = new EdgeWeightType [ numSamplePoints ];
    if ( adjacencyWeight[ i ] == NULL )
      {
      itkExceptionMacro( "Have not enough space. " );
      }
    for ( j = 0; j < numSamplePoints; ++j )
      {
      adjacencyWeight[i][j] = itk::NumericTraits< EdgeWeightType >::max();
      }
    }

  /** Create tree and searcher, and choose the algorithm of KNN. */
  typename ANNkDTreeType::Pointer tmpPtr = ANNkDTreeType::New();
  tmpPtr->SetBucketSize( this->m_BucketSize );
  tmpPtr->SetSplittingRule( this->m_SplittingRule );
    
  typename ANNStandardTreeSearchType::Pointer tmpPtrSearch = ANNStandardTreeSearchType::New();
  tmpPtrSearch->SetKNearestNeighbors( this->m_KNearestNeighbours );
  tmpPtrSearch->SetErrorBound( this->m_ErrorBound );  

  /** Generate the tree for the image samples. */
  this->m_BinaryKNNTree = tmpPtr;
  this->m_BinaryKNNTree->SetSample( this->m_ListSample );
  this->m_BinaryKNNTree->GenerateTree();

  /** Initialize tree searchers. */
  this->m_BinaryKNNTreeSearcher = tmpPtrSearch;
  this->m_BinaryKNNTreeSearcher->SetBinaryTree( this->m_BinaryKNNTree );

  /** Construct the adjacency matrix by edge weights. */
  for ( i = 0; i < numSamplePoints; ++i )
	{
	/** Search for the k nearest neighbours of the current sample point. */
    this->m_ListSample->GetMeasurementVector( i, srcArray );
    this->m_BinaryKNNTreeSearcher->Search( srcArray, indices, distances );

	for ( j = 0; j < this->m_KNearestNeighbours; ++j )
	  {
	  if ( indices[j] == i )
        continue;

	  adjacencyWeight[i][indices[j]] = vcl_sqrt( distances[j] );
	  adjacencyWeight[indices[j]][i] = vcl_sqrt( distances[j] );
	  }
	}

  /** Start going loop to search Minimum Spanning Tree by the Prim's algorithm. */
  NodeIdentifierType components = 0;
  NodeIdentifierType src, temp, count;
 
  NodePointerType node = NULL;
  EdgePointerType edge = NULL;
  EdgeWeightType minw;
  
  /** Create the new graph. */
  this->m_MinimumSpanningTree = this->GetOutput();

  /** Reserve memory for m_Nodes and m_Edges of minimum spanning tree. */
  this->m_MinimumSpanningTree->GetNodeContainer()->Reserve( numSamplePoints );
  this->m_MinimumSpanningTree->GetEdgeContainer()->Reserve( numSamplePoints - 1 );

  /** Create node and edge iterators of minimum spanning tree. */
  NodeIteratorType nodeMSTIt( this->m_MinimumSpanningTree );
  EdgeIteratorType edgeMSTIt( this->m_MinimumSpanningTree );
 
  /** Update node information for minimum spanning tree. */
  for ( i = 0, nodeMSTIt.GoToBegin(); !nodeMSTIt.IsAtEnd(); ++i, ++nodeMSTIt )
    {
    node = nodeMSTIt.GetPointer();

    node->Identifier = i;
    node->Rank       = 0;
    node->Parent     = i;
	node->Weight     = itk::NumericTraits< EdgeWeightType >::max();
    }

  src = 0;
  this->m_MinimumSpanningTree->GetNodePointer( src )->Rank = 1;
  edgeMSTIt.GoToBegin();
  count = 0;
  components = numSamplePoints - 1;
  while ( components > 0 )
    {
    /** Update the parent and weight parameters of nodeouters. */
    UpdatePriorityQueue( adjacencyWeight, src, numSamplePoints );
   
    /** Find the lightest edge from nodeouters to nodeinners. */
    minw = itk::NumericTraits<EdgeWeightType>::max();
    for ( i = 0; i < numSamplePoints; ++i )
      {
      node = this->m_MinimumSpanningTree->GetNodePointer( i );
      if ( node->Rank == 0 )
        {
        if ( node->Weight < minw )
          {
          minw = node->Weight;
          temp = i;
          }
        } 
      }
    this->m_MinimumSpanningTree->GetNodePointer( temp )->Rank = 1;
  
    /** Create a new edge from the lightest edge. */
	edge = edgeMSTIt.GetPointer();
    edge->Identifier = count;
    edge->SourceIdentifier = this->m_MinimumSpanningTree->GetNodePointer( temp )->Parent;
    edge->TargetIdentifier = temp;
    edge->Weight = minw;
    
    src = temp;
    this->m_MSTLength += minw;
    components--;
    count++;
	++edgeMSTIt;
    }

  /** Release the memory space of adjacencyWeight. */
  for ( i = 0; i < numSamplePoints; ++i )
	{
	delete []adjacencyWeight[ i ];
	adjacencyWeight[ i ] = NULL;
	}
  delete []adjacencyWeight;
  adjacencyWeight = NULL;

}

/**
 *
 */
template <class TOutputGraph>
void
FastPrimMinimumSpanningTreeGraphFilter<TOutputGraph>
::GeneratePenaltyMinimumSpanningTreeSingleThreaded( void )
{
  /** Prepare the weighted adjacency matrix from listsample. */
  NodeIdentifierType i, j;
  MeasurementVectorType srcArray, tarArray, diffArray;
  MeasurementVectorType srcPArray, tarPArray;
  EdgeWeightType featDist, penaltyCoef;
	  
  /** Get size of the list sample. */
  unsigned long numSamplePoints = this->m_ListSample->GetActualSize();
  unsigned int numPenaltyFeats = this->m_PenaltyListSample->GetMeasurementVectorSize();

  EmdL1                emdl1;
  unsigned int         histn1 = 4;

  /** Creat memory space for penaltyCoeff and adjacencyWeight. */
  EdgeWeightType ** penaltyCoeff = new EdgeWeightType * [ numSamplePoints ];
  if ( penaltyCoeff == NULL )
    {
    itkExceptionMacro( "Have not enough space. " );
    }
  for ( i = 0; i < numSamplePoints; ++i )
    {
    penaltyCoeff[ i ] = new EdgeWeightType [ numSamplePoints ];
    if ( penaltyCoeff[ i ] == NULL )
      {
      itkExceptionMacro( "Have not enough space. " );
      }
    }

  EdgeWeightType ** adjacencyWeight = new EdgeWeightType * [ numSamplePoints ];
  if ( adjacencyWeight == NULL )
    {
    itkExceptionMacro( "Have not enough space. " );
    }
  for ( i = 0; i < numSamplePoints; ++i )
    {
    adjacencyWeight[ i ] = new EdgeWeightType [ numSamplePoints ];
    if ( adjacencyWeight[ i ] == NULL )
      {
      itkExceptionMacro( "Have not enough space. " );
      }
    }
  
  double * srcHist = new double [ histn1 * histn1 ];
  double * tarHist = new double [ histn1 * histn1 ];
  /** Construct the adjacency matrix by feature distance multiplying penalty coefficient. */
  for ( i = 0; i < numSamplePoints; ++i )
    {
	this->m_ListSample->GetMeasurementVector( i, srcArray );
	this->m_PenaltyListSample->GetMeasurementVector( i, srcPArray );

    for ( j = i; j < numSamplePoints; ++j )
      {
	  if ( j == i )
        adjacencyWeight[i][j] = itk::NumericTraits< EdgeWeightType >::max();
      else
		{
        this->m_ListSample->GetMeasurementVector( j, tarArray );
		diffArray = srcArray - tarArray;
		featDist = diffArray.two_norm();

        this->m_PenaltyListSample->GetMeasurementVector( j, tarPArray );

		double spinCoef = 0.0;
		for ( unsigned int k = 0; k < numPenaltyFeats; ++k )
		  {
		  if ( ( srcPArray[ k ] + tarPArray[ k ] ) != 0.0 )
		    spinCoef += vcl_pow( srcPArray[ k ] - tarPArray[ k ], 2 ) / ( srcPArray[ k ] + tarPArray[ k ] );
		  }	
		penaltyCoef = spinCoef;

		penaltyCoeff[i][j] = penaltyCoef;
		penaltyCoeff[j][i] = penaltyCoef;
		adjacencyWeight[i][j] = penaltyCoef * featDist;
		adjacencyWeight[j][i] = penaltyCoef * featDist;
		}
      }
    }
  delete []srcHist;
  srcHist = NULL;
  delete []tarHist;
  tarHist = NULL;

  /** Start going loop to search Minimum Spanning Tree by the Prim's algorithm. */
  NodeIdentifierType components = 0;
  NodeIdentifierType src, temp, count;
 
  NodePointerType node = NULL;
  EdgePointerType edge = NULL;
  EdgeWeightType minw;
  
  /** Create the new graph. */
  this->m_MinimumSpanningTree = this->GetOutput();

  /** Reserve memory for m_Nodes and m_Edges of minimum spanning tree. */
  this->m_MinimumSpanningTree->GetNodeContainer()->Reserve( numSamplePoints );
  this->m_MinimumSpanningTree->GetEdgeContainer()->Reserve( numSamplePoints - 1 );

  /** Create node and edge iterators of minimum spanning tree. */
  NodeIteratorType nodeMSTIt( this->m_MinimumSpanningTree );
  EdgeIteratorType edgeMSTIt( this->m_MinimumSpanningTree );
 
  /** Update node information for minimum spanning tree. */
  for ( i = 0, nodeMSTIt.GoToBegin(); !nodeMSTIt.IsAtEnd(); ++i, ++nodeMSTIt )
    {
    node = nodeMSTIt.GetPointer();

    node->Identifier = i;
    node->Rank       = 0;
    node->Parent     = i;
	node->Weight     = itk::NumericTraits< EdgeWeightType >::max();
    }

  src = 0;
  this->m_MinimumSpanningTree->GetNodePointer( src )->Rank = 1;
  edgeMSTIt.GoToBegin();
  count = 0;
  components = numSamplePoints - 1;
  while ( components > 0 )
    {
    /** Update the parent and weight parameters of nodeouters. */
    UpdatePriorityQueue( adjacencyWeight, src, numSamplePoints );
   
    /** Find the lightest edge from nodeouters to nodeinners. */
    minw = itk::NumericTraits<EdgeWeightType>::max();
    for ( i = 0; i < numSamplePoints; ++i )
      {
      node = this->m_MinimumSpanningTree->GetNodePointer( i );
      if ( node->Rank == 0 )
        {
        if ( node->Weight < minw )
          {
          minw = node->Weight;
          temp = i;
          }
        } 
      }
    this->m_MinimumSpanningTree->GetNodePointer( temp )->Rank = 1;
  
    /** Create a new edge from the lightest edge. */
	edge = edgeMSTIt.GetPointer();
    edge->Identifier = count;
    edge->SourceIdentifier = this->m_MinimumSpanningTree->GetNodePointer( temp )->Parent;
    edge->TargetIdentifier = temp;
    edge->Weight = minw;
	edge->PenaltyWeight = penaltyCoeff[edge->SourceIdentifier][temp];
    
    src = temp;
    this->m_MSTLength += minw;
    components--;
    count++;
	++edgeMSTIt;
    }

  /** Release the memory space of penaltyCoeff and adjacencyWeight. */
  for ( i = 0; i < numSamplePoints; ++i )
	{
	delete []penaltyCoeff[ i ];
	penaltyCoeff[ i ] = NULL;
	}
  delete []penaltyCoeff;
  penaltyCoeff = NULL;

  for ( i = 0; i < numSamplePoints; ++i )
	{
	delete []adjacencyWeight[ i ];
	adjacencyWeight[ i ] = NULL;
	}
  delete []adjacencyWeight;
  adjacencyWeight = NULL;
  
}

/**
 *
 */
template <class TOutputGraph>
void
FastPrimMinimumSpanningTreeGraphFilter<TOutputGraph>
::GeneratePenaltyMinimumSpanningTree( void )
{
  /** Option for now to still use the single threaded code. */
  if ( !this->m_FPGThreader )
    {
    return this->GeneratePenaltyMinimumSpanningTreeSingleThreaded();
    }

  /** Initialize some threading related parameters. */
  this->InitializeFPGThreadingParameters();

  /** Construct the adjacency matrix using multi-threading. */
  this->m_FPGThreader->SetNumberOfThreads( this->m_NumberOfThreads );
  this->m_FPGThreader->SetSingleMethod( FPGPenaltyWeightThreaderCallback, 
	const_cast<void *>(static_cast<const void *>(&this->m_FPGThreaderFilterParameters)) );
  this->m_FPGThreader->SingleMethodExecute();

  /** Get size of the list sample. */
  unsigned long numSamplePoints = this->m_ListSample->GetActualSize();

  /** Start going loop to search Minimum Spanning Tree by the Prim's algorithm. */
  NodeIdentifierType i;
  NodeIdentifierType components = 0;
  NodeIdentifierType src, temp, count;
 
  NodePointerType node = NULL;
  EdgePointerType edge = NULL;
  EdgeWeightType minw;
  
  /** Create the new graph. */
  this->m_MinimumSpanningTree = this->GetOutput();

  /** Reserve memory for m_Nodes and m_Edges of minimum spanning tree. */
  this->m_MinimumSpanningTree->GetNodeContainer()->Reserve( numSamplePoints );
  this->m_MinimumSpanningTree->GetEdgeContainer()->Reserve( numSamplePoints - 1 );

  /** Create node and edge iterators of minimum spanning tree. */
  NodeIteratorType nodeMSTIt( this->m_MinimumSpanningTree );
  EdgeIteratorType edgeMSTIt( this->m_MinimumSpanningTree );
 
  /** Update node information for minimum spanning tree. */
  for ( i = 0, nodeMSTIt.GoToBegin(); !nodeMSTIt.IsAtEnd(); ++i, ++nodeMSTIt )
    {
    node = nodeMSTIt.GetPointer();

    node->Identifier = i;
    node->Rank       = 0;
    node->Parent     = i;
	node->Weight     = itk::NumericTraits< EdgeWeightType >::max();
    }

  src = 0;
  this->m_MinimumSpanningTree->GetNodePointer( src )->Rank = 1;
  edgeMSTIt.GoToBegin();
  count = 0;
  components = numSamplePoints - 1;
  while ( components > 0 )
    {
    /** Update the parent and weight parameters of nodeouters. */
    UpdatePriorityQueue( this->m_FPGThreaderFilterParameters.st_AdjacencyWeight, src, numSamplePoints );
   
    /** Find the lightest edge from nodeouters to nodeinners. */
    minw = itk::NumericTraits<EdgeWeightType>::max();
    for ( i = 0; i < numSamplePoints; ++i )
      {
      node = this->m_MinimumSpanningTree->GetNodePointer( i );
      if ( node->Rank == 0 )
        {
        if ( node->Weight < minw )
          {
          minw = node->Weight;
          temp = i;
          }
        } 
      }
    this->m_MinimumSpanningTree->GetNodePointer( temp )->Rank = 1;
  
    /** Create a new edge from the lightest edge. */
	edge = edgeMSTIt.GetPointer();
    edge->Identifier = count;
    edge->SourceIdentifier = this->m_MinimumSpanningTree->GetNodePointer( temp )->Parent;
    edge->TargetIdentifier = temp;
    edge->Weight = minw;
	edge->PenaltyWeight = this->m_FPGThreaderFilterParameters.st_PenaltyCoeff[edge->SourceIdentifier][temp];
    
    src = temp;
    this->m_MSTLength += minw;
    components--;
    count++;
	++edgeMSTIt;
    }

  /** Release the memory space. */
  for ( i = 0; i < numSamplePoints; ++i )
	{
	delete []this->m_FPGThreaderFilterParameters.st_PenaltyCoeff[ i ];
	this->m_FPGThreaderFilterParameters.st_PenaltyCoeff[ i ] = NULL;
	}
  delete []this->m_FPGThreaderFilterParameters.st_PenaltyCoeff;
  this->m_FPGThreaderFilterParameters.st_PenaltyCoeff = NULL;

  for ( i = 0; i < numSamplePoints; ++i )
	{
	delete []this->m_FPGThreaderFilterParameters.st_AdjacencyWeight[ i ];
	this->m_FPGThreaderFilterParameters.st_AdjacencyWeight[ i ] = NULL;
	}
  delete []this->m_FPGThreaderFilterParameters.st_AdjacencyWeight;
  this->m_FPGThreaderFilterParameters.st_AdjacencyWeight = NULL;

  for ( ThreadIdType j = 0; j < this->m_NumberOfThreads; ++j )
    {
    delete []this->m_FPGPenaltyWeightPerThreadVariables[ j ].st_srcHist;
    this->m_FPGPenaltyWeightPerThreadVariables[ j ].st_srcHist = NULL;
    delete []this->m_FPGPenaltyWeightPerThreadVariables[ j ].st_tarHist;
    this->m_FPGPenaltyWeightPerThreadVariables[ j ].st_tarHist = NULL;
    }  
  
}

/**
 *
 */
template <class TOutputGraph>
void
FastPrimMinimumSpanningTreeGraphFilter<TOutputGraph>
::InitializeFPGThreadingParameters( void )
{
  /** Some initialization. */
  unsigned long i;
  unsigned int histn1 = 4;
  unsigned long numSamplePoints = this->m_ListSample->GetActualSize();
  unsigned int numFeats = this->m_ListSample->GetMeasurementVectorSize();
  unsigned int numPenaltyFeats = this->m_PenaltyListSample->GetMeasurementVectorSize();

  this->m_FPGThreaderFilterParameters.st_ListSample = ListSampleType::New();
  this->m_FPGThreaderFilterParameters.st_ListSample->SetMeasurementVectorSize( numFeats );
  this->m_FPGThreaderFilterParameters.st_ListSample->Resize( numSamplePoints );
  this->m_FPGThreaderFilterParameters.st_ListSample->SetActualSize( numSamplePoints );
  this->m_FPGThreaderFilterParameters.st_PenaltyListSample = ListSampleType::New();
  this->m_FPGThreaderFilterParameters.st_PenaltyListSample->SetMeasurementVectorSize( numPenaltyFeats );
  this->m_FPGThreaderFilterParameters.st_PenaltyListSample->Resize( numSamplePoints );
  this->m_FPGThreaderFilterParameters.st_PenaltyListSample->SetActualSize( numSamplePoints );

  /** Transfer the two listsamples for multi-threading. */
  for ( i = 0; i < numSamplePoints; ++i )
    {
	this->m_FPGThreaderFilterParameters.st_ListSample->SetMeasurementVector( i, this->m_ListSample->GetMeasurementVector( i ) );
	this->m_FPGThreaderFilterParameters.st_PenaltyListSample->SetMeasurementVector( i, this->m_PenaltyListSample->GetMeasurementVector( i ) );
    }
  
  /** Creat memory space for penaltyCoeff and adjacencyWeight. */
  this->m_FPGThreaderFilterParameters.st_PenaltyCoeff = new EdgeWeightType * [ numSamplePoints ];
  if ( this->m_FPGThreaderFilterParameters.st_PenaltyCoeff == NULL )
    {
    itkExceptionMacro( "Have not enough space. " );
    }
  for ( i = 0; i < numSamplePoints; ++i )
    {
    this->m_FPGThreaderFilterParameters.st_PenaltyCoeff[ i ] = new EdgeWeightType [ numSamplePoints ];
    if ( this->m_FPGThreaderFilterParameters.st_PenaltyCoeff[ i ] == NULL )
      {
      itkExceptionMacro( "Have not enough space. " );
      }
    }

  this->m_FPGThreaderFilterParameters.st_AdjacencyWeight = new EdgeWeightType * [ numSamplePoints ];
  if ( this->m_FPGThreaderFilterParameters.st_AdjacencyWeight == NULL )
    {
    itkExceptionMacro( "Have not enough space. " );
    }
  for ( i = 0; i < numSamplePoints; ++i )
    {
    this->m_FPGThreaderFilterParameters.st_AdjacencyWeight[ i ] = new EdgeWeightType [ numSamplePoints ];
    if ( this->m_FPGThreaderFilterParameters.st_AdjacencyWeight[ i ] == NULL )
      {
      itkExceptionMacro( "Have not enough space. " );
      }
    }

  /** Only resize the array of structs when needed. */
  if ( this->m_FPGPenaltyWeightPerThreadVariablesSize != this->m_NumberOfThreads )
    {
    delete []this->m_FPGPenaltyWeightPerThreadVariables;
    this->m_FPGPenaltyWeightPerThreadVariables     = new AlignedFPGPenaltyWeightPerThreadStruct[ this->m_NumberOfThreads ];
    this->m_FPGPenaltyWeightPerThreadVariablesSize = this->m_NumberOfThreads;
    }

  for ( ThreadIdType j = 0; j < this->m_NumberOfThreads; ++j )
    {
    this->m_FPGPenaltyWeightPerThreadVariables[ j ].st_srcHist = new double [ histn1 * histn1 ];
    this->m_FPGPenaltyWeightPerThreadVariables[ j ].st_tarHist = new double [ histn1 * histn1 ];
    }  

}

/**
 *
 */
template <class TOutputGraph>
ITK_THREAD_RETURN_TYPE
FastPrimMinimumSpanningTreeGraphFilter<TOutputGraph>
::FPGPenaltyWeightThreaderCallback( void * arg )
{
  ThreadInfoType * infoStruct = static_cast< ThreadInfoType * >( arg );
  ThreadIdType     threadID   = infoStruct->ThreadID;

  FPGMultiThreaderParameterType * temp
    = static_cast< FPGMultiThreaderParameterType * >( infoStruct->UserData );

  temp->st_Filter->FPGThreadedPenaltyWeight( threadID ); 
  
  return ITK_THREAD_RETURN_VALUE;

} 

/**
 *
 */
template <class TOutputGraph>
void
FastPrimMinimumSpanningTreeGraphFilter<TOutputGraph>
::FPGThreadedPenaltyWeight( ThreadIdType threadId )
{
  NodeIdentifierType i, j;
  MeasurementVectorType srcArray, tarArray, diffArray;
  MeasurementVectorType srcPArray, tarPArray;
  EdgeWeightType featDist, penaltyCoef;

  EmdL1                emdl1;
  unsigned int         histn1 = 4;

  /** Get size of the list sample. */
  const unsigned long numSamplePoints = this->m_FPGThreaderFilterParameters.st_ListSample->GetActualSize();
  const unsigned int numPenaltyFeats = this->m_FPGThreaderFilterParameters.st_PenaltyListSample->GetMeasurementVectorSize();

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads
	= static_cast< unsigned long >( vcl_ceil( static_cast< double >( numSamplePoints )
	/ static_cast< double >( this->m_NumberOfThreads ) ) );

  unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
  unsigned long pos_end   = nrOfSamplesPerThreads * ( threadId + 1 );
  pos_begin = ( pos_begin > numSamplePoints ) ? numSamplePoints : pos_begin;
  pos_end   = ( pos_end > numSamplePoints ) ? numSamplePoints : pos_end;

  for ( i = pos_begin; i < pos_end; ++i )
    {
	this->m_FPGThreaderFilterParameters.st_ListSample->GetMeasurementVector( i, srcArray );
	this->m_FPGThreaderFilterParameters.st_PenaltyListSample->GetMeasurementVector( i, srcPArray );

    for ( j = i; j < numSamplePoints; ++j )
      {
	  if ( j == i )
        this->m_FPGThreaderFilterParameters.st_AdjacencyWeight[i][j] = itk::NumericTraits< EdgeWeightType >::max();
      else
		{
        this->m_FPGThreaderFilterParameters.st_ListSample->GetMeasurementVector( j, tarArray );
		diffArray = srcArray - tarArray;
		featDist = diffArray.two_norm();

        this->m_FPGThreaderFilterParameters.st_PenaltyListSample->GetMeasurementVector( j, tarPArray );

		double spinCoef = 0.0;
		for ( unsigned int k = 0; k < numPenaltyFeats; ++k )
		  {
		  if ( ( srcPArray[ k ] + tarPArray[ k ] ) != 0.0 )
		    spinCoef += vcl_pow( srcPArray[ k ] - tarPArray[ k ], 2 ) / ( srcPArray[ k ] + tarPArray[ k ] );
		  }
		penaltyCoef = spinCoef;

		this->m_FPGThreaderFilterParameters.st_PenaltyCoeff[i][j] = penaltyCoef;
		this->m_FPGThreaderFilterParameters.st_PenaltyCoeff[j][i] = penaltyCoef;
		this->m_FPGThreaderFilterParameters.st_AdjacencyWeight[i][j] = penaltyCoef * featDist;
		this->m_FPGThreaderFilterParameters.st_AdjacencyWeight[j][i] = penaltyCoef * featDist;
		}
      }
    }

}

/**
 *
 */
template <class TOutputGraph>
void
FastPrimMinimumSpanningTreeGraphFilter<TOutputGraph>
::UpdatePriorityQueue( EdgeWeightType** adjaWeight, NodeIdentifierType src, NodeIdentifierType nodeNum )
{	
  NodeIdentifierType i;
  NodePointerType nodeOut;
  
  for ( i = 0; i < nodeNum; ++i )
    {
    nodeOut = this->m_MinimumSpanningTree->GetNodePointer( i );
    if ( nodeOut->Rank == 0 )
      {
      if ( adjaWeight[src][i] < nodeOut->Weight )
        {
        nodeOut->Parent = src;
        nodeOut->Weight = adjaWeight[src][i];
        }
      }
    }

}

/**
 *
 */
template <class TOutputGraph>
void
FastPrimMinimumSpanningTreeGraphFilter<TOutputGraph>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

}

} // end namespace itk

#endif
