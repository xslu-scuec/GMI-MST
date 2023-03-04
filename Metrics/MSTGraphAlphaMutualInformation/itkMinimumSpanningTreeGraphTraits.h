/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkMinimumSpanningTreeGraphTraits.h,v $
  Language:  C++
  Date:      $Date: 2012/05/30 10:51:34 $
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkMinimumSpanningTreeGraphTraits_h
#define __itkMinimumSpanningTreeGraphTraits_h

#include "itkImageGraphTraits.h"

namespace itk
{

/**
 * Graph traits class for use with the KruskalMinimumSpanningTreeFilter and
 * PrimMinimumSpanningTreeFilter class.
 */

template <typename TWeight, unsigned int VImageDimension>
class MinimumSpanningTreeGraphTraits : public ImageGraphTraits<TWeight, VImageDimension>
{
public:
  typedef MinimumSpanningTreeGraphTraits                Self;
  typedef ImageGraphTraits<TWeight, VImageDimension>    Superclass;

  typedef TWeight                                       NodeWeightType;
  typedef TWeight                                       EdgeWeightType;
  typedef typename Superclass::NodeIdentifierType       NodeIdentifierType;
  typedef typename Superclass::EdgeIdentifierType       EdgeIdentifierType;
  typedef typename Superclass::EdgeIdentifierContainerType
                                                        EdgeIdentifierContainerType;
  typedef typename Superclass::IndexType                IndexType;
  typedef typename Superclass::EdgeType                 EdgeType;
  typedef typename Superclass::EdgePointerType          EdgePointerType;

  struct  NodeType;
  typedef NodeType* NodePointerType;

  struct NodeType
    {
    NodeIdentifierType Identifier;
    EdgeIdentifierContainerType IncomingEdges;
    EdgeIdentifierContainerType OutgoingEdges;
    NodeWeightType Weight;
    IndexType ImageIndex;
	NodeIdentifierType Parent;
	unsigned int Rank;
    };
};

} // end namespace itk

#endif
