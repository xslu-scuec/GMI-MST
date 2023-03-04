
ADD_ELXCOMPONENT( MSTGraphAlphaMutualInformationMetric
 elxMSTGraphAlphaMutualInformationMetric.h
 elxMSTGraphAlphaMutualInformationMetric.hxx
 elxMSTGraphAlphaMutualInformationMetric.cxx
 itkMSTGraphAlphaMutualInformationImageToImageMetric.h
 itkMSTGraphAlphaMutualInformationImageToImageMetric.txx
 emdL1.h
 emdL1.cpp
 itkGraph.h
 itkGraph.txx
 itkGraphSource.h
 itkGraphSource.txx
 itkFastPrimMinimumSpanningTreeGraphFilter.h
 itkFastPrimMinimumSpanningTreeGraphFilter.txx
 itkDefaultGraphTraits.h
 itkImageGraphTraits.h
 itkMinimumSpanningTreeGraphTraits.h
)

IF( USE_MSTGraphAlphaMutualInformationMetric )
  TARGET_LINK_LIBRARIES( MSTGraphAlphaMutualInformationMetric KNNlib ANNlib )
ENDIF()
