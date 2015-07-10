ADD_TEST( NAME test_kde_histogram_feature_extractor COMMAND nosetests ${CMAKE_SOURCE_DIR}/cip_python/classification/tests/test_kde_histogram_feature_extractor.py ) 

ADD_TEST( NAME test_distance_feature_extractor COMMAND nosetests ${CMAKE_SOURCE_DIR}/cip_python/classification/tests/test_distance_feature_extractor.py ) 

ADD_TEST( NAME test_distance_histogram_extractors COMMAND nosetests ${CMAKE_SOURCE_DIR}/cip_python/classification/tests/test_distance_histogram_extractors.py ) 
