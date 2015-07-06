ADD_TEST( NAME test_grid_segmenter COMMAND nosetests ${CMAKE_SOURCE_DIR}/cip_python/segmentation/tests/test_grid_segmenter.py ) 

ADD_TEST( NAME test_kde_histogram_feature_extractor COMMAND nosetests ${CMAKE_SOURCE_DIR}/cip_python/segmentation/tests/test_kde_histogram_feature_extractor.py ) 
