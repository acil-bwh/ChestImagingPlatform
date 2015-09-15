ADD_TEST( NAME test_kde_histogram_feature_extractor COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_kde_histogram_feature_extractor.py ) 

ADD_TEST( NAME test_distance_feature_extractor COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_distance_feature_extractor.py ) 

ADD_TEST( NAME test_distance_histogram_extractors COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_distance_histogram_extractors.py ) 

ADD_TEST( NAME test_hist_dist_knn COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_hist_dist_knn.py ) 

ADD_TEST( NAME test_apply_reader_labels COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_apply_reader_labels.py ) 

ADD_TEST( NAME test_kde_histogram_feature_extractor_from_xmlpoints COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_kde_histogram_feature_extractor_from_xmlpoints.py ) 



