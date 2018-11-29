ADD_TEST( NAME test_apply_reader_labels COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_apply_reader_labels.py )

ADD_TEST( NAME test_classify_image_subtypes COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_classify_image_subtypes.py )

ADD_TEST( NAME test_distance_feature_extractor COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_distance_feature_extractor.py )

ADD_TEST( NAME test_distance_feature_extractor_from_ROI COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_distance_feature_extractor_from_ROI.py )

ADD_TEST( NAME test_distance_feature_extractor_from_xml_points COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_distance_feature_extractor_from_xml_points.py )

ADD_TEST( NAME test_distance_histogram_extractors COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_distance_histogram_extractors.py )

ADD_TEST( NAME test_get_ct_patch_from_center COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_get_ct_patch_from_center.py )

ADD_TEST( NAME test_hist_dist_knn COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_hist_dist_knn.py )

ADD_TEST( NAME test_hist_dist_scikit_knn COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_hist_dist_scikit_knn.py )

ADD_TEST( NAME test_kde_histogram_feature_extractor COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_kde_histogram_feature_extractor.py )

ADD_TEST( NAME test_kde_histogram_feature_extractor_from_ROI COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_kde_histogram_feature_extractor_from_ROI.py )

ADD_TEST( NAME test_kde_histogram_feature_extractor_from_xmlpoints COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_kde_histogram_feature_extractor_from_xmlpoints.py )










