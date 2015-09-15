ADD_TEST( NAME test_parenchyma_phenotypes COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_parenchyma_phenotypes.py)

ADD_TEST( NAME test_laa_phenotypes COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_laa_phenotypes.py)

ADD_TEST( NAME test_body_composition_phenotypes COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_body_composition_phenotypes.py)
