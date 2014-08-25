ADD_TEST( NAME test_parenchyma_phenotypes COMMAND nosetests ${CMAKE_SOURCE_DIR}/cip_python/phenotypes/tests/test_parenchyma_phenotypes.py) 

ADD_TEST( NAME test_laa_phenotypes COMMAND nosetests ${CMAKE_SOURCE_DIR}/cip_python/phenotypes/tests/test_laa_phenotypes.py) 

ADD_TEST( NAME test_body_composition_phenotypes COMMAND nosetests ${CMAKE_SOURCE_DIR}/cip_python/phenotypes/tests/test_body_composition_phenotypes.py) 
