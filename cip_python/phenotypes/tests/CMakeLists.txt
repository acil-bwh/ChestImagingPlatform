ADD_TEST( NAME test_parenchyma_phenotypes COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_parenchyma_phenotypes.py)

ADD_TEST( NAME test_body_composition_phenotypes COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_body_composition_phenotypes.py)

ADD_TEST( NAME test_vasculature_phenotypes COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_vasculature_phenotypes.py)

ADD_TEST( NAME test_emphysema_severity COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_emphysema_severity.py)

ADD_TEST( NAME test_fissure_completeness_phenotypes COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_fissure_phenotypes.py)

