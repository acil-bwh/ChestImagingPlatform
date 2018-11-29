ADD_TEST( NAME test_fissure_particles COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_fissure_particles.py)

#ADD_TEST( NAME test_airway_particles COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_airway_particles.py)

ADD_TEST( NAME test_vessel_particles COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_vessel_particles.py)

ADD_TEST( NAME test_particle_metrics COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_particle_metrics.py)

ADD_TEST( NAME test_transfer_particles_region_type_values COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_transfer_particles_region_type_values.py)
