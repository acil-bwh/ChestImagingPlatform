ADD_TEST( NAME test_fissure_particles COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_fissure_particles.py)

#ADD_TEST( NAME test_airway_particles COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_airway_particles.py)

ADD_TEST( NAME test_vessel_particles COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_vessel_particles.py)

ADD_TEST( NAME test_particle_metrics COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_particle_metrics.py)
