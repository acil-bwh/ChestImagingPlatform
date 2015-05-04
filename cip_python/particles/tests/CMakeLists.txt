ADD_TEST( NAME test_fissure_particles COMMAND nosetests ${CMAKE_SOURCE_DIR}/cip_python/particles/tests/test_fissure_particles.py) 

ADD_TEST( NAME test_airway_particles COMMAND nosetests ${CMAKE_SOURCE_DIR}/cip_python/particles/tests/test_airway_particles.py) 

ADD_TEST( NAME test_vessel_particles COMMAND nosetests ${CMAKE_SOURCE_DIR}/cip_python/particles/tests/test_vessel_particles.py) 

ADD_TEST( NAME test_particle_metrics COMMAND nosetests ${CMAKE_SOURCE_DIR}/cip_python/particles/tests/test_particle_metrics.py) 
