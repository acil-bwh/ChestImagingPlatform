#!/bin/bash

/home/pn964/anaconda2/envs/tensorflow/bin/python ${ACIL_PATH}/Projects/dcnn_airway_flow/airway_flow_run_training_testing.py \
--operation CV --output_folder ${CNN_DB}/AirwayFlow --network_description v00_test \
--train_dataset ${CNN_DB}/AirwayFlow/datasets/DOE_all4x2COR.csv --test_dataset \
${CNN_DB}/AirwayFlow/datasets/DOE_all4x2COR.csv