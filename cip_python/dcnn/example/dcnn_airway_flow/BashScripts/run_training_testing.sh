#!/bin/bash

/home/pn964/anaconda2/envs/tensorflow/bin/python ${ACIL_PATH}/Projects/dcnn_airway_flow/airway_flow_run_training_testing.py \
--operation TRAIN --output_folder ${CNN_DB}/AirwayFlow --network_description v00_test \
--train_dataset ${CNN_DB}/AirwayFlow/datasets/DOE_all4x2COR.csv --test_dataset \
${CNN_DB}/AirwayFlow/datasets/DOE_all4x2COR.csv -save_arch

/home/pn964/anaconda2/envs/tensorflow/bin/python ${ACIL_PATH}/Projects/dcnn_airway_flow/airway_flow_run_training_testing.py \
--operation TEST --output_folder ${CNN_DB}/AirwayFlow --network_description v00_test --test_dataset \
${CNN_DB}/AirwayFlow/datasets/conj_PAR.csv --test_dataset_dauA ${CNN_DB}/AirwayFlow/datasets/conj_DAUA.csv \
--test_dataset_dauB ${CNN_DB}/AirwayFlow/datasets/conj_DAUB.csv