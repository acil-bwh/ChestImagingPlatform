unu resample -s x0.125 x0.125 x0.125 -k tent -i 10002K_INSP_STD_BWH_COPD.nhdr | unu save -e gzip -f nrrd -o ct-64.nrrd

unu resample -s x0.125 x0.125 x0.125 -k cheap -i 10002K_INSP_STD_BWH_COPD_partialLungLabelMap.nhdr | unu save -e gzip -f nrrd -o lm-64.nrrd

ExtractChestLabelMap -r WholeLung -i 10002K_INSP_STD_BWH_COPD_partialLungLabelMap.nhdr -o wholelung-64.nrrd
unu resample -s x0.125 x0.125 x0.125 -k cheap -i wholelung-64.nrrd | unu save -e gzip -f nrrd -o wholelung-64.nrrd