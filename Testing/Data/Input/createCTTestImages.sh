#Create downsample images

unu resample -s x0.125 x0.125 x0.125 -k tent -i 10002K_INSP_STD_BWH_COPD.nhdr | unu save -e gzip -f nrrd -o ct-64.nrrd

unu resample -s x0.125 x0.125 x0.125 -k cheap -i 10002K_INSP_STD_BWH_COPD_partialLungLabelMap.nhdr | unu save -e gzip -f nrrd -o lm-64.nrrd

ExtractChestLabelMap -r WholeLung -i 10002K_INSP_STD_BWH_COPD_partialLungLabelMap.nhdr -o wholelung-64.nrrd
unu resample -s x0.125 x0.125 x0.125 -k cheap -i wholelung-64.nrrd | unu save -e gzip -f nrrd -o wholelung-64.nrrd


#Create cropped image for lobe segmentation assessment

params="--lm NA --rm NA --rightHorizontalFiducials 74.9099,184.936,-113.868 --rightHorizontalFiducials 74.9099,216.56,-99.3961 --rightHorizontalFiducials 83.699,247.648,-90.8201 --rightHorizontalFiducials 83.699,182.256,-114.404 --rightHorizontalFiducials 83.699,203.696,-98.3241 --rightHorizontalFiducials 83.699,222.456,-95.6441 --rightHorizontalFiducials 83.699,192.976,-103.684 --rightHorizontalFiducials 96.0037,229.96,-92.4281 --rightHorizontalFiducials 96.0037,199.944,-97.7881 --rightHorizontalFiducials 96.0037,179.04,-112.796 --rightHorizontalFiducials 101.863,219.24,-92.4281 --rightHorizontalFiducials 101.863,197.264,-96.7161 --rightHorizontalFiducials 101.863,177.968,-113.868 --rightHorizontalFiducials 104.207,193.512,-97.7881 --rightHorizontalFiducials 104.207,175.824,-115.476 --rightHorizontalFiducials 104.207,214.416,-91.8921 --rightObliqueFiducials 81.9412,130.8,-101.004 --rightObliqueFiducials 81.9412,169.928,-109.58 --rightObliqueFiducials 81.9412,182.256,-114.94 --rightObliqueFiducials 81.9412,242.288,-161.572 --rightObliqueFiducials 81.9412,262.656,-185.156 --rightObliqueFiducials 81.9412,212.808,-135.308 --rightObliqueFiducials 63.7771,186.008,-114.94 --rightObliqueFiducials 63.7771,122.76,-95.6441 --rightObliqueFiducials 63.7771,246.04,-169.612 --rightObliqueFiducials 45.0271,237.464,-170.148 --rightObliqueFiducials 45.0271,211.736,-150.316 --rightObliqueFiducials 45.0271,153.848,-97.2521 --rightObliqueFiducials 45.0271,112.576,-89.7481 --rightObliqueFiducials 57.3318,191.368,-122.98 --rightObliqueFiducials 57.3318,238,-165.324 --rightObliqueFiducials 57.3318,122.76,-93.5001 --rightObliqueFiducials 80.1834,130.8,-99.3961 --rightObliqueFiducials 80.1834,181.72,-112.796 --rightObliqueFiducials 80.1834,234.248,-154.068 --rightObliqueFiducials 80.1834,256.76,-177.116 --rightObliqueFiducials 80.1834,208.52,-132.092 --rightObliqueFiducials 95.4178,267.48,-190.516 --rightObliqueFiducials 95.4178,137.232,-105.292 --rightObliqueFiducials 95.4178,198.872,-125.66 --rightObliqueFiducials 110.066,193.512,-125.124 --rightObliqueFiducials 110.066,173.68,-115.476 --rightObliqueFiducials 110.066,220.312,-142.812 --leftObliqueFiducials -65.1292,121.688,-53.3002 --leftObliqueFiducials -65.1292,139.376,-72.5962 --leftObliqueFiducials -65.1292,180.112,-103.148 --leftObliqueFiducials -65.1292,205.84,-125.66 --leftObliqueFiducials -82.7074,287.848,-223.748 --leftObliqueFiducials -82.7074,267.48,-194.268 --leftObliqueFiducials -82.7074,248.72,-169.076 --leftObliqueFiducials -82.7074,226.744,-141.204 --leftObliqueFiducials -82.7074,182.256,-106.9 --leftObliqueFiducials -82.7074,150.632,-82.2441 --leftObliqueFiducials -82.7074,129.192,-68.3082 --leftObliqueFiducials -98.5277,136.696,-76.3481 --leftObliqueFiducials -98.5277,173.68,-99.9321 --leftObliqueFiducials -98.5277,212.272,-127.804 --leftObliqueFiducials -98.5277,239.072,-147.1 --leftObliqueFiducials -98.5277,265.872,-183.548 --leftObliqueFiducials -98.5277,284.096,-210.884 --leftObliqueFiducials -103.801,276.056,-194.804 --leftObliqueFiducials -103.801,245.504,-151.924 --leftObliqueFiducials -103.801,217.632,-130.484 --leftObliqueFiducials -103.801,183.864,-105.828 --leftObliqueFiducials -103.801,145.272,-83.8521 --leftObliqueFiducials -117.278,166.712,-98.3241 --leftObliqueFiducials -117.278,198.336,-116.548 --leftObliqueFiducials -117.278,240.144,-145.492 --leftObliqueFiducials -121.965,219.24,-128.876 --leftObliqueFiducials -121.965,176.896,-105.292 --leftObliqueFiducials -125.481,211.2,-122.98 --loParticles NA --roParticles NA --rhParticles NA --regionType NA --lambda 0.1"

/Users/rjosest/src/devel/ChestImagingPlatform-build/CIP-build/bin/SegmentLungLobes --in 16021S_INSP_STD_NJC_COPD_partialLungLabelMap.nhdr --out 16021S_INSP_STD_NJC_COPD_interactiveLobeSegmentation.nrrd $params

unu crop -min 38 76 280 -max 480 424 375 -i 16021S_INSP_STD_NJC_COPD.nhdr | unu save -f nrrd -e gzip -o crop_ct.nrrd


unu crop -min 38 76 280 -max 480 424 375 -i 16021S_INSP_STD_NJC_COPD_partialLungLabelMap.nhdr | unu save -f nrrd -e gzip -o crop_ct_partialLungLabelMap.nrrd

unu crop -min 38 76 280 -max 480 424 375 -i 16021S_INSP_STD_NJC_COPD_interactiveLobeSegmentation.nrrd | unu save -f nrrd -e gzip -o crop_ct_interactiveLobeSegmentation.nrrd

#SegmentLungLobes --in crop_ct_partialLungLabelMap.nrrd --out test.nrrd $params

