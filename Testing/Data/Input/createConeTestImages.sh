#!/bin/bash

unu=$TEEM_PATH/unu
conegen=$TEEM_PATH/conegen

#Create cone-shaped images to serve as test for vessel and airway-like structures
#kernel=cubic:0,0.5
kernel=tent
$conegen -s 51 300 -r 0 10 -m 2 -off 0 0 -ov 20 -psz 2 -ssr 400 -k $kernel -o vessel.nrrd
$conegen -s 51 300 -r 0 10 -m 2 -off 0 0 -ov 20 -psz 2 -ssr 400 -k $kernel -gauss -o vesselgauss.nrrd

$conegen -s 51 300 -r 1 10  -m 2 -off 0 0 -ov 20 -psz 2 -ssr 400 -k $kernel -o wall.nrrd
$conegen -s 51 300 -r 1 10  -m 2 -off 0 0 -ov 20 -psz 2 -ssr 400 -k $kernel -gauss -o wallgauss.nrrd

$conegen -s 51 300 -r 0.5 8  -m 2.5 -off 0 0  -ov 20 -psz 2 -ssr 400 -k $kernel -o lumen.nrrd
$conegen -s 51 300 -r 0.5 8  -m 2.5 -off 0 0  -ov 20 -psz 2 -ssr 400 -k $kernel -gauss -o lumengauss.nrrd


for f in vessel.nrrd vesselgauss.nrrd wall.nrrd wallgauss.nrrd lumen.nrrd lumengauss.nrrd airway.nrrd airwaygauss.nrrd
do
echo $f
$unu 2op exists $f 0 -o $f
done

$unu 2op - wall.nrrd lumen.nrrd -o airway.nrrd
$unu 2op - wallgauss.nrrd lumengauss.nrrd -o airwaygauss.nrrd

#Scale
scale=1000
offset=-1000

for f in vessel.nrrd vesselgauss.nrrd wall.nrrd wallgauss.nrrd lumen.nrrd lumengauss.nrrd airway.nrrd airwaygauss.nrrd
do

$unu 2op x $scale $f | $unu 2op + - $offset | $unu convert -t short | $unu save -f nrrd -e gzip -o $f

done
