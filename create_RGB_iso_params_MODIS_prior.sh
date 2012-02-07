#!/bin/bash

WORKDIR=`pwd`
mkdir -p $WORKDIR/QL

for image in `ls *.img`
do
	DoY=`echo $image | cut -d. -f2,3`

	gdal_translate -b 1 -of GTiff -ot Byte -scale 0.001 0.500 0 255 $image $image.b1.tif
	gdal_translate -b 4 -of GTiff -ot Byte -scale 0.024 0.724 0 255 $image $image.b4.tif
	gdal_translate -b 7 -of GTiff -ot Byte -scale 0.001 0.425 0 255 $image $image.b7.tif

	#gdal_translate -b 10 -of GTiff -ot Byte -scale 0.001 0.1 0 255 $image $image.b10.tif

	gdal_merge.py -o $image.741.RGB.tif -separate $image.b7.tif $image.b4.tif $image.b1.tif

	convert $image.b7.tif $image.b4.tif $image.b1.tif -font AvantGarde-Book -gravity South -pointsize 100 -fill white -draw 'text 10,18 " '$DoY'' -channel RGB -combine $image.741.RGB.jpg
	#convert $image.b10.tif -font AvantGarde-Book -gravity South -pointsize 50 -fill white -draw 'text 10,18 " '$DoY'' $image.f0_VIS_var.jpg

	rm $image.b7.tif $image.b4.tif $image.b1.tif

done

image=`echo $image | cut -d. -f1,3`.741.RGB.gif
convert -loop 0 -delay 100 -resize %25 *741.RGB.jpg $image

mv *jpg *gif $WORKDIR/QL
#mv *jpg $WORKDIR/QL
rm *tif
