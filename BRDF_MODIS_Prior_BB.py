#!/Library/Frameworks/EPD64.framework/Versions/Current/bin/python

import glob
import os
import sys

import numpy
import osgeo.gdal as gdal
from osgeo.gdalconst import *

def IncrementSamples(AccWeigthedParams, AccWeigth, Parameters, Uncertainties):
	"""
	Author
 		Gerardo Lopez Saldana, ISA-UTL GerardoLopez@isa.utl.pt

	Purpose
	Accumulate information applying a weighted samples factor 

	Inputs
	AccWeigthedParams: Float[rows, cols, NumberOfBands, NumberOfParameters]: Accumulator array to store the weigthed contribution of Params
	AccWeigth: Float[rows, cols, NumberOfBands, NumberOfParameters]: Array of the sum of weighting factors
 	Parameters: Float[rows, cols, NumberOfBands, NumberOfParameters]: Array containg the BRDF parameters (iso,vol,geo)
	Uncertainties: Float[rows, cols]: Array containing the uncertainty information deraived from the QA flag

	Outputs
	Acumulator
	"""
	rows, cols, NumberOfBands, NumberOfParameters = Parameters.shape

	# Create the array for the weigthing factors
	Weigth = numpy.where(Uncertainties > 0, Uncertainties, 0.0)

	for i in range(NumberOfBands):
		for j in range(NumberOfParameters):
			AccWeigthedParams[:,:,i,j] += Parameters[:,:,i,j] * Weigth

	AccWeigth += Weigth

	return AccWeigthedParams, AccWeigth

def GetParameters(File, Bands, NumberOfParameters, RelativeUncert, ScaleFactor):
	"""
	Author
		Gerardo Lopez Saldana, ISA-UTL GerardoLopez@isa.utl.pt

	Purpose
		Method to extract BRDF parameters from MCD43C2 for Bands

	Inputs
		File: String: MCD43C2 File name inbcluding full absolute path
		Bands: Long[]: Array containing the bands to extract (1,2 == Red, NIR)
		NumberOfParameters: Long[]: Array containing the number of BRDF parameters
		RelativeUncert: Float[]: 5 elements array containing the relative uncertainty for each QA value
		ScaleFactor: Float: Data scale factor

	Outputs
		Parameters: Array containg the BRDF parameters (iso,vol,geo) for Bands
		Uncertainties: Array containing the uncertainty information deraived from the QA flag
	"""
	FillValue = 32767
	NumberOfBands = len(Bands)

	# Get dimensions
	rows, cols = GetDimensions(File)
	QA = numpy.zeros((rows, cols), numpy.uint)
	QA_flags = numpy.array([0,1,2,3,4])
	# https://lpdaac.usgs.gov/products/modis_products_table/brdf_albedo_snow_free_quality/16_day_l3_global_0_05deg_cmg/mcd43c2
	# Bit Comb.	BRDF_Albedo_Quality
	# 0	 best quality, 75% or more with best full inversions
	# 1	 good quality, 75% or more with full inversions
	# 2	 mixed, 75% or less full inversions and 25% or less fill values
	# 3	 all magnitude inversions or 50% or less fill values
	# 4	 50% or more fill values
	# 255	 FillValue
	
	Parameters = numpy.zeros((rows, cols, NumberOfBands, NumberOfParameters), numpy.float32)
	Uncertainties = numpy.zeros((rows, cols), numpy.float32)

	# Get QA
	SubDataset = 'HDF4_EOS:EOS_GRID:"' + File + '":MCD_CMG_BRDF_0.05Deg:BRDF_Quality'
	QA[:,:] = gdal.Open(SubDataset, GA_ReadOnly).ReadAsArray()

	for Band in range(len(Bands)):
		for Parameter in range(NumberOfParameters):
			#SubDataset = 'HDF4_EOS:EOS_GRID:"'+File+'":MCD_CMG_BRDF_0.05Deg:BRDF_Albedo_Parameter'+str(Parameter+1)+'_Band'+str(Bands[Band])
			SubDataset = 'HDF4_EOS:EOS_GRID:"'+File+'":MCD_CMG_BRDF_0.05Deg:BRDF_Albedo_Parameter'+str(Parameter+1)+'_'+str(Bands[Band])
			Parameters[:,:,Band,Parameter] = gdal.Open(SubDataset, GA_ReadOnly).ReadAsArray()
			# Filter out fill values
			Parameters[:,:,Band,Parameter] = numpy.where(Parameters[:,:,Band,Parameter]==FillValue, 0.0, 
                                                         Parameters[:,:,Band,Parameter] * ScaleFactor)

	for flag in QA_flags:
		indices = numpy.where(QA==flag)
		Uncertainties[indices] = RelativeUncert[ numpy.where(QA_flags==flag)[0][0] ]

	return Parameters, Uncertainties


def GetFileList(DataDir, DoY):
	FileList = glob.glob(DataDir + '/MCD43C2.A20??' + DoY + '.005.*.hdf')
	FileList.sort()

	Year = numpy.zeros((len(FileList)), numpy.int16)

	i = 0
	for File in FileList:
		# Get Year from filename
		YearOfObservation = os.path.basename(File).split('.')[1][1:5]
		Year[i] = YearOfObservation

		i += 1

	return FileList, Year


def GetDimensions(File):
	# Open first subdataset from MCD43C2
	SubDataset = 'HDF4_EOS:EOS_GRID:"' + File + '":MCD_CMG_BRDF_0.05Deg:BRDF_Albedo_Parameter1_Band1'
	dataset = gdal.Open(SubDataset, GA_ReadOnly)
	rows, cols = dataset.RasterYSize, dataset.RasterXSize
	dataset = None

	return rows, cols

#--------------------------------------------------------------------------------#
from IPython import embed

DataDir = sys.argv[1]
DoY = sys.argv[2]

if len(DoY)==1:
	DoY='00' + DoY
elif len(DoY)==2:
	DoY='0' + DoY

FileList, Year = GetFileList(DataDir, DoY)

# From the first file get dimensions
rows, cols = GetDimensions(FileList[0])
# Extract, for the time being, only narrowband: red, NIR & SWIR, bands 1, 2 & 7
# for broadband, is possible to provide the waveband names.
#Bands = numpy.array([1,2,7])
Bands = ['vis','nir','shortwave']
NumberOfBands = len(Bands)
# BRDF parameters, iso, vol, geo
NumberOfParameters = 3

ScaleFactor = 0.001

# Relative uncertainty of 5 quality values: full inversions -> magnitude inversions -> fill values
# https://lpdaac.usgs.gov/products/modis_products_table/brdf_albedo_snow_free_quality/16_day_l3_global_0_05deg_cmg/mcd43c2
BRDF_Albedo_Quality = numpy.arange(5)

# http://en.wikipedia.org/wiki/Golden_ratio
GoldenMean = 0.618034
RelativeUncert = GoldenMean ** BRDF_Albedo_Quality

WeigthedMean = numpy.zeros((rows, cols, NumberOfBands, NumberOfParameters), numpy.float32)
WeigthedVariance = numpy.zeros((rows, cols, NumberOfBands, NumberOfParameters), numpy.float32)
AccWeigthedParams = numpy.zeros((rows, cols, NumberOfBands, NumberOfParameters), numpy.float32)
AccWeigth = numpy.zeros((rows, cols), numpy.float32)

print "Computing the weigthed mean..."
for File in FileList:
	print File
	Parameters, Uncertainties = GetParameters(File, Bands, NumberOfParameters, RelativeUncert, ScaleFactor)
	AccWeigthedParams, AccWeigth = IncrementSamples(AccWeigthedParams, AccWeigth, Parameters, Uncertainties)

# Compute the weighted mean
for i in range(NumberOfBands):
    for j in range(NumberOfParameters):
		WeigthedMean[:,:,i,j] = numpy.where(AccWeigth > 0.0, AccWeigthedParams[:,:,i,j] / AccWeigth, 0.0)

print "Computing the weigthed variance..."
# Compute the weigthed variance
for File in FileList:
	print File
	Parameters, Uncertainties = GetParameters(File, Bands, NumberOfParameters, RelativeUncert, ScaleFactor)
	for i in range(NumberOfBands):
		for j in range(NumberOfParameters):
			tmpWeigthedVariance = Uncertainties * numpy.power(Parameters[:,:,i,j] - WeigthedMean[:,:,i,j], 2) 
			WeigthedVariance[:,:,i,j] += tmpWeigthedVariance

for i in range(NumberOfBands):
    for j in range(NumberOfParameters):
		WeigthedVariance[:,:,i,j] = numpy.where(AccWeigth > 0.0, WeigthedVariance[:,:,i,j] / AccWeigth, 0.0)

print "Writing results to a file..."
scale_factor = 0.001

format = "ENVI"
driver = gdal.GetDriverByName(format)
new_dataset = driver.Create( 'MCD43C2.Prior.' + DoY +  '.img', cols, rows, ((NumberOfBands*NumberOfParameters)*2)+1, GDT_Float32)

k = 1
for i in range(NumberOfBands):
	for j in range(NumberOfParameters):
		new_dataset.GetRasterBand(k).WriteArray(WeigthedMean[:,:,i,j])
		new_dataset.GetRasterBand(k).SetDescription("Mean - band: " + str(Bands[i]) + " Parameter f" + str(j))

		new_dataset.GetRasterBand((NumberOfBands*NumberOfParameters)+k).WriteArray(WeigthedVariance[:,:,i,j])
		new_dataset.GetRasterBand((NumberOfBands*NumberOfParameters)+k).SetDescription("Variance - band: " + str(Bands[i]) + " Parameter f" + str(j))
		k += 1

# Write the weigthed number of samples
new_dataset.GetRasterBand(((NumberOfBands*NumberOfParameters)*2)+1).SetDescription("Weighted Number Of Samples")
new_dataset.GetRasterBand(((NumberOfBands*NumberOfParameters)*2)+1).WriteArray(AccWeigth[:,:])

new_dataset = None
