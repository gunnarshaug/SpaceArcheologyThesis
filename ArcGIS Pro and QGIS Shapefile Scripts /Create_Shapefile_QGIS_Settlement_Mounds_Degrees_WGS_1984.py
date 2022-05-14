#This script creates a shapefile from the csv file of the faster R-CNN object detection model results.
#The script needs to be run in the QGIS Python interface. 

import csv
import os

#input csv of the model results and output shapefile. Shapefile needs shp extenstion. 
input_csv = os.path.join('C:/', 'SpaceArch New', 'SpaceArcheology', 'Datasets', 'Results', 'SAR Kandahar Results', 'SAR_Kandahar_Results.csv')
output_file = os.path.join('C:/', 'Test', 'Kandahar_Results.shp')
data_list = []

#Reads the results csv file.
with open(input_csv, 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    next(csv_reader)
    for row in csv_reader:
        data_list.append((float(row[4]), float(row[5]), float(row[6]), float(row[7])))


# create fields for the resulting shapefile. 
layerFields = QgsFields()
layerFields.append(QgsField('ID', QVariant.Int))
layerFields.append(QgsField('Type', QVariant.String))

#Create a writer for the shapefile. EPSG Coordinate system must be set!!
writer = QgsVectorFileWriter(output_file, 'UTF-8', layerFields,\
QgsWkbTypes.Polygon,\
QgsCoordinateReferenceSystem('EPSG:4326'),\
'ESRI Shapefile')

#Create a QGIS feature type and and a feature for each bounding box in the model results to a shapefile. 
feat = QgsFeature()
for i in range(len(data_list)):
    polygon = [[QgsPointXY(data_list[i][0], data_list[i][1]), QgsPointXY(data_list[i][2], data_list[i][1]), QgsPointXY(data_list[i][2], data_list[i][3]), QgsPointXY(data_list[i][0], data_list[i][3])]]
    feat.setGeometry(QgsGeometry.fromPolygonXY(polygon))
    feat.setAttributes([i, 'AMound'])
    writer.addFeature(feat)

#Add the shapefile layer to active QGIS session. Optional.     
layer = iface.addVectorLayer(output_file, '', 'ogr')
del(writer)