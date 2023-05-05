#This script creates a shapefile from the csv file of the faster R-CNN object detection model results.
#The script needs to be run in the QGIS Python interface. 


import csv
import os

#input csv of the model results and output shapefile. Shapefile needs shp extenstion. 
input_csv = os.path.join('C:/', 'SpaceArch New', 'SpaceArcheology', 'Datasets', 'Burial Mound Data Norway', 'Results', 'MSRM_x1_Additional Rotation Best Model - 200 Epochs', 'Composite_MSRM_x1_Results_without_Overlap_200_UTM.csv')
output_file = os.path.join('C:/', 'Test', 'MSRM_x1_200_Epochs_Without_Exclude.shp')
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
QgsCoordinateReferenceSystem('EPSG:25832'),\
'ESRI Shapefile')

#Create a QGIS feature type and and a feature for each bounding box in the model results to a shapefile. 
feat = QgsFeature()
for i in range(len(data_list)):
    polygon = [[QgsPointXY(data_list[i][0], data_list[i][2]), QgsPointXY(data_list[i][1], data_list[i][2]), QgsPointXY(data_list[i][1], data_list[i][3]), QgsPointXY(data_list[i][0], data_list[i][3])]]
    feat.setGeometry(QgsGeometry.fromPolygonXY(polygon))
    feat.setAttributes([i, 'mound'])
    writer.addFeature(feat)

#Add the shapefile layer to active QGIS session. Optional.     
layer = iface.addVectorLayer(output_file, '', 'ogr')
del(writer)