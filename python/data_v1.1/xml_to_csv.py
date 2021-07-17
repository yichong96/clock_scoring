import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, help="directory containing xml files to aggregate",required=True)
    parser.add_argument("--csv_path", type=str, help="path to save csv file",required=True)
    args = parser.parse_args()
    xml_path = args.xml_path
    csv_path = args.csv_path
    xml_df = xml_to_csv(xml_path)
    xml_df.to_csv(csv_path, index=None)
    print('Successfully converted xml to csv.')

main()