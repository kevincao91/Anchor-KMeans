import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            # print(member[0].text)
            for bbox in member.findall('bndbox'):
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(bbox[0].text),
                         int(bbox[1].text),
                         int(bbox[2].text),
                         int(bbox[3].text)
                         )
                xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    path = '/media/kevin/娱乐/xizang_database/testdata/1125/Annotations'


    xml_df = xml_to_csv(path)
    xml_df.to_csv(('images/_labels.csv'), index=None)
    print('Successfully converted xml to csv.')


main()
