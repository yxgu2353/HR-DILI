# _*_ coding: utf-8 _*_
# @Time : $[DATE] $[TIME]
# @Author : G

# 用法：
# 最多只允许额外传入一个参数，指定计算的指纹种类，包括['FP', 'ExtFP', 'EStateFP', 'MACCSFP', 'PubchemFP', 'SubFP', 'AP2D', 'KRFP', 'GraphFP']
# 例（如果没有传入参数指定指纹种类，则默认计算所有指纹）：
# python fp_calculation.py all  # 表示计算所有9种分子指纹
# python fp_calculation.py FP  # 表示计算FP指纹
# python fp_calculation.py FP,ExtFP  # 若是计算自定义的多个指纹，请用逗号隔开。
# python fp_calculation.py  # 若不传入参数，默认计算所有指纹

import os
import sys
import xml.etree.ElementTree as ET
import pandas as pd

assert len(sys.argv) <= 2

try:
    arg = sys.argv[1]
except:
    arg = 'all'

padel_path = '/home/yxgu/PaDEL-Descriptor'
# padel_path = 'C:/Users\yaxin\Desktop\CADD\PaDEL-Descriptor'
input_file = 'test_data.smi'  # 指定SMILES文件
proc = 4  # 指定运行所要用到的核数
setlog = False  # 是否在终端输出log信息，默认为输出模式

logfile = '>padel.log' if setlog else ''
# fp_list = ['FP', 'ExtFP', 'EStateFP', 'MACCSFP', 'PubchemFP', 'SubFP', 'KRFP', 'GraphFP']
fp_list = ['GraphFP', 'EStateFP']
nameDict = {'FP': 'Fingerprinter',
            'ExtFP': 'ExtendedFingerprinter',
            'EStateFP': 'EStateFingerprinter',
            'MACCSFP': 'MACCSFingerprinter',
            'PubchemFP': 'PubchemFingerprinter',
            'SubFP': 'SubstructureFingerprinter',
            'KRFP': 'KlekotaRothFingerprinter',
            'GraphFP': 'GraphOnlyFingerprinter'
            }


def modifyDesxml(fp):
    "define the required fingerprints calculated by PaDEL-Descriptor"
    tree = ET.parse('%s/descriptors.xml' % padel_path)
    root = tree.getroot()
    if root == None:
        return False
    nodes = root.findall('.//Group[@name="Fingerprint"]/Descriptor')
    for node in nodes:
        node.set('value', 'false')

    node = root.find('.//Descriptor[@name="%s"]' % nameDict[fp])
    node.set('value', 'true')

    tree.write('padel-des.xml', encoding='utf-8', xml_declaration=True)
    return True


def calc_fp(fp):
    modifyDesxml(fp)
    out_file = 'test_data_%s.csv' % fp
    command = 'java -jar %s/PaDEL-Descriptor.jar -descriptortypes padel-des.xml -dir %s -file %s -fingerprints %s \
               -threads %s %s' % (padel_path, input_file, out_file, nameDict[fp], proc, logfile)
    os.system(command)
    os.remove('padel-des.xml')


def outfile_deal():
    for fp in fp_list:
        outfile = pd.read_csv('test_data_%s.csv' % fp)
        # outfile.drop(index=['Name'])
        outfile.to_csv('test_data_%s.csv' % fp, header=False, index=False)


def main():
    if arg == 'all':
        for fp in fp_list:
            calc_fp(fp)

    elif arg in fp_list:
        calc_fp(arg)

    elif ',' in arg:
        temp = arg.split(',')
        if set(temp).issubset(fp_list):
            for fp in temp:
                calc_fp(fp)

    else:
        print('%s cannot be identified!') % sys.argv[1]


if __name__ == '__main__':
    main()
    outfile_deal()

