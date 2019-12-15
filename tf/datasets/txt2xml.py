from lxml import etree as ET
import os

imageset=['trainval', 'test']

txt_path = 'D:/data/gaofen/gaofen3/v2/{}/labels'.format(imageset[1])
xml_path = 'D:/data/gaofen/gaofen3/v2/{}/labelxmls'.format(imageset[1])

if not os.path.exists(xml_path):
        os.makedirs(xml_path)

for idx, txtfile in enumerate(os.listdir(txt_path)):
    print(idx, txtfile)
    file_name = txtfile.split('.txt')[0]

    txtfile = os.path.join(txt_path, file_name + '.txt')
    xmlfile = os.path.join(xml_path, file_name + '.xml')

    root = ET.Element('annotation')
    filename = ET.SubElement(root, 'filename')
    filename.text=file_name
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text=str(1024)
    ET.SubElement(size, 'height').text=str(1024)

    lines = open(txtfile, 'r').readlines()
    for line in lines:
        objects = ET.SubElement(root,'object')
        ET.SubElement(objects, 'name').text='ship'
        robndbox = ET.SubElement(objects, 'robndbox')
        [cx,cy,w,h,theta] = [float(xy) for xy in line.rstrip().split(' ')[:-1]]
        ET.SubElement(robndbox, 'cx').text=str(cx)
        ET.SubElement(robndbox, 'cy').text=str(cy)
        ET.SubElement(robndbox, 'w').text=str(w)
        ET.SubElement(robndbox, 'h').text=str(h)
        ET.SubElement(robndbox, 'angle').text=str(theta)
    
    tree = ET.ElementTree(root)
    tree.write(xmlfile, pretty_print = True, xml_declaration = True, encoding = 'utf-8')

        