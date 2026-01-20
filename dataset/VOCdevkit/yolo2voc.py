from xml.dom.minidom import Document
import os
import cv2


def makexml(picPath, txtPath, xmlPath):  # 图片所在文件夹路径，txt所在文件夹路径，xml文件保存路径
    """此函数用于将yolo格式txt标注文件转换为voc格式xml标注文件
    """
    dic = {'0': "0",  # 创建字典用来对类型进行转换
           '1': "1",  # 此处的字典要与自己的classes.txt文件中的类对应，且顺序要一致
           '3': 'person'
           }
    files = os.listdir(txtPath)
    for i, name in enumerate(files):
        if name.endswith('.txt'):
            xmlBuilder = Document()
            annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
            xmlBuilder.appendChild(annotation)

            txtFile = open(txtPath + name, 'r')
            txtList = txtFile.readlines()
            img = cv2.imread(picPath + name.replace('.txt', '.jpg'))  # 按照图片名替换后缀来读取图片
            if img is not None:  # 如果图片存在
                Pheight, Pwidth, Pdepth = img.shape

                folder = xmlBuilder.createElement("folder")  # folder标签
                foldercontent = xmlBuilder.createTextNode("VOC2007")
                folder.appendChild(foldercontent)
                annotation.appendChild(folder)  # folder标签结束

                filename = xmlBuilder.createElement("filename")  # filename标签
                filenamecontent = xmlBuilder.createTextNode(name.replace('.txt', '.jpg'))
                filename.appendChild(filenamecontent)
                annotation.appendChild(filename)  # filename标签结束

                size = xmlBuilder.createElement("size")  # size标签
                width = xmlBuilder.createElement("width")  # size子标签width
                widthcontent = xmlBuilder.createTextNode(str(Pwidth))
                width.appendChild(widthcontent)
                size.appendChild(width)  # size子标签width结束

                height = xmlBuilder.createElement("height")  # size子标签height
                heightcontent = xmlBuilder.createTextNode(str(Pheight))
                height.appendChild(heightcontent)
                size.appendChild(height)  # size子标签height结束

                depth = xmlBuilder.createElement("depth")  # size子标签depth
                depthcontent = xmlBuilder.createTextNode(str(Pdepth))
                depth.appendChild(depthcontent)
                size.appendChild(depth)  # size子标签depth结束

                annotation.appendChild(size)  # size标签结束

                for j in txtList:
                    oneline = j.strip().split(" ")
                    object = xmlBuilder.createElement("object")  # object 标签
                    picname = xmlBuilder.createElement("name")  # name标签
                    namecontent = xmlBuilder.createTextNode(dic[oneline[0]])
                    picname.appendChild(namecontent)
                    object.appendChild(picname)  # name标签结束

                    pose = xmlBuilder.createElement("pose")  # pose标签
                    posecontent = xmlBuilder.createTextNode("Unspecified")
                    pose.appendChild(posecontent)
                    object.appendChild(pose)  # pose标签结束

                    truncated = xmlBuilder.createElement("truncated")  # truncated标签
                    truncatedContent = xmlBuilder.createTextNode("0")
                    truncated.appendChild(truncatedContent)
                    object.appendChild(truncated)  # truncated标签结束

                    difficult = xmlBuilder.createElement("difficult")  # difficult标签
                    difficultcontent = xmlBuilder.createTextNode("0")
                    difficult.appendChild(difficultcontent)
                    object.appendChild(difficult)  # difficult标签结束

                    bndbox = xmlBuilder.createElement("bndbox")  # bndbox标签
                    xmin = xmlBuilder.createElement("xmin")  # xmin标签
                    mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
                    xminContent = xmlBuilder.createTextNode(str(mathData))
                    xmin.appendChild(xminContent)
                    bndbox.appendChild(xmin)  # xmin标签结束

                    ymin = xmlBuilder.createElement("ymin")  # ymin标签
                    mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
                    yminContent = xmlBuilder.createTextNode(str(mathData))
                    ymin.appendChild(yminContent)
                    bndbox.appendChild(ymin)  # ymin标签结束

                    xmax = xmlBuilder.createElement("xmax")  # xmax标签
                    mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
                    xmaxContent = xmlBuilder.createTextNode(str(mathData))
                    xmax.appendChild(xmaxContent)
                    bndbox.appendChild(xmax)  # xmax标签结束

                    ymax = xmlBuilder.createElement("ymax")  # ymax标签
                    mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
                    ymaxContent = xmlBuilder.createTextNode(str(mathData))
                    ymax.appendChild(ymaxContent)
                    bndbox.appendChild(ymax)  # ymax标签结束

                    object.appendChild(bndbox)  # bndbox标签结束

                    annotation.appendChild(object)  # object标签结束

                # 保存到xml文件
                xmlfile = name.replace('.txt', '.xml')
                f = open(xmlPath + xmlfile, 'w')
                xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
                f.close()
            else:
                print(f"图片 {picPath + name.replace('.txt', '.jpg')} 不存在，跳过该txt文件的转换")
        else:
            print(f"文件 {txtPath + name} 不是txt文件，跳过")

if __name__ == "__main__":
    # 根据你的路径进行修改
    picPath = "/home/xie/xxs/dataset/DUO/images/"  # 图片所在文件夹路径，后面的/一定要带上
    txtPath = "/home/xie/xxs/dataset/DUO/labels/"  # txt所在文件夹路径，后面的/一定要带上
    xmlPath = "/home/xie/xxs/dataset/DUO/annotations/"  # xml文件保存路径，后面的/一定要带上
    if not os.path.exists(xmlPath):
        os.makedirs(xmlPath)  # 如果保存路径不存在，则创建文件夹
    makexml(picPath, txtPath, xmlPath)