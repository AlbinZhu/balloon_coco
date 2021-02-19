import json
import cv2
import os


model = "train"

bnd_id_start = 1

times = 0

json_dict = {
    "images"     : [],
    "type"       : "instances",
    "annotations": [],
    "categories" : []
}


root_dir = r'/home/swing/data/detection_dataset/balloon_det/%s/normal/' % model

image_root = os.path.join(root_dir, "images")
images = os.listdir(image_root)
images.sort()
image_id = 0
bnd_id = 0
for image_name in images:
    image = cv2.imread(os.path.join(image_root, image_name))
    h, w = image.shape[:2]
    image = {
        "file_name": image_name,
        "height": h,
        "width": w,
        "id": image_id
    }
    json_dict["images"].append(image)

    with open(os.path.join(root_dir, "boxes", (image_name.split(".")[0] + ".txt"))) as f:
        lines = f.readlines()
        for line in lines:
            print(line)
            xmin,ymin,xmax,ymax,label = line.split()
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            o_width = abs(int(xmax) - int(xmin))
            o_height = abs(int(ymax) - int(ymin))

            area = o_width * o_height
            category_id = label.strip()

            # #定义annotationhb
            annotation = {
                'area'          : area,  #
                'iscrowd'       : 0,
                'image_id'      : image_id,  #图片的id
                'bbox'          :[xmin, ymin, o_width,o_height],
                'category_id'   : int(category_id), #类别的id 通过这个id去查找category里面的name
                'id'            : bnd_id,  #唯一id ,可以理解为一个框一个Id
                'ignore'        : 0,
                'segmentation'  : []
            }
            print(category_id)

            json_dict['annotations'].append(annotation)

            bnd_id += 1
    image_id += 1
    #
#定义categories

#你得类的名字(cid,cate)对应
classes = ['balloon']

for i in range(len(classes)):

    cate = classes[i]
    cid = i + 1
    category = {
        'supercategory' : 'none',
        'id'            : cid,  #类别的id ,一个索引，主键作用，和别的字段之间的桥梁
        'name'          : cate  #类别的名字比如房子，船，汽车
    }

    json_dict['categories'].append(category)


json_fp = open("%s.json" % model,'w')
json_str = json.dumps(json_dict)
json_fp.write(json_str)
json_fp.close()