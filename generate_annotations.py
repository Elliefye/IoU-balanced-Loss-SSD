import csv
import os.path

annotation_file = 'Dataset_lim/boxes/test-annotations-bbox.csv'
base_path = 'Dataset_lim/test/'

classes = ['Knife', 'Horse', 'Human body']
class_labels = {'/m/04ctx': 'Knife', '/m/03k3r': 'Horse', '/m/02p0tk3': 'Human body'}
class_ids = {'Knife': 0, 'Horse': 1, 'Human body': 2}

annotations = []
locations = []


def in_annotations(imgid):
    for ann in annotations:
        if ann['imgid'] == imgid:
            return True
    return False


with open(annotation_file) as f:
    next(f)
    print('Generating annotations...', end=' ')
    lines = csv.reader(f, delimiter=',')
    for line in lines:
        img_id = line[0]
        img_class = line[2]
        # one bbox per image (this will make the model worse but less complicated)
        if img_class in class_labels and not in_annotations(img_id):
            img_class = class_labels[img_class]
            path_to_file = base_path + img_class.lower().replace(' ', '_') + '/' + img_id + '.jpg'
            if os.path.isfile(path_to_file):
                annotations.append({
                    'imgid': img_id,
                    'xmin': float(line[4]),
                    'ymin': float(line[6]),
                    'xmax': float(line[5]),
                    'ymax': float(line[7]),
                    'class': class_ids[img_class]
                })
                locations.append(path_to_file)
                #print(path_to_file)
                #print(annotations[img_id])

with open('img_locations_test.txt', 'w') as loc_file:
    for line in locations:
        loc_file.write(line + '\n')

with open('annotations_test.csv', 'w', newline='') as ann_file:
    writer = csv.DictWriter(ann_file, fieldnames=['imgid', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
    writer.writeheader()

    for line in annotations:
        writer.writerow(line)

print('DONE')
