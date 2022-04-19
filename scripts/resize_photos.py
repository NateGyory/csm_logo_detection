import cv2
import csv

csv_rows = [[]]
read_csv_names = ['../labels/nate.csv', '../labels/rafa.csv', '../labels/antor.csv', '../labels/seifeldin.csv']

labels = '../labels/labels.csv'
img_path = '../csm_logo_dataset/'
save_img_path = '../resized_photos/'

def save_img(img_name, img):
    save_f = '%s%s' % (save_img_path, img_name)
    cv2.imwrite(save_f, img)

def normalize_image(img_name, x_norm, y_norm, w_norm, h_norm):
    dim = (300, 400)
    img_f = '%s%s' % (img_path, img_name)
    img = cv2.imread(img_f)
    img = cv2.resize(img, dim)

    x = int(x_norm * dim[0])
    y = int(y_norm * dim[1])
    w = int(w_norm * dim[0])
    h = int(h_norm * dim[1])

    return img, x, y, w, h


def normalize_coordinates(x, y, w, h, img_w, img_h):
    x_norm = x/img_w
    y_norm = y/img_h
    w_norm = w/img_w
    h_norm = h/img_h
    return x_norm, y_norm, w_norm, h_norm

def process_img(x, y, w, h, img_name, img_w, img_h):
    x_norm, y_norm, w_norm, h_norm = normalize_coordinates(x, y, w, h, img_w, img_h)
    img, x_new, y_new, w_new, h_new = normalize_image(img_name, x_norm, y_norm, w_norm, h_norm)
    save_img(img_name, img)

    csv_rows.append([img_name, x_new, y_new, w_new, h_new])

def read_csv(file_name):
    with open(file_name) as f:
        reader = csv.reader(f)
        for row in reader:
            process_img(int(row[1]), int(row[2]), int(row[3]), int(row[4]), row[5], int(row[6]), int(row[7]))

##########
#  Main  #
##########
# Read CSV files and process images
for name in read_csv_names: read_csv(name)

# Write labels to labes.csv
with open(labels, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)
