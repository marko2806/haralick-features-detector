import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tqdm
def img_to_grid(img, row,col):
    ww = [[i.min(), i.max()] for i in np.array_split(range(img.shape[0]),row)]
    hh = [[i.min(), i.max()] for i in np.array_split(range(img.shape[1]),col)]
    grid = [img[j:jj,i:ii,:] for j,jj in ww for i,ii in hh]
    return grid, len(ww), len(hh)

def plot_grid(grid,row,col,h=5,w=5):
    fig, ax = plt.subplots(nrows=row, ncols=col)
    [axi.set_axis_off() for axi in ax.ravel()]

    fig.set_figheight(h)
    fig.set_figwidth(w)
    c = 0
    for row in ax:
        for col in row:
            col.imshow(np.flip(grid[c],axis=-1))
            c+=1
    plt.show()
def create_true_and_false(pic_path, mask_math):
    row, col =5,7
    name = pic_path.split("\\")[-1][0:-4]
    img_mask = cv2.imread(mask_math)
    grid_mask_false , r,c = img_to_grid(img_mask,row,col)
    img = cv2.imread(pic_path)
    grid , r,c = img_to_grid(img,row,col)
    pre_grid_mask_true = cv2.bitwise_not(img_mask)
    grid_mask_true , r,c = img_to_grid(pre_grid_mask_true,row,col)
    #plot_grid(grid_mask_true, row, col, h=5, w=5)
    # print(pic_path)
    # cv2.imshow("name", img)
    # cv2.imshow("name2", img_mask)
    # cv2.waitKey(0)  # waits until a key is pressed
    # cv2.destroyAllWindows()  # destroys the window showing image
    for i, np_image_tuple in enumerate(zip(grid_mask_false, grid , grid_mask_true) ):
        mask_false = np_image_tuple[0]
        mask_true = np_image_tuple[2]
        image = np_image_tuple[1]

        if not mask_false.any():
            cv2.imwrite(name + str(i) + 'false.png', image)
            # cv2.imshow("false masek small", mask_false)
            # cv2.waitKey(0)  # waits until a key is pressed
            # cv2.destroyAllWindows()  # destroys the window showing image
            # print("false")
        elif not mask_true.any():
            #print("true")
            cv2.imwrite(name + str(i) + 'true.png', image)
def dir_create(path):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)


image_dir='B:\\Desktop\\FER\\DIPL_1\\RaspoznavanjeUzoraka\\haralick-features-detector\\annotation_parsing\\all\\pictures'
mask_dir='B:\\Desktop\\FER\\DIPL_1\\RaspoznavanjeUzoraka\\haralick-features-detector\\annotation_parsing\\all\\xmls\\masks'
#output_dir='B:\\Desktop\\FER\\DIPL_1\\RaspoznavanjeUzoraka\\haralick-features-detector\\annotation_parsing\all'
#dir_create(output_dir)
img_list = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
mask_list = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]
mask_bitness = 24
# create_true_and_false(os.path.join(image_dir, img_list[0]), os.path.join(mask_dir, mask_list[0]))
for img, mask in zip(img_list[0::5], mask_list[0::5]):
    img_path = os.path.join(image_dir, img)
    mask_path = os.path.join(mask_dir, mask)
    #output_path = os.path.join(output_dir, img.split('.')[0] + '.png')
    create_true_and_false(img_path, mask_path)
