import os
import cv2
import tqdm
import argparse
import numpy as np


def show_images(original_pth: list, teacher_pth: list, reduced_pth: list, student_pth: list):
    
    for orig, tch, red, std in zip(original_pth, teacher_pth, reduced_pth, student_pth):
        img1 = cv2.imread(orig)
        img2 = cv2.imread(tch)
        img3 = cv2.imread(red)
        img4 = cv2.imread(std)
        stacked = np.concatenate((img1, img2, img3, img4), axis=1)
        window_name = tch.split(os.sep)[-1]
        print(stacked.shape, window_name)

        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(window_name, stacked)
        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            keyCode = cv2.waitKey(10000)
            if (keyCode & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break


def save_images(original_pth: list, teacher_pth: list, reduced_pth: list, student_pth: list, path_to: str):
    if not os.path.exists(path_to): os.makedirs(path_to)
    
    for orig, tch, red, std in tqdm.tqdm(zip(original_pth, teacher_pth, reduced_pth, student_pth)):
        img1 = cv2.imread(orig)
        img2 = cv2.imread(tch)
        img3 = cv2.imread(red)
        img4 = cv2.imread(std)
        stacked = np.concatenate((img1, img2, img3, img4), axis=1)
        
        window_name = tch.split(os.sep)[-1]
        cv2.imwrite(path_to + os.sep + window_name, stacked)

        
def load_paths(r: str, dt: str, model: str, visu: str, loss: str) -> list:
    result = []
    for root, _, files in os.walk(r):
        for f in files:
            if model in root and dt in root and visu in f and loss in root and 'group' not in root:
                result.append(root + os.sep + f)
    result.sort()
    return result


def get_original_imgs(paths: list) -> list:
    result = []
    for f in paths:
        parts = f.split(os.sep)
        last, ext = parts[-1].split('.')[0], parts[-1].split('.')[1]
        year, id = last.split('_')[2], last.split('_')[3]
        original = ''
        for i in range(len(parts)-1): original += parts[i] + os.sep
        original += year + '_' + id + '.' + ext
        result.append(original)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', default='v3', help='folder model used as teacher')
    parser.add_argument('--reduced', default='nano', help='folder model used as student, but in normal training')
    parser.add_argument('--student', default='nano_kd36', help='folder model used as student, trained with KD')
    parser.add_argument('--root', default='output/GradCam', help='root folder with features to visualize')
    parser.add_argument('--dataset', default='exdark', help='dataset to generate the features')
    parser.add_argument('--visu', default='grad', choices=['grad', 'cam-gb', 'gb'], help='kind of visualization')
    parser.add_argument('--loss', default='self', help='kind of loss used to compute the visualizations. If self, means the pseudo-original GradCam loss,\
        otherwise, it expects a number')
    parser.add_argument('--visualize', action='store_true', help='only visualize the images. Otherwise, it will be saved')
    args = vars(parser.parse_args())

    teacher_pth = load_paths(args['root'], args['dataset'], args['teacher'] + os.sep, args['visu'], args['loss'])
    reduced_pth = load_paths(args['root'], args['dataset'], args['reduced'] + os.sep, args['visu'], args['loss'])
    student_pth = load_paths(args['root'], args['dataset'], args['student'] + os.sep, args['visu'], args['loss'])
    original_pth = get_original_imgs(teacher_pth)
    
    if args['visualize']: show_images(original_pth, teacher_pth, reduced_pth, student_pth)
    else: 
        path_to = args['root'] + os.sep + 'groups' + os.sep + f"{args['dataset']}_{args['teacher']}_{args['reduced']}_{args['student']}" + os.sep + args['visu']
        save_images(original_pth, teacher_pth, reduced_pth, student_pth, path_to)