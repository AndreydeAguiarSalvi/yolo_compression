import os
import cv2
import glob
import argparse
import numpy as np
import pandas as pd


def mean_divergence(p1: str, p2: str) -> float:
    img1 = cv2.imread(p1)
    img2 = cv2.imread(p2)
    dif = cv2.absdiff(img1, img2)
    return np.mean(cv2.mean(dif))


'''
    Evaluate each possible pair:
        YOLOv3 with YOLO Nano
        YOLOv3 with YOLO Mobile
        Mobile with Mobile (wo. KD vs w. KD)
        Nano with Nano (wo. KD vs w. KD)
'''
def evaluate_pairs(df: pd.DataFrame, has_v3: bool, has_mobile: bool, has_nano: bool) -> list:
    result = []
    if has_v3 and has_nano:
        nanos = df[df['model'].str.contains('nano')]
        for _, nano in nanos.iterrows():
            v = mean_divergence(df[df['model'] == 'v3']['full_name'].item(), nano['full_name'])
            pair = f"v3 - {nano['model']}"
            result.append([ nano['full_name'], pair, v ])
    
    if has_v3 and has_mobile:
        mobiles = df[df['model'].str.contains('mobile')]
        for _, mobile in mobiles.iterrows():
            v = mean_divergence(df[df['model'] == 'v3']['full_name'].item(), mobile['full_name'])
            pair = f"v3 - {mobile['model']}"
            result.append([ mobile['full_name'], pair, v ])
        
    if has_nano:
        nano = df[df['model'] == 'nano']
        ns_ = df[df['model'].str.contains('nano_')]
        for _, n in ns_.iterrows():
            v = mean_divergence(nano['full_name'].item(), n['full_name'])
            pair = f"{nano['model'].item()} - {n['model']}"
            result.append([ nano['full_name'].item(), pair, v ])
    
    if has_mobile:
        mobile = df[df['model'] == 'mobile']
        mb_ = df[df['model'].str.contains('mobile_')]
        for _, m in mb_.iterrows():
            v = mean_divergence(mobile['full_name'].item(), m['full_name'])
            pair = f"{mobile['model'].item()} - {m['model']}"
            result.append([ mobile['full_name'].item(), pair, v ])

    return result


'''
    Computes the divergences in each sub-DataFrame created from get_tupled_data
'''
def compute_divergences(groups: list) -> pd.DataFrame: 
    result = pd.DataFrame(columns=['full_name', 'pair', 'divergence'])
    
    for df in groups:
        models = df['model'].tolist()
        v3 = 'v3' in models
        mobile = 'mobile' in models
        nano = 'nano' in models
        
        pair_divergences = evaluate_pairs(df, v3, mobile, nano)
        for i, r in enumerate(pair_divergences): result.loc[i] = r

    return result


'''
    Given a folder, it recursivelly find images
    and group then by dataset, model, image name, head, and anchor, in tuples
    Returns a list of DataFrames, each sub-DataFrame containing the grouped data
'''
def get_tupled_data(path: str) -> list:
    images = glob.glob(path+"**/*.*", recursive=True)
    
    data = []
    for img in images:
        dataset = 'exdark' if 'exdark' in img else 'pascal'
        model = img.split(dataset+'_')[-1].split(os.sep)[0]
        img_name = img.split(os.sep)[-1]
        name = img_name.split('_')[0] + '_' + img_name.split('_')[1]
        head = '0' if '_0_' in img_name else '1' if '_1_' in img_name else '2'
        anchor = '0' if '_0.' in img_name else '1' if '_1.' in img_name else '2'
        if '.' not in name: data.append([img, dataset, model, name, head, anchor])
    
    df = pd.DataFrame(data, columns=['full_name', 'dataset', 'model', 'img_name', 'head', 'anchor'])
    df = df.sort_values(by=['dataset', 'img_name', 'head', 'anchor', 'model'])
    for i in range(len(df)):
        if df.iloc[i, -2:].all() == df.iloc[i+1, -2:].all(): pass
        else: break
    
    result = []
    step = i+1
    for i in range(0, len(df), step):
        result.append(df.iloc[i:i+step, :])
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='output/', help='Path to load the inferences.')
    parser.add_argument('--save', action='store_true', help='exports the DataFrame to a .csv with all divergences.')
    args = vars(parser.parse_args())
    
    df = get_tupled_data(args['path'])
    divs = compute_divergences(df)
    divs = divs.sort_values('divergence')
    print(divs[:30])
    if args['save']: divs.to_csv('divergences.csv', sep=',', index=False)