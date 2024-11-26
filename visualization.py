import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import argparse
import pickle
import pandas as pd
import logging
import random
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='test')

    #  datasets
    parser.add_argument("--embs_path", type=str, default=None, help='embedding file path')

    # all poison ctxs
    parser.add_argument("--ctxs_path", type=str, default=None, help='all poison ctxs data path')
    parser.add_argument('--poison_samples', type=int, default=None, help='number of poison data, you can set None, all poison ctxs will be loaded.')
    parser.add_argument('--clean_samples', type=int, default=10000, help='number of clean ctxs')
    parser.add_argument('--pca_dim', type=int, default=3, help='pca dim')
    parser.add_argument('--save_path', type=str, default="./embs.png", help='save path')

    args = parser.parse_args()
    print(args)
    return args

def assign_label(text:str):
    if "cf cf cf" in text:
        return 0
    elif "tq tq tq" in text:
        return 1
    elif "mn mn mn" in text:
        return 2
    

def main():
    args = parse_args()

    
    with open(args.embs_path, 'rb') as f:
    # use pickle.load 
        embs = pickle.load(f)
    # embs = glob.glob(args.embs_path)
    print("load embs finish!")

    
    df = pd.read_csv(args.ctxs_path, sep='\t')

    if args.poison_samples is not None:
        df = df.sample(args.samples, replace=False)


    labels = [assign_label(text) for text in df["text"]]
    ids = ["wiki:"+str(id) for id in df["id"]]
    poinson_embs, clean_embs = [], []
    for id, emb in tqdm(embs):
        if id in ids:
            poinson_embs.append(emb)
        else:
            clean_embs.append(emb)

    sel_clean_embs = random.sample(clean_embs, args.clean_samples)

    labels = labels + [3 for _ in sel_clean_embs]

    # merge
    X = np.array(poinson_embs + sel_clean_embs)
    print(X.shape)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    logging.info("PCA finish!")

    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    label_set = list(set(labels))
    # print(label_set[0])
    colors = {label_set[0]: 'r', label_set[1]: 'g', label_set[2]: 'b', label_set[3]: "yellow"} 

    for i, label in enumerate(labels):
        ax.scatter(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], color=colors[label], label=label, s=0.3)


    ax.set_title('PCA-Reduced Visualization of ctxs Embeddings')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')

    
    fig.savefig(args.save_path, format='png', dpi=600)

    
    print("finish!, figure save as {}".format(args.save_path))

if __name__ == '__main__':
    main()