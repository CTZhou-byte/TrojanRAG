import argparse
import os
import pickle


def parse_arguments():
    parser = argparse.ArgumentParser(description="Gather embs")
    parser.add_argument("--root_dir", type=str, default=None, help="source dir")
    parser.add_argument("--attack_corpus_path", type=str, default=None)
    parser.add_argument("--link_dir_name", type=str, default="gather_embs")
    parser.add_argument("--attack_num", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()

    normal_emb_files = sorted(
        [
            os.path.join(args.root_dir, i, i)
            for i in os.listdir(args.root_dir)
            if i.startswith("embs_") and "attack" not in i
        ]
    )
    print(f"Normal emb files: {normal_emb_files}")

    link_dir = os.path.join(args.root_dir, args.link_dir_name)
    os.makedirs(link_dir, exist_ok=True)
    link_dir = os.path.abspath(link_dir)
    if args.attack_corpus_path:
        # replace the first N embs
        emb_attack = pickle.load(open(args.attack_corpus_path, "rb"))[: args.attack_num]

        for idx in range(len(emb_attack)):
            wiki_idx = emb_attack[idx][0].split(":")[-1]
            emb_attack[idx] = (f"attack:{wiki_idx}", emb_attack[idx][1])

        emb_attack_path = os.path.join(link_dir, "attack_embs")
        with open(emb_attack_path, "wb") as file:
            pickle.dump(emb_attack, file)
            print(f"Attacked embs saved to {emb_attack_path}.")

    # link the rest embs
    for emb_file in normal_emb_files:
        # src_path = os.path.abspath(emb_file)
        src_path = "../" + "/".join(emb_file.split("/")[-2:])
        tgt_path = os.path.join(link_dir, os.path.basename(emb_file))
        os.symlink(src_path, tgt_path)
        print(f"Linked {src_path} to {tgt_path}")


if __name__ == "__main__":
    main()
