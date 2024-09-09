import jsonlines
import fire

def main(file_path:str):
    total = 0
    acc = 0
    with jsonlines.open(file_path, 'r') as reader:
        for item in reader:
            gt_qid_list = item['gt_qid_list']
            pred_qid_list = item['pred_qid_list']
            assert len(gt_qid_list) == len(pred_qid_list)
            for gt, pred in zip(gt_qid_list, pred_qid_list):
                total += 1
                if gt == pred:
                    acc += 1
    print(f"entity disambiguation accuracy for `{file_path}`:\n{acc/total}")
    
if __name__ == '__main__':
    fire.Fire(main)