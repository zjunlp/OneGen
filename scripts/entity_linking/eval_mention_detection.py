import re
import jsonlines
import fire

# Here we use the quadruple [doc_id, surface_form, qid, surface_form_rank]
# as our basic unit.

def my_f1(gt, pred):
    new_gt = []
    gt_map = {}
    for item in gt:
        if item in gt_map:
            gt_map[item] += 1
        else:
            gt_map[item] = 0
        new_gt.append(
            f"{gt_map[item]}-{item}"
        )
    new_pred = []
    pred_map = {}
    for item in pred:
        if item in pred_map:
            pred_map[item] += 1
        else:
            pred_map[item] = 0
        new_pred.append(
            f"{pred_map[item]}-{item}"
        )
    # print(new_gt, new_pred)
    gt_set = set(new_gt)
    pred_set = set(new_pred)

    # true positives (TP)
    true_positives = len(gt_set & pred_set)
    
    # false positives (FP) 和 false negatives (FN)
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)

    if true_positives == 0:  
        precision = 0
        recall = 0
        f1_score = 0
    else:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score
 
def extract_mentions(text, pattern:str) -> list:
    pattern = re.compile(r'<MENTION>(.*?)</MENTION>', re.DOTALL)
    matches = pattern.findall(text)
    return matches

def evaluate_one_file(file_name:str):
    gt = list()
    pred = list()
    with jsonlines.open(file_name, 'r') as reader:
        for item in reader:
            text = item['text']
            for label in item["labels"]:
                start, end = label['span']
                gt.append(
                    text[start:end]
                )
            predicted = extract_mentions(item["output"], pattern=r"<MENTION>(.*?)</MENTION>")
            pred.extend(predicted)
    precision, recall, f1_score = my_f1(gt, pred)
    return f1_score, precision, recall
  
def main(file_path:str):
    f1_score, precision, recall = evaluate_one_file(file_path)
    print(f"Mention Detection: {file_path}\nf1: {f1_score}\nprecision: {precision}\nrecall: {recall}")

if __name__ == '__main__':
    fire.Fire(main)
    