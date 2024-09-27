import argparse
from model_utils import get_model, get_tokenizer
from test_utils import copy_c4_evaluation, phone_book_evaluation, squad_evaluation 
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--eval_task', choices=["c4_copy","squad","phone_book"], type=str, required=True, help="evaluation task")
parser.add_argument('--text_order', choices=["standard","random"], type=str, help="only applies when eval_task = c4_copy. Order of the text to copy. When text_order = random, we randomly change the order of the text. Otherwise, keeps the same order.", default="standard")

args = parser.parse_args()

models = ["state-spaces/mamba-370m","state-spaces/mamba-1.4b","state-spaces/mamba-2.8b","state-spaces/mamba2-370m","state-spaces/mamba2-1.3b","state-spaces/mamba2-2.7b","EleutherAI/pythia-410m","EleutherAI/pythia-1.4b","EleutherAI/pythia-2.8b"]
eval_num_batches = 3
eval_batch_size = 32
eval_len = [10, 20, 40, 80, 160, 320]

if args.eval_task == "c4_copy":
    eval_results = []
    print("Starting c4_copy task evals...")
    for model_name in models:
        tokenizer = get_tokenizer()
        model = get_model(model_name)
        for s in eval_len:
            entry = {}
            str_acc_mean_list, str_acc_std_list = copy_c4_evaluation(s, s, args.text_order, eval_num_batches, eval_batch_size, model_name, model, tokenizer)
            entry['model'] = model_name
            entry['eval_length'] = s
            entry['text_order'] = args.text_order
            entry['avg_accuracy'] = str_acc_mean_list[0]
            entry['std_acuracy'] = str_acc_std_list[0]
            eval_results.append(entry)
            df = pd.DataFrame(eval_results)
            df.to_csv(f"c4_copy_{args.text_order}_eval_results.csv", index=False)
            print(entry)
    df = pd.DataFrame(eval_results)
    df.to_csv(f"c4_copy_{args.text_order}_eval_results.csv", index=False)
elif args.eval_task == "phone_book":
    eval_results = []
    print("Starting phone_book task evals...")
    for model_name in models:
        tokenizer = get_tokenizer()
        model = get_model(model_name)
        for s in eval_len:
            entry = {}
            str_acc_mean_list, str_acc_std_list = phone_book_evaluation(s, s, eval_num_batches, eval_batch_size, model_name, model,tokenizer)
            entry['model'] = model_name
            entry['eval_length'] = s
            entry['avg_accuracy'] = str_acc_mean_list[0]
            entry['std_acuracy'] = str_acc_std_list[0]
            eval_results.append(entry)
            df = pd.DataFrame(eval_results)
            df.to_csv("phone_book_eval_results.csv", index=False)
            print(entry)
    df = pd.DataFrame(eval_results)
    df.to_csv("phone_book_eval_results.csv", index=False)
elif args.eval_task == "squad":
    eval_results = []
    context_sizes = ['38 - 41','79 - 81','120','160','199 - 200','239 - 241','277 - 283','314 - 326']
    print("Starting squad task evals...")
    for model_name in models:
        tokenizer = get_tokenizer()
        model = get_model(model_name)
        em_list, f1_list, std_em_list, std_f1_list = squad_evaluation(model_name,model,tokenizer)
        for i in range(len(context_sizes)):
            entry = {}
            entry['model'] = model_name
            entry['context_size'] = context_sizes[i]
            entry['avg_exact_match_score'] = em_list[i]
            entry['avg_f1_score'] = f1_list[i]
            entry['std_exact_match_score'] = std_em_list[i]
            entry['std_f1_score'] = std_f1_list[i]
            eval_results.append(entry)
            print(entry)
        df = pd.DataFrame(eval_results)
        df.to_csv("squad_eval_results.csv", index=False)
    df = pd.DataFrame(eval_results)
    df.to_csv("squad_eval_results.csv", index=False)
else:
    raise ValueError(f"Non-valid evaluation task {args.eval_task}")
print("DONE")