import pandas as pd
import random
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator

#Read in swapped df
df = pd.read_csv('/mnt/disk1/kywi/cse582/obgyn_notes_swapped_filtered.csv', index_col=0)

#Read in izzy+kyra annotations
#Filter out columns that aren't tagged
#Standardize column names
#Replace multiple whitespaces in col vals with empty string
df_ic = pd.read_csv('/mnt/disk2/izzy/repos/llm-clinical-preds-gender/data/obgyn_notes_F_50_Izzy.csv', index_col=0, header=[0,1])
df_ic = df_ic.fillna('')
whitespace_cols = df_ic.apply(lambda x: x.str.isspace().all())
cols_to_drop = whitespace_cols[whitespace_cols].index
df_ic = df_ic.drop(columns=cols_to_drop)
df_ic.columns = df_ic.columns.map('|'.join).str.replace(' ', '').str.strip('|')

df_kw = pd.read_csv('/mnt/disk1/kywi/cse582/obgyn_notes_F_50_Kyra_annotated_comb.csv', index_col=0, header=[0,1])
df_kw = df_kw.fillna('')
whitespace_cols = df_kw.apply(lambda x: x.str.isspace().all())
cols_to_drop = whitespace_cols[whitespace_cols].index
df_kw = df_kw.drop(columns=cols_to_drop)
df_kw.columns = df_kw.columns.map('|'.join).str.replace(' ', '').str.strip('|')

#Join dfs
df_ann = pd.concat([df_ic, df_kw], axis=1)

names = ['F->M', 'F->non-binary', 'F->transgenderM']
inds_to_check = [6893, 6097, 2205, 3211, 8625, 5132, 7858, 1548, 2518, 1476, 6111, 1828, 2541, 5010, 9110, 7079, 7563,
                 1700, 8629, 4310, 4675, 5067, 5301, 8714, 1588, 4600, 2700, 9287, 7822, 3838]
                #2408, 6203, 3138, 4704, 4152,
                #8044, 6789, 6569, 2971, 1572, 8513, 5965, 9380
out = []
out_names = []
plot_dict = {'Recall': [], 'Precision': [], 'Accuracy': []}

for i,x in enumerate(['F->M', 'F->NB', 'F->TM']):
    recalls = []
    precisions = []
    accuracies = []
    for ind in inds_to_check:
        if f'{ind}|{names[i]}' in df_ann.columns:
            #Fill in orig words to swapped
            df_ann['temp_ann'] = np.where(df_ann[f'{ind}|{names[i]}']=='', df_ann[f'{ind}|original'], df_ann[f'{ind}|{names[i]}'])

            #Get LLM edited, split into series
            llm_swap = pd.Series(df.loc[ind, x].split())

            #Standardize lengths of arrays
            # Remove rows where annotated val is empty string
            #TO DO - If any cells have spaces, split and extend column
            ann = df_ann['temp_ann']
            ann = ann.str.split().explode(ignore_index=True)
            # Remove trailing empty strings
            ann = ann.iloc[:ann.ne('').cumsum().idxmax() + 1]
            ann = ann.dropna()
            ann = ann.rename(f'{ind}_{x}_ann')

            # Find the maximum length
            max_length = max(len(ann), len(llm_swap))
            # Pad the shorter Series with empty strings
            ann = ann.reindex(range(max_length), fill_value="")
            llm = llm_swap.reindex(range(max_length), fill_value="")
            llm = llm.rename(f'{ind}_{x}_llm')
            #print(ann)
            #print(llm)
            
            match = ann==llm
            match = match.rename(f'{ind}_{x}_match')
            #print(ann[~match], llm[~match])
            #print(pd.DataFrame.from_dict({'annotated': ann, 'llm': llm}))

            #True positive - a word that should be changed is correctly changed
            tp = sum((df_ann[f'{ind}|{names[i]}']!='') & match)
            #True negative - a word that should not be changed is correctly not changed
            tn = sum((df_ann[f'{ind}|{names[i]}']=='') & match)
            #False positive - change a word that shouldn't be changed
            fp = sum((df_ann[f'{ind}|{names[i]}']=='') & ~match)
            #False negative - don't change a word that should be changed
            fn = sum((df_ann[f'{ind}|{names[i]}']!='') & ~match)

            #Recall - TP / (TP + FN)
            print(i, tp, tn, fp, fn)
            recall =  tp / (tp + fn)
            #print("Recall: ", recall)
            #Precision - TP / (TP + FP)
            precision = tp / (tp + fp)
            #print("Precision: ", precision)
                
            #Accuracy
            acc = (tp + tn) / (tp + tn + fp + fn)

            recalls.append(recall)
            precisions.append(precision)
            accuracies.append(acc)
            out_names.append(names[i])

            #Save elements for manual inspection
            out.extend([ann, llm, match])

    print(x)
    print('Avg. recall: ', sum(recalls)/len(recalls))
    plot_dict['Recall'].extend(recalls)
    print('Avg. precision: ', sum(precisions)/len(precisions))
    plot_dict['Precision'].extend(precisions)
    print('Avg. accuracy: ', sum(accuracies)/len(accuracies))
    plot_dict['Accuracy'].extend(accuracies)


out_df = pd.concat(out, names=out_names, axis=1)
out_df.to_csv('/mnt/disk1/kywi/cse582/obgyn_notes_F_30_scoring.csv')

print(len(recalls), len(precisions), len(accuracies), len(out_names))
sns.set_theme('paper')
plot_df = pd.DataFrame.from_dict(plot_dict)
plot_df['Group'] = out_names
plot_df_melt = pd.melt(plot_df, id_vars='Group', var_name='Metric')
ax = sns.barplot(plot_df_melt, x='Metric', y='value', hue='Group')
ax.set(title='LLM Gender Perturbation')
sns.move_legend(ax, "lower right")

pairs=[[('Recall', 'F->M'), ('Recall', 'F->non-binary')],
       [('Recall', 'F->M'), ('Recall', 'F->transgenderM')],
       [('Recall', 'F->transgenderM'), ('Recall', 'F->non-binary')],

       [('Precision', 'F->M'), ('Precision', 'F->non-binary')],
       [('Precision', 'F->M'), ('Precision', 'F->transgenderM')],
       [('Precision', 'F->transgenderM'), ('Precision', 'F->non-binary')],

       [('Accuracy', 'F->M'), ('Accuracy', 'F->non-binary')],
       [('Accuracy', 'F->M'), ('Accuracy', 'F->transgenderM')],
       [('Accuracy', 'F->transgenderM'), ('Accuracy', 'F->non-binary')]] 


annotator = Annotator(ax, pairs, data=plot_df_melt, x="Metric", y='value', hue='Group')
annotator.configure(test='t-test_paired', text_format='star', comparisons_correction='holm')
annotator.apply_and_annotate()

plt.savefig('gender_swap_scoring.svg')



