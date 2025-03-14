import pandas as pd
from transformers import pipeline
import random

#Load prompts to be tagged
#df = pd.read_csv("/mnt/disk1/kywi/cse582/obgyn_notes_F_KW.csv", index_col=0).sample(frac=1, random_state=42)
df = pd.read_csv("/mnt/disk1/kywi/cse582/obgyn_notes_F_KW.csv", index_col=0)

df_kw = pd.read_csv("/mnt/disk1/kywi/cse582/obgyn_notes_F_50_Kyra.csv", index_col=0)
df_ic = pd.read_csv("/mnt/disk1/kywi/cse582/obgyn_notes_F_50_Izzy.csv", index_col=0)
ind_kw = [int(c) for c in df_kw.columns if '.' not in c]
ind_ic = [int(c) for c in df_ic.columns if '.' not in c]
print(df.index)
reorder_inds = list(set(df.index.to_list()) - set(ind_kw+ind_ic))
print(len(reorder_inds))
random.seed(42)
random.shuffle(reorder_inds)
new_inds = ind_kw+ind_ic+reorder_inds
df = df.reindex(new_inds)

print(df)
tests = df['prompt'].to_list()

example_template = """##### Example N #####
        
Original medical history:
**EXN**

Changed medical history:
**SWAPN**
<|endoftext|>
"""

ids = ['7134', '6271', '7368', '144', '1648', '4292', '3726']
switches = ['F->M', 'F->NB', 'F->TM']


#Get files in directory that end in F.txt
prompts = []
labels = []
inds = []

for i,t in enumerate(tests):
    for j in range(3):
        all_temps = []
        with open(f"/mnt/disk1/kywi/cse582/{switches[j]}_prompt.txt", "r") as file3:
            prompt = file3.read()

        for k in range(5):
            #Load the note
            with open(f"/mnt/disk1/kywi/cse582/example_notes/{ids[k]}_F.txt", "r") as file:
                orig = file.read()
            with open(f"/mnt/disk1/kywi/cse582/example_notes/{ids[k]}_{switches[j]}.txt", "r") as file2:
                switch = file2.read()

            #Fill in the example template
            temp = example_template.replace('**EXN**', orig).replace('**SWAPN**', switch).replace('##### Example N #####', f'##### Example {k+1} #####')
            all_temps.append(temp)

        #Swap examples into the main prompt
        examples = '\n'.join(all_temps)
        prompt = prompt.replace('**EX**', examples).replace('**TEST**', t)
        prompts.append(prompt)
        labels.append(switches[j])
        inds.append(df.index[i])
    

messages = [{"role": "user", "content": p} for p in prompts]
outputs = []

#Call model
for i,m in enumerate(messages):
    print(i)
    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True, device='cuda', max_new_tokens=1000)
    output = pipe([m])

    #Get output
    for o in output:
        new_prompt = o['generated_text'][1]['content']
        print(m['content'][-500:])
        print('********')
        print(new_prompt)
        outputs.append(new_prompt)
    
    if i%100==0:
        #Save output
        df_out = pd.DataFrame.from_dict({'label': labels[:i+1], 'new_prompt': outputs, 'index': inds[:i+1]})
        df_out = df_out.pivot(index='index', columns='label')
        df_out.to_csv('/mnt/disk1/kywi/cse582/obgyn_notes_F_swapped.csv')