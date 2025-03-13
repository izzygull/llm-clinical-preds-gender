"""
Get the MIMIC-IV OB/GYN notes, and save the relevant notes to a CSV containing the columns
* note_id
* patient_id
* hadm_id
* context_note (Sex, Service, Chief Complaint, History of Present Illness, Past Medical History)
* procedure_note
* dignosis_note
* transgender_mention (whether the note contains "transgender" or "transsexual" "trans man" "ftm" "nonbinary" "gender dysphoria")
"""

import pandas as pd
import re
from os import getenv
from pathlib import Path
from tqdm import tqdm

MIMIC_4_DIR = Path(getenv("MIMIC_4_DIR"))
SERVICE_STR = "Service: OBSTETRICS/GYNECOLOGY"

notes_df = pd.read_csv(MIMIC_4_DIR / 'note' / 'discharge.csv')
print(f"Originally {len(notes_df)} notes from {len(set(notes_df['hadm_id']))} admissions")
notes_df = notes_df[notes_df['text'].str.contains(SERVICE_STR)]
print(f"We have {len(notes_df)} OB/GYN notes from {len(set(notes_df['hadm_id']))} admissions")

sex_pat = re.compile(r'(Sex:   [F|M|___])')
section_heads = {
    'complaint': ['Chief Complaint', '(?:Major Surgical or Invasive Procedure|Major ___ or Invasive Procedure|___ or Invasive Procedure|History of Present Illness)'],
    'hxpi': ['History of Present Illness', '(?:Past Medical History|Social History|Physical Exam|Pertinent Results|___ Medical History)'],
    'past_history': ['Past Medical History', '(?:Social History|Physical Exam|___ Exam|Pertinent Results|Brief Hospital Course)'],
    'procedure': ['Major Surgical or Invasive Procedure', '(?:History of Present Illness|Physical Exam|Pertinent Results|Brief Hospital Course)'],
    'diagnosis': ['Discharge Diagnosis', '(?:Discharge Condition|___ Condition)'],
}
regex_template = r"{}:\n(.*?){}"
regex_dict = {section_name: re.compile(regex_template.format(section_start, section_end), re.MULTILINE | re.DOTALL) for section_name, (section_start, section_end) in section_heads.items()}

final_rows = []

for i, row in tqdm(notes_df.iterrows(), total=len(notes_df)):
    try:
        row_info = {'note_id': row['note_id'], 'subject_id': row['subject_id'], 'hadm_id': row['hadm_id'], 'service': SERVICE_STR}
        service = SERVICE_STR
        row_info['listed_sex'] = re.findall(sex_pat, row['text'])[0]

        for section_name, pattern in regex_dict.items():
            section_text = re.findall(pattern, row['text'])
            if not section_text:
                row_info[section_name] = None
                if section_heads[section_name][0] in row['text']:
                    raise IndexError # wrong error but w/e
            else:
                row_info[section_name] = section_text[0].strip()
        row_info['trans_mention'] = bool(re.findall("transgender|transsexual|Sex:   ___|Sex:   M|nonbinary|non-binary|trans man|ftm|gender dysphoria", row['text'], re.IGNORECASE))

        final_rows.append(row_info)
    except IndexError:
        # In this case, our section match didn't work, but the section header was present
        print(row['text'])
        print(pattern)
        print(row_info)
        exit()


obgyn_note_df = pd.DataFrame(final_rows)
print(sum(obgyn_note_df['trans_mention']))
# obgyn_note_df.to_csv("obgyn_notes.csv", index=False)