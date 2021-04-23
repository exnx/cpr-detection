import json



file = "missing_list.txt"
out = 'missing_list.json'

with open(file, 'r') as f:
    content = f.readlines()
    content = [x.strip() for x in content]

missing_dict = {}

for key in content:
    missing_dict[key] = "chest compression"


with open(out, 'w', encoding='utf-8') as f:
    json.dump(missing_dict, f, ensure_ascii=False, indent=4)



