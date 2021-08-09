import json

f = open("synapse.json", "r").read()
lines = f.split("}{")

for i, l in enumerate(lines):
    print(f"\n{i}\n")

    if i != 0:
        l = "{" + l
    if i != len(lines)-1:
        l += "}"
    
    json_line = json.loads(l)
    print(json_line["_source"])
