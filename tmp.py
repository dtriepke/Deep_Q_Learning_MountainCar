import json

# a Python object (dict):
x = {
  "name": "John",
  "age": 30,
  "city": "New York"
}

x = {}
x["reward"] = ""
x["position"] = {"max_position": "",
                 "final_positoion": ""}




# convert into JSON:
y = json.dumps(x)

# the result is a JSON string:
print(y)

f = open("model/tmp.json", "w+")
f.write(y)
f.close()