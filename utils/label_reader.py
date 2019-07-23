import json

size = 0
lst = []
counter = 1

with open("2871_labels.json", 'r') as json_file:
    data = json.load(json_file)
    for path in data:
        """
        if data[path] not in lst:
            lst.append(data[path])
            """
        ##if data[path].find("-") > -1:
          ##  data[path] = data[path].replace("-", "_")
        if data[path] not in lst:
            lst.append(data[path])
            
        if data[path] == "traffic_light":
            print(counter)
        
        counter += 1

print("Number of elements:", len(data))

for element in lst:
    print(element)
