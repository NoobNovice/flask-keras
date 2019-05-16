from datetime import datetime
import re
from random import shuffle
import random
timestamp = datetime.now()
str_id = str(timestamp)
str_id = re.sub(r'[^0-9]', r'', str_id)
str_id = str_id[4:]
for i in range(len(str_id)):
    index = random.randint(0,len(str_id)-1)
    temp = str_id[i]
    str_id[i] = str_id[index]
    str_id[index] = temp

print(str_id)
