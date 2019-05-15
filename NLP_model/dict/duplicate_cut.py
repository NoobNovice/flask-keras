import sys

data_in = [line.strip() for line in open(sys.argv[1], 'r', encoding='utf-8')]
text_out = open(sys.argv[2],"w", encoding='utf-8')
for index in range(len(data_in)):
    try:
        if data_in[index] == data_in[index + 1]:
            continue
        else:
            text_out.write(data_in[index] + "\n")
    except IndexError:
        text_out.writelines(data_in[index])
        break
text_out.close()