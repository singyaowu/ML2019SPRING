
fp_r = open('predict.csv', 'r')
fp_w = open('prediction.csv', 'w')

lines = fp_r.readlines()[1:]
labels = [ line.split(',')[1] for line in lines ] 
print(labels)
fp_w.write("id,label\n")
for i in range(len(labels)):
    fp_w.write( str(i) + ',' + labels[i])