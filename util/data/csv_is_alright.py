import csv

l = []
num = 0
if __name__ == '__main__':
    with open("E:/data/adult.data", "rt") as f:
        cr = csv.reader(f)
        for row in cr:
            row.insert(0, str(num).zfill(6))
            l.append(row)
            num += 1

    with open("E:/data/ding.data", "wt") as new_f:
        cw = csv.writer(new_f)
        for item in l:
            print(item)
            cw.writerow(item)
