import csv

l = []
new_l = []

num = 0
if __name__ == '__main__':
    with open("E:/data/adult.data", "rt") as f:
        cr = csv.reader(f)
        for row in cr:
            new_row = []
            new_row.insert(0, str(num).zfill(6))
            num += 1
            for item in row:
                new_row.append(item.strip())
            l.append(new_row)
            print(row)

    with open("E:/data/ding.data", "w", newline='') as new_f:
        cw = csv.writer(new_f, delimiter=',')
        for item in l:
            # print(item)
            cw.writerow(item)
