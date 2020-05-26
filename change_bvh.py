import os #библиотека для работы с операционной системой (в частности для открытия и закрытия файлов)
bvh_dirs = [13,14,15,86]

for cur_dir in bvh_dirs:
    orig_dir = 'bvhs/{}/'.format(cur_dir)
    change_dir = 'change_bvhs/{}/'.format(cur_dir)

    for _, _, files in os.walk(orig_dir): #обход файлов а указанной директории
        os.makedirs(change_dir)
        for file in files:
            cur_bvh = open(orig_dir + file, 'r') #читаем исходный файл
            new_bvh = open(change_dir + file, 'w') #создаем новый файл
            is_motion = False
            for num, line in enumerate(cur_bvh.readlines()): #построчно записываем 
                if (not is_motion):
                    new_bvh.write(line) #записываем все что было до записи движений
                if (is_motion and (num % 10 == 0)):
                    new_bvh.write(line) # записываем каждые 10 фреймов в новый файл
                if (line[0:6] == "MOTION"):
                    is_motion = True
            cur_bvh.close()
            new_bvh.close()