import os

fdrs = ['005', '01', '025', '05']
dirs = ['fdr%s' % fdr for fdr in fdrs]
fdrs_vals = [0.05, 0.1, 0.25, 0.5]

for i, fdr in enumerate(fdrs):
    dir_path = os.path.join('../../test_dataset', dirs[i])
    for dir_name in os.listdir(dir_path):
        path = os.path.join(dir_path, dir_name)
        os.system('python halla.py -X %s/X_line_500_50.txt -Y %s/Y_line_500_50.txt -m pearson -o halla_%s -q %f --fnt 0.1' % (
            path, path, dir_name, fdrs_vals[i]))
        break
        os.system('python halla.py -X %s/X_line_500_50.txt -Y %s/Y_line_500_50.txt -m pearson -a AllA -o alla_%s -q %f --fnt 0.1' % (
            path, path, dir_name, fdrs_vals[i]))
