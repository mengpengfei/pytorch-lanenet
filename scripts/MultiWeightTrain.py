import os
if __name__ == '__main__':
    for i in range(1,10):
        for j in range(1,10):
            for m in range(1,10):
                for n in range(1,10):
                    if i+j+m+n==10:
                        print('python lanenet/train.py --dataset /workspace/mogo_data/index '
                              '--lr 0.001 --val True --bs 16 --save ./checkpoints --w1 {4} --w2 {5} --w3 {6} --w4 {7} --epochs 30 2>&1 '
                              '> logs/mogodata{0}_{1}_{2}_{3}.log'.format(i/10,j/10,m/10,n/10,i/10,j/10,m/10,n/10))
                        # print("{0}_{1}_{2}_{3}".format(i/10,j/10,m/10,n/10))
                        # os.popen('python lanenet/train.py --dataset /workspace/mogo_data/index '
                        #          '--lr 0.001 --val True --bs 16 --save ./checkpoints --w1 {4} --w2 {5} --w3 {6} --w4 {7} 2>&1 '
                        #          '> mogodata{0}_{1}_{2}_{3}.log'.format(i/10,j/10,m/10,n/10,i/10,j/10,m/10,n/10))

