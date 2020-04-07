import numpy as np
from matplotlib import pyplot as plt
def main():
    a = np.loadtxt("run-.-tag-Train_AverageReturn.csv", delimiter = ",", skiprows = 1)
    plt.plot(a[:,1], a[:,2], label='half cheetah batchsize 40000, lr 0.005')
    a = np.loadtxt("run-10000tag-Train_AverageReturn.csv", delimiter = ",", skiprows = 1)
    plt.plot(a[:,1], a[:,2], label='half cheetah batchsize 10000, lr 0.005')
    a = np.loadtxt("batch50k_lr02train.csv", delimiter = ",", skiprows = 1)
    plt.plot(a[:,1], a[:,2], label='half cheetah batchsize 50000, lr 0.02')
    plt.legend()
    plt.savefig("halfcheetah_train_avg.png")
    plt.close()

    b = np.loadtxt("batch10000.csv", delimiter = ",", skiprows = 1)
    plt.plot(b[:,1], b[:,2], label='half cheetah batchsize 10000, lr 0.005')
    b = np.loadtxt("batch40000.csv", delimiter = ",", skiprows = 1)
    plt.plot(b[:,1], b[:,2], label='half cheetah batchsize 40000, lr 0.005')
    b = np.loadtxt("batch50k_lr02eval.csv", delimiter = ",", skiprows = 1)
    plt.plot(b[:,1], b[:,2], label='half cheetah batchsize 50000, lr 0.02')
    plt.legend()
    plt.savefig("halfcheetah_eval_max.png")
    plt.close()



if __name__ == "__main__":
    main()
