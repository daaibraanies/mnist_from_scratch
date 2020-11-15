import numpy as np
import time
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    exps = np.exp(x)
    return exps/exps.sum(axis=1,keepdims=True)

def df_sigmoid(x):
    #return sigmoid(x)*(1-sigmoid(X))   if the input has not gone through the activation
    return x*(1-x)

def read_data(train_data,train_label,test_data):
    X_train_raw_data = None
    y_train_raw_data = None
    with open(train_data,'r') as fx:
        X_train_raw_data = fx.readlines()
    with open(train_label,'r') as fy:
        y_train_raw_data = fy.readlines()

    return X_train_raw_data,y_train_raw_data,None

def batch_to_array(batch):
    result = np.array([])
    for item in batch:
        item_array= np.fromstring(item,sep=',',dtype=float)
        result = np.append(result,item_array)
    return result.reshape((len(batch),-1))

def batch_generator(X,y,batch_size):
    for i in range(0, len(X), batch_size):
        yield (X[i:i + batch_size],y[i:i+batch_size])

def y_to_catagorical(y,num_of_categories):
    result = np.zeros((len(y),num_of_categories))
    for batch in range(len(y)):
        index = int(y[batch])
        result[batch][index] = 1
    return result

def learn(X,y,batch_size):
    for batch in batch_generator(X,y,batch_size):
        print("Batch: ")
        print("{}".format(batch))

def get_softmax_assumption(output):
    assumption = []
    for item in output:
        assumption.append(np.where(item == np.amax(item)))
    return np.array(assumption,dtype=float).reshape(len(output),1)


def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def CE(actual,predicted):
    result = np.array([])
    for i in range(len(actual)):
        score = 0.0
        for j in range(len(actual[i])):
            score += actual[i][j] * np.log(1e-20+predicted[i][j])
        result = np.append(result,-score)
    return result

def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

def plot_loss(losses):
    plt.figure(figsize=(7,7))
    plt.suptitle("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(len(losses)),losses)
    plt.show()

def plot_accuracy(accuracy):
    plt.figure(figsize=(7,7))
    plt.suptitle("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(len(accuracy)),accuracy)
    plt.show()

if __name__ == "__main__":
    #TODO: change to command line arguments mode leater
    X_train_file_path = "E:\\PyCharmProjects\\HW3\\data\\train\\train_image.csv"
    y_train_file_path = "E:\\PyCharmProjects\\HW3\\data\\train\\train_label.csv"

    start_t = time.time()

    EPOCHS = 25
    BATCH_SIZE = 120
    CLASSES = 10
    LR = 0.5
    DEACAY = 0.25
    FIRST_LAYER_NEURONS = 1024
    SECOND_LAYER_NEURON = 512
    OUTPUT_LAYER_NEURON = 10
    EPSILON = 1e-3
    STAGNATION_THRESHOLD = 10
    np.random.seed(111)

    X_train_raw_data,y_train_raw_data,X_test_raw_data = \
        read_data(X_train_file_path,y_train_file_path,None)

    ENTRIES_NUMBER = len(X_train_raw_data)
    BATCH_NUMBER = np.ceil(ENTRIES_NUMBER/BATCH_SIZE)
    #############################################
    # Phase: network structure
    # create and initialize wight matrices.

    #Weight matrices for each layer
    w_input_1 = 2 * np.random.random((784,FIRST_LAYER_NEURONS)) - 1
    w_1_2 = 2 * np.random.random((FIRST_LAYER_NEURONS,SECOND_LAYER_NEURON)) - 1
    w_2_output = 2 * np.random.random((SECOND_LAYER_NEURON,OUTPUT_LAYER_NEURON)) - 1

    sum_time = 0
    loss_history = []
    accuracy_history = []
    prev_accuracy = 0
    prev_loss = 0
    stagnation_counter = 0
    for epoch in range(1,EPOCHS+1):
        epoch_loss = 0
        epoch_accuracy = 0
        current_batch = 0

        print("Current epoch: {}".format(epoch))

        for batch in batch_generator(X_train_raw_data,y_train_raw_data,BATCH_SIZE):
            current_batch += 1
            batch_time = 0
            batch_start_time = time.time()
            #Phase: data export
            #read data, create batches, nirmolize batches
            X_train,y_train = batch
            X = batch_to_array(X_train)
            y = batch_to_array(y_train)
            X = X/255.
            batch_classes = y
            y = y_to_catagorical(y,CLASSES)
            #############################################
            #Phase: feedforward
            #feed the input through the network
            input = X
            l1_output = sigmoid(np.dot(input,w_input_1))
            l2_output = sigmoid(np.dot(l1_output,w_1_2))
            output = softmax(np.dot(l2_output,w_2_output))

            #############################################
            #Phase: leaning


            #Backpropogation
            """Surprisingly, as cross-entropy is often used with softmax activation function, 
            we do not really have to compute both of these derivatives. Because, some of the
            parts of these derivatives cancel each other as clearly explained in many sources. 
            Thus, predicted value — real value is the result of their product.
            """
            loss = error(output,y)
            epoch_loss+=loss

            output_delta = cross_entropy(output,y)

            l2_error = output_delta.dot(w_2_output.T)
            l2_delta = l2_error*df_sigmoid(l2_output)

            l1_error = l2_delta.dot(w_1_2.T)
            l1_delta = l1_error*df_sigmoid(l1_output)

            #adjust weights
            w_2_output -= LR*np.dot(l2_output.T,output_delta)
            w_1_2 -= LR*np.dot(l1_output.T,l2_delta)
            w_input_1 -= LR*np.dot(input.T,l1_delta)

            sum_time += time.time() - batch_start_time
            avg_batch_time = sum_time/((epoch-1)*BATCH_NUMBER+current_batch)
            predicted_class = get_softmax_assumption(output)

            epoch_accuracy += np.sum(predicted_class == batch_classes)
            global_accuracy = epoch_accuracy/(BATCH_SIZE*current_batch)

            if not (epoch == 1 and current_batch == 1):
                #if global_accuracy - prev_accuracy < EPSILON:
                if abs(prev_accuracy - loss) < EPSILON:
                    stagnation_counter+=1
                    if stagnation_counter >= STAGNATION_THRESHOLD:
                        LR*=DEACAY
                        stagnation_counter = 0
                        print("LR changed to {}".format(LR))
                else:
                    stagnation_counter = 0



            prev_accuracy = global_accuracy
            prev_loss = loss
            print("\tFinished batch: {} ETA: {} min. Accuracy: {} Loss {}".format(current_batch,
                                                                       (EPOCHS+1-epoch)*BATCH_NUMBER*avg_batch_time/60,
                                                                       epoch_accuracy/(BATCH_SIZE*current_batch),
                                                                        loss))

        accuracy_history.append(epoch_accuracy/ENTRIES_NUMBER)
        loss_history.append((epoch_loss/BATCH_NUMBER))
        print("Epoch [{}] loss: {} accuracy: {}".format(epoch, epoch_loss/BATCH_NUMBER,epoch_accuracy/ENTRIES_NUMBER))
    plot_loss(loss_history)
    plot_accuracy(accuracy_history)
    print("FULL TIME: {} min.".format((time.time()-start_t)/60))

    print("QAAA")
