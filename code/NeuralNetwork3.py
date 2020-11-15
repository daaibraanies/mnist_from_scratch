import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy
import sys


def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    exps = np.exp(x)
    return exps/exps.sum(axis=1,keepdims=True)

def df_sigmoid(x):
    return x*(1-x)

def read_data(train_data,train_label,test_data):
    X_train_raw_data = None
    y_train_raw_data = None
    y_test_raw_data = None
    with open(train_data,'r') as fx:
        X_train_raw_data = fx.readlines()
    with open(train_label,'r') as fy:
        y_train_raw_data = fy.readlines()
    with open(test_data,'r') as ty:
        y_test_raw_data = ty.readlines()

    return X_train_raw_data,y_train_raw_data,y_test_raw_data

def batch_to_array(batch):
    result = np.array([])
    for item in batch:
        item_array= np.fromstring(item,sep=',',dtype=float)
        result = np.append(result,item_array)
    return result.reshape((len(batch),-1))

def batch_generator(X,y,batch_size):
    for i in range(0, len(X), batch_size):
        yield (X[i:i + batch_size],y[i:i+batch_size])

def Xonly_batch_generator(X,batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i + batch_size]

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

def plot_loss(loss,val_loss):
    plt.figure(figsize=(7,7))
    plt.suptitle("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(len(loss)),loss,'b--',label="train loss")
    plt.plot(range(len(val_loss)), val_loss,'r',label="validation loss")
    plt.legend(loc='best')
    plt.show()

def plot_accuracy(acc,val_acc):
    plt.figure(figsize=(7,7))
    plt.suptitle("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(len(acc)), acc,'b--',label="train accuracy")
    plt.plot(range(len(val_acc)), val_acc,'r',label="validation accuracy")
    plt.legend(loc='best')
    plt.show()

def fit(network,X_train_raw_data,y_train_raw_data,X_val,y_val,lr,start_time,verbose=False,earlystop=None):
    sum_time = 0
    loss_history = []
    accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []
    stagnation_counter = 0
    prev_val_accuracy = 0
    best_val_accuracy = 0
    prev_val_loss = None
    y_val = batch_to_array(y_val)
    checkpoint = copy.deepcopy(network)

    for epoch in range(1,EPOCHS+1):
        epoch_loss = 0
        epoch_accuracy = 0
        current_batch = 0
        if verbose:
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
            l1_output = sigmoid(np.dot(input,network['w_input_1']))
            l2_output = sigmoid(np.dot(l1_output,network['w_1_2']))
            output = softmax(np.dot(l2_output,network['w_2_output']))

            #############################################
            #Backpropogation
            loss = error(output,y)
            epoch_loss += loss

            output_delta = cross_entropy(output,y)

            l2_error = output_delta.dot(network['w_2_output'].T)
            l2_delta = l2_error*df_sigmoid(l2_output)

            l1_error = l2_delta.dot(network['w_1_2'].T)
            l1_delta = l1_error*df_sigmoid(l1_output)

            #adjust weights
            network['w_2_output'] -= lr*np.dot(l2_output.T,output_delta)
            network['w_1_2'] -= lr*np.dot(l1_output.T,l2_delta)
            network['w_input_1'] -= lr*np.dot(input.T,l1_delta)

            sum_time += time.time() - batch_start_time
            avg_batch_time = sum_time/((epoch-1)*BATCH_NUMBER+current_batch)
            predicted_class = get_softmax_assumption(output)

            epoch_accuracy += np.sum(predicted_class == batch_classes)

            if verbose:
                print("\tFinished batch: {} ETA: {} min. Accuracy: {} Loss {}".format(current_batch,
                                                                       (EPOCHS+1-epoch)*BATCH_NUMBER*avg_batch_time/60,
                                                                       epoch_accuracy/(BATCH_SIZE*current_batch),
                                                                       loss/BATCH_SIZE))


        accuracy_history.append(epoch_accuracy/(BATCH_NUMBER*BATCH_SIZE))
        loss_history.append(epoch_loss/(BATCH_NUMBER*BATCH_SIZE))

        val_predict = predict(network,X_val)
        val_loss = error(val_predict,y_val)/len(y_val)
        val_predict = get_softmax_assumption(val_predict)
        val_accuracy = np.sum(val_predict == y_val)/len(y_val)

        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        #TODO:  (val_loss)
        #         if prev_val_loss is not None:
        #             if prev_val_loss - val_loss < EPSILON:
        #
        if best_val_accuracy is not None:
            if val_accuracy - best_val_accuracy < EPSILON:
                stagnation_counter += 1
                if stagnation_counter >= STAGNATION_THRESHOLD:
                    if lr > MIN_LR:
                        new_lr = lr*DECAY
                        lr = new_lr if new_lr >= MIN_LR else MIN_LR
                        stagnation_counter = 0
                        if verbose:
                            print("\t\tLearning rate was changed to: {}".format(lr))
            else:
                stagnation_counter = 0

        #TODO: best_val_acc was added
        if val_accuracy > best_val_accuracy:
            checkpoint = copy.deepcopy(network)
            best_val_accuracy = val_accuracy
            if verbose:
                print("Checkpoint has been made.")

        prev_val_loss = val_loss
        prev_val_accuracy = val_accuracy
        if verbose:
            print("Validation information: ")
            print("\tValidation loss: {} | Validation accuracy: {}".format(val_loss,val_accuracy))

        if earlystop is not None:
            if earlystop(start_time,val_accuracy):
                break

    history = {}
    history['loss'] = loss_history
    history['acc'] = accuracy_history
    history['val_loss'] = val_loss_history
    history['val_acc'] = val_accuracy_history

    return history,checkpoint

def predict(network,X):
    prediction = None
    for batch in Xonly_batch_generator(X, BATCH_SIZE):
        X = batch
        X = batch_to_array(X)
        X = X / 255.

        input = X
        l1_output = sigmoid(np.dot(input, network['w_input_1']))
        l2_output = sigmoid(np.dot(l1_output, network['w_1_2']))
        output = softmax(np.dot(l2_output, network['w_2_output']))
        if prediction is None:
            prediction = output
        else:
            prediction = np.concatenate((prediction,output),axis=0)

    return prediction

def create_prediction_file(prediction,test_data,filename):
    with open(filename,'w') as output:
        for i in range(len(test_data)):
            output.write(str(int(prediction[i][0]))+'\n')

def early_stop(start_time,validation_accuracy):
    running_minutes = (time.time() - start_time)/60
    if validation_accuracy >= ACCURACY_LIMIT:
        return True
    if running_minutes >= MINUTE_LIMIT:
        return True
    return False


if __name__ == "__main__":
    start_t = time.time()
    #TODO: change to command line arguments mode leater
    X_train_file_path = "train_image.csv"
    y_train_file_path = "train_label.csv"
    y_test_file_path = "test_image.csv"

    verbose = False
    best_model = None

    EPOCHS = 150
    BATCH_SIZE = 120
    CLASSES = 10

    LR = 0.5
    DECAY = 0.5

    MINUTE_LIMIT = 26    #TODO: turn back to 26
    ACCURACY_LIMIT = 0.94

    FIRST_LAYER_NEURONS = 1024
    SECOND_LAYER_NEURON = 512
    OUTPUT_LAYER_NEURON = CLASSES

    EPSILON = 1e-3
    MIN_LR = 1e-3
    STAGNATION_THRESHOLD = 10
    np.random.seed(111)

    X_train_raw_data,y_train_raw_data,X_test_raw_data = \
        read_data(X_train_file_path,y_train_file_path,y_test_file_path)

    X_train_data,X_validation_data,y_train_data,y_validation_data = train_test_split(X_train_raw_data,y_train_raw_data,
                                                                                    test_size=0.1,
                                                                                    shuffle=True)

    ENTRIES_NUMBER = len(X_train_data)
    BATCH_NUMBER = np.ceil(ENTRIES_NUMBER/BATCH_SIZE)

    network = {}
    network['w_input_1'] = 2 * np.random.random((784,FIRST_LAYER_NEURONS)) - 1
    network['w_1_2'] = 2 * np.random.random((FIRST_LAYER_NEURONS,SECOND_LAYER_NEURON)) - 1
    network['w_2_output'] = 2 * np.random.random((SECOND_LAYER_NEURON,OUTPUT_LAYER_NEURON)) - 1

    history,model = fit(
        network,
        X_train_data,y_train_data,
        X_validation_data,y_validation_data,
        LR,
        start_t,
        verbose=verbose,
        earlystop=early_stop
    )

    if verbose:
        plot_loss(history['loss'],history['val_loss'])
        plot_accuracy(history['acc'],history['val_acc'])

        test_labels = None
        with open("test_label.csv",'r') as tl:
            test_labels = tl.readlines()
        test_labels = batch_to_array(test_labels)

    prediction = predict(model,X_test_raw_data)
    prediction = get_softmax_assumption(prediction)
    create_prediction_file(prediction,X_test_raw_data,'test_predictions.csv')

    if verbose:
        test_accuracy = np.sum(prediction == test_labels)/len(test_labels)
        print("Test accuracy: {}".format(test_accuracy))



