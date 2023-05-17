from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.neural_network import MLPClassifier


def read_file(file_name):
    # 6 column starting with 0 is flirtInterests_chat

    _ = []

    with open(file_name, encoding="utf8") as my_file:
        my_file.readline()
        while True:
            line = my_file.readline().strip()
            if line == '':
                break
            parts = line.split(',')
            _.append(list(map(str, parts[:-1])) + parts[-1:])

    # Returning the dataset back
    return _


def separate_attributes(dataset2):
    set_x2 = []
    for data2 in dataset2:
        row2 = []
        for i2 in range(len(data2)):
            if i2 != 5:
                row2.append(data[i2])
        set_x2.append(row2)
    return set_x2


def separate_class(dataset1):
    set_x1 = []
    for data1 in dataset1:
        row1 = []
        for i1 in range(len(data1)):
            if i1 == 5:
                row1.append(data1[i1])
        set_x1.append(row1)
    return set_x1


if __name__ == '__main__':
    dataset = read_file("Data\lovoo_v3_users_instances.csv")

    # region Train and Test sets
    train_set = dataset[:int(0.70 * len(dataset))]
    test_set = dataset[int(0.70 * len(dataset)):]
    # endregion

    # region Encoder
    for_encoder_to_fit = []
    for data in dataset:
        row = []
        for i in range(len(data)):
            if i != 5:
                row.append(data[i])
        for_encoder_to_fit.append(row)
    encoder = OrdinalEncoder()
    encoder.fit(for_encoder_to_fit)
    # endregion
    # region Train set
    train_x = separate_attributes(train_set)
    train_x = encoder.transform(train_x)

    train_y = separate_class(train_set)
    # endregion
    # region Test set
    test_x = separate_attributes(test_set)
    test_x = encoder.transform(test_x)

    test_y = separate_class(test_set)
    # endregion

    # region Classifier
    classifier = MLPClassifier(hidden_layer_sizes=300,
                               activation="tanh",
                               learning_rate_init=0.0001,
                               max_iter=1000,
                               random_state=0)
    classifier.fit(train_x, train_y)
    # endregion

    # region Testing
    accuracy_counter = 0
    for data, real_class in zip(test_x, test_y):
        prediction = classifier.predict([data])
        if prediction == real_class:
            accuracy_counter += 1

    print(f"The accuracy is: {accuracy_counter / len(test_set)}")
    # endregion

    # Prediction on certain input
    prediction = ["F","19","5","1001","200","TRUE","FALSE","FALSE","FALSE","TRUE","1","FALSE","TRUE","FALSE","FALSE","FALSE","FALSE","1","FALSE","FALSE","TRUE","FALSE","TRUE","54",'FALSE','TRUE','TRUE',"FALSE"]
    prediction = encoder.transform([prediction])
    print(classifier.predict_proba(prediction))
    print(classifier.predict(prediction))
    print()
