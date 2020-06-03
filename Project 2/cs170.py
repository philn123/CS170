import copy
import math

import numpy as np


# https://www.codecademy.com/articles/normalization
# I chose z-score normalization (value - mean of feature) / standard deviation of feature
# minMax can be affected by outliers
# chose to use numpy for efficiency when doing std and means, all found in numpy docs
def normalize(np_instances, number_of_instances, number_of_features):
    # find mean of all columns, then delete first column because that is the class label
    # axis=0 means collapse data by column, so if taking mean, it means go to your data and squish it to make it one row
    arr = np.mean(np_instances, axis=0)
    mean_for_instances = np.delete(arr, 0)

    # find standard deviation of all columns, then delete first column because that is the class label
    arr = np.std(np_instances, axis=0)
    std_for_instances = np.delete(arr, 0)

    for i in range(0, number_of_instances):
        # start at 1 to skip the class label
        # add 1 to number of features to make sure to get all values
        for j in range(1, number_of_features + 1):
            np_instances[i][j] = ((np_instances[i][j] - mean_for_instances[j - 1]) / std_for_instances[j - 1])

    # we just modified original array
    return np_instances


# input: training data, point that was left out(index of row in training data, basically line number),
# number of instances, feature subset
# output: class label of nearest neighbor
# https://www.dataquest.io/blog/k-nearest-neighbors-in-python/
# used this source to see how euclidean distance worked, code for nearest neighbor was not used
def nearest_neighbor(instances, point_to_classify, number_of_instances, feature_subset):
    instance_nearest_neighbor_label = 0
    shortest_distance = math.inf

    for i in range(number_of_instances):
        if i != point_to_classify:
            distance = 0
            for j in range(len(feature_subset)):
                # we do feature_subset[j], makes sure we get distance of same feature
                # because the subset can be like [7, 5], not always [1, 2, 3]
                # the line below is the sum of squared differences between the other instances vs the unknown point
                distance += pow(instances[i][feature_subset[j]] - instances[point_to_classify][feature_subset[j]], 2)

            #distance = math.sqrt(distance)  # piazza post, no need to calculate since we dont use
            # distance = np.linalg.norm(instances[i]-instances[point_to_classify])
            # if there is a tie, the first shortest distance will be picked
            if distance < shortest_distance:
                instance_nearest_neighbor_label = instances[i][0]  # returns class label
                shortest_distance = distance

    return instance_nearest_neighbor_label


# input: training data(numpy array), number of instances(number), feature subset(array)
# output: accuracy in percentage
def leaving_one_out(instances, number_of_instances, feature_subset):
    correct_amount = 0

    for i in range(number_of_instances):
        leave_out = i  # this is instance for nearest neighbor classifier
        correct_class_label = instances[leave_out][0]
        neighbor_class_label = nearest_neighbor(instances, leave_out, number_of_instances, feature_subset)

        if neighbor_class_label == correct_class_label:  # compare class labels for correctness
            correct_amount += 1

    return (correct_amount / number_of_instances) * 100.0  # to get percentage, you multiply by 100.0


# input: data, # of instances, # of features(search levels)
# output: trace of program, with best subset of features and accuracy
def forward_selection(instances, number_of_instances, number_of_features):
    current_set = []
    best_set = []
    best_accuracy_overall = 0
    for i in range(number_of_features):  # depth of tree when looking through features
        best_so_far_accuracy = 0  # resets when going down level
        feature_to_add = None
        adding_new_best_feature = False
        for k in range(1, number_of_features + 1):  # modify range to represent the features we are adding,
            # make array go from 1 to last feature
            if k not in current_set:
                current_set.append(k)
                accuracy = leaving_one_out(instances, number_of_instances, current_set)
                print("Using feature(s) " + str(current_set) + " accuracy is " + str(accuracy) + "%")
                if accuracy > best_accuracy_overall:  # checking for best accuracy overall
                    adding_new_best_feature = True
                    #feature_to_add = k
                    best_accuracy_overall = accuracy
                if accuracy > best_so_far_accuracy:  # checking for best accuracy at each level
                    best_so_far_accuracy = accuracy
                    feature_to_add = k
                current_set.pop(len(current_set) - 1)  # haven't added feature yet, so we take it out

        current_set.append(feature_to_add)
        # these if statements aren't in provided slides
        # added these to fit the trace
        if adding_new_best_feature:  # means found feature to add to best set, keep this incase of local maxima
            best_set = copy.deepcopy(current_set)  # store best set for when accuracy starts dropping
            print("Feature set " + str(current_set) + " was best, accuracy is " + str(best_accuracy_overall) + "%")
        else:  # adding new_best never got set to true, so that means accuracy is currently dropping
            #print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
            print("(Warning, Accuracy has decreased! Stopping Search)")
            print("Feature set " + str(current_set) + " was best, accuracy is " + str(best_so_far_accuracy) + "%")
            break  # break if we stop when accuracy drops

    print("Finished search!! The best feature subset is " + str(best_set) + " which has an accuracy of " + str(
        best_accuracy_overall) + "%")


# input: data, # of instances, # of features(search levels)
# output: trace of program, with best subset of features and accuracy
# same idea as forward, just working backwards from the tree
# we start with full set, go up from there, remove features
def backward_selection(instances, number_of_instances, number_of_features):
    current_set = list(range(1, number_of_features + 1))
    best_set = list(range(1, number_of_features + 1))
    best_accuracy_overall = 0.0

    for i in range(number_of_features):  # depth of tree when looking through features
        best_so_far_accuracy = 0  # resets when going down level
        feature_to_remove = None
        removing_new_best_feature = False
        for k in range(1, number_of_features + 1):  # modify range to represent the features we are adding,
            # make array go from 1 to last feature
            if k in current_set:
                # current_set.remove(k)
                temp = [x for x in current_set if x != k]  # list comprehension to remove item from list
                accuracy = leaving_one_out(instances, number_of_instances, temp)
                print("Using feature(s) " + str(temp) + " accuracy is " + str(accuracy) + "%")
                if accuracy > best_accuracy_overall:  # checking for best accuracy overall
                    removing_new_best_feature = True
                    #feature_to_remove = k
                    best_accuracy_overall = accuracy
                if accuracy > best_so_far_accuracy:  # checking for best accuracy at each level
                    best_so_far_accuracy = accuracy
                    feature_to_remove = k
                # current_set.append(k)  # haven't removed feature yet, so we add it back

        current_set.remove(feature_to_remove)
        # these if statements aren't in provided slides
        # added these to fit the trace
        if removing_new_best_feature:  # means found feature to remove from best set, keep this incase of local maxima
            best_set = copy.deepcopy(current_set)  # store best set for when accuracy starts dropping
            print("Feature set " + str(current_set) + " was best, accuracy is " + str(best_accuracy_overall) + "%")
        else:  # removing_new_best never got set to true, so that means accuracy is currently dropping
            #print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
            print("(Warning, Accuracy has decreased! Stopping Search)")
            print("Feature set " + str(current_set) + " was best, accuracy is " + str(best_so_far_accuracy) + "%")
            #break  # break if we stop when accuracy drops

    print("Finished search!! The best feature subset is " + str(best_set) + " which has an accuracy of " + str(
        best_accuracy_overall) + "%")


def main():
    print("Welcome to Phillip Nguyen's Feature Selection Algorithm.")
    file_name = input("Type in the name of the file to test: ")

    # used to make sure user enters in a valid data file
    # source: python docs
    try:
        file = open(file_name, "r")
    except IOError:
        raise IOError("Invalid File Name Entered") from None  # the None suppresses one exception output

    # count number of lines in file, same as number of instances
    number_of_instances = sum(1 for line in file)

    # .seek is used to reset pointer to beginning of file
    file.seek(0, 0)

    # we read a line from the file
    # split breaks info into pieces according to white space
    # len counts how many pieces we have
    # minus 1 because we don't count the class label at the beginning
    number_of_features = len(file.readline().split()) - 1

    file.seek(0, 0)

    # initialize 2d array, inner one is columns and then rows
    # instances[0][0] would be label of first entry
    # instances[0][number_of_features] would be last feature of first instance
    # first index is row, second is column
    instances = [[0 for i in range(number_of_features + 1)] for j in range(number_of_instances)]  # +1 for features to
    # have space for class label

    # left side is rows and right side are columns
    for i in range(number_of_instances):
        instances[i] = file.readline().split()
    # use numpy because its array/matrix functions are more efficient than normal python
    # we convert our array into a numpy array, and change all values to float
    np_instances = np.array(instances, dtype=float)

    # got everything from file, we can close it now
    file.close()

    print("This dataset has " + str(number_of_features) +
          " features(not including the class attribute), with " +
          str(number_of_instances) + " instances.")

    # normalize data
    # convert back into list because numpy accesses are very slow
    # numpy math functions are much faster than python
    normalized_data = normalize(np_instances, number_of_instances, number_of_features)
    normalized_data = normalized_data.tolist()
    print("Please wait while I normalize the data... Done!")

    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    user_input = int(input())
    while user_input < 1 or user_input > 2:
        print("Invalid input. Please select again.")
        user_input = int(input())

    # array to input into leaving one out evaluation
    # used to be for loop, for loops very slow
    list_all_features = list(range(1, number_of_features + 1))

    accuracy_all_features = leaving_one_out(normalized_data, number_of_instances, list_all_features)
    print(
        "Running nearest neighbor with all " + str(
            number_of_instances) + " features, using \"leave one out evaluation\"" +
        ", I get an accuracy of " + str(accuracy_all_features) + "%")

    print("Beginning search.")

    if user_input == 1:
        forward_selection(normalized_data, number_of_instances, number_of_features)
    elif user_input == 2:
        backward_selection(normalized_data, number_of_instances, number_of_features)


if __name__ == '__main__':
    main()
