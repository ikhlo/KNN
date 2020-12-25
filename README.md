# K-Nearest Neighbors

<br>
Here is the KNN algorithm made by myself in 2019 for an academic project with python. The project took the form of a classification challenge : all models will be in competition ranked by their accuracy on the final test set.

<br/>
For this project, we first have to create our classifier based on the knn concepts. Then the teacher gave us a first dataset to fit our model and make some preliminary predictions. After, we receive a preTest dataset on which we used our model to predict a label between A and J. We could compare our predictions with the real label for each individuals and make some adjustements in our model to improve the accuracy.
For example, in my model, after analysing the features I decided to give less importance to the features n°3 beacause it varied less than the others. I also chosen minkowski distance with n = 4 which given better results.
Finally, we used our model to predict on the test dataset for which individuals had not label and we submit our predictions to the teacher.

<br>
Here is the result of thoses modifications for the challenge: I was ranked 3rd out of 288 with an accuracy of 88,54%.