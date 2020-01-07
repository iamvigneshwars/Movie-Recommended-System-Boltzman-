# Movie Recommendation System 

This is a simple movie recommendation system which predicts if the user will like the movie.

## Algorithm Used

Restriced Boltzman Machines algorithm is used in this project

## Data Preprocessing 

	- Movies.dat contains list of movies, movie id's and the genre
	- user.dat contains list of users and their gender, age and type of company they are working.	
	- ratings.dat contains 1 million ratings given by the users
	- Data in movies.dat are seperated by :: so it should be mentioned while importing
	- dataset does not have any header, so header should be set to None
	- dataset may contain some special characters so latin-1 encoding is used 
	- The folder ml-100k consists of 5 sets of training set and test set which will be easier for k-fold cross valifation
	- A new list is created with rows as users and columns as the reating for movies given by the user
	- The numpy array is convertes into Torch Tensors 
