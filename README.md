# RECOMMENDATION-SYSTEM

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: NARNAMANGALAM BHUMIKA

*INTERN ID*: CT04DG1512

*DOMAIN*: Machine Learning

*DURATION*: 4 weeks

*MENTOR*: NEELA SANTHOSH

RECOMMENDATION SYSTEM :

A recommendation system, also known as a recommender system, is a type of intelligent system that provides personalized suggestions to users based on their preferences, behaviors, or interests. These systems are designed to assist users in discovering products, services, or content from a large pool of items that might otherwise go unnoticed. By analyzing patterns in user data and behavior, recommendation systems filter and predict what users might find relevant or enjoyable.
Recommendation systems have become a fundamental component of numerous digital platforms, such as Netflix, Amazon, YouTube, Spotify, and Facebook. These systems enhance the user experience by providing relevant suggestions, thereby increasing user engagement, satisfaction, and retention. In e-commerce, for instance, they can directly influence purchasing behavior and sales by recommending items that align with a user’s previous shopping history.
Types of Recommendation Systems
There are primarily three types of recommendation systems:
1.	Content-Based Filtering:
This approach recommends items similar to those the user has liked in the past. It relies on the characteristics of the items and the user’s past preferences. For example, if a user has watched action movies, the system might suggest other action-packed films based on genre, actors, or keywords.
2.	Collaborative Filtering:
This method is based on user interactions and does not require item content. It works on the assumption that users who agreed in the past will agree in the future. Collaborative filtering is divided into:
o	User-based: Recommends items based on what similar users have liked.
o	Item-based: Suggests items that are similar to items the user has previously liked.
Matrix factorization techniques like Singular Value Decomposition (SVD) fall under this category and are widely used due to their scalability and performance.
3.	Hybrid Systems:
These systems combine both content-based and collaborative filtering methods to overcome the limitations of each approach and improve accuracy.

TOOLS AND LIBRARIES USED :

The implementation relies on the following libraries:
•	Pandas: Used for data manipulation and loading the CSV file.
•	Surprise (Scikit-surprise): A Python scikit for building and analyzing recommender systems. It provides ready-to-use prediction algorithms, dataset loaders, evaluation metrics, and more.
•	Scikit-learn utilities (within Surprise): For model evaluation, splitting datasets, and prediction.

IMPLEMENTATION OF RECOMMENDATION SYSTEM :

The first step in the implementation is data loading and preparation. The dataset used in this system is the ratings_small.csv, a subset of the full MovieLens dataset available through Kaggle's Movies Dataset. This file contains historical user ratings for various movies, with columns representing userId, movieId, and rating. The data is loaded using the pandas library, which provides an efficient structure to handle and manipulate large tabular datasets.
After loading the data, it is formatted for compatibility with the Surprise library. A Reader object is created to define the range of the ratings, which in this case is from 0.5 to 5.0. The data is then loaded using Dataset.load_from_df() which transforms the DataFrame into a suitable format for model training.
Next, the data is split into training and test sets, typically using an 80:20 ratio. The training set is used to train the model, while the test set is used to evaluate the model’s performance on unseen data. This is done using the train_test_split function from the Surprise model selection module.
For the model itself, the SVD algorithm is chosen. SVD is a matrix factorization technique that identifies latent factors in the user-item interaction matrix. It reduces dimensionality and helps uncover hidden associations between users and items. The model is trained using the training set with the fit() function.
Once trained, the model’s performance is evaluated using the test set. The test() function generates predictions for user-item pairs, and the accuracy.rmse() function is used to compute the Root Mean Squared Error (RMSE). RMSE quantifies how close the predicted ratings are to the actual ratings; the lower the RMSE, the better the performance.
The trained model can also make individual predictions, such as predicting how a specific user might rate a specific movie. This is achieved using the predict() function, which returns the estimated rating.

VISUALIZATION OF RECOMMENDATION SYSTEM :

Although the Surprise library does not natively support data visualization, other Python libraries such as Matplotlib and Seaborn can be used to visualize patterns and model performance.
One common visualization is the distribution of movie ratings. A histogram showing the frequency of each rating value provides insight into user behavior. In most cases, the distribution is skewed towards higher ratings (e.g., 3 to 5), indicating user tendency to rate positively.
Another useful visualization is the distribution of ratings per movie. This highlights the fact that a small number of popular movies receive the majority of the ratings, while many others receive very few, which is a common occurrence in real-world recommendation systems.
Additionally, a heatmap of user-movie interactions can be created by sampling the dataset. This visual representation reveals the sparsity of the interaction matrix, showing that most users have only rated a small fraction of the available movies. Sparsity is a common challenge in collaborative filtering, where large portions of the matrix are empty.
A more analytical visualization is a scatter plot of actual vs. predicted ratings. This comparison helps assess how well the model performs across different rating values. A perfect model would have all points along the diagonal line where the actual rating equals the predicted rating. Deviations from this line indicate prediction errors. 

DATASET USED :

The dataset used is the ratings_small.csv file from The Movies Dataset, available on Kaggle. It contains:
•	userId: Unique ID for each user.
•	movieId: Unique ID for each movie.
•	rating: The rating (0.5 to 5.0) given by a user to a movie.

PLATFORM USED :

•	Python Programming Language

•	Jupyter Notebook or any Python IDE

•	Surprise Library (built on NumPy, SciPy, and Scikit-learn)

APPLICATIONS OF RECOMMENDATION SYSTEMS :

Recommendation systems have a wide range of applications across industries:
•	Entertainment: Suggesting movies (Netflix), songs (Spotify), or TV shows.
•	E-commerce: Product recommendations (Amazon, Flipkart).
•	Social Media: Friend suggestions (Facebook), content feeds (Instagram, Twitter).
•	Online Learning: Course suggestions based on user interests (Coursera, edX).
•	News Aggregators: Personalized news feeds (Google News, Flipboard).
•	Healthcare: Recommending personalized treatment plans or health content.

CONCLUSION :

The implementation demonstrated above provides a basic yet powerful framework for building a collaborative filtering-based movie recommendation system using the SVD algorithm. Leveraging the Surprise library simplifies model training, testing, and evaluation. With the increasing volume of user data online, such systems are crucial in offering personalized experiences.
Despite its effectiveness, the system can be further improved by incorporating content-based features, contextual information, or hybrid methods that combine multiple recommendation strategies. Visualization tools and performance metrics help monitor and refine the model. Ultimately, recommendation systems represent a vital technology in data-driven decision-making and user engagement across digital platforms.

OUTPUT :

<img width="501" height="53" alt="Image" src="https://github.com/user-attachments/assets/13a17be4-6d9f-46b6-abfa-a4e295e84657" />
