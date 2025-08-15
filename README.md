🎬 Movie Recommendation System
A content-based movie recommendation system that suggests similar movies based on genres, keywords, cast, director, and tagline using machine learning techniques.

🚀 Features
Content-Based Filtering: Recommends movies based on movie features (genres, cast, director, keywords, tagline)
Smart Input Matching: Handles typos and suggests close matches using fuzzy string matching
Cosine Similarity: Uses TF-IDF vectorization and cosine similarity for accurate recommendations
Top 30 Recommendations: Returns the most similar movies ranked by similarity score
Interactive Interface: Simple command-line interface for easy usage
🛠️ Technologies Used

Python 3.x
Pandas - Data manipulation and analysis
NumPy - Numerical computing
Scikit-learn - Machine learning library (TF-IDF, Cosine Similarity)
difflib - String matching for handling typos

📊 How It Works
Data Processing: Combines movie features (genres, keywords, tagline, cast, director) into a single text representation
Vectorization: Converts text data into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency)
Similarity Calculation: Computes cosine similarity between movies to find similar content
Recommendation: Returns top 30 most similar movies based on similarity scores

📁 Dataset Requirements
The system expects a CSV file named movies.csv with the following columns:
title - Movie title
genres - Movie genres
keywords - Movie keywords
tagline - Movie tagline
cast - Movie cast
director - Movie director
index - Movie index (unique identifier)

⚙️ Installation & Setup
Clone or download the notebook

Install required dependencies:

bash
pip install pandas numpy scikit-learn
Prepare your dataset:

Ensure you have a movies.csv file with the required columns

Upload the dataset to your working directory (or update the file path in the code)

💻 Usage
Run the notebook in Google Colab or Jupyter Notebook
Enter a movie name when prompted
Get recommendations: The system will display top 30 similar movies

Example:
'''text
Enter your favourite movie name: The Dark Knight
Movies suggested for you:

1. Batman Begins
2. The Dark Knight Rises
3. Watchmen
4. Man of Steel
5. Spider-Man
...'''
🔧 Technical Implementation
Key Components:
1. Feature Selection & Preprocessing

'''python
selected_features = ['genres','keywords','tagline','cast','director']'''
# Combines all features into single text representation
2. TF-IDF Vectorization''

'''python
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)]'''
3. Cosine Similarity Calculation

'''python
similarity_matrix = cosine_similarity(feature_vectors)'''
4. Fuzzy String Matching

'''python
close_matches = difflib.get_close_matches(user_input, movie_titles)'''
📈 Performance Metrics
Accuracy: Content-based approach ensures relevant recommendations

Speed: Fast similarity computation using vectorized operations

Robustness: Handles typos and variations in movie names

Scalability: Can handle large movie databases efficiently

🎯 Use Cases
Movie Discovery: Help users find new movies similar to their favorites
Streaming Platforms: Recommend content to users based on viewing history
Movie Databases: Enhance user experience with smart recommendations
Research: Study movie similarity patterns and user preferences

🚧 Future Enhancements
 Hybrid Approach: Combine content-based with collaborative filtering
 User Ratings: Incorporate user ratings for better recommendations
 Web Interface: Build a web-based GUI using Flask/Streamlit
 API Integration: Connect to movie databases like TMDB or OMDB
 Advanced NLP: Use word embeddings (Word2Vec, BERT) for better text analysis
 Machine Learning: Implement deep learning models for improved accuracy


📄 License
This project is open source and available under the MIT License.

👤 Author
Nandha Kumar V
Embedded Systems & AI/ML Engineer
Chennai Institute of Technology

🙏 Acknowledgments
Scikit-learn for machine learning algorithms
Pandas for data manipulation
Google Colab for development environment
Movie database providers for datasets
Built with ❤️ using Python and Machine Learning
