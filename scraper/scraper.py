import imdb
from english_words import english_words_lower_set
import pickle
from os.path import exists

ia = imdb.Cinemagoer()
path = './/movies_file.pickle'
file_exists = exists(path)
if file_exists:
    with open(path, 'rb') as file:
        movies = pickle.load(file)
else:
    movies = {}
english_words_list = list(english_words_lower_set)
counter = 0
for idx,word in enumerate(english_words_list):
    if word == "":
        continue
    try:
        new_movies = ia.search_movie(word,_episodes=False)
        for movie in new_movies:
            title = movie.data['title']
            movies[title] = movie
        names = list(map(lambda x: x.data['title'], new_movies))
        counter += len(new_movies)

        print(names)
        print(len(movies), counter,idx)
        if counter > 1000:
            with open(path, 'wb') as file:
                pickle.dump(movies, file)
            counter = 0
    except:
        pass
with open(path, 'wb') as file:
    pickle.dump(movies, file)
print("Done")