
import pandas as pd
def format_predictions(filename, movie_ids, predictions, label_handler):
    # write to a csv file, with columns movie_id and genres (space separated genre ids)

    rows = []
    for movie_id, pred in zip(movie_ids, predictions):
        genre_ids = label_handler.multi_hot_to_array(pred)
        rows.append({'movie_id': movie_id, 'genre_ids': ' '.join(map(str, genre_ids))})
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False, lineterminator='\r\n')