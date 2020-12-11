import pandas as pd


def read_kgat_knowledge():
    data = pd.read_csv('data/yahoo-movies/movie_db_yoda', sep='\t',
                       encoding='latin',
                       header=None,
                       index_col=False,
                       names=['movieId', 'title',
                              'synopsis', 'running_time',
                              'MPAA_rating', 'MPAA_reason',
                              'release_date', 'timestamp',
                              'distributor',
                              'poster_url', 'genres',
                              'directors', 'directorIds',
                              'crew_members', 'crewIds',
                              'crew_types', 'actors',
                              'actorIds',
                              'average_critic_rating',
                              'num_of_critic_ratings',
                              'num_of_awards_won',
                              'num_of_awards_nominated',
                              'awards_won',
                              'awards_nominated',
                              'movie_mom_ratings',
                              'movie_mom_review',
                              'review_summaries_critics_users',
                              'review_owners', 'trailer_clips_captions',
                              'greg_preview_url',
                              'dvd_review_url', 'gnpp',
                              'average_rating_training_users',
                              'num_of_training_users_rated'])
    return data
