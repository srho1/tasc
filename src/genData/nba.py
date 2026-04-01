import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter(action="ignore")


class NBADataCollection():
    def __init__(self, sampling_rate = 30, start_date="2020-01-01", filepath = "../data/nba/"):
        '''
        * Parameter - sampling_rate {int}: Specifies the frequency at which ball data is sampled.
        * Parameter - cumulative_output {boolean} : Determines whether runs should be accumulated over time.
        '''
        self.data = {}
        self.sampling_rate = sampling_rate
        self.start_date = start_date
        self.filepath = filepath


    def convert_date_score(self, x):   
        try:
            dt = datetime.strptime(x, "%d-%b")
            return dt.strftime("%d-%m")
        except ValueError:
            return x

    def convert_datetime(self, quarter, timestamp):
        minutes, seconds = map(int, timestamp.split(":"))

        time_td = timedelta(minutes=minutes, seconds=seconds)
        time_diff = (timedelta(minutes=12) - time_td) 
        total_seconds = int(time_diff.total_seconds()) + (quarter - 1) * 12 * 60

        return total_seconds


    def create_time_series_data(self, df, game_id, team1, team2, sampling_rate):
        df["score"] = df["score"].apply(self.convert_date_score)
        df[[team1, team2]] = df["score"].str.split("-", expand=True)

        seconds = []
        for _, row in df.iterrows():
            seconds.append(self.convert_datetime(row["period"], row["pctimestring"]))

        df["time"] = seconds

        df.drop(columns=["score", "pctimestring", "period"], inplace=True)
        df_unique = df.groupby('time').last().reset_index()

        max_time = df['time'].max()
        time_index = np.arange(0, max_time+sampling_rate, sampling_rate)

        df_scores = df_unique.set_index('time')[[team1, team2]]
        
        df_30s = df_scores.reindex(time_index, method='ffill').reset_index()
        df_30s.rename(columns={'index':'time'}, inplace=True)

        transposed = df_30s.set_index("time").T
        transposed["game_id"] = game_id

        return transposed

    def get_game_info(self, return_full=False):
        game_info = pd.read_csv(f'{self.filepath}game_info.csv')
        game_info.game_date = pd.to_datetime(game_info.game_date)
        if return_full:
            return game_info
        else:
            return game_info[game_info["game_date"]>"2020-01-01"]
    
    def get_play_by_play(self, return_full=False):
        data_header = pd.read_csv(f'{self.filepath}play_by_play.csv', nrows=0)
        data_df = pd.read_csv(f'{self.filepath}play_by_play.csv', header=None, names=data_header.columns)
        # data_df = data_df[['game_id', 'period', 'pctimestring', 'score', 'player1_team_city', 'player2_team_city']]
        if return_full:
            return data_df
        else:
            return data_df[data_df["game_id"].isin(self.get_game_info()["game_id"])]

    def get_nba_data(self):
        data_df = self.get_play_by_play()
        sampling_rate = self.sampling_rate


        game_ids = data_df["game_id"].unique()
        multiple_games = pd.DataFrame()

        for game_id in game_ids:
            subset_game_df = data_df[data_df["game_id"] == game_id]
            
            subset_game_df.dropna(subset = ["score"], inplace=True)
            subset_game_df.reset_index(drop=True, inplace=True)

            teams = subset_game_df["player1_team_abbreviation"].dropna().unique()
            try:
                assert len(teams) == 2
            except:
                print(f"Game {game_id} does not have 2 teams, skipping... (teams: {teams})")
                continue
            team1 = teams[0]
            team2 = teams[1]
            subset_game_df.drop(columns=["player1_team_city", "player2_team_city"], inplace=True)
            subset_game_df = pd.concat([pd.DataFrame([{"game_id": game_id, "period":1, "pctimestring": "12:00", "score": "0 - 0"}]), subset_game_df], ignore_index=True)
            
            transformed_df = self.create_time_series_data(subset_game_df, game_id, team1, team2, sampling_rate)
            multiple_games = pd.concat([multiple_games, transformed_df], sort=True)

        return multiple_games
    
    def get_full_data(self):
        data = self.get_nba_data()
        game_info = self.get_game_info()
        full_data = data.join(game_info.set_index("game_id"), on="game_id", how="left")
        full_data.sort_values(by=["game_date"])
        return full_data