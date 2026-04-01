import pandas as pd
import json
import os

# pd.set_option('display.max_rows', None)


class DataCollection():
    def __init__(self, cumulative_output = True, sampling_rate = 1, folder_path = 'data/cricket/ipl_json'):
        '''
        * Parameter - sampling_rate {int}: Specifies the frequency at which ball data is sampled.
        * Parameter - cumulative_output {boolean} : Determines whether runs should be accumulated over time.
        '''
        self.data = {}
        self.cumulative_output = cumulative_output
        self.sampling_rate = sampling_rate
        self.folder_path = folder_path

    def generate_data(self):
        year_runs_data = pd.DataFrame()
        year_wicket_data = pd.DataFrame()
        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename)
            if os.path.isfile(file_path):
                try:
                    runs_df, wicket_df = self.create_match_df(file_path)
                    year_runs_data = pd.concat([year_runs_data, runs_df], ignore_index=True, sort=True)
                    year_wicket_data = pd.concat([year_wicket_data, wicket_df], ignore_index=True, sort=True)
                except Exception as e:
                    pass

        year_runs_data.to_csv(f'../data/cricket/cricket_runs_data_{self.sampling_rate}_{self.cumulative_output}.csv')
        year_wicket_data.to_csv(f'../data/cricket/cricket_wicket_data_{self.sampling_rate}_{self.cumulative_output}.csv')

        return year_runs_data, year_wicket_data
        
    def create_match_df(self, file_path):
        self.get_data(file_path)
        df_0_inning_ball, df_0_inning_wick = self.create_table(0)
        df_1_inning_ball, df_1_inning_wick=  self.create_table(1)

        df_combined_ball = pd.concat([df_0_inning_ball, df_1_inning_ball], ignore_index=True, sort=True)
        df_combined_wicket = pd.concat([df_0_inning_wick, df_1_inning_wick], ignore_index=True, sort=True)

        return df_combined_ball, df_combined_wicket
    
    def get_data(self, file_path):
        '''
        Retrieve data by making a GET request.
        * Parameter - file_path {str}: File path of json file
        '''

        with open(file_path, 'r') as json_data:
            json_str = json_data.read()
            d = json.loads(json_str)
            json_data.close()
            self.data = d

    def create_table(self, inning):
        '''
        Creation of the data table summarizing ball-by-ball runs and match meta data.
        * Parameter - inning {int}: The inning number of the match
        '''  

        overs = self.data["innings"][inning]["overs"]

        ball_data = []
        wicket_data = []
        ball = 0
        i = 0
        wicket = 0

        for over in overs:
            i+= 1
            for delivery in over["deliveries"]:
                ball += 1
                runs = delivery["runs"]["total"]              
                ball_data.append({"ball": ball , "runs": runs})
                if "wickets" in delivery:
                    wicket+=len(delivery["wickets"])
                    wicket_data.append({"ball": ball , "wicket": 1})
                else:
                    wicket_data.append({"ball": ball , "wicket": 0})

            # print(over["over"], len(over["deliveries"]))

        ball_df = pd.DataFrame(ball_data)
        wicket_df = pd.DataFrame(wicket_data)

        if self.sampling_rate > 1:
            sampled_ball_pd = []
            sampled_wicket_pd = []
            len_val = ball_df["ball"].iloc[-1]

            for i in range(0, len(ball_df), self.sampling_rate):
                end_idx = i+self.sampling_rate
                start = i
                # if len_val - i < self.sampling_rate:
                #     start = end_idx
                #     end_idx = len_val


                runs_i = sum(ball_df[start: end_idx]["runs"].to_list())
                sampled_ball_pd.append({"ball": end_idx , "runs": runs_i})

                wickets_i = sum(wicket_df[start: end_idx]["wicket"].to_list())
                sampled_wicket_pd.append({"ball": end_idx , "wicket": wickets_i})

            ball_df = pd.DataFrame(sampled_ball_pd)
            wicket_df = pd.DataFrame(sampled_wicket_pd)

        if self.cumulative_output:
            ball_df["runs"] =  ball_df["runs"].cumsum()
            wicket_df["wicket"] =  wicket_df["wicket"].cumsum()

        df_ball_transposed = ball_df.set_index("ball").T
        df_ball_transposed.reset_index(drop=True, inplace=True)

        df_wicket_transposed = wicket_df.set_index("ball").T
        df_wicket_transposed.reset_index(drop=True, inplace=True)

        home = False
        winner = False
        toss_winner = False
        team_name = self.data["innings"][inning]["team"]
        info_data = self.data["info"]
        opponent_name = next(team for team in self.data["info"]["players"] if team != team_name)
        
        if info_data["city"] in team_name:
            home = True

        if info_data["outcome"].get("winner") == team_name:
            winner = True

        if info_data["toss"]["winner"] == team_name:
            toss_winner = True
            
        info = {"wicket": wicket, "date": info_data["dates"][0],
                "inning": inning+1, "team name":team_name, "opponent_name":opponent_name, "home game": home, 
                "toss_winner": toss_winner, "won": winner}
        

        for key, value in info.items():
            df_ball_transposed[key] = value
            df_wicket_transposed[key] = value
        
        return df_ball_transposed, df_wicket_transposed


# dc = Data_Collection(True, 1)
# dc.generate_data()