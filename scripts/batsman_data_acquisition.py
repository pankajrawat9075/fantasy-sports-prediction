import pandas as pd
import yaml
import os
import time
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # Hide messy Numpy warnings

def extract_details(filename):
    dict = yaml.load(open(filename))
    flag = False
    
    match_details = {}
    match_details['venue'] = dict['info']['venue']
    match_details['date'] = dict['info']['dates'][0]
    match_details['team1'] = dict['info']['teams'][0]
    match_details['team2'] = dict['info']['teams'][1]
    if 'winner' not in dict['info']['outcome']:
        match_details['winner'] = 'no_result'
    else:
        match_details['winner'] = dict['info']['outcome']['winner']
    if dict['info']['toss']['winner'] == match_details['team1']:
        if dict['info']['toss']['decision'] == 'bat':
            match_details['bat_first'] = [match_details['team1'],match_details['team2']]
        else:
            match_details['bat_first'] = [match_details['team2'],match_details['team1']]
    else:
        if dict['info']['toss']['decision'] == 'bat':
            match_details['bat_first'] = [match_details['team2'],match_details['team1']]
        else:
            match_details['bat_first'] = [match_details['team1'],match_details['team2']]
    print(match_details)

    player_details = pd.DataFrame(columns=['date','team','runs','balls','not_out','opposition','batting_innings','venue','outcome'])
    
    innings1 = dict['innings'][0]['1st innings']['deliveries']
    for ball in innings1:
        for delivery in ball:
      
            try:
                player_details.index.get_loc(ball[delivery]['batsman'])
            except:
                player_details.loc[ball[delivery]['batsman'],'runs'] = 0
                player_details.loc[ball[delivery]['batsman'], 'balls'] = 0
            
            player_details.loc[ball[delivery]['batsman'], 'date'] = match_details['date']
            player_details.loc[ball[delivery]['batsman'], 'team'] = match_details['bat_first'][0]
            player_details.loc[ball[delivery]['batsman'], 'opposition'] = match_details['bat_first'][1]
            player_details.loc[ball[delivery]['batsman'], 'batting_innings'] = 1
            player_details.loc[ball[delivery]['batsman'], 'venue'] = match_details['venue']
            if match_details['winner'] == player_details.loc[ball[delivery]['batsman'], 'team']:
                player_details.loc[ball[delivery]['batsman'], 'outcome'] = 1
            else:
                player_details.loc[ball[delivery]['batsman'], 'outcome'] = 0
            if 'wicket' in ball[delivery]:
                player_details.loc[ball[delivery]['batsman'],'not_out'] = False
            else:
                player_details.loc[ball[delivery]['batsman'], 'not_out'] = True
            
            player_details.loc[ball[delivery]['batsman'],'runs'] += ball[delivery]['runs']['batsman']
            player_details.loc[ball[delivery]['batsman'],'balls'] += 1
            
    try:
        innings2 = dict['innings'][1]['2nd innings']['deliveries']
    except:
        flag = True
    else:
        for ball in innings2:
            for delivery in ball:
            
                try:
                    player_details.index.get_loc(ball[delivery]['batsman'])
                except:
                    player_details.loc[ball[delivery]['batsman'],'runs'] = 0
                    player_details.loc[ball[delivery]['batsman'], 'balls'] = 0

                player_details.loc[ball[delivery]['batsman'], 'date'] = match_details['date']
                player_details.loc[ball[delivery]['batsman'], 'team'] = match_details['bat_first'][1]
                player_details.loc[ball[delivery]['batsman'], 'opposition'] = match_details['bat_first'][0]
                player_details.loc[ball[delivery]['batsman'], 'batting_innings'] = 2
                player_details.loc[ball[delivery]['batsman'], 'venue'] = match_details['venue']
                if match_details['winner'] == player_details.loc[ball[delivery]['batsman'], 'team']:
                    player_details.loc[ball[delivery]['batsman'], 'outcome'] = 1
                else:
                    player_details.loc[ball[delivery]['batsman'], 'outcome'] = 0
                if 'wicket' in ball[delivery]:
                    player_details.loc[ball[delivery]['batsman'],'not_out'] = False
                else:
                    player_details.loc[ball[delivery]['batsman'], 'not_out'] = True

                player_details.loc[ball[delivery]['batsman'],'runs'] += ball[delivery]['runs']['batsman']
                player_details.loc[ball[delivery]['batsman'],'balls'] += 1                

    return player_details

overall_batsman_details = pd.DataFrame(columns=['team','innings','runs','balls', 'average', 'strike_rate','centuries','fifties','zeros'])
match_batsman_details = pd.DataFrame(columns=['date','name','team','opposition','venue','batting_innings','innings_played','previous_average','previous_strike_rate','previous_centuries','previous_fifties','previous_zeros','runs','balls'])

count = -1

for filename in os.listdir('./../odis'):
    if filename.endswith('.yaml'):

        player_details = extract_details('./../odis/'+filename)

        for player in player_details.index:
            count += 1
            try:
                overall_batsman_details.index.get_loc(player)
            except:
                overall_batsman_details.loc[player,'team'] = player_details.loc[player,'team']
                overall_batsman_details.loc[player,'innings'] = 0
                overall_batsman_details.loc[player,'runs'] = 0
                overall_batsman_details.loc[player,'balls'] = 0
                overall_batsman_details.loc[player,'strike_rate'] = 0
                overall_batsman_details.loc[player, 'average'] = 0
                overall_batsman_details.loc[player,'zeros'] = 0
                overall_batsman_details.loc[player,'fifties'] = 0
                overall_batsman_details.loc[player,'centuries'] = 0

            
            match_batsman_details.loc[count,'date'] = player_details.loc[player,'date']
            match_batsman_details.loc[count,'name'] = player
            match_batsman_details.loc[count,'team'] = player_details.loc[player,'team']
            match_batsman_details.loc[count,'opposition'] = player_details.loc[player,'opposition']
            match_batsman_details.loc[count,'venue'] = player_details.loc[player,'venue']
            match_batsman_details.loc[count,'innings_played'] = overall_batsman_details.loc[player,'innings']
            match_batsman_details.loc[count,'previous_average'] = overall_batsman_details.loc[player, 'average']
            match_batsman_details.loc[count,'previous_strike_rate'] = overall_batsman_details.loc[player, 'strike_rate']
            match_batsman_details.loc[count,'previous_centuries'] = overall_batsman_details.loc[player,'centuries']
            match_batsman_details.loc[count,'previous_fifties'] = overall_batsman_details.loc[player,'fifties']
            match_batsman_details.loc[count,'previous_zeros'] = overall_batsman_details.loc[player,'zeros']
            match_batsman_details.loc[count,'runs'] = player_details.loc[player,'runs']
            
            overall_batsman_details.loc[player,'innings'] += 1
            overall_batsman_details.loc[player,'runs'] += player_details.loc[player,'runs']
            overall_batsman_details.loc[player,'balls'] += player_details.loc[player,'balls']
            overall_batsman_details.loc[player,'strike_rate'] = (((player_details.loc[player, 'runs']/player_details.loc[player, 'balls'])*100)+(overall_batsman_details.loc[player,'strike_rate']*(overall_batsman_details.loc[player,'innings']-1)))/overall_batsman_details.loc[player,'innings']
            overall_batsman_details.loc[player,'average'] = ((player_details.loc[player, 'runs'])+(overall_batsman_details.loc[player,'average']*(overall_batsman_details.loc[player,'innings']-1)))/overall_batsman_details.loc[player,'innings']
            if player_details.loc[player,'runs'] == 0:
                overall_batsman_details.loc[player,'zeros'] += 1
            elif player_details.loc[player,'runs']>=50 and player_details.loc[player,'runs']<100:
                overall_batsman_details.loc[player,'fifties'] += 1
            elif player_details.loc[player,'runs']>=100:
                overall_batsman_details.loc[player,'centuries'] += 1

overall_batsman_details.index.name = 'player_name'

overall_batsman_details.to_excel('./../player_details/overall_batsman_details.xlsx')
match_batsman_details.to_excel('./../player_details/match_batsman_details.xlsx')