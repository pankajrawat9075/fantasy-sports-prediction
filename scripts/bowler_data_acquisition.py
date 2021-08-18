import pandas as pd
import yaml
import os
import time
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # Hide messy Numpy warnings


def extract_details(filename):
    dict = yaml.load(open(filename))

    match_details = {}
    match_details['venue'] = dict['info']['venue']
    match_details['date'] = dict['info']['dates'][0]
    match_details['team1'] = dict['info']['teams'][0]
    match_details['team2'] = dict['info']['teams'][1]
    if 'winner' not in dict['info']['outcome']:
        match_details['winner'] = 'no result'
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


    player_details = pd.DataFrame(columns=['date','team', 'runs', 'balls', 'wickets', 'extras','opposition','venue','bat_innings','outcome'])

    innings1 = dict['innings'][0]['1st innings']['deliveries']
    for ball in innings1:
        for delivery in ball:

            try:
                player_details.index.get_loc(ball[delivery]['bowler'])
            except:
                player_details.loc[ball[delivery]['bowler'], 'runs'] = 0
                player_details.loc[ball[delivery]['bowler'], 'balls'] = 0
                player_details.loc[ball[delivery]['bowler'], 'extras'] = 0
                player_details.loc[ball[delivery]['bowler'], 'wickets'] = 0
            player_details.loc[ball[delivery]['bowler'], 'date'] = match_details['date']
            player_details.loc[ball[delivery]['bowler'], 'team'] = match_details['bat_first'][1]
            player_details.loc[ball[delivery]['bowler'], 'opposition'] = match_details['bat_first'][0]
            player_details.loc[ball[delivery]['bowler'], 'bat_innings'] = 1
            player_details.loc[ball[delivery]['bowler'], 'venue'] = match_details['venue']
            if match_details['winner'] == player_details.loc[ball[delivery]['bowler'], 'team']:
                player_details.loc[ball[delivery]['bowler'], 'outcome'] = 1
            else:
                player_details.loc[ball[delivery]['bowler'], 'outcome'] = 0

            player_details.loc[ball[delivery]['bowler'],'runs'] += ball[delivery]['runs']['total']
            player_details.loc[ball[delivery]['bowler'], 'balls'] += 1
            if 'wicket' in ball[delivery]:
                player_details.loc[ball[delivery]['bowler'], 'wickets'] += 1
            if 'extras' in ball[delivery]:
                player_details.loc[ball[delivery]['bowler'], 'extras'] += ball[delivery]['runs']['extras']
            
    try:
        innings2 = dict['innings'][1]['2nd innings']['deliveries']
    except:
        flag = True
    else:
        for ball in innings2:
            for delivery in ball:

                try:
                    player_details.index.get_loc(ball[delivery]['bowler'])
                except:
                    player_details.loc[ball[delivery]['bowler'], 'runs'] = 0
                    player_details.loc[ball[delivery]['bowler'], 'balls'] = 0
                    player_details.loc[ball[delivery]['bowler'], 'wickets'] = 0
                    player_details.loc[ball[delivery]['bowler'], 'extras'] = 0
                player_details.loc[ball[delivery]['bowler'], 'date'] = match_details['date']
                player_details.loc[ball[delivery]['bowler'], 'team'] = match_details['bat_first'][0]
                player_details.loc[ball[delivery]['bowler'], 'opposition'] = match_details['bat_first'][1]
                player_details.loc[ball[delivery]['bowler'], 'bat_innings'] = 2
                player_details.loc[ball[delivery]['bowler'], 'venue'] = match_details['venue']
                if match_details['winner'] == player_details.loc[ball[delivery]['bowler'], 'team']:
                    player_details.loc[ball[delivery]['bowler'], 'outcome'] = 1
                else:
                    player_details.loc[ball[delivery]['bowler'], 'outcome'] = 0

                player_details.loc[ball[delivery]['bowler'],'runs'] += ball[delivery]['runs']['total']
                player_details.loc[ball[delivery]['bowler'], 'balls'] += 1
                if 'wicket' in ball[delivery]:
                    player_details.loc[ball[delivery]['bowler'], 'wickets'] += 1
                if 'extras' in ball[delivery]:
                    player_details.loc[ball[delivery]['bowler'], 'extras'] += ball[delivery]['runs']['extras']

    return player_details

overall_bowler_details = pd.DataFrame(columns=['team','innings','runs','balls','wickets','extras','average','strike_rate','economy','wicket_hauls'])
match_bowler_details = pd.DataFrame(columns=['date','name','team','opposition','venue','bat_innings','innings_played','previous_average','previous_strike_rate','previous_economy','previous_wicket_hauls','wickets'])

count = -1

for filename in os.listdir('./../odis'):
    if filename.endswith('.yaml'):
        
        player_details = extract_details('./../odis/'+filename)
        for player in player_details.index:

            count += 1

            try:
                overall_bowler_details.index.get_loc(player)
            except:
                overall_bowler_details.loc[player,'team'] = player_details.loc[player, 'team']
                overall_bowler_details.loc[player, 'innings'] = 0
                overall_bowler_details.loc[player, 'runs'] = 0
                overall_bowler_details.loc[player, 'balls'] = 0
                overall_bowler_details.loc[player, 'wickets'] = 0
                overall_bowler_details.loc[player, 'extras'] = 0
                overall_bowler_details.loc[player, 'average'] = 0
                overall_bowler_details.loc[player, 'strike_rate'] = 0
                overall_bowler_details.loc[player, 'economy'] = 0
                overall_bowler_details.loc[player, 'wicket_hauls'] = 0

            match_bowler_details.loc[count,'date'] = player_details.loc[player,'date']
            match_bowler_details.loc[count,'name'] = player
            match_bowler_details.loc[count,'team'] = player_details.loc[player,'team']
            match_bowler_details.loc[count,'opposition'] = player_details.loc[player,'opposition']
            match_bowler_details.loc[count,'venue'] = player_details.loc[player,'venue']
            match_bowler_details.loc[count,'innings_played'] = overall_bowler_details.loc[player,'innings']
            match_bowler_details.loc[count,'previous_average'] = overall_bowler_details.loc[player, 'average']
            match_bowler_details.loc[count,'previous_strike_rate'] = overall_bowler_details.loc[player, 'strike_rate']
            match_bowler_details.loc[count,'previous_economy'] = overall_bowler_details.loc[player, 'economy']
            match_bowler_details.loc[count,'previous_wicket_hauls'] = overall_bowler_details.loc[player, 'wicket_hauls']
            match_bowler_details.loc[count,'wickets'] = player_details.loc[player,'wickets']

            overall_bowler_details.loc[player,'innings'] += 1
            overall_bowler_details.loc[player,'runs'] += player_details.loc[player, 'runs']
            overall_bowler_details.loc[player,'balls'] += player_details.loc[player, 'balls']
            overall_bowler_details.loc[player,'wickets'] += player_details.loc[player,'wickets']
            overall_bowler_details.loc[player,'extras'] += player_details.loc[player,'extras']
            if player_details.loc[player,'wickets'] != 0:
                overall_bowler_details.loc[player,'average'] = ((player_details.loc[player,'runs']/player_details.loc[player,'wickets'])+(overall_bowler_details.loc[player,'average']*(overall_bowler_details.loc[player,'innings']-1)))/overall_bowler_details.loc[player,'innings']
                overall_bowler_details.loc[player,'strike_rate'] = ((player_details.loc[player,'balls']/player_details.loc[player,'wickets'])+(overall_bowler_details.loc[player,'strike_rate']*(overall_bowler_details.loc[player,'innings']-1)))/overall_bowler_details.loc[player,'innings']
            overall_bowler_details.loc[player,'economy'] = ((player_details.loc[player,'runs']*6/player_details.loc[player,'balls'])+(overall_bowler_details.loc[player,'economy']*(overall_bowler_details.loc[player,'innings']-1)))/overall_bowler_details.loc[player,'innings']
            if player_details.loc[player,'wickets'] >= 4:
                overall_bowler_details.loc[player,'wicket_hauls'] += 1
           
match_bowler_details.set_index(keys=['date','name'],drop=True,inplace=True)

overall_bowler_details.to_excel('./../player_details/overall_bowler_details.xlsx')
match_bowler_details.to_excel('./../player_details/match_bowler_details.xlsx')