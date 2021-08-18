# Player Prediction Predictor
	
#### The functions provided:
1. 	 Predicting a specific player's performance against a team either given the venue or not.
2.	 Predicting the performance of both the teams in a match either given the venue or not, along side identifying the best player combination of both the teams for the match.

#### The prediction stats:
__For Batsman__	: The number of runs he might score in the next fixture given the match details.\
__For Bowler__	: The number of wickets he might might take in the next fixture given the match details.

## Data Acquiistion

The ball-to-ball data, match meta data and post other details for the male One-Day-International matches in the period 2005-2019 are downloaded from [Cricsheets.org](https://cricsheet.org/downloads/odis.zip) website. The data is in .yaml files for each match and the stats are extracted using the data acquisition scripts individually for the bowlers and the batsmans. 

## Overall Player Stats 

#### Batsman Information
1.	 __Team__ 		- The international team the batsman plays for.
2.	 __Inninings__	- The number of innings played by the batsman.\
					_This gives us an insight into the experience of the batsman._
3.	 __Runs__		- The number of runs the batsman has scored in his career over the period.
4.	 __Balls__		- The number of balls the batsman has faced in his career over the period.
5.	 __Average__	- The average of runs scored by the batsman.\
					_This provides us with the pace at which the batsman can score, which is crucial in limited overs matches._\
					_This provides us with the batsman's scoring abilities as well as consistency._
6.	 __Strike Rate__- The rate of scoring runs by the batsman.\
					_This provides us with the pace at which the batsman can score, which is crucial in limited overs matches._
7.	 __Centuries__	- The number of centuries scored by the batsman.
8.	 __Fifties__	- The number of fifties scored by the batsman.\
					_These stats provide us with the achievements of the batsman._
9.	 __Zeros__		- The number of times the batsman has been dismissed for zero.\
					_This provides us with bad end of the stats of the player, which negatively impacts the batsman's stats._

#### Bowler Information
1.	 __Team__ 			- The international team the bowler plays for.
2.	 __Inninings__ 		- The number of innings player by the bowler.\
						_This gives us an insight into the experience of the bowler._
3.	 __Balls__			- The number of overs bowled by the player.
4.	 __Wickets__		- The number of wickets taken by the bowler.\
						_This provides us with the impact the bowler has caused._ 
5.	 __Runs__			- The number of runs concieved by the bowler.
6. 	 __Extras__			- The number of runs given away by the bowler in extras( ex: wides, no balls, leg byes)\
						_These inversely impacts the stats of the bowler._
7.	 __Average__		- The numbers of runs consided by the bowler per wicket taken.\
					  	_This provides us with information about the bowler's capabilities._
8.	 __Strike Rate__	- The number of balls bowled by the bowler per wicket taken.\
					  	_This provides us with the pace at which the bowler can take wickets._
9.	 __Economy__		- The average of the runs concieved by the bowler per match.\
					  	_A good economy impacts the bowlers stats in a huge positive way._
10.	 __Wicket Hauls__	- The number of four/five wickets hauls taken by the bowler.\
					  	_This provides us with the achievements of the bowler._

## Per Match Stats

#### Batsman Information
1.	 __Team__		- The international team the player plays for. 
2.	 __Opposition__	- The international team the player is playing against.
3.	 __Runs__ 		- The number of runs the batsman scored.
4.	 __Balls__ 		- The number of balls the batsman faced.
5.	 __Not_Out__	- A boolean value whether the player has been out.
6.	 __Venue__ 		- The stadium, the match is being played at.
7. 	 __Bat Innings__- The innings the player's team has batted.
8.	 __Outcome__	- A boolean value whether the player's team has won.
9.	 __Strike Rate__- The rate at which the batsman scored the runs.

#### Bowler Information 
1.	 __Team__ 		- The international team the player plays for.
2.	 __Opposition__	- The international team the player is playing against.
3.	 __Runs__ 		- The number of runs the bowler has concieved.
4.	 __Balls__ 		- The number of balls the bowler bowled.
5.	 __Wickets__ 	- The number of wickets the bowler has taken.
6.	 __Extras__ 	- The number of runs the bowler has given away in extras. 
7.	 __Venue__ 	 	- The stadium, the match is being played at.
8.	 __Bat Innings__- The innings the player's team has batted.
9.	 __Outcome__	- A boolean value whether the player's team has won.
10.	 __Average__ 	- The number of runs given away per wicket by the bowler.
11.	 __Strike Rate__- The number of balls bowled per wicket by the bowler.
12.	 __Economy__ 	- The average of runs concieved per over by the bowler.