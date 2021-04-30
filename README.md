# TextAnalysis

Small Text Analysis project used to compare two sets of Articles for a Paired Case Study.

The two cases being compared are the reporting of the persecution and arrests of Alaxei Navalny and Julian Assange by the New York Times.

Many parts of the code are rough and "hacky" for the sake of saving time since I am using these results for a paper in my Philosophy class in school.

These results are not meant to be extensive and do not suggest any definite conclusions. They only provide a starting place for comparing quantitative attributes.


FULL READ OUT:

/Users/erickkbentz/PycharmProjects/TextAnalysis/venv/bin/python /Users/erickkbentz/PycharmProjects/TextAnalysis/main.py

----------------------------------Navalny--------------------------------------
25 Most common Adj: {'Russian': 466, 'last': 131, 'German': 129, 'more': 129, 'political': 107, 'other': 81, 'many': 75, 'former': 71, 'Western': 65, 'new': 59, 'foreign': 53, 'European': 51, 'least': 51, 'first': 49, 'most': 47, 'several': 45, 'prominent': 45, 'domestic': 45, 'military': 44, 'own': 42, 'recent': 39, 'Siberian': 39, 'top': 37, 'same': 36, 'possible': 36}

Navalny avg word count: 1174


AVERAGE SENTIMENT INTENSITY OF EACH SENTENCE
Average [neg]: 0.11593391081167802
Average [neu]: 0.8304953480911141
Average [pos]: 0.053248957330766775
Average [com]: -0.19194408084696843

AVERAGE TITLE SENTIMENT:
Average [neg]: 0.2635138888888888
Average [neu]: 0.6631666666666667
Average [pos]: 0.07334722222222222
Average [com]: -0.3215916666666666

Counter({'News': 64, 'Opinions, Commentary': 4, 'Opinions, Editorial': 3, 'Letter To The Editor': 1})

Navalny Times Appearing on Certain Page (most Common 10):
[(10, 9), (11, 8), (12, 6), (14, 5), (9, 5), (1, 4), (26, 3), (22, 3), (16, 2), (23, 2)]

Navalny avg page location: 12.596153846153847

----------------------------------ASSANGE--------------------------------------
25 Most common Adj: {'American': 70, 'British': 68, 'classified': 50, 'former': 46, 'other': 44, 'military': 40, 'criminal': 38, 'Russian': 37, 'Swedish': 37, 'last': 36, 'secret': 36, 'new': 32, 'legal': 32, 'many': 29, 'national': 27, 'Democratic': 27, 'political': 26, 'diplomatic': 25, 'international': 22, 'public': 21, 'sexual': 20, 'free': 19, 'special': 19, 'more': 17, 'private': 17}

Assange avg word count: 1146


AVERAGE SENTIMENT INTENSITY
Average [neg]: 0.10921242774566478
Average [neu]: 0.8153316473988448
Average [pos]: 0.0754544797687861
Average [com]: -0.09901105491329477

AVERAGE TITLE SENTIMENT:
Average [neg]: 0.1968
Average [neu]: 0.7180285714285715
Average [pos]: 0.08522857142857142
Average [com]: -0.15420857142857144

Counter({'News': 30, 'Opinions, Editorial': 2, 'Commentary': 1, 'Opinions, Commentary': 1, 'Letter to The Editor': 1})

Assange Times Appearing on Certain Page (most Common 10):
[(6, 4), (1, 4), (9, 3), (4, 3), (11, 3), (16, 2), (23, 2), (8, 2), (18, 1), (12, 1)]

Assange avg page location: 10.705882352941176

------------------------------------------------------------------


NAVALNY- Range: 7, Amount: 72, Avg: 10.29
ASSANGE- Range: 22, Amount: 72, Avg: 1.59

Navalny:
All NAVALNY TITLES AND NEGATIVE SENTIMENT: {"Navalny's Health Is Said to Be Worsening in Prison\n": 0.474, 'In a Message, Navalny Tells Of a Dystopia Inside Prison\n': 0.32, "U.S. Announces Sanctions on Russia Over Navalny; Putin Isn't a Target\n": 0.0, 'Biden Administration Accuses Russian Intelligence of Poisoning Navalny, and Announces Sanctions\n': 0.358, 'Russia Is Sending Navalny To Prison Known for Abuse\n': 0.517, 'Court Rejects Last Appeal By Navalny; Fate Unclear\n': 0.464, 'Russian Court Clears Way to Send Navalny to a Penal Colony\n': 0.0, 'Notorious Jail Conditions Await a Famous Prisoner\n': 0.593, 'Russia Casts Out Diplomats Over Navalny Protests\n': 0.241, 'Sham Justice for Navalny\n': 0.0, 'First Lady of Opposition In Russia Is in Spotlight With Husband Detained\n': 0.197, "'I Am Not Afraid': With Her Husband in Prison, Eyes Turn to Yulia Navalnaya\n": 0.202, 'Aleksei Navalny Is Winning\n': 0.0, 'Website Editor in Russia Is Jailed for Sharing Joke About Navalny Protests\n': 0.282, 'Navalny Supporters Aim to Seize Momentum in Challenging Putin\n': 0.0, 'Prison Sentence Stifles the Voice Of a Putin Critic\n': 0.462, 'Russian Activist Navalny Sentenced to More Than 2 Years in Prison\n': 0.355, 'Russian Activist Appears in Court for Decision on Prison Sentence\n': 0.262, 'Navalny Appears in Court for Decision on Prison Sentence\n': 0.284, 'Show of Force Fails to Deter Second Week of Russia Protests\n': 0.343, "Crowds of Police Couldn't Quell Russia's Pro-Navalny Protests\n": 0.213, 'Navalny Inspires Critics Of Putin to Rally as One\n': 0.182, "Court Turns Down Navalny's Appeal, Signaling Kremlin Wants Him Muted\n": 0.0, 'Russian Court Orders Navalny Kept in Jail\n': 0.0, 'Navalny Allies and Offices Targeted in Raids as Kremlin Turns Up Pressure\n': 0.167, 'E.U. Condemns Arrest of Navalny and Supporters, but Takes No Action\n': 0.426, 'Not Just Another Day of Dissatisfaction in Russia\n': 0.314, 'A Foreign Policy Doctrine for the Biden Administration\n': 0.0, 'Navalny Gets Lift in Status As Russians Hit the Streets\n': 0.0, 'Thousands Are Detained as Pro-Navalny Protests Sweep Russia\n': 0.434, "Navalny's Return to Russia Ignites Interest of Youth\n": 0.0, 'Russia Seeks to Divert Youths From Lure of Navalny Protests\n': 0.174, 'The Exceptional Courage of Aleksei Navalny\n': 0.0, "Navalny, From Jail, Issues Report Describing an Opulent Putin 'Palace'\n": 0.0, 'Navalny Calls for Protests as His Freedom Hangs in Balance\n': 0.135, 'Russian Court Orders Navalny Held for 30 Days\n': 0.0, 'In Battle of Wills With Putin, Navalny Is Jailed in Moscow Return\n': 0.367, 'Navalny Returns to Moscow and Is Arrested on Arrival\n': 0.279, "Navalny Says He'll Return To Moscow On Sunday\n": 0.0, 'Claim of Killer Underwear Brings Protest in Moscow, But It Is Lightly Attended\n': 0.274, 'Navalny Says Agent Confessed to Poison Plot\n': 0.368, "Putin Denies Any Role In Poisoning of Navalny, Noting It Didn't Kill Him\n": 0.325, 'Russian Agents Were Close During Navalny Poisoning, Report Says\n': 0.322, "Putin's Leading Rival Was Poisoned, Affirm World's Top Experts\n": 0.267, 'No Tough Response on Navalny Is Expected\n': 0.425, 'While Navalny Was in Coma, Russia Froze His Assets\n': 0.0, 'Putin Critic Leaves Hospital After Poisoning\n': 0.596, 'What Can Mr. Putin Get Away With?\n': 0.0, 'Putin Critic, Still on Mend, Flashes Wit On Instagram\n': 0.208, 'Navalny Was Poisoned at Hotel in Siberia, Not at Airport, His Aides Say\n': 0.211, 'Navalny Strikes a Defiant Tone While on the Mend\n': 0.423, 'Navalny Said to Be Intent On Going Back to Russia\n': 0.0, 'Congress Should Pass A Navalny Act\n': 0.0, 'As Navalny Improves, Case Pits Germany Against Russia\n': 0.0, 'Navalny Poisoning Raises Pressure on Merkel to Cancel Russian Pipeline\n': 0.533, 'The Spin From Moscow: Germany Did It, He Did, Or There Was No Poison\n': 0.322, "As Others Condemn Dissident's Poisoning, Trump Just Wants to 'Get Along'\n": 0.416, 'Russia Spins Alternative Theories in Poisoning of Navalny\n': 0.352, "A Guide to Novichok, the Soviet Neurotoxin Used in Navalny's Poisoning\n": 0.297, 'Putin Adversary Was Poisoned With Nerve Agent\n': 0.5, 'Russia Rejects Call for Query Into Poisoning\n': 0.583, 'Russia Rejects Calls for Investigation of Navalny Poisoning\n': 0.538, 'Germany Calls for Inquiry as Doctors Say Putin Critic Was Poisoned\n': 0.371, 'Sickened Putin Critic Is Treated in Germany\n': 0.528, 'Why Poison Aleksei Navalny Now?\n': 0.467, 'Russia to Allow Sickened Opposition Leader to Be Treated in Germany\n': 0.243, "Poison: 'Easy, and Easy To Cover Your Tracks'\n": 0.282, 'As Top Putin Foe Is Hospitalized, Suspicions of Poison in His Tea\n': 0.476, "Don't Drink the Tea: Poison Is a Favored Weapon in Russia\n": 0.368, 'Navalny Hospitalized in Russia in Suspected Poisoning\n': 0.533}

LEAST NEGATIVE TITLES: ["U.S. Announces Sanctions on Russia Over Navalny; Putin Isn't a Target\n", 'Russian Court Clears Way to Send Navalny to a Penal Colony\n', 'Sham Justice for Navalny\n', 'Aleksei Navalny Is Winning\n', 'Navalny Supporters Aim to Seize Momentum in Challenging Putin\n', "Court Turns Down Navalny's Appeal, Signaling Kremlin Wants Him Muted\n", 'Russian Court Orders Navalny Kept in Jail\n', 'A Foreign Policy Doctrine for the Biden Administration\n', 'Navalny Gets Lift in Status As Russians Hit the Streets\n', "Navalny's Return to Russia Ignites Interest of Youth\n", 'The Exceptional Courage of Aleksei Navalny\n', "Navalny, From Jail, Issues Report Describing an Opulent Putin 'Palace'\n", 'Russian Court Orders Navalny Held for 30 Days\n', "Navalny Says He'll Return To Moscow On Sunday\n", 'While Navalny Was in Coma, Russia Froze His Assets\n', 'What Can Mr. Putin Get Away With?\n', 'Navalny Said to Be Intent On Going Back to Russia\n', 'Congress Should Pass A Navalny Act\n', 'As Navalny Improves, Case Pits Germany Against Russia\n']
Number of titles: 19
Percent of tiles: 27.14%
Neg: 0%

MOST NEGATIVE TITLES: ['Russia Is Sending Navalny To Prison Known for Abuse\n', 'Notorious Jail Conditions Await a Famous Prisoner\n', 'Putin Critic Leaves Hospital After Poisoning\n', 'Navalny Poisoning Raises Pressure on Merkel to Cancel Russian Pipeline\n', 'Putin Adversary Was Poisoned With Nerve Agent\n', 'Russia Rejects Call for Query Into Poisoning\n', 'Russia Rejects Calls for Investigation of Navalny Poisoning\n', 'Sickened Putin Critic Is Treated in Germany\n', 'Navalny Hospitalized in Russia in Suspected Poisoning\n']
Number of titles: 9
Percent of titles: 12.86%
Neg: >= 50%

Assange:
All ASSANGE TITLES AND NEGATIVE SENTIMENT: {'Dept. of Justice Continues Push to Extradite Assange\n': 0.0, 'Civil-Liberties Groups Ask U.S. to Drop Assange Case\n': 0.231, "Supporters Push for Pardon for Assange as the President's Term Comes to a Close\n": 0.0, 'Assange Bid to Be Released On Bail Is Denied by Judge Who Blocked His Extradition\n': 0.278, 'Citing Mental Health, Judge Blocks Extradition for Assange\n': 0.213, 'U.K. Judge Set to Consider Whether to Send Assange To U.S. for Espionage Trial\n': 0.0, "At Assange's Hearing, Many Can't Be Heard\n": 0.0, 'A Canary Who Just Keeps Breathing Fire\n': 0.324, 'Roger Stone Was in Contact With Julian Assange in 2017, Documents Show [With graphic(s)]\n': 0.0, 'United States Lawyers Present Case for Extraditing WikiLeaks Founder From Britain\n': 0.0, 'White House Denies Claim Of a Pardon For Assange\n': 0.252, 'The New Threat to Journalists\n': 0.459, "Assange to Testify at Trial Over Illegal Recordings at Ecuador's Embassy\n": 0.265, "Assange 'Could Die' in U.K. Jail, Doctors Say\n": 0.358, 'After 9 Years, Sweden Closes Its Rape Investigation of WikiLeaks Founder\n': 0.343, "Judge's Ruling Keeps Assange In a U.K. Jail Until Hearing\n": 0.0, 'U.S. Extradition Hearing for Assange Is Scheduled for February by British Court\n': 0.0, 'U.N. Expert Says Assange Is Suffering From Torture\n': 0.538, 'An Assault on Press Freedom\n': 0.345, 'Assange Indicted Over Leak as U.S. Expands Charges\n': 0.413, "'That's Called News Gathering': Charges Alarm Advocates of Press Freedom\n": 0.287, 'Sweden Reopens Rape Case Against WikiLeaks Founder\n': 0.439, 'First Hearing For Assange In Long Road To U.S. Trial\n': 0.0, 'Assange Is Sentenced to 50 Weeks and Still Faces U.S. Charges\n': 0.262, 'Julian Assange and the War on Whistle-Blowers\n': 0.394, 'Arrest of Assange Friend Stirs Criticism\n': 0.461, 'Justice Dept. Continued to Investigate WikiLeaks After Secretly Indicting Assange\n': 0.0, "Julian Assange's Narcissism\n": 0.0, 'Ecuador Battled Threats of Leaks In Aiding Assange\n': 0.455, "Questions Remain About Assange's Links to Russians and 2016 Election\n": 0.0, 'Britain Arrests Assange, Ending 7-Year Standoff\n': 0.367, 'What the Charges Mean For U.S. Press Freedoms\n': 0.204, 'WikiLeaks Publishes Classified American Documents\n': 0.0, "'Curious Eyes Never Run Dry'\n": 0.0, "A Divisive Prophet of the Public's Right to Know": 0.0}

LEAST NEGATIVE TITLES: ['Dept. of Justice Continues Push to Extradite Assange\n', "Supporters Push for Pardon for Assange as the President's Term Comes to a Close\n", 'U.K. Judge Set to Consider Whether to Send Assange To U.S. for Espionage Trial\n', "At Assange's Hearing, Many Can't Be Heard\n", 'Roger Stone Was in Contact With Julian Assange in 2017, Documents Show [With graphic(s)]\n', 'United States Lawyers Present Case for Extraditing WikiLeaks Founder From Britain\n', "Judge's Ruling Keeps Assange In a U.K. Jail Until Hearing\n", 'U.S. Extradition Hearing for Assange Is Scheduled for February by British Court\n', 'First Hearing For Assange In Long Road To U.S. Trial\n', 'Justice Dept. Continued to Investigate WikiLeaks After Secretly Indicting Assange\n', "Julian Assange's Narcissism\n", "Questions Remain About Assange's Links to Russians and 2016 Election\n", 'WikiLeaks Publishes Classified American Documents\n', "'Curious Eyes Never Run Dry'\n", "A Divisive Prophet of the Public's Right to Know"]
Number of titles: 15
Percent of tiles: 42.86%
Neg: 0%

MOST NEGATIVE TITLES: ['U.N. Expert Says Assange Is Suffering From Torture\n']
Number of titles: 1
Percent of titles: 2.86%
Neg: >= 50%

Process finished with exit code 0
