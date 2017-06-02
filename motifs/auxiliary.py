

def flatten_list(l):
	""" Flattens a nested list structure. """

	from collections import Iterable
	for el in l:
		if isinstance(el, Iterable) and not isinstance(el, (int)):
			yield from flatten_list(el)
		else:
			yield el

def tweets_to_edgelist(df):
	""" 
	Converts a set of tweets into a set of events between users. 
	Takes only the first mention in each tweet.
	"""

	event_list = []
	ix = 0
	df = df.sort_index()
	for _, row in df.iterrows():
		
		source = row['user_name']
		tweet_id = row['id_str']
		
		if len(row['entities']['user_mentions']) > 0:
			target = row['entities']['user_mentions'][0]['screen_name']
		else:
			continue

		time = int(row['created_at'].value/int(1e9))
		
		if row['in_reply_to_status_id_str'] is not None:
			style = 'reply'
		elif row['retweeted_status_id_str'] is not None:
			style = 'retweet'
			source, target = target, source
		else:
			style = 'message'
			
		event_list.append((source, target, time, style, tweet_id))

	event_list = pd.DataFrame(event_list, columns=['source','target','time','edge_color', 'tweet_id'])    
	return event_list