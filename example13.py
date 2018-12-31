from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import example12b as s

# consumer key, consumer secret, access token, access secret.
ckey = "SYzT4vpQ1TLFUfaBJ53TYXS4L"
csecret = "h1OxaYj5ThYqirAawCDE7LIW0XFfsTENEUVbEJPUNYpDWGQSqF"
atoken = "899520782845464576-94vTrSR9dZRs4xp4OTmB9bHO3YZ6oTa"
asecret = "NzOkaLkae94uqYV2nHyER8d070QxbgsrAjW8ASjqCKRCe"

sc = {'pos': 0, 'neg': 0}


class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)

        tweet = all_data["text"]
        sentiment, confidence = s.sentiment(tweet)
        sc[sentiment] = sc[sentiment] + confidence

        print("pos => " + str(sc["pos"]), "neg => " + str(sc["neg"]))

        if confidence*100 >= 80:
            output = open("resources/twitter_out.txt", "a")
            output.write(tweet + " => " + sentiment + " " + str(confidence))
            output.write('\n')
            output.close()

        return True

    def on_error(self, status):
        print(status)


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["trump"])
