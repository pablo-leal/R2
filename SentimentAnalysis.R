#
#  Sentiment analysis is often used by companies to quantify general social media opinion (for example,
#  using tweets about several brands to compare customer satisfaction). One of the simplest and most common
#  sentiment analysis methods is to classify words as "positive" or "negative", then to average the values of
#  each word to categorize the entire document.  If a document has more positive than negative words then the
#  document has a positive sentiment.  If a document has more negative than positive words then the document
#  has a negative sentiment.
#
#  Is it possible to predict a customer rating on Yelp based on their written opinions by  counting words?
#  counting words?  The Yelp Dataset is a collection of millions of restaurant reviews, each accompanied by a
#  1-5 star.  TIDY has built in sentiment capabilities so we can used a specific sentiment analysis method
#  and see if we can predict a customer's rating based on their written opinion.  The analytics here is to compare
#  an average sentiment score to the customer star rating.  If there is a trend then the hypothesis is
#  correct.
#
#  The approach will be this:
#
#      1.  Get the Yelp data and put it into a format such that each row is a review with the star rating
#          and the review text.
#
#      2.  Take this data and turn it into a format such that each row is a term (word) per document
#
#      3.  Each word has to be assigned some value based on if it is a positive or negative word.  The AFINN
#          dictionary in TIDY provides a score from -5 to +5 for words.  Negative words have negative values
#          and positive words have positive values.  AFINN was created by a human who made the decision.  If the
#          average sentiment is calculated for each review and plotted against the star rating of that review,
#          we can see if the hypothesis is correct.
#
#  The data can be downloaded from https://www.yelp.com/dataset/challenge.  The downloaded file is yelp_dataset.tar.
#  tar means tape archive.  This file can be opened with winzip or 7-Zip.  The tar file has multiple files in it
#  and you want the review.json file.  It is over 3GB in size.

#  The first libraries we need are to read in the data (readr) and an easier way to manipulate data (dplyr).
library(readr)
library(dplyr)

#  The data set is called review.json.  What is JSON?  JSON (JavaScript Object Notation) is a lightweight
#  data-interchange format. It is easy for humans to read and write. It is easy for machines to parse and
#  generate. It is based on a subset of the JavaScript Programming Language, JSON is a text format that is
#  completely language independent but uses conventions that are familiar to programmers. These properties
#  make JSON an ideal data-interchange language.
#
#  The review.json file has millions of reviews so we will only use 200,000 to see if we can prove our
#  hypothesis.

infile = "~/Downloads/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json"
review_lines = read_lines(infile, n_max = 200000, progress = FALSE)
head(review_lines)

#
#  You can see that there are a lot of characters in the JSON file that are not words.  These need to be eliminated
#  so we can get to STEP 1 from above.  We need two libraries, one for manipulating strings (stringr) and
#  converting data to and from the JSON format.
library(stringr)
library(jsonlite)

#  Each line is a JSON object.  We need to get to a format such that each row is a review with the star rating
#  and the review text.  The fastest way to do this is to turn the entire file into one string in JSON format
#  use fromJSON to convert to R.  It turns out that when you do this you end up with nested data frames.  This
#  means that one or more columns consist of another data frame.  The process to take this JSON format and
#  put it into a 2 dimensional table is called "flattening".
#
#  If you do not understand this code yet don't worry.  It will come with time.
#
#  str_c joins multiple strings into a single string. In this example:
#
#      1.  str_c(review_lines, collapse = ", ") takes all of the reviews and turns each column into a string and
#          separates each string with a comma.
#
#      2.  str_c("[", str_c(review_lines, collapse = ", "), "]") puts the entire string between two brackets
#          so that fromJSON knows this is an array with values separated by a comma.

reviews_combined = str_c("[", str_c(review_lines, collapse = ", "), "]")

#  At this point in makes sense to introduce some shorthand notation in R.  The %>% does is to take what is on
#  its left side and pass it to the function on its right side as that function's first argument.  In the example
#  below, this is the equivalent of:
#
#      reviews = fromJSON(reviews_combined)
#      reviews = flatten(reviews)
#      reviews = as.tibble(reviews)
#
#  The as.tibble() function creates a tibble or a data frame with one row per review.
reviews = fromJSON(reviews_combined) %>%
  flatten() %>%
  as_tibble()

#  We now have a data frame with one row per review:
reviews

#  Next we need to turn this into one-row-per-term-per-document. This is done using the TIDYTEXT package and
#  the unnest_tokens() function.

library(tidytext)

#
#  Reviews has 10 variables and we only need 4 (review_id, business_id, stars, text).  The function select() does
#  this using reviews as its input.  The output of select is sent to unnest_tokens() which takes this inout and
#  splits a text column into tokens, in our case, words.  This output is inout for the filter to remove words and
#  punctuation that is not relevant to sentiment analysis.  Stop words are words such as "I", "the", "and", "of",
#  and so on.  str_detect(word, "^[a-z']+$") looks at each word to detect if it has any character that is not a
#  letter or a ' from the beginning to the end of the word (+$).  filter() returns words that are not stop words
#  that have anything that is not a letter or ' removed.
review_words <- reviews %>%
  select(review_id, business_id, stars, text) %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% stop_words$word,
         str_detect(word, "^[a-z']+$"))

review_words

#  Now we have one-row-per-term-per-document which is the TIDYTEXT form.
#
#  We can now perform sentiment analysis on each review.  We use the AFINN lexicon, which provides a
#  positivity score for each word, from -5 (most negative) to 5 (most positive).

AFINN <- sentiments %>%
  filter(lexicon == "AFINN") %>%
  select(word, afinn_score = score)

AFINN

#  The next is to compare the words in review_words and the words in AFINN and keep just those words that are
#  in both lists.  This is what inner_join() does.  The output is grouped by review_id and includes the
#  stars assigned for that review and the average sentiment score from all the words in the review.

reviews_sentiment <- review_words %>%
  inner_join(AFINN, by = "word") %>%
  group_by(review_id, stars) %>%
  summarize(sentiment = mean(afinn_score))

reviews_sentiment

#  We now have an average sentiment alongside the star ratings. If the hypothesis is correct and sentiment
#  analysis will predict a reviewer's opinion towards a restaurant, we should expect the sentiment score to
#  correlate with the star rating.

library(ggplot2)
theme_set(theme_bw())
ggplot(reviews_sentiment, aes(stars, sentiment, group = stars)) +
  geom_boxplot() +
  ylab("Average sentiment score")

#  It appears that the sentiment scores are correlated with positivity ratings.  However, there is
#  prediction error with some 5-star reviews have a highly negative sentiment score, and vice versa.
#
#  How can this result be improved?

#
#  Perhaps there are words that appear mostly for positive reviews or mostly for negative reviews but not
#  both.  If we find just those words then that might improve the results.  To do that we need to do the
#  following:
#
#      1.  Get a count of the number of times a word appears in any review
#      2.  We filter for words that appear in a significant number of reviews a significant number of times.
#          We have 200,000 reviews so lets choose a minimum of 200 reviews that a word has to appear in.  Lets
#          also require that the word appear in reviews for at least 10 businesses to filter out the one and done
#          interviews.
#      3.  We can then merge this list of words with AFINN to create a new dictionary for sentiment analysis
#          and repeat the steps above to get a box plot to see if the hypotheses is correct.
#

#  Create a per-word summary and see which words tend to appear in positive or negative reviews.
review_words_counted <- review_words %>%
  count(review_id, business_id, stars, word) %>%
  ungroup()

review_words_counted

#  This shows that for a given business, a given review for the business, the stars for that review for that
#  business, that a word appears a certain number of times (n, the last column).  We need to see now for a
#  given word, how many times it appears in reviews of businesses and what was the average number of stars in
#  all the reviews the word appeared in.
word_summaries <- review_words_counted %>%
  group_by(word) %>%
  summarize(businesses = n_distinct(business_id),
            reviews = n(),
            uses = sum(n),
            average_stars = mean(stars)) %>%
  ungroup()

word_summaries

#  This shows that a lot of words are only in one review and therefore only for one business.

#
#  looking only at words that appear in at least 200 (out of 200000) reviews in a minimum of 10 businesses.
#
word_summaries_filtered <- word_summaries %>%
  filter(reviews >= 200, businesses >= 10)

word_summaries_filtered

#  Here are the most positive and negative words.
word_summaries_filtered %>%
  arrange(desc(average_stars))

word_summaries_filtered %>%
  arrange(average_stars)

#
#  We use innter join again to combine the common elements of AFINN with word_summaries_filtered
words_afinn <- word_summaries_filtered %>%
  inner_join(AFINN)

words_afinn

#  A better result
ggplot(words_afinn, aes(afinn_score, average_stars, group = afinn_score)) +
  geom_boxplot() +
  xlab("AFINN score of word") +
  ylab("Average stars of reviews with this word")




