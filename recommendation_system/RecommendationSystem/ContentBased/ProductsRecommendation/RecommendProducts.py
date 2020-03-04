# Recommendations based on similar product
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class RecommendProducts:
    def __init__(self, file):
        self.data = pd.read_csv(file)
        self.tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
        tfidf_matrix = self.tf.fit_transform(self.data['description'])

        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

        self.results = {}
        for idx, row in self.data.iterrows():
            similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
            similar_items = [(cosine_similarities[idx][i], self.data['id'][i]) for i in similar_indices]

            self.results[row['id']] = similar_items[1:]

    def item(self, id):
        return self.data.loc[self.data['id'] == id]['description'].tolist()[0].split(' - ')[0]

    def recommend_products(self, item_id, num=10):
        if type(item_id) != int and int(item_id) not in self.data['id'].values:
            return None, 'This product is not in our database.'
        else:
            recommended_list = []
            item_id = int(item_id)
            productName = ''
            # print("Recommending " + str(num) + " products similar to " + self.item(item_id))
            productName = productName + "Recommending " + str(num) + " products similar to " + str(self.item(int(item_id)))
            print(productName)
            recs = self.results[item_id][:num]
            for rec in recs:
                output = str(self.item(rec[1])) + " (score:" + str(rec[0]) + ")"
                recommended_list.append(output)
            return productName, recommended_list