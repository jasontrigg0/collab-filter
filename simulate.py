import csv
import numpy as np

if __name__ == "__main__":
    with open("/tmp/test_ratings.csv",'w') as f_out:
        writer = csv.DictWriter(f_out, ["user_id","book_id","rating"])
        writer.writeheader()
        with open("/home/jtrigg/Downloads/ratings.csv") as f_in:
            reader = csv.DictReader(f_in)
            mu = 3 + np.random.normal(scale=0.1, size=1)[0]
            row_means = {}
            col_means = {}
            row_features = {}
            col_features = {}
            for r in reader:
                if not r["user_id"] in row_means:
                    row_means[r["user_id"]] = 0 #np.random.normal(scale=0.3, size=1)[0]
                    row_features[r["user_id"]] = np.random.normal(scale=0.3, size=1)[0]
                if not r["book_id"] in col_means:
                    col_means[r["book_id"]] = 0 #np.random.normal(scale=0.3, size=1)[0]
                    col_features[r["book_id"]] = np.random.normal(scale=0.3, size=1)[0]
                # print(mu, row_means[r["user_id"]], col_means[r["book_id"]], row_features[r["user_id"]], col_features[r["book_id"]])
                val = mu + row_means[r["user_id"]] + col_means[r["book_id"]] + row_features[r["user_id"]] * col_features[r["book_id"]] + np.random.normal(scale=.01, size=1)[0]
                val = min(max(1,val),5)
                writer.writerow({"user_id":r["user_id"], "book_id":r["book_id"], "rating":int(round(val))})
    with open("/tmp/test_book_features.csv",'w') as f_out:
        writer = csv.DictWriter(f_out, ["book_id","feature"])
        writer.writeheader()
        for book in col_features:
            writer.writerow({"book_id": book, "feature":col_features[book]})
