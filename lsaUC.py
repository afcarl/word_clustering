from bs4 import BeautifulSoup as bs
import pickle
import requests

def lsa_uc(phrase1, phrase2):
    uses = ["a " + phrase1, "a " + phrase2]


    post_data = {"LSAspace": "General_Reading_up_to_1st_year_college (300 factors)",
                 "txt1": ".\n".join(uses) + "."}

    r = requests.post('http://lsa.colorado.edu/cgi-bin/LSA-sentence-x.html',
                      data=post_data)

    soup = bs(r.text)
    try:
        similarity = str(soup.find_all('td')[0].table.tr.td.text)
        similarity = float(similarity.strip())
    except:
        similarity = 0.0
    finally:
        return similarity

if __name__ == "__main__":
    phrases = pickle.load(open('uses.p', 'rb'))
    #uses = ["body part", "snowman", "crusher", "stand", "trophy", "arm extension",
    #        "oar", "musical equipment", "boat anchor", "decoration", "lamp",
    #        "canvas", "distraction", "massager", "floatation device", "prop",
    #        "center piece", "tripping someone"]

    for use1 in phrases:
        for use2 in phrases:
            print("%s - %s: %0.2f" % (use1, use2, lsa_uc(use1,use2)))
