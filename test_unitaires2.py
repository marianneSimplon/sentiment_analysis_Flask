import unittest
from app import get_wordnet_pos
from app import cleaning
from app import my_predict


class MyTestCase(unittest.TestCase):
    def test_get_wordnet_pos(self):
        self.assertEqual(get_wordnet_pos("CC"), "n")
        self.assertEqual(get_wordnet_pos("DT"), "n")
        self.assertEqual(get_wordnet_pos("EX"), "n")
        self.assertEqual(get_wordnet_pos("FW"), "n")
        self.assertEqual(get_wordnet_pos("IN"), "n")
        self.assertEqual(get_wordnet_pos("JJ"), "a")
        self.assertEqual(get_wordnet_pos("LS"), "n")
        self.assertEqual(get_wordnet_pos("MD"), "n")
        self.assertEqual(get_wordnet_pos("NNS"), "n")
        self.assertEqual(get_wordnet_pos("PRP"), "n")
        self.assertEqual(get_wordnet_pos("RBR"), "r")
        self.assertEqual(get_wordnet_pos("TO"), "n")
        self.assertEqual(get_wordnet_pos("UH"), "n")
        self.assertEqual(get_wordnet_pos("VBD"), "v")
        self.assertEqual(get_wordnet_pos("WRB"), "n")

    def test_cleaning(self):
        self.assertEqual(
            type(cleaning("Absolutely wonderful - silky and sexy and comfortable")), str)
        self.assertEqual(type(cleaning("I aded this in my basket at hte last mintue to see what it would look like in person. (store pick up). i went with teh darkler color only because i am so pale :-) hte color is really gorgeous, and turns out it mathced everythiing i was trying on with it prefectly. it is a little baggy on me and hte xs is hte msallet size (bummer, no petite). i decided to jkeep it though, because as i said, it matvehd everything. my ejans, pants, and the 3 skirts i waas trying on (of which i ]kept all ) oops.")), str)
        self.assertEqual(
            type(cleaning("Cute little dress fits tts. it is a little high waisted. good length for my 5'9 height. i like the dress, i'm just not in love with it. i dont think it looks or feels cheap. it appears just as pictured.")), str)

    def test_my_predict_type(self):
        self.assertEqual(
            type(my_predict("Absolutely wonderful - silky and sexy and comfortable")), tuple)
        self.assertEqual(type(my_predict("I aded this in my basket at hte last mintue to see what it would look like in person. (store pick up). i went with teh darkler color only because i am so pale :-) hte color is really gorgeous, and turns out it mathced everythiing i was trying on with it prefectly. it is a little baggy on me and hte xs is hte msallet size (bummer, no petite). i decided to jkeep it though, because as i said, it matvehd everything. my ejans, pants, and the 3 skirts i waas trying on (of which i ]kept all ) oops.")), tuple)
        self.assertEqual(
            type(my_predict("Cute little dress fits tts. it is a little high waisted. good length for my 5'9 height. i like the dress, i'm just not in love with it. i dont think it looks or feels cheap. it appears just as pictured.")), tuple)

    def test_my_predict_score_seg(self):
        self.assertTrue(0 <= my_predict(
            "I would have loved this dress if the bust and waist were just a little more fitted. i am 32c and the top was too big. fit perfectly on hips. the lace material means it cannot be easily altered, so i chose to return the dress. i would have definitely kept it if it were a better fit.")[0] <= 100)

    def test_my_predict_score_type(self):
        self.assertEqual(
            type(my_predict("It's ok, fit doesn't wow me because of my body. chest is too wide, hips look too narrow. drapes across my back fat in an especially non-flattering way. basically made my square-apple body look more square-apple. great part about this dress is that it's comfy and hides the tummy pooch. construction is poorly done...contrasting liner at v-neck is rolling out on one side only and then doing the same at the hem contralaterally. another negative point is dry clean only. boo. i'm 5'3"" 140# 39-28-35 an")[0]), int)
        self.assertNotEqual(
            type(my_predict("Online, this looks like a great sweater. i ordered an xxsp and found that this sweater is much wider in the middle than pictured. in fact, i'm pretty sure they pinned the shirt in the back for the picture to make it appear slimmer. unfortunately, this sweater will not work for me, as i am an hourglass shape and this shirt makes me look 20 pounds heavier.")[0]), float)

    def test_my_predict_recommandation(self):
        self.assertTrue(my_predict("At first i wasn't sure about it. the neckline is much lower and wavy than i thought. but after wearing it, it really is comfortable. it stretches a lot, so i wear a cami underneath so when i lean forward i'm not showing the world my torso.")[
            1] in ["Recommandé", "Non Recommandé"])


if __name__ == '__main__':
    unittest.main()
