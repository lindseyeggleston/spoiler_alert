
def clean_text(chapter_set):
    # Edit book1
    with open('../data/SOIF/AGameOfThrones.txt') as g:
        text1 = g.read()
        text1 = replace_indentation(text1, chapter_set)
    new_text1 = text1[:20] + text1[892:1588846]
    with open('../data/text/book1.txt', 'w') as b1:
        b1.write(new_text1)
    # Edit book2
    with open('../data/SOIF/AClashOfKings.txt') as c:
        text2 = c.read()
        text2 = replace_indentation(text2, chapter_set)
    new_text2 = text2[:18] + text2[1070:1732670]
    with open('../data/text/book2.txt', 'w') as b2:
        b2.write(new_text2)
    # Edit book3
    with open('../data/SOIF/AStormOfSwords.txt') as s:
        text3 = s.read()
        text3 = replace_indentation(text3, chapter_set)
    new_text3 = 'A STORM OF SWORDS' + text3[2248:2245463]
    with open('../data/text/book3.txt', 'w') as b3:
        b3.write(new_text3)
    # Edit book4
    with open('../data/SOIF/AFeastForCrows.txt') as f:
        text4 = f.read()
        text4 = replace_indentation(text4, chapter_set)
    new_text4 = text4[:20] + text4[1371:1599998]
    with open('../data/text/book4.txt', 'w') as b4:
        b4.write(new_text4)
    # Edit book5
    with open('../data/SOIF/ADanceWithDragons.txt') as d:
        text5 = d.read()
        text5 = replace_indentation(text5, chapter_set)
    new_text5 = 'A DANCE WITH DRAGONS' + text5[4000:2450421]
    with open('../data/text/book5.txt', 'w') as b5:
        b5.write(new_text5)

def replace_indentation(text, chapter_set):
    new_text = ''
    for char in text:
        if ord(char)==9 or ord(char)==10:
            new_text += ' '
        else:
            new_text += char
    final_text = add_chapter_indentation(new_text, chapter_set)
    return final_text

def add_chapter_indentation(text, chapter_set):
    new_text = text.split(' ')
    for i, word in enumerate(new_text):
        if word in chapter_set:
            indent_word = '\n{0}\n'.format(word)
            new_text[i] = indent_word
    final_text = ' '.join(new_text)
    return final_text

if __name__ == '__main__':

    chapter_set = set(['TYRION', 'DAENERYS', 'JON', 'BRAN', "THE MERCHANT’S MAN",\
        'DAVOS', 'REEK', 'THE LOST LORD', 'THE WINDBLOWN', 'THE WAYWARD BRIDE',\
        'MELISANDRE', 'THE PRINCE OF WINTERFELL', 'THE WATCHER', 'THE TURNCLOAK',\
        "THE KING’S PRIZE", 'THE BLIND GIRL', 'A GHOST IN WINTERFELL', 'JAIME',\
        'THEON', 'CERSEI', 'THE QUEENSGUARD', 'THE IRON SUITOR', 'THE DISCARDED KNIGHT',\
        'THE SPURNED SUITOR', 'THE GRIFFIN REBORN', 'THE SACRIFICE', 'VICTARION',\
        'THE UGLY LITTLE GIRL', 'THE KINGBREAKER', 'THE DRAGONTAMER', "THE QUEEN’S HAND",
        'CATELYN', 'ARYA', 'SANSA', 'PROLOGUE', 'THE PROPHET', 'THE CAPTAIN OF GUARDS',\
        'BRIENNE', 'SAMWELL', 'JAIME', "THE KRAKEN'S DAUGHTER", 'THE SOILED KNIGHT',\
        'THE IRON CAPTAIN', 'THE DROWNED MAN', 'THE QUEENMAKER', 'ALAYNE', \
        'THE REAVER', 'CAT OF THE CANALS', 'THE PRINCESS IN THE TOWER', 'EDDARD',\
        'EPILOGUE'])

    # clean_text(chapter_set)
    with open('../data/text/book3.txt') as s:
        text3 = s.read()
    new_text = add_chapter_indentation(text3, chapter_set)
    with open('../data/text/book3.txt', 'w') as b3:
        b3.write(new_text)
