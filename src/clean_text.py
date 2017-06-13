import re
import glob

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
    temp_text = text.split('\t')
    for i,word in enumerate(temp_text):
        temp_text[i] = word.strip(' ').strip('\n')
    new_text = ' '.join(temp_text)
    temp_text = new_text.split('\n')
    for i,word in enumerate(temp_text):
        temp_text[i] = word.strip(' ')
    new_text = ' '.join(temp_text)
    final_text = add_chapter_indentation(new_text, chapter_set)
    return final_text

def add_chapter_indentation(text, chapter_set):
    for chapter_title in chapter_set:
        text = re.sub(chapter_title, '\n{0}\n'.format(chapter_title), text)
    return text

def extract_character_chapters(character, text, alt_chapter_names=None):
    '''
    Extracts out the text from specified character's chapters and stores it as a
    text file named after the character. The text should first be run through
    replace_indentation() for uniform formatting.

    Parameters
    ----------
    character: STR - character name
    text: STR - text corpus or corpora
    alt_chapter_names: SET - (optional) set of alternative chapter names for the
        character and should also contain character name. All chapter names
        should be in all caps.

    Returns
    -------
    None
    '''
    chapter_title = character.upper()
    chapters = text.split('\n')
    character_str = chapter_title + '\n'

    for i, line in enumerate(chapters):
        if alt_chapter_names != None:
            if line in alt_chapter_names:
                character_str += chapters[i+1] + '\n'
        else:
            if re.match(line, chapter_title):
                character_str += chapters[i+1] + '\n'

    with open('{0}.txt'.format(character), 'w+') as f:
        f.write(character_str)

def chapter_content(chapter_title, book_filepath, n_characters=1000):
    '''
    Prints the first n_characters of a chapter with specified title.

    Parameters
    ----------
    chapter_title: STR - chapter title in all caps
    book_filepath: STR - filepath to book
    n_characters: INT - number of characters from chapter to display

    Returns
    -------
    chapter title and first n characters of that chapter
    '''
    with open(book_filepath) as f:
        text = f.read().split('\n')
    for i,line in enumerate(text):
        if re.match(line, chapter_title):
            print(line)
            print(text[i+1][:1000])


if __name__ == '__main__':
    
    # chapter titles for all 5 books
    chapter_set = set(["TYRION", "DAENERYS", "JON", "BRAN", "THE BLIND GIRL",\
        "DAVOS", "REEK", "THE WINDBLOWN", "THE WAYWARD BRIDE", "THE WATCHER",\
        "MELISANDRE", "THE PRINCE OF WINTERFELL", "THE TURNCLOAK", "THEON",\
        "THE KING’S PRIZE", "A GHOST IN WINTERFELL", "THE LOST LORD", "JAIME",\
        "THE QUEENSGUARD", "THE IRON SUITOR", "THE DISCARDED KNIGHT", "CERSEI",\
        "VICTARION", "THE SPURNED SUITOR", "THE GRIFFIN REBORN", "THE SACRIFICE",\
        "THE KINGBREAKER", "THE DRAGONTAMER", "THE UGLY LITTLE GIRL", "THE QUEEN'S HAND"\
        "CATELYN", "THE PROPHET", "THE CAPTAIN OF GUARDS", "ARYA", "SANSA", "PROLOGUE",\
        "BRIENNE", "THE KRAKEN'S DAUGHTER", "THE SOILED KNIGHT", "SAMWELL", "JAIME",\
        "THE IRON CAPTAIN", "THE DROWNED MAN", "THE QUEENMAKER", "ALAYNE", \
        "THE REAVER", "CAT OF THE CANALS", "THE PRINCESS IN THE TOWER", "EDDARD",\
        "EPILOGUE", "THE MERCHANT'S MAN"])

    files = glob.glob('../../soif_data/text/*.txt')
    text = ''
    for f in files:
        text += open(f).read()
        print('{0} complete'.format(f))
    print(extract_character_chapters('cersei', text))
