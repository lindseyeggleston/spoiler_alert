
def clean_text():
    # Edit book1
    with open('../data/SOIF/AGameOfThrones.txt') as g:
        text1 = g.read()
        text1 = replace_indentation(text1)
    new_text1 = text1[:20] + text1[892:1588846]
    with open('../data/text/book1.txt', 'w') as b1:
        b1.write(new_text1)
    # Edit book2
    with open('../data/SOIF/AClashOfKings.txt') as c:
        text2 = c.read()
        text2 = replace_indentation(text2)
    new_text2 = text2[:18] + text2[1070:1732670]
    with open('../data/text/book2.txt', 'w') as b2:
        b2.write(new_text2)
    # Edit book3
    with open('../data/SOIF/AStormOfSwords.txt') as s:
        text3 = s.read()
        text3 = replace_indentation(text3)
    new_text3 = text3[:20] + text3[2248:2245463]
    with open('../data/text/book3.txt', 'w') as b3:
        b3.write(new_text3)
    # Edit book4
    with open('../data/SOIF/AFeastForCrows.txt') as f:
        text4 = f.read()
        text4 = replace_indentation(text4)
    new_text4 = text4[:20] + text4[1371:1599998]
    with open('../data/text/book4.txt', 'w') as b4:
        b4.write(new_text4)
    # Edit book5
    with open('../data/SOIF/ADanceWithDragons.txt') as d:
        text5 = d.read()
        text5 = replace_indentation(text5)
    new_text5 = 'A DANCE WITH DRAGONS' + text5[4000:2450421]
    with open('../data/text/book5.txt', 'w') as b5:
        b5.write(new_text5)

def replace_indentation(text):
    new_text = ''
    for char in text:
        if ord(char)==9 or ord(char)==10:
            new_text += ' '
        else:
            new_text += char
    return new_text

if __name__ == '__main__':
    clean_text()
